import asyncio
import functools
import gc
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
import whisperx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

HF_TOKEN = os.environ["HF_TOKEN"]
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

CPU_THREADS = int(os.environ.get("CPU_THREADS", 0)) or None
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 1))

if CPU_THREADS:
    torch.set_num_threads(CPU_THREADS)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs: dict[str, dict] = {}
job_queue: asyncio.Queue = None


def run_whisperx(audio_path: str, min_speakers: Optional[int], max_speakers: Optional[int]) -> dict:
    model_kwargs = {"compute_type": COMPUTE_TYPE}
    if CPU_THREADS:
        model_kwargs["num_workers"] = CPU_THREADS
    model = whisperx.load_model("large-v3", DEVICE, **model_kwargs)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    del model
    gc.collect()

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
    del model_a
    gc.collect()

    diarize_kwargs = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    diarize_model = whisperx.diarize.DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio, **diarize_kwargs)
    result = whisperx.diarize.assign_word_speakers(diarize_segments, result)

    return result


async def job_worker():
    loop = asyncio.get_event_loop()
    while True:
        job_id, audio_path, min_speakers, max_speakers = await job_queue.get()
        jobs[job_id]["status"] = "processing"
        try:
            fn = functools.partial(run_whisperx, audio_path, min_speakers, max_speakers)
            result = await loop.run_in_executor(executor, fn)

            lines = []
            for seg in result["segments"]:
                speaker = seg.get("speaker", "UNKNOWN")
                text = seg["text"].strip()
                start = seg["start"]
                lines.append(f"[{speaker} {start:.1f}s] {text}")

            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = {
                "segments": result["segments"],
                "text": "\n".join(lines),
            }
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global job_queue
    job_queue = asyncio.Queue()
    worker_task = asyncio.create_task(job_worker())
    yield
    worker_task.cancel()


app = FastAPI(title="STT Server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "queued": job_queue.qsize() if job_queue else 0}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    job_id = str(uuid.uuid4())
    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise HTTPException(status_code=500, detail="Failed to write uploaded file")

    jobs[job_id] = {"status": "queued", "filename": file.filename}
    await job_queue.put((job_id, tmp.name, min_speakers, max_speakers))

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs")
def list_jobs():
    return {jid: {"status": j["status"], "filename": j.get("filename")} for jid, j in jobs.items()}
