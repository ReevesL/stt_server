import asyncio
import functools
import gc
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import torch
import whisperx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

HF_TOKEN = os.environ["HF_TOKEN"]
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

_omp_threads = int(os.environ.get("OMP_NUM_THREADS", 0))
if _omp_threads:
    torch.set_num_threads(_omp_threads)

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 1))
ARCHIVE_PATH = os.environ.get("ARCHIVE_PATH")
LOCAL_TRANSCRIPT_PATH = Path(os.environ.get("LOCAL_TRANSCRIPT_PATH", "/app/transcripts"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", 2))

LOCAL_TRANSCRIPT_PATH.mkdir(parents=True, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs: dict[str, dict] = {}
job_queue: asyncio.Queue = None


def build_transcript_text(segments: list) -> str:
    speakers = sorted({seg.get("speaker", "UNKNOWN") for seg in segments if seg.get("speaker")})
    lines = ["UNMAPPED SPEAKERS"]
    for sp in speakers:
        lines.append(f"  {sp}: ")
    lines.append("")
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg["text"].strip()
        start = seg["start"]
        lines.append(f"[{speaker} {start:.1f}s] {text}")
    return "\n".join(lines)


def write_job_files(job_id: str, stem: str, transcript_text: str, segments: list, log_data: dict) -> Path:
    job_dir = LOCAL_TRANSCRIPT_PATH / job_id
    job_dir.mkdir(exist_ok=True)

    txt_file = job_dir / f"{stem}.txt"
    seg_file = job_dir / f"{stem}_segments.json"
    log_file = job_dir / f"{stem}.log"

    txt_file.write_text(transcript_text, encoding="utf-8")
    seg_file.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    log_file.write_text("\n".join(f"{k}: {v}" for k, v in log_data.items()), encoding="utf-8")

    zip_path = job_dir / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_file, txt_file.name)
        zf.write(seg_file, seg_file.name)
        zf.write(log_file, log_file.name)

    if ARCHIVE_PATH:
        archive_dir = Path(ARCHIVE_PATH)
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, archive_dir / zip_path.name)

    return job_dir


def run_whisperx(audio_path: str, min_speakers: Optional[int], max_speakers: Optional[int]) -> dict:
    start_time = time.time()

    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE)
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

    result["_processing_seconds"] = time.time() - start_time
    return result


def cleanup_old_transcripts():
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    for job_dir in LOCAL_TRANSCRIPT_PATH.iterdir():
        if not job_dir.is_dir():
            continue
        mtime = datetime.utcfromtimestamp(job_dir.stat().st_mtime)
        if mtime < cutoff:
            shutil.rmtree(job_dir, ignore_errors=True)


async def cleanup_loop():
    while True:
        await asyncio.sleep(6 * 60 * 60)
        await asyncio.get_event_loop().run_in_executor(None, cleanup_old_transcripts)


async def job_worker():
    loop = asyncio.get_event_loop()
    while True:
        job_id, audio_path, original_filename, min_speakers, max_speakers = await job_queue.get()
        jobs[job_id]["status"] = "processing"
        started_at = datetime.utcnow().isoformat()
        try:
            fn = functools.partial(run_whisperx, audio_path, min_speakers, max_speakers)
            result = await loop.run_in_executor(executor, fn)

            elapsed = result.pop("_processing_seconds", 0)
            transcript_text = build_transcript_text(result["segments"])
            stem = Path(original_filename).stem if original_filename else job_id
            speakers_detected = len({seg.get("speaker") for seg in result["segments"] if seg.get("speaker")})

            log_data = {
                "job_id": job_id,
                "filename": original_filename,
                "started_at": started_at,
                "completed_at": datetime.utcnow().isoformat(),
                "processing_time": f"{elapsed:.1f}s",
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
                "speakers_detected": speakers_detected,
                "segments": len(result["segments"]),
            }

            job_dir = await loop.run_in_executor(
                None, write_job_files, job_id, stem, transcript_text, result["segments"], log_data
            )

            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = {
                "text": transcript_text,
                "processing_time": f"{elapsed:.1f}s",
                "speakers_detected": speakers_detected,
                "files": str(job_dir),
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
    cleanup_old_transcripts()
    worker_task = asyncio.create_task(job_worker())
    cleanup_task = asyncio.create_task(cleanup_loop())
    yield
    worker_task.cancel()
    cleanup_task.cancel()


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
    await job_queue.put((job_id, tmp.name, file.filename, min_speakers, max_speakers))

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs")
def list_jobs():
    return {jid: {"status": j["status"], "filename": j.get("filename")} for jid, j in jobs.items()}


class SpeakerMapping(BaseModel):
    mapping: dict[str, str]


@app.post("/jobs/{job_id}/apply-mapping")
def apply_mapping(job_id: str, body: SpeakerMapping):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] != "done":
        raise HTTPException(status_code=400, detail="Job is not complete")

    job_dir = LOCAL_TRANSCRIPT_PATH / job_id
    txt_files = list(job_dir.glob("*.txt"))
    if not txt_files:
        raise HTTPException(status_code=404, detail="Transcript file not found")

    txt_path = txt_files[0]
    stem = txt_path.stem
    text = txt_path.read_text(encoding="utf-8")

    for speaker_id, name in body.mapping.items():
        text = text.replace(speaker_id, name)

    txt_path.write_text(text, encoding="utf-8")

    seg_file = job_dir / f"{stem}_segments.json"
    log_file = job_dir / f"{stem}.log"
    zip_path = job_dir / f"{stem}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, txt_path.name)
        if seg_file.exists():
            zf.write(seg_file, seg_file.name)
        if log_file.exists():
            zf.write(log_file, log_file.name)

    if ARCHIVE_PATH:
        shutil.copy2(zip_path, Path(ARCHIVE_PATH) / zip_path.name)

    return {"status": "ok", "applied": body.mapping}
