# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hosts

- aionic: 192.168.1.98, SSH user `reeves`
- TrueNAS (NasGul): 192.168.1.149, SSH user `reeves`

## What this is

A self-hosted speech-to-text server that runs WhisperX (Whisper large-v3-turbo + wav2vec2 alignment + pyannote speaker diarization) as a FastAPI service in a Docker container. It runs CPU-only. The production instance is on NasGul (TrueNAS, 192.168.1.149, 2x Xeon, 48 cores). aionic (192.168.1.98) also runs it on CPU; its NVIDIA M40 is unused because Maxwell (sm_52) is not supported by the PyTorch/WhisperX diarization stack.

Primary use case: batch transcription of meeting recordings and audio files, with speaker labels.

## Image architecture

The build is split into three layered images, each published to ghcr.io/reevesl/stt_server:

- **:base** (Dockerfile.base) -- CUDA 12.1 + cuDNN 8 + Python + ffmpeg (CUDA is currently unused since inference runs on CPU)
- **:torch** (Dockerfile.torch) -- :base + PyTorch 2.5.1+cu121
- **:latest** (Dockerfile) -- :torch + whisperx + fastapi + app code

This layering exists so CI can cache torch separately (it's ~3GB and changes rarely). Changes to app code only rebuild the final layer.

## API

The server exposes three endpoints:

- `POST /transcribe` -- multipart form upload; optional `min_speakers`/`max_speakers` int fields; returns `{job_id, status}`
- `GET /jobs/{job_id}` -- poll for result; status is `queued`, `processing`, `done`, or `error`; done response includes `segments` (WhisperX format) and `text` (formatted speaker+timestamp lines)
- `GET /health` -- returns status and queue depth

Jobs are processed by a `ThreadPoolExecutor` sized by `MAX_WORKERS` (default 1). The in-memory `jobs` dict is not persisted across restarts.

## Environment

One required env var: `HF_TOKEN` -- a Hugging Face token with access to pyannote gated models. The pyannote models (speaker-diarization-3.1 and segmentation-3.0) must be accepted on HF before the token will work. Set via `.env` file (see `.env.example`).

Optional tuning vars:

- `WHISPER_MODEL` -- Whisper model name, default `large-v3-turbo`. Set to `large-v3` for maximum accuracy at several times the transcription cost.
- `OMP_NUM_THREADS` -- CPU threads per job, used both for PyTorch (alignment, diarization) and passed as `threads` to faster-whisper. Default 14. 24 crashed CTranslate2 4.8.3 with a segfault on NasGul; 18 worked but pushed core temps above 80C. `MAX_WORKERS x OMP_NUM_THREADS` should not exceed the core count.

The HuggingFace model cache is mounted as a named volume (`hf_cache`) so models survive container restarts.

## Running locally

```bash
cp .env.example .env  # fill in HF_TOKEN
docker compose up
```

The server listens on port 8001 (mapped from internal 8000).

## CI

GitHub Actions (`.github/workflows/docker-publish.yml`) builds and pushes all three images on every push to main. The three jobs run sequentially (base -> torch -> app) with registry-side build caches. Images are public on ghcr.io.

## Performance notes

Inference runs on CPU with `compute_type = "int8"`. Transcription is usually the slowest stage. faster-whisper defaults to 4 threads if `threads` is not passed to `whisperx.load_model`, regardless of `OMP_NUM_THREADS`, so the app passes it explicitly. The app calls `gc.collect()` between model load/unload steps (transcribe -> align -> diarize). Models are loaded fresh for every job.
