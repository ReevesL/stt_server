# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-hosted speech-to-text server that runs WhisperX (Whisper large-v3 + wav2vec2 alignment + pyannote speaker diarization) as a FastAPI service in a GPU-accelerated Docker container. The target deployment host is aionic (aionic.little.local / 192.168.1.98), an Ubuntu server with an NVIDIA M40 24GB GPU.

Primary use case: batch transcription of meeting recordings and audio files, with speaker labels.

## Image architecture

The build is split into three layered images, each published to ghcr.io/reevesl/stt_server:

- **:base** (Dockerfile.base) -- CUDA 12.1 + cuDNN 8 + Python + ffmpeg
- **:torch** (Dockerfile.torch) -- :base + PyTorch 2.5.1+cu121
- **:latest** (Dockerfile) -- :torch + whisperx + fastapi + app code

This layering exists so CI can cache torch separately (it's ~3GB and changes rarely). Changes to app code only rebuild the final layer.

## API

The server exposes three endpoints:

- `POST /transcribe` -- multipart form upload; optional `min_speakers`/`max_speakers` int fields; returns `{job_id, status}`
- `GET /jobs/{job_id}` -- poll for result; status is `queued`, `processing`, `done`, or `error`; done response includes `segments` (WhisperX format) and `text` (formatted speaker+timestamp lines)
- `GET /health` -- returns status and queue depth

Jobs are processed one at a time via a single `ThreadPoolExecutor` worker. The in-memory `jobs` dict is not persisted across restarts.

## Environment

One required env var: `HF_TOKEN` -- a Hugging Face token with access to pyannote gated models. The pyannote models (speaker-diarization-3.1 and segmentation-3.0) must be accepted on HF before the token will work. Set via `.env` file (see `.env.example`).

The HuggingFace model cache is mounted as a named volume (`hf_cache`) so models survive container restarts.

## Running locally (with GPU)

```bash
cp .env.example .env  # fill in HF_TOKEN
docker compose up
```

The server listens on port 8001 (mapped from internal 8000).

## CI

GitHub Actions (`.github/workflows/docker-publish.yml`) builds and pushes all three images on every push to main. The three jobs run sequentially (base -> torch -> app) with registry-side build caches. Images are public on ghcr.io.

## VRAM notes

Peak usage is ~8GB on M40 24GB -- well within headroom. The app calls `gc.collect()` between model load/unload steps (transcribe -> align -> diarize) to keep peak usage down. `compute_type = "float32"` is used -- the M40 (Maxwell, compute capability 5.2) does not support efficient float16 or int8 computation.
