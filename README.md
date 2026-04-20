# stt_server

A self-hosted speech-to-text server with speaker diarization. Submit an audio file, get back a timestamped transcript with each speaker labeled.

Built on [WhisperX](https://github.com/m-bain/whisperX) (Whisper large-v3 + wav2vec2 alignment + pyannote diarization), served via FastAPI.

## Requirements

- Docker with Docker Compose
- A free [Hugging Face](https://huggingface.co) account and token (see below)

## Hugging Face setup

The speaker diarization models are gated on Hugging Face. Before the server will work you need to:

1. Create a free account at https://huggingface.co
2. Generate a token at https://huggingface.co/settings/tokens (read access is sufficient)
3. Accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1
4. Accept the terms at https://huggingface.co/pyannote/segmentation-3.0

The token is only used to download the models on first run. Once cached, it is not used again.

## Configuration

Edit `docker-compose.yml` before deploying:

| Variable | Description |
|---|---|
| `HF_TOKEN` | Your Hugging Face token |
| `OMP_NUM_THREADS` | CPU threads for Whisper and diarization. Set to roughly half your available cores to leave headroom for the OS and other services. |
| `MAX_WORKERS` | Number of concurrent transcription jobs. Each job uses `OMP_NUM_THREADS` threads, so `MAX_WORKERS x OMP_NUM_THREADS` should not exceed your core count. |

The Hugging Face model cache is stored in a named Docker volume (`hf_cache`) so models survive container restarts.

## Running

```bash
docker compose up -d
```

The server listens on port 8001.

## Usage

Submit an audio file:

```bash
curl -X POST http://your-server:8001/transcribe \
  -F "file=@/path/to/audio.m4a" \
  -F "min_speakers=2" \
  -F "max_speakers=2"
```

`min_speakers` and `max_speakers` are optional but improve diarization accuracy when you know how many speakers are present.

The response contains a `job_id`:

```json
{"job_id": "abc-123", "status": "queued"}
```

Poll for the result:

```bash
curl http://your-server:8001/jobs/abc-123
```

When complete, the response includes a `text` field with the formatted transcript:

```
[SPEAKER_00 0.5s] Hello, how are you?
[SPEAKER_01 3.2s] I'm doing well, thanks.
```

Other endpoints:

- `GET /health` - server status and queue depth
- `GET /jobs` - list all jobs and their statuses
