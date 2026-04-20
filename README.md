# stt_server

A self-hosted speech-to-text server with speaker diarization. Submit an audio file, get back a timestamped transcript with each speaker labeled. Enrolled speakers are automatically identified by name.

Built on [WhisperX](https://github.com/m-bain/whisperX) (Whisper large-v3 + wav2vec2 alignment + pyannote diarization), served via FastAPI.

## Requirements

- Docker with Docker Compose
- A free [Hugging Face](https://huggingface.co) account and token (see below)

## Hugging Face setup

The speaker diarization and enrollment models are gated on Hugging Face. Before the server will work you need to:

1. Create a free account at https://huggingface.co
2. Generate a token at https://huggingface.co/settings/tokens (read access is sufficient)
3. Accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1
4. Accept the terms at https://huggingface.co/pyannote/segmentation-3.0
5. Accept the terms at https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM (required for voice enrollment)

The token is only used to download the models on first run. Once cached, it is not used again.

## Configuration

Edit `docker-compose.yml` before deploying. All settings live in the `environment` section:

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | required | Your Hugging Face token |
| `OMP_NUM_THREADS` | all cores | CPU threads for Whisper and diarization. Set to roughly half your available cores to leave headroom for other services. |
| `MAX_WORKERS` | `1` | Number of concurrent transcription jobs. `MAX_WORKERS x OMP_NUM_THREADS` should not exceed your core count. |
| `ARCHIVE_PATH` | none | Path inside the container where completed ZIP archives are delivered. Mount your NAS share to this path (see below). |
| `RETENTION_DAYS` | `2` | How many days to keep local transcript files before automatic cleanup. NAS archives are not affected. |
| `ENROLLMENT_THRESHOLD` | `0.75` | Cosine similarity threshold for automatic speaker identification. Lower values match more loosely. |

### NAS output delivery

To deliver completed transcript archives to a NAS share, add a bind mount and set `ARCHIVE_PATH`:

```yaml
environment:
  - ARCHIVE_PATH=/mnt/stt_files

volumes:
  - /path/on/nas:/mnt/stt_files
```

After each job, a ZIP file is written to that directory. The app user must have write access to the NAS path.

### Local file retention

Local transcript files are stored in a Docker volume (`transcripts`) and cleaned up automatically every 6 hours. Files older than `RETENTION_DAYS` days are deleted. NAS archives are never automatically deleted.

## Running

```bash
docker compose up -d
```

The server listens on port 8001.

## Output files

Each completed transcription job produces a ZIP archive containing three files, named after the original audio file:

| File | Contents |
|---|---|
| `filename.txt` | Formatted transcript with speaker labels and timestamps. Starts with an `UNMAPPED SPEAKERS` block listing any unidentified speaker IDs. |
| `filename_segments.json` | Raw WhisperX segments: start/end times, text, speaker label, and word-level timestamps with confidence scores. |
| `filename.log` | Job metadata: original filename, start/end times, processing duration, device settings, speaker count. |

## Transcription

Submit an audio file:

```bash
curl -X POST http://your-server:8001/transcribe \
  -F "file=@/path/to/audio.m4a" \
  -F "min_speakers=2" \
  -F "max_speakers=2"
```

`min_speakers` and `max_speakers` are optional but improve diarization accuracy when you know how many speakers are present. The response contains a `job_id`:

```json
{"job_id": "abc-123", "status": "queued"}
```

Poll for the result:

```bash
curl http://your-server:8001/jobs/abc-123
```

When complete, `status` will be `done` and the response includes the transcript text, processing time, and a `speakers_identified` map showing any enrolled speakers that were automatically matched.

Other job endpoints:

- `GET /jobs` - list all jobs and their statuses
- `GET /health` - server status and queue depth

## Speaker name mapping

If a speaker was not automatically identified, the transcript `.txt` file starts with an `UNMAPPED SPEAKERS` block:

```
UNMAPPED SPEAKERS
  SPEAKER_00: 
  SPEAKER_01: 

[SPEAKER_00 0.5s] Hello, how are you?
...
```

To apply names after the fact, POST a mapping to the job:

```bash
curl -X POST http://your-server:8001/jobs/abc-123/apply-mapping \
  -H "Content-Type: application/json" \
  -d '{"mapping": {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}}'
```

This updates the `.txt` file in place, regenerates the ZIP, and re-copies it to the NAS archive path.

## Voice enrollment

Enroll a speaker so they are automatically identified in future transcriptions. Use 30-60 seconds of clean, single-speaker audio with no background noise or music.

```bash
curl -X POST http://your-server:8001/enroll \
  -F "file=@/path/to/my-voice.m4a" \
  -F "name=Reeves"
```

List enrolled speakers:

```bash
curl http://your-server:8001/speakers
```

Remove an enrolled speaker:

```bash
curl -X DELETE http://your-server:8001/speakers/Reeves
```

Enrolled speaker data (voice embeddings) is stored in `speakers.json` at the `ARCHIVE_PATH` location so enrollments survive container restarts. The similarity threshold for a match defaults to `0.75` and can be tuned via `ENROLLMENT_THRESHOLD` in `docker-compose.yml`.

Note: voice enrollment requires accepting the terms for `pyannote/wespeaker-voxceleb-resnet34-LM` on Hugging Face (see setup above). The model is downloaded on first use.
