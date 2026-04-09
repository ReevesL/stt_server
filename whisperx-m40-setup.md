# Local STT + Speaker Diarization with WhisperX

## Context

This document summarizes a research conversation and provides an implementation spec for setting up local speech-to-text with speaker diarization on a TrueNAS home lab with an NVIDIA M40 24GB GPU.

---

## Background: STT + Diarization Options

### Recommended: WhisperX

WhisperX is the best all-in-one option for batch transcription with diarization. It combines three components:

- **faster-whisper** -- CTranslate2-optimized Whisper backend (70x realtime speed with large-v2/v3)
- **wav2vec2** -- for accurate word-level timestamp alignment
- **pyannote-audio** -- for speaker diarization (labels speakers as SPEAKER_00, SPEAKER_01, etc.)

GitHub: https://github.com/m-bain/whisperX

### Alternative: WhisperLiveKit

For real-time/streaming use cases rather than batch processing. Supports Sortformer (SOTA 2025) for real-time diarization. More setup complexity. Not needed unless live transcription is required.

GitHub: https://github.com/QuentinFuxa/WhisperLiveKit

### Diarization Backends

- **pyannote-audio** (recommended) -- efficient PyTorch-based, works well with 24GB VRAM
- **NVIDIA Sortformer** -- more accurate for overlapping speech, higher compute cost, requires NeMo framework

---

## Compute Load Context

From lightest to heaviest GPU workload:

1. STT alone (small models, bounded 30-second input windows)
2. STT + diarization (two model passes, still well below LLM)
3. LLM inference (scales with context length, autoregressive)
4. Image generation
5. Video generation (heaviest)

The M40 24GB handles WhisperX large-v3 + pyannote diarization comfortably. Run sequentially, not concurrently with LLMs.

### Why audio data is lightweight

Whisper processes audio in 30-second windows regardless of file length. After mel spectrogram conversion, a 30-second chunk at 16kHz (480,000 samples) becomes an 80x3000 matrix. Compare that to LLM inference operating on tensors shaped by model dimension (e.g., 4096x4096+ for a 7B model) with autoregressive token generation.

---

## Hugging Face Token Note

The pyannote diarization models are gated on Hugging Face. A free HF token is required to download them because you must accept pyannote's terms of use on the model card page. Once downloaded, the token is not used again -- all inference runs locally.

Model cards requiring acceptance:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

---

## Implementation Spec

### Target Environment

- Host: TrueNAS (192.168.1.149)
- GPU: NVIDIA M40 24GB
- Deployment: Docker container
- Use case: Batch transcription of audio files with speaker labels

### Recommended Model

`whisper large-v3` -- best accuracy, fits comfortably in 24GB VRAM with pyannote running alongside it.

Use `compute_type = "float16"` for speed. Fall back to `"int8"` if VRAM issues arise.

### Docker Setup

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install whisperx
```

### Python Usage

```python
import whisperx
import gc

device = "cuda"
audio_file = "audio.mp3"
batch_size = 16        # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem

# Step 1: Transcribe
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

# Free GPU memory before alignment
del model
gc.collect()

# Step 2: Word-level alignment
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
result = whisperx.align(
    result["segments"], model_a, metadata, audio, device,
    return_char_alignments=False
)

del model_a
gc.collect()

# Step 3: Speaker diarization
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token="YOUR_HF_TOKEN", device=device
)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

# Output
for segment in result["segments"]:
    print(f"[{segment['speaker']}] {segment['text'].strip()}")
```

### CLI Usage

```bash
# Basic transcription with diarization
whisperx audio.wav --model large-v3 --diarize --hf_token YOUR_HF_TOKEN

# If speaker count is known (improves accuracy)
whisperx audio.wav --model large-v3 --diarize --min_speakers 2 --max_speakers 2 --hf_token YOUR_HF_TOKEN

# CPU fallback (slower, no VRAM needed)
whisperx audio.wav --model large-v3 --compute_type int8 --device cpu
```

### Docker Compose (suggested)

```yaml
services:
  whisperx:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - /path/to/audio:/audio
      - /path/to/output:/output
      - huggingface_cache:/root/.cache/huggingface
    command: >
      whisperx /audio/input.mp3
      --model large-v3
      --diarize
      --hf_token ${HF_TOKEN}
      --output_dir /output

volumes:
  huggingface_cache:
```

Store your HF token in a `.env` file:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### Pre-flight Checklist

1. Create a free account at https://huggingface.co
2. Generate a token at https://huggingface.co/settings/tokens (read access is sufficient)
3. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
4. Accept terms at https://huggingface.co/pyannote/segmentation-3.0
5. Confirm NVIDIA container toolkit is installed on TrueNAS host
6. Confirm `nvidia-smi` is accessible from within containers

### Expected VRAM Usage

| Component | Approx VRAM |
|---|---|
| Whisper large-v3 (float16) | ~6-8 GB |
| pyannote diarization | ~2-3 GB |
| wav2vec2 alignment | ~1-2 GB |
| **Total (with gc between steps)** | **~8 GB peak** |

The M40's 24GB provides comfortable headroom. The gc.collect() calls in the Python example above free each model before loading the next, keeping peak usage well under the limit.
