# Herald

Internal speech-to-text service powered by open-source Whisper models. Auto-detects hardware: Apple Silicon (MLX) for development, NVIDIA GPUs (faster-whisper/CTranslate2) for production.

## Install — Mac (Apple Silicon)

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg
brew install ffmpeg

# Clone and install
cd ~/herald
uv venv && uv pip install -e ".[mlx,dev]"
source .venv/bin/activate
```

## Install — Linux (NVIDIA GPU)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart shell to pick up ~/.local/bin

# Install ffmpeg
sudo apt update && sudo apt install -y ffmpeg

# Verify CUDA is available
nvidia-smi

# Clone and install
cd ~/herald
uv venv && uv pip install -e ".[cuda,dev]"
source .venv/bin/activate
```

The backend is auto-detected at runtime. Or force it with `WHISPER_BACKEND=cuda`.

## CLI Usage

```bash
# Basic transcription (auto-detects backend)
whisper-transcribe recording.mp3

# Choose model
whisper-transcribe meeting.wav --model large-v3-turbo

# Output formats: text, json, verbose_json, srt, vtt
whisper-transcribe call.m4a --format srt --output subtitles.srt

# Domain prompt for better accuracy
whisper-transcribe standup.mp3 --language en --prompt "HPC, GPU, CUDA, Kubernetes"

# Translate non-English audio to English
whisper-transcribe japanese_meeting.wav --task translate

# Explicit CUDA backend with GPU selection
whisper-transcribe audio.wav --backend cuda --gpu 0 --compute-type float16
whisper-transcribe audio.wav --backend cuda --gpu 2 --compute-type int8_float16
```

## API Server

```bash
# Start (auto-detects backend)
whisper-serve

# NVIDIA GPU with specific config
WHISPER_BACKEND=cuda WHISPER_DEVICE_INDEX=0 WHISPER_COMPUTE_TYPE=float16 whisper-serve

# Custom port
WHISPER_PORT=9000 whisper-serve
```

### API Endpoints

OpenAI-compatible — drop-in replacement for `/v1/audio/transcriptions`.

```bash
# Transcribe
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@recording.mp3 \
  -F model=large-v3-turbo \
  -F response_format=verbose_json

# Translate to English
curl -X POST http://localhost:8000/v1/audio/translations \
  -F file=@foreign_audio.mp3

# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
```

Works with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("recording.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="large-v3-turbo",
        file=f,
        response_format="verbose_json",
    )
print(transcript.text)
```

## Architecture

```
src/whisper_service/
  api/app.py              # FastAPI server, OpenAI-compatible endpoints
  backends/
    base.py               # Abstract backend interface
    factory.py            # Auto-detect hardware, create backend
    mlx_backend.py        # MLX Whisper (Apple Silicon)
    cuda_backend.py       # faster-whisper (NVIDIA CUDA)
  preprocessing/
    audio.py              # FFmpeg conversion, resampling, normalization
  postprocessing/
    formatters.py         # JSON, text, SRT, VTT output formatting
  config.py               # Settings (env vars with WHISPER_ prefix)
  cli.py                  # CLI tool
  main.py                 # Server entry point
```

## Models

| Alias | Params | Best For |
|---|---|---|
| `large-v3` | 1.55B | Highest accuracy |
| `large-v3-turbo` | 809M | Balanced speed/quality (default) |
| `distil-large-v3` | 756M | Fastest, near-large quality |
| `small` | 244M | Quick testing |
| `base` | 74M | Rapid iteration |
| `tiny` | 39M | Smoke tests |

Models download automatically from HuggingFace on first use. MLX backend pulls MLX-optimized weights; CUDA backend pulls CTranslate2-optimized weights.

## Configuration

All settings via environment variables with `WHISPER_` prefix:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_BACKEND` | `auto` | Backend: `auto`, `mlx`, or `cuda` |
| `WHISPER_DEFAULT_MODEL` | `large-v3-turbo` | Model alias to load on startup |
| `WHISPER_HOST` | `0.0.0.0` | Server bind address |
| `WHISPER_PORT` | `8000` | Server port |
| `WHISPER_WORKERS` | `1` | Uvicorn workers |
| `WHISPER_DEVICE_INDEX` | `0` | GPU index (CUDA only) |
| `WHISPER_COMPUTE_TYPE` | `float16` | Quantization: `float16`, `int8_float16`, `int8` (CUDA only) |
| `WHISPER_MAX_FILE_SIZE_MB` | `500` | Max upload size |
| `WHISPER_DEFAULT_LANGUAGE` | (auto-detect) | Default language code |

## Benchmarks

| Hardware | Model | Audio | RTF | Notes |
|---|---|---|---|---|
| M5 Max 48GB | large-v3-turbo | 51s | 0.73 | First run, includes model download |
| A40 48GB | large-v3-turbo | TBD | ~0.03-0.05 expected | |

## Deployment on NVIDIA Server

```bash
# On the Supermicro 4x A40 server:
git clone <repo> ~/herald && cd ~/herald
uv venv && uv pip install -e ".[cuda]"
source .venv/bin/activate

# Quick test on GPU 0
whisper-transcribe test_audio.mp3 --backend cuda --gpu 0 --format verbose_json

# Test each GPU
for gpu in 0 1 2 3; do
  echo "=== GPU $gpu ==="
  whisper-transcribe test_audio.mp3 --backend cuda --gpu $gpu --format verbose_json
done

# Start API server on GPU 0
WHISPER_BACKEND=cuda WHISPER_DEVICE_INDEX=0 whisper-serve
```

## Roadmap

- [x] Phase 1: Local POC on Apple Silicon
- [x] Phase 2: NVIDIA GPU backend via faster-whisper/CTranslate2
