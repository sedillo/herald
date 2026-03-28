# Herald

Internal speech-to-text service. Auto-detects hardware: Apple Silicon (MLX) or NVIDIA GPU (faster-whisper/CTranslate2).

## Install — Mac (Apple Silicon)

```bash
# Install uv and ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install ffmpeg

# Install herald
cd ~/herald
uv venv && uv pip install -e ".[mlx,dev]"
source .venv/bin/activate

# Test it
whisper-transcribe examples/long.mp3
```

## Install — Linux (NVIDIA GPU)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install system deps
sudo apt update && sudo apt install -y ffmpeg

# Verify GPU
nvidia-smi

# Install herald
cd ~/herald
uv venv && uv pip install -e ".[cuda,dev]"
source .venv/bin/activate

# Test it
whisper-transcribe examples/long.mp3
```

## CLI

```bash
# Transcribe (auto-detects backend)
whisper-transcribe examples/long.mp3

# Choose model: large-v3 | large-v3-turbo (default) | distil-large-v3
whisper-transcribe examples/long.mp3 --model large-v3

# Output formats: text (default) | json | verbose_json | srt | vtt
whisper-transcribe examples/long.mp3 --format srt --output subtitles.srt

# Domain prompt for accuracy on technical terms
whisper-transcribe meeting.mp3 --prompt "HPC, GPU, CUDA, Kubernetes"

# Pick a specific GPU
whisper-transcribe examples/long.mp3 --gpu 2

# Translate non-English audio to English
whisper-transcribe foreign.mp3 --task translate
```

## API Server

```bash
# Start (auto-detects backend)
whisper-serve

# With specific GPU
WHISPER_DEVICE_INDEX=0 whisper-serve

# Custom port
WHISPER_PORT=9000 whisper-serve
```

OpenAI-compatible endpoints:

```bash
# Transcribe
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@examples/long.mp3 \
  -F response_format=verbose_json

# Health check
curl http://localhost:8000/health
```

Works with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("examples/long.mp3", "rb") as f:
    result = client.audio.transcriptions.create(model="large-v3-turbo", file=f)
print(result.text)
```

## Models

| Alias | Params | Use Case |
|---|---|---|
| `large-v3` | 1.55B | Highest accuracy |
| `large-v3-turbo` | 809M | Balanced (default) |
| `distil-large-v3` | 756M | Fastest |

Downloaded automatically from HuggingFace on first use.

## Configuration

Environment variables with `WHISPER_` prefix:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_BACKEND` | `auto` | `auto`, `mlx`, or `cuda` |
| `WHISPER_DEFAULT_MODEL` | `large-v3-turbo` | Model to load |
| `WHISPER_PORT` | `8000` | Server port |
| `WHISPER_DEVICE_INDEX` | `0` | GPU index (CUDA) |
| `WHISPER_COMPUTE_TYPE` | `float16` | `float16`, `int8_float16`, `int8` (CUDA) |

## Benchmarks

| Hardware | Model | Audio | RTF | Speed |
|---|---|---|---|---|
| M5 Max 48GB | large-v3-turbo | 51s | 0.73 | ~1.4x real-time |
| A40 48GB | large-v3-turbo | 51s | TBD | ~20-30x real-time expected |

## Architecture

```
src/whisper_service/
  api/app.py              # FastAPI, OpenAI-compatible endpoints
  backends/
    factory.py            # Auto-detect hardware, create backend
    mlx_backend.py        # Apple Silicon (MLX)
    cuda_backend.py       # NVIDIA GPU (faster-whisper)
  preprocessing/audio.py  # FFmpeg conversion, resampling
  postprocessing/         # JSON, text, SRT, VTT formatters
  cli.py                  # CLI tool
  main.py                 # Server entry point
```

## Roadmap

- [x] Phase 1: Local POC on Apple Silicon
- [x] Phase 2: NVIDIA GPU backend via faster-whisper/CTranslate2
- [ ] Phase 3: Integration with internal inference gateway
- [ ] Phase 4: Streaming transcription (WebSocket)
- [ ] Phase 5: LoRA fine-tuning for domain vocabulary
- [ ] Phase 6: Production deployment (containers, CI/CD, monitoring)
