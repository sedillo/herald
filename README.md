# Herald

Internal speech-to-text service. Auto-detects hardware: Apple Silicon (MLX) or NVIDIA GPU (faster-whisper/CTranslate2).

---

## Deploy — A40 Linux Server (UI + CUDA)

This is the primary deployment target. Runs two processes on the same host: the FastAPI inference backend and the Flask web UI. HAProxy sits in front and exposes a single port to the network.

### 1. System dependencies

```bash
sudo apt update && sudo apt install -y ffmpeg haproxy
```

Verify the GPU is visible:

```bash
nvidia-smi
```

### 2. Clone and install

```bash
git clone <repo-url> ~/herald
cd ~/herald

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# Create venv and install CUDA + UI extras
uv venv && uv pip install -e ".[cuda,ui]"
source .venv/bin/activate
```

### 3. Start the inference backend

```bash
WHISPER_DEVICE_INDEX=0 whisper-serve
# Listening on http://localhost:8000 — internal only, not exposed to network
```

### 4. Start the Flask web UI

Open a second terminal:

```bash
cd ~/herald && source .venv/bin/activate
WHISPER_BACKEND_URL=http://localhost:8000 whisper-ui
# Listening on http://localhost:9090 — internal only
```

### 5. Configure HAProxy

A ready-to-use HAProxy stanza is at `deploy/haproxy-herald.cfg`. Edit it first to bind on port **80** so DNS routes cleanly without a port number:

```bash
# Edit the stanza — change bind *:9999 to bind *:80
sudo nano deploy/haproxy-herald.cfg
```

Then append and reload:

```bash
# Append the herald stanza to your existing haproxy config
sudo cat deploy/haproxy-herald.cfg >> /etc/haproxy/haproxy.cfg

# Validate the config
sudo haproxy -c -f /etc/haproxy/haproxy.cfg

# Reload HAProxy (zero-downtime)
sudo systemctl reload haproxy
```

If HAProxy isn't running yet:

```bash
sudo systemctl enable haproxy
sudo systemctl start haproxy
```

### 6. DNS

DNS resolves hostnames to IPs only — port is handled by HAProxy on port 80. Pick whichever DNS method fits your network:

**Internal DNS (Pi-hole, CoreDNS, corporate DNS):**
```
herald.internal  A  <a40-ip>
```

**No internal DNS — `/etc/hosts` on each client machine:**
```bash
echo "<a40-ip>  herald.internal" | sudo tee -a /etc/hosts
```

**Public domain:**
Add an A record in your registrar/Cloudflare:
```
herald.yourdomain.com  A  <a40-public-ip>
```
Then open port 80 in the firewall:
```bash
sudo ufw allow 80
```

Access the UI at `http://herald.internal` (or whatever hostname you chose).

> **Mic access:** Chrome blocks mic on plain HTTP for non-localhost origins. Easiest fix: open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add your URL (e.g. `http://herald.internal`), relaunch Chrome. For a permanent fix, see HTTPS below.

### 7. HTTPS / TLS (recommended for mic access)

SSL terminates at HAProxy. The two services (Flask UI, FastAPI backend) stay on plain HTTP internally — only the inbound connection gets TLS.

#### Option A — Internal CA cert (best for team use)

If your network has a corporate CA, request a cert for your hostname. Everyone on the network already trusts that CA — no browser warnings.

```bash
# You'll receive cert.pem + key.pem (or a .p12 to convert)
# HAProxy wants them concatenated into a single .pem
cat cert.pem key.pem > /etc/haproxy/herald.pem
chmod 600 /etc/haproxy/herald.pem
```

#### Option B — Self-signed cert (single machine or quick setup)

```bash
openssl req -x509 -newkey rsa:4096 -days 825 -nodes \
  -keyout /etc/haproxy/herald-key.pem \
  -out    /etc/haproxy/herald-cert.pem \
  -subj "/CN=herald.internal" \
  -addext "subjectAltName=DNS:herald.internal,IP:<a40-ip>"

# Concatenate for HAProxy
cat /etc/haproxy/herald-cert.pem /etc/haproxy/herald-key.pem > /etc/haproxy/herald.pem
chmod 600 /etc/haproxy/herald.pem
```

Browser will show an untrusted cert warning — click **Advanced → Proceed**. Once accepted, mic works with no Chrome flags needed.

To suppress the warning on all team machines, distribute and trust `herald-cert.pem`:
```bash
# On each client (Ubuntu/Debian)
sudo cp herald-cert.pem /usr/local/share/ca-certificates/herald.crt
sudo update-ca-certificates
```

#### HAProxy config for HTTPS

In `/etc/haproxy/haproxy.cfg`, update the herald frontend bind line:

```
frontend herald_front
    bind *:443 ssl crt /etc/haproxy/herald.pem
    # Optionally redirect HTTP → HTTPS
    bind *:80
    redirect scheme https if !{ ssl_fc }
    ...
```

```bash
sudo haproxy -c -f /etc/haproxy/haproxy.cfg && sudo systemctl reload haproxy
sudo ufw allow 443
```

Access at `https://herald.internal`. Mic works natively, no Chrome flags needed.

### 8. Run as a service (optional)

To survive reboots, create systemd units for both processes:

```bash
# /etc/systemd/system/herald-api.service
[Unit]
Description=Herald Whisper API
After=network.target

[Service]
User=<your-user>
WorkingDirectory=/home/<your-user>/herald
Environment=WHISPER_DEVICE_INDEX=0
ExecStart=/home/<your-user>/herald/.venv/bin/whisper-serve
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
# /etc/systemd/system/herald-ui.service
[Unit]
Description=Herald Web UI
After=herald-api.service

[Service]
User=<your-user>
WorkingDirectory=/home/<your-user>/herald
Environment=WHISPER_BACKEND_URL=http://localhost:8000
ExecStart=/home/<your-user>/herald/.venv/bin/whisper-ui
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable herald-api herald-ui
sudo systemctl start herald-api herald-ui
```

---

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

## Install — Mac Client + Linux Server (Typical Setup)

The most common workflow: **inference runs on the A40 Linux server**, **demo UI runs on your Mac**.

### Step 1 — Linux server (A40)

```bash
# Clone and install
git clone <repo-url> ~/herald
cd ~/herald

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
sudo apt update && sudo apt install -y ffmpeg

uv venv && uv pip install -e ".[cuda]"
source .venv/bin/activate

# Verify GPU
nvidia-smi

# Start the API server (binds to all interfaces)
WHISPER_DEVICE_INDEX=0 whisper-serve
# Server is up at http://0.0.0.0:8000
```

### Step 2 — Mac (demo UI only)

```bash
# Clone and install — UI only, no inference on Mac
git clone <repo-url> ~/herald
cd ~/herald

curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv && uv pip install -e ".[demo]"
source .venv/bin/activate

# Point at the A40 server and launch the demo
WHISPER_BACKEND_URL=http://<a40-host>:8000 streamlit run demo/app.py
```

> **Network**: the Mac and A40 must be on the same network (or use SSH tunnel).
> SSH tunnel if needed: `ssh -L 8000:localhost:8000 user@a40-host`
> Then use `WHISPER_BACKEND_URL=http://localhost:8000`.

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

## Demo UI

Streamlit frontend for live demos — mic recording or file upload, displays RTF front-and-center.

```bash
# Install demo extras (on Mac, alongside mlx)
uv pip install -e ".[mlx,demo]"

# Run — point at A40 backend
WHISPER_BACKEND_URL=http://<a40-host>:8000 streamlit run demo/app.py

# Or leave URL blank and set it in the sidebar at runtime
streamlit run demo/app.py
```

Sidebar shows live backend health (model, backend type, GPU index). Two tabs:
- **Record** — mic capture via `st.audio_input` → transcribe
- **Upload** — drag-and-drop audio file → transcribe

Key demo metric: **RTF** displayed after each transcription. RTF 0.022 = 51.5s audio in 1.15s on A40 (45x real-time).

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
