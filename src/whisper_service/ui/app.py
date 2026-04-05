"""Herald hosted web UI — Flask application.

Deployment strategy: hosted-centralized-ui

Provides a browser-accessible front-end for Herald's Whisper backend.
Users can record audio directly in the browser or upload a file; every
request is logged (timestamp, IP, transcription) to SQLite.

Configuration (env vars):
    WHISPER_BACKEND_URL   URL of the Herald FastAPI backend  [default: http://localhost:8000]
    HERALD_UI_PORT        Port this Flask app listens on     [default: 9090]
    HERALD_UI_HOST        Bind address                       [default: 0.0.0.0]

Run locally (Mac):
    uv pip install -e ".[ui]"
    WHISPER_BACKEND_URL=http://localhost:8000 whisper-ui

With HAProxy in front (production):
    HAProxy:9999 → this app on HERALD_UI_PORT → Herald FastAPI backend
"""

import io
import os

import requests
from flask import Flask, jsonify, render_template, request

from whisper_service.ui.logger import init_db, log_transcription, get_logs


def _get_client_ip() -> str:
    """Return the real client IP, unwrapping X-Forwarded-For from HAProxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def create_app() -> Flask:
    app = Flask(__name__)
    init_db()

    backend_url = os.getenv("WHISPER_BACKEND_URL", "http://localhost:8000").rstrip("/")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _backend_health() -> dict | None:
        try:
            r = requests.get(f"{backend_url}/health", timeout=3)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        health = _backend_health()
        return render_template("index.html", health=health, backend_url=backend_url)

    @app.route("/transcribe", methods=["POST"])
    def transcribe():
        client_ip = _get_client_ip()

        audio_file = request.files.get("file")
        if not audio_file:
            return jsonify({"error": "No audio file in request"}), 400

        model = request.form.get("model", "large-v3-turbo")
        language = request.form.get("language") or None
        filename = audio_file.filename or "recording.webm"
        audio_bytes = audio_file.read()
        file_size_kb = round(len(audio_bytes) / 1024, 1)

        try:
            resp = requests.post(
                f"{backend_url}/v1/audio/transcriptions",
                data={
                    "model": model,
                    "response_format": "verbose_json",
                    **({"language": language} if language else {}),
                },
                files={
                    "file": (
                        filename,
                        io.BytesIO(audio_bytes),
                        audio_file.content_type or "audio/webm",
                    )
                },
                timeout=180,
            )
            resp.raise_for_status()
            result = resp.json()

            log_transcription(
                client_ip=client_ip,
                filename=filename,
                file_size_kb=file_size_kb,
                model=model,
                language=result.get("language"),
                duration_s=result.get("duration"),
                processing_time_s=result.get("processing_time"),
                rtf=result.get("rtf"),
                transcription=result.get("text", ""),
            )

            return jsonify(result)

        except requests.exceptions.ConnectionError:
            msg = f"Cannot reach Herald backend at {backend_url}"
            log_transcription(
                client_ip=client_ip, filename=filename, file_size_kb=file_size_kb,
                model=model, error=msg,
            )
            return jsonify({"error": msg}), 503

        except requests.exceptions.HTTPError as exc:
            msg = exc.response.text
            log_transcription(
                client_ip=client_ip, filename=filename, file_size_kb=file_size_kb,
                model=model, error=msg,
            )
            return jsonify({"error": msg}), exc.response.status_code

        except Exception as exc:
            msg = str(exc)
            log_transcription(
                client_ip=client_ip, filename=filename, file_size_kb=file_size_kb,
                model=model, error=msg,
            )
            return jsonify({"error": msg}), 500

    @app.route("/logs")
    def logs():
        entries = get_logs()
        return render_template("logs.html", entries=entries)

    @app.route("/health")
    def health():
        """Lightweight health endpoint — used by HAProxy check."""
        return jsonify({"status": "ok"}), 200

    return app


def run():
    port = int(os.getenv("HERALD_UI_PORT", "9090"))
    host = os.getenv("HERALD_UI_HOST", "0.0.0.0")
    app = create_app()
    # threaded=True so slow transcription POSTs don't block the logs page
    app.run(host=host, port=port, threaded=True)
