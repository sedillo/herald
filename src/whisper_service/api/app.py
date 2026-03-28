"""FastAPI application with OpenAI-compatible transcription API."""

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from whisper_service.backends import create_backend, detect_backend
from whisper_service.config import OutputFormat, settings
from whisper_service.postprocessing.formatters import format_output
from whisper_service.preprocessing.audio import preprocess_audio

logger = logging.getLogger(__name__)

# Global backend instance — created at startup based on hardware detection
backend = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Detect hardware, create backend, load model on startup."""
    global backend

    backend_type = settings.backend if settings.backend != "auto" else None
    detected = backend_type or detect_backend()

    logger.info(f"Backend: {detected} | Model: {settings.default_model}")

    if detected == "cuda":
        logger.info(
            f"CUDA config: GPU:{settings.device_index} "
            f"compute_type:{settings.compute_type}"
        )

    backend = create_backend(
        backend_type=backend_type,
        device_index=settings.device_index,
        compute_type=settings.compute_type,
    )

    await backend.load_model(settings.default_model)
    logger.info(f"Ready: {backend.model_name()} on {detected}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Whisper STT Service",
    description="Internal speech-to-text API, OpenAI-compatible.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": backend.model_name() if backend else None,
        "model_loaded": backend.is_loaded() if backend else False,
        "backend": settings.backend,
        "device_index": settings.device_index,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    # Common aliases available across all backends
    aliases = {
        "large-v3": "Whisper Large V3 (1.55B params, highest accuracy)",
        "large-v3-turbo": "Whisper Large V3 Turbo (809M params, balanced)",
        "distil-large-v3": "Distil Whisper Large V3 (756M params, fastest)",
        "small": "Whisper Small (244M params)",
        "base": "Whisper Base (74M params)",
        "tiny": "Whisper Tiny (39M params)",
    }

    models = []
    for alias, description in aliases.items():
        models.append({
            "id": alias,
            "object": "model",
            "owned_by": "internal",
            "description": description,
        })
    return {"object": "list", "data": models}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="large-v3-turbo", description="Model name or alias"),
    language: str | None = Form(default=None, description="ISO 639-1 language code"),
    prompt: str | None = Form(default=None, description="Optional prompt for domain vocabulary"),
    response_format: str = Form(default="json", description="Output format: json, verbose_json, text, srt, vtt"),
    temperature: float = Form(default=0.0, description="Sampling temperature (0 = greedy)"),
):
    """Transcribe audio file. OpenAI API-compatible endpoint.

    Drop-in replacement for OpenAI's /v1/audio/transcriptions.
    """
    if not backend or not backend.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate format
    try:
        fmt = OutputFormat(response_format)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format: {response_format}. "
            f"Must be one of: {[f.value for f in OutputFormat]}",
        )

    # Validate file extension
    suffix = Path(file.filename or "audio.wav").suffix.lstrip(".")
    if suffix not in settings.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: .{suffix}. "
            f"Supported: {settings.supported_formats}",
        )

    # Save uploaded file to temp directory
    with tempfile.TemporaryDirectory(prefix="whisper_upload_") as tmp_dir:
        tmp_path = Path(tmp_dir) / (file.filename or "upload.wav")
        content = await file.read()

        # Check file size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {size_mb:.1f}MB. Max: {settings.max_file_size_mb}MB",
            )

        tmp_path.write_bytes(content)

        # Preprocess audio (convert to 16kHz mono WAV)
        try:
            processed_path = preprocess_audio(tmp_path, output_dir=Path(tmp_dir))
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=f"Audio preprocessing failed: {e}")

        # Transcribe
        try:
            result = await backend.transcribe(
                audio_path=processed_path,
                language=language,
                task="transcribe",
                word_timestamps=(fmt == OutputFormat.VERBOSE_JSON),
                initial_prompt=prompt,
            )
        except Exception as e:
            logger.exception("Transcription failed")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        # Format output
        output = format_output(result, fmt)

        if isinstance(output, dict):
            return JSONResponse(content=output)
        else:
            return PlainTextResponse(content=output)


@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(..., description="Audio file to translate to English"),
    model: str = Form(default="large-v3-turbo", description="Model name or alias"),
    prompt: str | None = Form(default=None, description="Optional prompt"),
    response_format: str = Form(default="json", description="Output format"),
):
    """Translate audio to English. OpenAI API-compatible endpoint."""
    if not backend or not backend.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        fmt = OutputFormat(response_format)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid response_format: {response_format}")

    with tempfile.TemporaryDirectory(prefix="whisper_upload_") as tmp_dir:
        tmp_path = Path(tmp_dir) / (file.filename or "upload.wav")
        content = await file.read()
        tmp_path.write_bytes(content)

        try:
            processed_path = preprocess_audio(tmp_path, output_dir=Path(tmp_dir))
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=f"Audio preprocessing failed: {e}")

        result = await backend.transcribe(
            audio_path=processed_path,
            task="translate",
            initial_prompt=prompt,
        )

        output = format_output(result, fmt)
        if isinstance(output, dict):
            return JSONResponse(content=output)
        else:
            return PlainTextResponse(content=output)
