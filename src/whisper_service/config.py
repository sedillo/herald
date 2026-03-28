"""Service configuration."""

from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings


class ModelTier(str, Enum):
    QUALITY = "large-v3"
    BALANCED = "large-v3-turbo"
    SPEED = "distil-large-v3"


class OutputFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    VERBOSE_JSON = "verbose_json"


class Settings(BaseSettings):
    """Service settings, configurable via environment variables."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Backend: "auto", "mlx", or "cuda"
    backend: str = "auto"

    # Model — alias like "large-v3-turbo" or full HuggingFace name
    # The backend resolves this to the appropriate model format.
    default_model: str = "large-v3-turbo"
    model_cache_dir: Path = Path.home() / ".cache" / "whisper-service"

    # CUDA-specific
    device_index: int = 0  # GPU index (0-3 for 4x A40)
    compute_type: str = "float16"  # float16 | int8_float16 | int8

    # Audio
    max_file_size_mb: int = 500
    supported_formats: list[str] = [
        "wav", "mp3", "flac", "ogg", "m4a", "webm", "mp4", "mpeg", "mpga"
    ]

    # Processing
    default_language: str | None = None  # None = auto-detect
    default_response_format: OutputFormat = OutputFormat.JSON

    model_config = {"env_prefix": "WHISPER_"}


settings = Settings()
