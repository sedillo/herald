"""Inference backends for Whisper models."""

from whisper_service.backends.base import TranscriptionBackend, TranscriptionResult
from whisper_service.backends.factory import create_backend, detect_backend

__all__ = [
    "TranscriptionBackend",
    "TranscriptionResult",
    "create_backend",
    "detect_backend",
]
