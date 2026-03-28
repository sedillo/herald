"""Backend factory — auto-detects hardware and selects the right backend.

Detection order:
1. Explicit WHISPER_BACKEND env var (mlx | cuda | cpu)
2. CUDA available → CUDAWhisperBackend
3. Apple Silicon → MLXWhisperBackend
4. Fallback → error (CPU-only Whisper is too slow to be useful)
"""

import logging
import os
import platform
import sys

from whisper_service.backends.base import TranscriptionBackend

logger = logging.getLogger(__name__)


def _has_cuda() -> bool:
    """Check if CUDA is available without importing torch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass

    # Check for faster-whisper's ctranslate2 CUDA support
    try:
        import ctranslate2
        return "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except (ImportError, RuntimeError):
        return False


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def detect_backend() -> str:
    """Auto-detect the best available backend."""
    # Explicit override
    forced = os.environ.get("WHISPER_BACKEND", "").lower()
    if forced in ("mlx", "cuda", "cpu"):
        logger.info(f"Backend forced via WHISPER_BACKEND={forced}")
        return forced

    if _has_cuda():
        logger.info("CUDA detected → using faster-whisper backend")
        return "cuda"

    if _is_apple_silicon():
        logger.info("Apple Silicon detected → using MLX backend")
        return "mlx"

    logger.warning("No GPU detected. CPU inference will be very slow.")
    return "cpu"


def create_backend(
    backend_type: str | None = None,
    device_index: int = 0,
    compute_type: str = "float16",
) -> TranscriptionBackend:
    """Create the appropriate backend instance.

    Args:
        backend_type: "mlx", "cuda", or None for auto-detect.
        device_index: GPU index for CUDA backend (0-3 for 4x A40).
        compute_type: Quantization for CUDA backend ("float16", "int8_float16", "int8").

    Returns:
        Configured TranscriptionBackend instance.
    """
    if backend_type is None:
        backend_type = detect_backend()

    if backend_type == "mlx":
        from whisper_service.backends.mlx_backend import MLXWhisperBackend
        return MLXWhisperBackend()

    elif backend_type == "cuda":
        from whisper_service.backends.cuda_backend import CUDAWhisperBackend
        return CUDAWhisperBackend(
            device_index=device_index,
            compute_type=compute_type,
        )

    elif backend_type == "cpu":
        # CPU fallback uses faster-whisper with device="cpu"
        # We reuse the CUDA backend class but it would need modification
        # For now, raise — CPU is not a realistic deployment target
        raise RuntimeError(
            "CPU-only inference is not supported. "
            "Install MLX (Apple Silicon) or faster-whisper[cuda] (NVIDIA GPU)."
        )

    else:
        raise ValueError(f"Unknown backend: {backend_type}. Use 'mlx' or 'cuda'.")
