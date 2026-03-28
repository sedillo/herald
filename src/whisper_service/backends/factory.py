"""Backend factory — auto-detects hardware and selects the right backend.

Detection order:
1. Explicit WHISPER_BACKEND env var (mlx | cuda)
2. NVIDIA GPU present (nvidia-smi) → CUDAWhisperBackend
3. Apple Silicon → MLXWhisperBackend
4. Fallback → error
"""

import logging
import os
import platform
import shutil
import subprocess
import sys

from whisper_service.backends.base import TranscriptionBackend

logger = logging.getLogger(__name__)


def _has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is present via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = result.stdout.strip().split("\n")
            logger.info(f"NVIDIA GPUs detected: {gpus}")
            return True
    except (subprocess.TimeoutExpired, OSError):
        pass
    return False


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def detect_backend() -> str:
    """Auto-detect the best available backend."""
    # Explicit override
    forced = os.environ.get("WHISPER_BACKEND", "").lower()
    if forced in ("mlx", "cuda"):
        logger.info(f"Backend forced via WHISPER_BACKEND={forced}")
        return forced

    if _has_nvidia_gpu():
        logger.info("NVIDIA GPU detected — using CUDA backend")
        return "cuda"

    if _is_apple_silicon():
        logger.info("Apple Silicon detected — using MLX backend")
        return "mlx"

    raise RuntimeError(
        "No supported GPU detected. Herald requires either:\n"
        "  - NVIDIA GPU (install with: pip install -e '.[cuda]')\n"
        "  - Apple Silicon Mac (install with: pip install -e '.[mlx]')\n"
        "Or set WHISPER_BACKEND=cuda to force CUDA backend."
    )


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

    else:
        raise ValueError(f"Unknown backend: {backend_type}. Use 'mlx' or 'cuda'.")
