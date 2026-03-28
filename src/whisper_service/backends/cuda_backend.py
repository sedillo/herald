"""faster-whisper backend for NVIDIA GPU inference via CTranslate2.

This is the production backend for NVIDIA hardware (A40, H100, etc.).
Uses CTranslate2 under the hood for optimized CUDA inference.
"""

import logging
import os
import site
import sys
import time
import warnings
from pathlib import Path

# Suppress HuggingFace unauthenticated request warnings
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")


def _preload_cuda_libs():
    """Preload pip-installed NVIDIA shared libraries.

    nvidia-cublas-cu12 and nvidia-cudnn-cu12 install .so files into
    site-packages/nvidia/*/lib/ but ctranslate2 can't find them because
    they're not on LD_LIBRARY_PATH. We preload them with ctypes so they're
    already in memory when ctranslate2 tries to use them.
    """
    import ctypes
    import glob

    logger = logging.getLogger(__name__)

    # Libraries ctranslate2 needs, in dependency order
    lib_patterns = [
        "nvidia/cublas/lib/libcublasLt.so*",
        "nvidia/cublas/lib/libcublas.so*",
        "nvidia/cudnn/lib/libcudnn*.so*",
        "nvidia/cufft/lib/libcufft.so*",
    ]

    site_dirs = site.getsitepackages()
    try:
        site_dirs.append(site.getusersitepackages())
    except AttributeError:
        pass

    loaded = []
    for sp in site_dirs:
        for pattern in lib_patterns:
            matches = sorted(glob.glob(os.path.join(sp, pattern)))
            for lib_path in matches:
                # Skip symlinks to avoid double-loading, prefer the versioned .so
                if os.path.islink(lib_path):
                    continue
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(lib_path))
                except OSError:
                    pass

    if loaded:
        logger.info(f"Preloaded CUDA libs: {loaded}")
    else:
        # Fallback: also set LD_LIBRARY_PATH for any subprocess or lazy loads
        nvidia_dirs = []
        for sp in site_dirs:
            nvidia_base = Path(sp) / "nvidia"
            if nvidia_base.is_dir():
                for subdir in nvidia_base.iterdir():
                    lib_dir = subdir / "lib"
                    if lib_dir.is_dir():
                        nvidia_dirs.append(str(lib_dir))
        if nvidia_dirs:
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_dirs) + (
                f":{existing}" if existing else ""
            )
            logger.info(f"Set LD_LIBRARY_PATH with: {nvidia_dirs}")


# Must run before any ctranslate2/faster_whisper import
_preload_cuda_libs()

from whisper_service.backends.base import Segment, TranscriptionBackend, TranscriptionResult

logger = logging.getLogger(__name__)

# Model name mappings — faster-whisper uses different naming
MODEL_ALIASES: dict[str, str] = {
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "distil-large-v3": "distil-large-v3",
    "small": "small",
    "base": "base",
    "tiny": "tiny",
}


def _resolve_model_name(name: str) -> str:
    """Resolve alias to faster-whisper model name."""
    return MODEL_ALIASES.get(name, name)


class CUDAWhisperBackend(TranscriptionBackend):
    """Whisper inference via faster-whisper on NVIDIA GPUs.

    Uses CTranslate2 for optimized CUDA inference. Supports multi-GPU
    via device_index parameter.

    Args:
        device_index: GPU index (0-3 for 4x A40). Default: 0.
        compute_type: Quantization level. Options:
            - "float16": Best quality, ~3GB VRAM (default for A40's 48GB)
            - "int8_float16": Faster, slightly lower quality, ~1.5GB VRAM
            - "int8": Fastest, lowest quality, ~1GB VRAM
        num_workers: Number of concurrent transcription workers per GPU.
    """

    def __init__(
        self,
        device_index: int = 0,
        compute_type: str = "float16",
        num_workers: int = 1,
    ) -> None:
        self._model = None
        self._model_name: str | None = None
        self._device_index = device_index
        self._compute_type = compute_type
        self._num_workers = num_workers

    async def load_model(self, model_name: str) -> None:
        """Load model onto specified GPU."""
        from faster_whisper import WhisperModel

        resolved = _resolve_model_name(model_name)
        logger.info(
            f"Loading faster-whisper model: {resolved} "
            f"(GPU:{self._device_index}, compute:{self._compute_type})"
        )

        self._model = WhisperModel(
            resolved,
            device="cuda",
            device_index=self._device_index,
            compute_type=self._compute_type,
            num_workers=self._num_workers,
        )

        self._model_name = resolved
        logger.info(
            f"Model ready: {resolved} on GPU:{self._device_index} "
            f"({self._compute_type})"
        )

    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper on CUDA."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        logger.info(
            f"Transcribing: {audio_path.name} with {self._model_name} "
            f"(GPU:{self._device_index})"
        )
        t0 = time.perf_counter()

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            beam_size=5,
            vad_filter=True,  # Built-in Silero VAD — filters silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        # Consume the generator to get all segments
        segments = []
        full_text_parts = []
        for i, seg in enumerate(segments_iter):
            segments.append(
                Segment(
                    id=i,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                )
            )
            full_text_parts.append(seg.text.strip())

        processing_time = time.perf_counter() - t0
        duration = segments[-1].end if segments else 0.0

        transcription = TranscriptionResult(
            text=" ".join(full_text_parts),
            language=info.language,
            duration=duration,
            segments=segments,
            processing_time=processing_time,
        )

        logger.info(
            f"Done: {audio_path.name} — "
            f"{transcription.duration:.1f}s audio in {processing_time:.2f}s "
            f"(RTF: {transcription.rtf:.3f}, GPU:{self._device_index})"
        )

        return transcription

    def is_loaded(self) -> bool:
        return self._model is not None

    def model_name(self) -> str | None:
        return self._model_name
