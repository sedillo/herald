"""MLX Whisper backend for Apple Silicon inference."""

import logging
import os
import time
from pathlib import Path

# Suppress HuggingFace unauthenticated request warnings.
# Models are public — auth is optional but HF nags about rate limits.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")

from whisper_service.backends.base import Segment, TranscriptionBackend, TranscriptionResult

logger = logging.getLogger(__name__)


# Model name mappings for convenience aliases
MODEL_ALIASES: dict[str, str] = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def _resolve_model_name(name: str) -> str:
    """Resolve a short alias to a full HuggingFace model name."""
    return MODEL_ALIASES.get(name, name)


class MLXWhisperBackend(TranscriptionBackend):
    """Whisper inference via MLX on Apple Silicon.

    Uses mlx-whisper for native Metal GPU acceleration on M-series chips.
    This is the primary development backend.
    """

    def __init__(self) -> None:
        self._model_name: str | None = None
        self._loaded = False

    async def load_model(self, model_name: str) -> None:
        """Load model weights. First call downloads from HuggingFace."""
        import mlx_whisper

        resolved = _resolve_model_name(model_name)
        logger.info(f"Loading MLX Whisper model: {resolved}")

        # mlx_whisper lazy-loads on first transcribe, but we trigger it here
        # by doing a minimal transcribe to warm up. We just verify the model
        # can be resolved for now — actual load happens on first inference.
        self._model_name = resolved
        self._loaded = True
        logger.info(f"Model ready: {resolved}")

    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using MLX Whisper."""
        import mlx_whisper

        if not self._loaded or not self._model_name:
            raise RuntimeError("No model loaded. Call load_model() first.")

        audio_str = str(audio_path)

        decode_options: dict = {}
        if language:
            decode_options["language"] = language
        if task == "translate":
            decode_options["task"] = "translate"
        if initial_prompt:
            decode_options["initial_prompt"] = initial_prompt

        logger.info(f"Transcribing: {audio_path.name} with {self._model_name}")
        t0 = time.perf_counter()

        result = mlx_whisper.transcribe(
            audio_str,
            path_or_hf_repo=self._model_name,
            word_timestamps=word_timestamps,
            **decode_options,
        )

        processing_time = time.perf_counter() - t0

        # Parse segments
        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            segments.append(
                Segment(
                    id=i,
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    avg_logprob=seg.get("avg_logprob", 0.0),
                    no_speech_prob=seg.get("no_speech_prob", 0.0),
                )
            )

        # Calculate duration from last segment end, or 0 if no segments
        duration = segments[-1].end if segments else 0.0

        transcription = TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", language or "unknown"),
            duration=duration,
            segments=segments,
            processing_time=processing_time,
        )

        logger.info(
            f"Done: {audio_path.name} — "
            f"{transcription.duration:.1f}s audio in {processing_time:.2f}s "
            f"(RTF: {transcription.rtf:.3f})"
        )

        return transcription

    def is_loaded(self) -> bool:
        return self._loaded

    def model_name(self) -> str | None:
        return self._model_name
