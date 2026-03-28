"""Base interface for transcription backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    """A single transcription segment with timing."""

    id: int
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class TranscriptionResult:
    """Complete transcription output."""

    text: str
    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)
    processing_time: float = 0.0

    @property
    def rtf(self) -> float:
        """Real-time factor: processing_time / audio_duration. Lower is faster."""
        if self.duration == 0:
            return 0.0
        return self.processing_time / self.duration


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends.

    All backends must implement this interface. This ensures we can swap
    MLX (dev/Mac) for CTranslate2/TensorRT (prod/GPU) without changing
    the API layer.
    """

    @abstractmethod
    async def load_model(self, model_name: str) -> None:
        """Load or download a model. Call once at startup."""
        ...

    @abstractmethod
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to preprocessed audio file (16kHz mono WAV).
            language: ISO 639-1 language code, or None for auto-detect.
            task: "transcribe" or "translate" (to English).
            word_timestamps: Whether to include word-level timestamps.
            initial_prompt: Optional prompt to guide the model (domain vocabulary).

        Returns:
            TranscriptionResult with text, segments, and metadata.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        ...

    @abstractmethod
    def model_name(self) -> str | None:
        """Return the name of the currently loaded model, or None."""
        ...
