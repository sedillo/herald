"""Audio preprocessing: format conversion, resampling, normalization.

Converts any input audio to Whisper's expected format: 16kHz mono WAV.
Uses ffmpeg for broad format support.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1


def _check_ffmpeg() -> str:
    """Find ffmpeg binary, raise if not installed."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg"
        )
    return ffmpeg


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of an audio file in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found. Install with: brew install ffmpeg")

    result = subprocess.run(
        [
            ffprobe,
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    return float(result.stdout.strip())


def preprocess_audio(
    input_path: Path,
    output_dir: Path | None = None,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    channels: int = WHISPER_CHANNELS,
    normalize: bool = True,
) -> Path:
    """Convert audio to Whisper-compatible format (16kHz mono WAV).

    Args:
        input_path: Path to input audio file (any format ffmpeg supports).
        output_dir: Directory for output file. Uses temp dir if None.
        sample_rate: Target sample rate (default: 16000).
        channels: Target channel count (default: 1 / mono).
        normalize: Apply loudness normalization.

    Returns:
        Path to preprocessed WAV file.
    """
    ffmpeg = _check_ffmpeg()

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="whisper_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_preprocessed.wav"

    # Build ffmpeg command
    cmd = [
        ffmpeg,
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-sample_fmt", "s16",  # 16-bit PCM
    ]

    if normalize:
        # Two-pass loudness normalization to -16 LUFS
        # For simplicity, use single-pass dynaudnorm filter
        cmd.extend(["-af", "dynaudnorm=p=0.9:s=5"])

    cmd.extend([
        "-y",  # Overwrite output
        str(output_path),
    ])

    logger.info(f"Preprocessing: {input_path.name} -> {output_path.name}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg preprocessing failed: {result.stderr}")

    logger.info(f"Preprocessed: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    return output_path
