"""Format transcription results into various output formats.

Supports: JSON, verbose JSON, plain text, SRT, VTT.
"""

import json
from whisper_service.backends.base import TranscriptionResult
from whisper_service.config import OutputFormat


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_json(result: TranscriptionResult) -> dict:
    """OpenAI-compatible JSON response."""
    return {
        "text": result.text,
    }


def format_verbose_json(result: TranscriptionResult) -> dict:
    """Detailed JSON with segments, timestamps, and metadata."""
    return {
        "text": result.text,
        "language": result.language,
        "duration": round(result.duration, 2),
        "processing_time": round(result.processing_time, 3),
        "rtf": round(result.rtf, 4),
        "segments": [
            {
                "id": seg.id,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text,
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            }
            for seg in result.segments
        ],
    }


def format_text(result: TranscriptionResult) -> str:
    """Plain text output."""
    return result.text


def format_srt(result: TranscriptionResult) -> str:
    """SubRip subtitle format."""
    lines = []
    for seg in result.segments:
        lines.append(str(seg.id + 1))
        lines.append(
            f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}"
        )
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def format_vtt(result: TranscriptionResult) -> str:
    """WebVTT subtitle format."""
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(
            f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}"
        )
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def format_output(result: TranscriptionResult, fmt: OutputFormat) -> dict | str:
    """Format a transcription result into the requested output format.

    Args:
        result: The transcription result to format.
        fmt: Desired output format.

    Returns:
        dict for JSON formats, str for text/subtitle formats.
    """
    formatters = {
        OutputFormat.JSON: format_json,
        OutputFormat.VERBOSE_JSON: format_verbose_json,
        OutputFormat.TEXT: format_text,
        OutputFormat.SRT: format_srt,
        OutputFormat.VTT: format_vtt,
    }

    formatter = formatters.get(fmt)
    if not formatter:
        raise ValueError(f"Unsupported format: {fmt}")

    return formatter(result)
