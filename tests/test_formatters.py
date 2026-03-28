"""Tests for output formatters."""

from whisper_service.backends.base import Segment, TranscriptionResult
from whisper_service.config import OutputFormat
from whisper_service.postprocessing.formatters import format_output


def _make_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="Hello world. This is a test.",
        language="en",
        duration=5.0,
        processing_time=0.5,
        segments=[
            Segment(id=0, start=0.0, end=2.5, text="Hello world."),
            Segment(id=1, start=2.5, end=5.0, text="This is a test."),
        ],
    )


def test_format_json():
    result = _make_result()
    output = format_output(result, OutputFormat.JSON)
    assert isinstance(output, dict)
    assert output["text"] == "Hello world. This is a test."


def test_format_verbose_json():
    result = _make_result()
    output = format_output(result, OutputFormat.VERBOSE_JSON)
    assert isinstance(output, dict)
    assert output["language"] == "en"
    assert output["duration"] == 5.0
    assert output["rtf"] == 0.1
    assert len(output["segments"]) == 2
    assert output["segments"][0]["start"] == 0.0
    assert output["segments"][1]["text"] == "This is a test."


def test_format_text():
    result = _make_result()
    output = format_output(result, OutputFormat.TEXT)
    assert output == "Hello world. This is a test."


def test_format_srt():
    result = _make_result()
    output = format_output(result, OutputFormat.SRT)
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello world." in output
    assert "2\n00:00:02,500 --> 00:00:05,000\nThis is a test." in output


def test_format_vtt():
    result = _make_result()
    output = format_output(result, OutputFormat.VTT)
    assert output.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in output


def test_rtf_calculation():
    result = _make_result()
    assert result.rtf == 0.1  # 0.5s processing / 5.0s audio
