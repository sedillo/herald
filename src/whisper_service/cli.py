"""CLI tool for local transcription.

Usage:
    whisper-transcribe audio.mp3
    whisper-transcribe audio.wav --model large-v3-turbo --format srt
    whisper-transcribe meeting.m4a --language en --prompt "HPC, GPU, CUDA"
    whisper-transcribe audio.wav --backend cuda --gpu 0 --compute-type float16
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from whisper_service.backends import create_backend, detect_backend
from whisper_service.config import OutputFormat
from whisper_service.postprocessing.formatters import format_output
from whisper_service.preprocessing.audio import preprocess_audio

MODEL_LIST = {
    "large-v3": "Whisper Large V3 (1.55B, highest accuracy)",
    "large-v3-turbo": "Whisper Large V3 Turbo (809M, balanced)",
    "distil-large-v3": "Distil Whisper Large V3 (756M, fastest)",
    "small": "Whisper Small (244M)",
    "base": "Whisper Base (74M)",
    "tiny": "Whisper Tiny (39M)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  whisper-transcribe recording.mp3
  whisper-transcribe meeting.wav --model large-v3-turbo --format verbose_json
  whisper-transcribe call.m4a --language en --prompt "domain-specific terms here"
  whisper-transcribe audio.wav --format srt --output subtitles.srt
  whisper-transcribe audio.wav --backend cuda --gpu 2 --compute-type int8_float16

Available models:
"""
        + "\n".join(f"  {alias:20s} {desc}" for alias, desc in MODEL_LIST.items()),
    )

    parser.add_argument("audio", type=Path, help="Path to audio file")
    parser.add_argument(
        "--model", "-m",
        default="large-v3-turbo",
        help="Model name or alias (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["auto", "mlx", "cuda"],
        default="auto",
        help="Inference backend (default: auto-detect)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index for CUDA backend (default: 0)",
    )
    parser.add_argument(
        "--compute-type",
        choices=["float16", "int8_float16", "int8"],
        default="float16",
        help="Compute type for CUDA backend (default: float16)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (e.g., en, es, ja). Auto-detect if not set.",
    )
    parser.add_argument(
        "--format", "-f",
        default="text",
        choices=[f.value for f in OutputFormat],
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path. Prints to stdout if not set.",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Optional prompt with domain vocabulary to improve accuracy.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task: transcribe (default) or translate to English.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip audio preprocessing (assume 16kHz mono WAV).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


async def _run(args):
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    audio_path = args.audio
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    fmt = OutputFormat(args.format)

    # Preprocess
    if args.no_preprocess:
        processed_path = audio_path
    else:
        print(f"Preprocessing: {audio_path.name}", file=sys.stderr)
        try:
            processed_path = preprocess_audio(audio_path)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Detect and create backend
    backend_type = args.backend if args.backend != "auto" else None
    detected = backend_type or detect_backend()
    print(f"Backend: {detected} | GPU: {args.gpu} | Compute: {args.compute_type}", file=sys.stderr)

    backend = create_backend(
        backend_type=backend_type,
        device_index=args.gpu,
        compute_type=args.compute_type,
    )

    # Load model
    print(f"Loading model: {args.model}", file=sys.stderr)
    await backend.load_model(args.model)

    # Transcribe
    print(f"Transcribing: {audio_path.name}", file=sys.stderr)
    t0 = time.perf_counter()

    result = await backend.transcribe(
        audio_path=processed_path,
        language=args.language,
        task=args.task,
        word_timestamps=(fmt == OutputFormat.VERBOSE_JSON),
        initial_prompt=args.prompt,
    )

    elapsed = time.perf_counter() - t0

    # Format output
    output = format_output(result, fmt)

    if isinstance(output, dict):
        output_str = json.dumps(output, indent=2, ensure_ascii=False)
    else:
        output_str = output

    # Write output
    if args.output:
        args.output.write_text(output_str, encoding="utf-8")
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output_str)

    # Stats to stderr
    print(
        f"\n--- {result.duration:.1f}s audio | {elapsed:.2f}s processing | "
        f"RTF: {result.rtf:.3f} | Language: {result.language} | "
        f"Backend: {detected} ---",
        file=sys.stderr,
    )


def main():
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
