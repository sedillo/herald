"""Entry point for the Whisper STT service."""

import logging
import uvicorn

from whisper_service.config import settings


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run():
    """Run the FastAPI server."""
    setup_logging()
    uvicorn.run(
        "whisper_service.api.app:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
    )


if __name__ == "__main__":
    run()
