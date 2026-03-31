"""SQLite-backed usage logger for the Herald web UI.

Each transcription attempt (success or failure) is recorded with:
- UTC timestamp
- Client IP (X-Forwarded-For aware)
- Filename and file size
- Model used, detected language
- Audio duration, processing time, RTF
- Transcription text
- Error message if failed
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".cache" / "whisper-service" / "ui_log.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS transcription_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    client_ip         TEXT,
    filename          TEXT,
    file_size_kb      REAL,
    model             TEXT,
    language          TEXT,
    duration_s        REAL,
    processing_time_s REAL,
    rtf               REAL,
    transcription     TEXT,
    error             TEXT
)
"""


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_TABLE)
        conn.commit()


def log_transcription(
    *,
    client_ip: str | None,
    filename: str | None,
    file_size_kb: float | None,
    model: str | None,
    language: str | None = None,
    duration_s: float | None = None,
    processing_time_s: float | None = None,
    rtf: float | None = None,
    transcription: str | None = None,
    error: str | None = None,
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO transcription_log
                (timestamp, client_ip, filename, file_size_kb, model, language,
                 duration_s, processing_time_s, rtf, transcription, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                client_ip,
                filename,
                file_size_kb,
                model,
                language,
                duration_s,
                processing_time_s,
                rtf,
                transcription,
                error,
            ),
        )
        conn.commit()


def get_logs(limit: int = 500) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM transcription_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
