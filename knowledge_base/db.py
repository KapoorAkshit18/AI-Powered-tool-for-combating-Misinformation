"""
Knowledge Base / Database Module for the Misinformation Detection Tool.

Manages a SQLite database at ``data/misinformation_kb.db`` for persisting
analysis history. Provides functions to initialise the database, save new
analyses, retrieve recent history, and search past results.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Default database path (project_root/data/misinformation_kb.db)
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "misinformation_kb.db"


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a connection to the SQLite database, creating directories as needed."""
    path = db_path or _DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn


# ── Public API ──────────────────────────────────────────────────────

def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initialise the database, creating the ``analyses`` table if it does not exist.

    Args:
        db_path: Optional override for the database file location.
    """
    conn = _get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT    NOT NULL,
                claim_text          TEXT    NOT NULL,
                source_url          TEXT,
                credibility_score   REAL    NOT NULL,
                verdict             TEXT    NOT NULL,
                full_analysis_json  TEXT    NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_analysis(
    claim: str,
    verdict: str,
    score: float,
    details: dict[str, Any],
    source_url: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Persist a completed analysis to the database.

    Args:
        claim:      The original claim or article text (truncated if very long).
        verdict:    The final verdict string.
        score:      The overall credibility score (0-100).
        details:    The full analysis dictionary to be stored as JSON.
        source_url: Optional source URL.
        db_path:    Optional override for the database file location.

    Returns:
        The row ``id`` of the newly inserted record.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO analyses (timestamp, claim_text, source_url,
                                  credibility_score, verdict, full_analysis_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                claim[:2000],  # truncate very long claims
                source_url,
                score,
                verdict,
                json.dumps(details, default=str),
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def get_history(
    limit: int = 50,
    db_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the most recent analyses.

    Args:
        limit:   Maximum number of records to return.
        db_path: Optional override for the database file location.

    Returns:
        A list of analysis records (most recent first), each as a dict.
    """
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def search_history(
    query: str,
    db_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """
    Search past analyses by claim text or verdict.

    Args:
        query:   Search term (case-insensitive substring match).
        db_path: Optional override for the database file location.

    Returns:
        A list of matching analysis records.
    """
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM analyses
            WHERE claim_text LIKE ? OR verdict LIKE ?
            ORDER BY id DESC
            """,
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row into a plain dictionary, parsing the JSON field."""
    d = dict(row)
    try:
        d["full_analysis_json"] = json.loads(d["full_analysis_json"])
    except (json.JSONDecodeError, TypeError):
        pass  # keep as raw string if parsing fails
    return d
