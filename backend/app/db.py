"""SQLite persistence for books, sections, AI cache, and chat history."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def data_dir() -> Path:
    override = os.environ.get("RC_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / ".data"


def db_path() -> Path:
    return data_dir() / "reading_companion.sqlite3"


_local = threading.local()
_init_lock = threading.Lock()
_initialized = False


def get_connection() -> sqlite3.Connection:
    """Per-thread connection with row factory and FK enforcement."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        path = db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        _local.conn = conn
    return conn


def init_db() -> None:
    """Create tables if missing. Safe to call multiple times."""
    global _initialized
    with _init_lock:
        path = db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS books (
                book_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                language TEXT,
                last_section_idx INTEGER DEFAULT 0,
                app_mode TEXT
            );

            CREATE TABLE IF NOT EXISTS sections (
                book_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                title TEXT,
                text TEXT,
                page TEXT,
                PRIMARY KEY (book_id, idx),
                FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ai_cache (
                book_id TEXT NOT NULL,
                section_idx INTEGER NOT NULL,
                kind TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (book_id, section_idx, kind, language, model),
                FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL,
                section_idx INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sections_book ON sections(book_id);
            CREATE INDEX IF NOT EXISTS idx_ai_cache_book ON ai_cache(book_id);
            CREATE INDEX IF NOT EXISTS idx_chat_book_section ON chat_messages(book_id, section_idx);
            """
        )
        conn.commit()
        _initialized = True


def _page_to_db(page: Any) -> str | None:
    if page is None:
        return None
    if isinstance(page, str):
        return page
    return json.dumps(page, ensure_ascii=False)


def _page_from_db(page: str | None) -> Any:
    if page is None:
        return None
    try:
        return json.loads(page)
    except (json.JSONDecodeError, TypeError):
        return page


# --- books / sections -------------------------------------------------


def insert_book(
    book_id: str,
    filename: str,
    sections: list[dict[str, Any]],
    *,
    language: str | None = None,
    app_mode: str | None = None,
    last_section_idx: int = 0,
) -> dict[str, Any]:
    now = _utc_now()
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO books (book_id, filename, created_at, updated_at, language, last_section_idx, app_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (book_id, filename, now, now, language, last_section_idx, app_mode),
        )
        for s in sections:
            conn.execute(
                """
                INSERT INTO sections (book_id, idx, title, text, page)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    book_id,
                    int(s["index"]),
                    s.get("title") or f"Section {s['index']}",
                    s.get("text") or "",
                    _page_to_db(s.get("page")),
                ),
            )
    return get_book(book_id)  # type: ignore[return-value]


def list_books() -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT b.book_id, b.filename, b.created_at, b.updated_at,
               b.last_section_idx, b.language, b.app_mode,
               (SELECT COUNT(*) FROM sections s WHERE s.book_id = b.book_id) AS section_count
        FROM books b
        ORDER BY b.updated_at DESC
        """
    ).fetchall()
    return [
        {
            "book_id": r["book_id"],
            "filename": r["filename"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "last_section_idx": r["last_section_idx"] or 0,
            "language": r["language"],
            "app_mode": r["app_mode"],
            "section_count": r["section_count"],
        }
        for r in rows
    ]


def get_book(book_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM books WHERE book_id = ?", (book_id,)
    ).fetchone()
    if not row:
        return None
    sections = conn.execute(
        """
        SELECT idx, title, text, page FROM sections
        WHERE book_id = ? ORDER BY idx
        """,
        (book_id,),
    ).fetchall()
    return {
        "book_id": row["book_id"],
        "filename": row["filename"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "language": row["language"],
        "last_section_idx": row["last_section_idx"] or 0,
        "app_mode": row["app_mode"],
        "sections": [
            {
                "index": s["idx"],
                "title": s["title"],
                "text": s["text"] or "",
                "page": _page_from_db(s["page"]),
            }
            for s in sections
        ],
    }


def get_book_meta(book_id: str) -> dict[str, Any] | None:
    """Metadata + section list without full text (char_count only)."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM books WHERE book_id = ?", (book_id,)
    ).fetchone()
    if not row:
        return None
    sections = conn.execute(
        """
        SELECT idx, title, LENGTH(COALESCE(text, '')) AS char_count
        FROM sections WHERE book_id = ? ORDER BY idx
        """,
        (book_id,),
    ).fetchall()
    return {
        "book_id": row["book_id"],
        "filename": row["filename"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "language": row["language"],
        "last_section_idx": row["last_section_idx"] or 0,
        "app_mode": row["app_mode"],
        "sections": [
            {
                "index": s["idx"],
                "title": s["title"],
                "char_count": s["char_count"],
            }
            for s in sections
        ],
    }


def get_section(book_id: str, idx: int) -> dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT idx, title, text, page FROM sections
        WHERE book_id = ? AND idx = ?
        """,
        (book_id, idx),
    ).fetchone()
    if not row:
        return None
    text = row["text"] or ""
    return {
        "book_id": book_id,
        "index": row["idx"],
        "title": row["title"],
        "text": text,
        "page": _page_from_db(row["page"]),
        "char_count": len(text),
    }


def patch_book(
    book_id: str,
    *,
    last_section_idx: int | None = None,
    language: str | None = None,
    app_mode: str | None = None,
) -> dict[str, Any] | None:
    conn = get_connection()
    existing = conn.execute(
        "SELECT book_id FROM books WHERE book_id = ?", (book_id,)
    ).fetchone()
    if not existing:
        return None
    fields: list[str] = []
    values: list[Any] = []
    if last_section_idx is not None:
        fields.append("last_section_idx = ?")
        values.append(last_section_idx)
    if language is not None:
        fields.append("language = ?")
        values.append(language)
    if app_mode is not None:
        fields.append("app_mode = ?")
        values.append(app_mode)
    if not fields:
        return get_book_meta(book_id)
    fields.append("updated_at = ?")
    values.append(_utc_now())
    values.append(book_id)
    with conn:
        conn.execute(
            f"UPDATE books SET {', '.join(fields)} WHERE book_id = ?",
            values,
        )
    return get_book_meta(book_id)


def delete_book(book_id: str) -> bool:
    conn = get_connection()
    with conn:
        # Explicit child deletes in case FK cascade is off for older DBs
        conn.execute("DELETE FROM chat_messages WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM ai_cache WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM sections WHERE book_id = ?", (book_id,))
        cur = conn.execute("DELETE FROM books WHERE book_id = ?", (book_id,))
        return cur.rowcount > 0


def touch_book(book_id: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE books SET updated_at = ? WHERE book_id = ?",
            (_utc_now(), book_id),
        )


# --- AI cache ---------------------------------------------------------


def get_ai_cache(
    book_id: str,
    section_idx: int,
    kind: str,
    language: str,
    model: str,
) -> dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT payload, updated_at FROM ai_cache
        WHERE book_id = ? AND section_idx = ? AND kind = ? AND language = ? AND model = ?
        """,
        (book_id, section_idx, kind, language or "", model),
    ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row["payload"])
    except json.JSONDecodeError:
        payload = row["payload"]
    return {"payload": payload, "updated_at": row["updated_at"]}


def put_ai_cache(
    book_id: str,
    section_idx: int,
    kind: str,
    language: str,
    model: str,
    payload: dict[str, Any] | list[Any] | str,
) -> None:
    conn = get_connection()
    now = _utc_now()
    blob = json.dumps(payload, ensure_ascii=False)
    with conn:
        conn.execute(
            """
            INSERT INTO ai_cache (book_id, section_idx, kind, language, model, payload, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id, section_idx, kind, language, model)
            DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
            """,
            (book_id, section_idx, kind, language or "", model, blob, now),
        )
        conn.execute(
            "UPDATE books SET updated_at = ? WHERE book_id = ?",
            (now, book_id),
        )


# --- chat -------------------------------------------------------------


def get_chat_messages(book_id: str, section_idx: int) -> list[dict[str, str]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT role, content FROM chat_messages
        WHERE book_id = ? AND section_idx = ?
        ORDER BY id ASC
        """,
        (book_id, section_idx),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def put_chat_messages(
    book_id: str,
    section_idx: int,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Replace the full chat thread for a book section."""
    conn = get_connection()
    now = _utc_now()
    with conn:
        conn.execute(
            "DELETE FROM chat_messages WHERE book_id = ? AND section_idx = ?",
            (book_id, section_idx),
        )
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content") or ""
            conn.execute(
                """
                INSERT INTO chat_messages (book_id, section_idx, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (book_id, section_idx, role, content, now),
            )
        conn.execute(
            "UPDATE books SET updated_at = ? WHERE book_id = ?",
            (now, book_id),
        )
    return get_chat_messages(book_id, section_idx)
