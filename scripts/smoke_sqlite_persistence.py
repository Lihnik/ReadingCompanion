#!/usr/bin/env python3
"""Smoke test: SQLite book persistence + AI cache roundtrip (no live Ollama)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Isolate DB for this run
tmpdir = tempfile.mkdtemp(prefix="rc_smoke_")
os.environ["RC_DATA_DIR"] = tmpdir

from backend.app import db  # noqa: E402


def test_db_roundtrip() -> None:
    # Fresh connection path
    db._local.conn = None  # type: ignore[attr-defined]
    db._initialized = False  # type: ignore[attr-defined]
    db.init_db()

    book_id = "smokebook01"
    sections = [
        {"index": 0, "title": "Intro", "text": "Hello world section zero.", "page": 1},
        {"index": 1, "title": "Next", "text": "Second section text.", "page": None},
    ]
    db.insert_book(book_id, "smoke.txt", sections, language="Italian", last_section_idx=0)

    # Simulate restart: drop thread-local connection
    if getattr(db._local, "conn", None) is not None:
        db._local.conn.close()
        db._local.conn = None

    db.init_db()
    books = db.list_books()
    assert any(b["book_id"] == book_id for b in books), books
    meta = db.get_book_meta(book_id)
    assert meta is not None
    assert meta["filename"] == "smoke.txt"
    assert len(meta["sections"]) == 2
    assert meta["sections"][0]["char_count"] == len(sections[0]["text"])

    sec = db.get_section(book_id, 1)
    assert sec is not None
    assert sec["text"] == "Second section text."

    db.patch_book(book_id, last_section_idx=1, language="German")
    meta2 = db.get_book_meta(book_id)
    assert meta2["last_section_idx"] == 1
    assert meta2["language"] == "German"

    payload = {"summary": "Cached gist"}
    db.put_ai_cache(book_id, 0, "ll_summary", "Italian", "llama3.1:8b", payload)
    hit = db.get_ai_cache(book_id, 0, "ll_summary", "Italian", "llama3.1:8b")
    assert hit is not None
    assert hit["payload"]["summary"] == "Cached gist"

    db.put_chat_messages(
        book_id,
        0,
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
    )
    chat = db.get_chat_messages(book_id, 0)
    assert len(chat) == 2

    assert db.delete_book(book_id) is True
    assert db.get_book_meta(book_id) is None
    print("OK db roundtrip + restart simulation")


def test_api_books() -> None:
    from fastapi.testclient import TestClient

    # Reset connection for TestClient process
    if getattr(db._local, "conn", None) is not None:
        db._local.conn.close()
        db._local.conn = None
    db._initialized = False  # type: ignore[attr-defined]

    from backend.app.main import app

    sample = Path("/home/box/decameron-sample.pdf")
    if not sample.exists():
        print("SKIP api upload (no sample PDF)")
        return

    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200, health.text

        with sample.open("rb") as f:
            res = client.post(
                "/api/books/upload",
                files={"file": ("decameron-sample.pdf", f, "application/pdf")},
            )
        assert res.status_code == 200, res.text
        body = res.json()
        book_id = body["book_id"]
        assert body["sections"], body

        listed = client.get("/api/books")
        assert listed.status_code == 200
        ids = [b["book_id"] for b in listed.json()["books"]]
        assert book_id in ids

        meta = client.get(f"/api/books/{book_id}")
        assert meta.status_code == 200
        assert meta.json()["filename"] == "decameron-sample.pdf"

        idx = body["sections"][0]["index"]
        sec = client.get(f"/api/books/{book_id}/sections/{idx}")
        assert sec.status_code == 200
        assert len(sec.json()["text"]) > 0

        patched = client.patch(
            f"/api/books/{book_id}",
            json={"last_section_idx": idx, "language": "Italian"},
        )
        assert patched.status_code == 200
        assert patched.json()["language"] == "Italian"

        # AI cache without Ollama: write via db then hit API
        db.put_ai_cache(
            book_id,
            idx,
            "ll_summary",
            "Italian",
            "llama3.1:8b",
            {"summary": "from-cache"},
        )
        summary = client.post(
            "/api/ai/summary",
            json={
                "book_id": book_id,
                "section_idx": idx,
                "language": "Italian",
                "model": "llama3.1:8b",
            },
        )
        assert summary.status_code == 200, summary.text
        assert summary.json()["summary"] == "from-cache"

        chat_put = client.put(
            f"/api/ai/chat/{book_id}/{idx}",
            json={"messages": [{"role": "user", "content": "ciao"}, {"role": "assistant", "content": "salve"}]},
        )
        assert chat_put.status_code == 200
        chat_get = client.get(f"/api/ai/chat/{book_id}/{idx}")
        assert chat_get.status_code == 200
        assert len(chat_get.json()["messages"]) == 2

        deleted = client.delete(f"/api/books/{book_id}")
        assert deleted.status_code == 200
        assert client.get(f"/api/books/{book_id}").status_code == 404

    print("OK API books + AI cache hit + chat")


if __name__ == "__main__":
    test_db_roundtrip()
    test_api_books()
    print(f"DB dir used: {tmpdir}")
    print("ALL SMOKE CHECKS PASSED")
