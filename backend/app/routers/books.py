"""Book upload, list, section, patch, and delete endpoints (SQLite-backed)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from companion.parsing import parse_epub, parse_pdf

from backend.app import db
from backend.app.store import store

router = APIRouter()


class PatchBookBody(BaseModel):
    last_section_idx: int | None = None
    language: str | None = None
    app_mode: str | None = None


@router.post("/upload")
async def upload_book(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    name = file.filename.lower()
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    try:
        if name.endswith(".pdf"):
            chunks = parse_pdf(data)
        elif name.endswith(".epub"):
            chunks = parse_epub(data)
        else:
            raise HTTPException(400, "Only PDF and EPUB are supported")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Parse failed: {e}") from e

    if not chunks:
        raise HTTPException(400, "No readable text found in file")

    book_id = store.new_book_id()
    sections: list[dict[str, Any]] = []
    for c in chunks:
        sections.append(
            {
                "index": int(c["index"]),
                "title": c.get("title") or f"Section {c['index']}",
                "text": c.get("text") or "",
                "page": c.get("page"),
            }
        )

    first_idx = sections[0]["index"] if sections else 0
    db.insert_book(book_id, file.filename, sections, last_section_idx=first_idx)

    return {
        "book_id": book_id,
        "filename": file.filename,
        "last_section_idx": first_idx,
        "sections": [
            {
                "index": s["index"],
                "title": s["title"],
                "char_count": len(s["text"]),
            }
            for s in sections
        ],
    }


@router.get("")
@router.get("/")
def list_books():
    return {"books": db.list_books()}


@router.get("/{book_id}")
def get_book(book_id: str):
    meta = db.get_book_meta(book_id)
    if not meta:
        raise HTTPException(404, "Book not found")
    return meta


@router.get("/{book_id}/sections/{idx}")
def get_section(book_id: str, idx: int):
    section = db.get_section(book_id, idx)
    if not section:
        # Distinguish missing book vs missing section
        if not db.get_book_meta(book_id):
            raise HTTPException(404, "Book not found")
        raise HTTPException(404, f"Section {idx} not found")
    return {
        "book_id": section["book_id"],
        "index": section["index"],
        "title": section["title"],
        "text": section["text"],
        "char_count": section["char_count"],
    }


@router.patch("/{book_id}")
def patch_book(book_id: str, body: PatchBookBody):
    updated = db.patch_book(
        book_id,
        last_section_idx=body.last_section_idx,
        language=body.language,
        app_mode=body.app_mode,
    )
    if not updated:
        raise HTTPException(404, "Book not found")
    return updated


@router.delete("/{book_id}")
def delete_book(book_id: str):
    if not db.delete_book(book_id):
        raise HTTPException(404, "Book not found")
    return {"ok": True, "book_id": book_id}
