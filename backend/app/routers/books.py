"""Book upload and section endpoints."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from companion.parsing import parse_epub, parse_pdf

from backend.app.store import BookRecord, store

router = APIRouter()


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
    sections = []
    for c in chunks:
        sections.append(
            {
                "index": int(c["index"]),
                "title": c.get("title") or f"Section {c['index']}",
                "text": c.get("text") or "",
                "page": c.get("page"),
            }
        )

    store.books[book_id] = BookRecord(book_id=book_id, filename=file.filename, sections=sections)

    return {
        "book_id": book_id,
        "filename": file.filename,
        "sections": [
            {
                "index": s["index"],
                "title": s["title"],
                "char_count": len(s["text"]),
            }
            for s in sections
        ],
    }


@router.get("/{book_id}/sections/{idx}")
def get_section(book_id: str, idx: int):
    book = store.books.get(book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    for s in book.sections:
        if s["index"] == idx:
            return {
                "book_id": book_id,
                "index": s["index"],
                "title": s["title"],
                "text": s["text"],
                "char_count": len(s["text"]),
            }
    raise HTTPException(404, f"Section {idx} not found")
