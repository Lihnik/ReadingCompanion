"""Language-learning AI endpoints (summary + vocab)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from companion.constants import SYSTEM_PROMPT_LL_SUMMARY, SYSTEM_PROMPT_LL_VOCAB
from companion.ollama import (
    build_ll_summary_prompt,
    build_ll_vocab_prompt,
    call_ollama,
    parse_vocab_response,
    stream_ollama,
)

from backend.app.store import store

router = APIRouter()


class AiRequest(BaseModel):
    book_id: str
    section_idx: int
    language: str = "Italian"
    model: str = "llama3.1:8b"
    stream: bool = False


def _section_text(book_id: str, section_idx: int) -> str:
    book = store.books.get(book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    for s in book.sections:
        if s["index"] == section_idx:
            return s["text"]
    raise HTTPException(404, f"Section {section_idx} not found")


@router.post("/summary")
def english_summary(body: AiRequest):
    text = _section_text(body.book_id, body.section_idx)
    prompt = build_ll_summary_prompt(text, body.language)

    if body.stream:
        def event_gen():
            for token in stream_ollama(prompt, body.model, SYSTEM_PROMPT_LL_SUMMARY):
                # SSE: data lines; empty tokens skipped
                if token:
                    yield f"data: {token.replace(chr(10), chr(10) + 'data: ')}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    summary = call_ollama(prompt, body.model, SYSTEM_PROMPT_LL_SUMMARY)
    if summary.startswith("ERROR:"):
        raise HTTPException(502, summary)
    return {"summary": summary}


@router.post("/vocab")
def vocabulary(body: AiRequest):
    text = _section_text(body.book_id, body.section_idx)
    prompt = build_ll_vocab_prompt(text, body.language)
    raw = call_ollama(prompt, body.model, SYSTEM_PROMPT_LL_VOCAB)
    if raw.startswith("ERROR:"):
        raise HTTPException(502, raw)
    entries = parse_vocab_response(raw)
    return {
        "vocab": entries,
        "raw_preview": (raw[:400] if not entries else None),
        "parse_ok": bool(entries),
    }
