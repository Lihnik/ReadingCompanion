"""AI endpoints: LL summary/vocab, streaming chat, reading-mode commentary/Q&A."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from companion.constants import (
    SYSTEM_PROMPT_CHAT,
    SYSTEM_PROMPT_COMMENTARY,
    SYSTEM_PROMPT_LL_SUMMARY,
    SYSTEM_PROMPT_LL_VOCAB,
    SYSTEM_PROMPT_QUESTION,
)
from companion.ollama import (
    build_chat_prompt,
    build_commentary_prompt,
    build_feedback_prompt,
    build_ll_summary_prompt,
    build_ll_vocab_prompt,
    build_question_prompt,
    build_summary_prompt,
    call_ollama,
    parse_vocab_response,
    stream_ollama,
)

from backend.app import db

router = APIRouter()


class AiRequest(BaseModel):
    book_id: str
    section_idx: int
    language: str = "Italian"
    model: str = "llama3.1:8b"
    stream: bool = False
    force: bool = False


class SectionAiRequest(BaseModel):
    book_id: str
    section_idx: int
    model: str = "llama3.1:8b"
    force: bool = False


class FeedbackRequest(BaseModel):
    book_id: str
    section_idx: int
    question: str
    answer: str
    model: str = "llama3.1:8b"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    book_id: str
    section_idx: int
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    model: str = "llama3.1:8b"


class ChatHistoryBody(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)


def _section_text(book_id: str, section_idx: int) -> str:
    section = db.get_section(book_id, section_idx)
    if not section:
        if not db.get_book_meta(book_id):
            raise HTTPException(404, "Book not found")
        raise HTTPException(404, f"Section {section_idx} not found")
    return section["text"]


def _sse_token_stream(tokens):
    """Yield SSE events with JSON-encoded token strings; finish with [DONE]."""
    for token in tokens:
        if token:
            yield f"data: {json.dumps(token, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _cache_get(book_id: str, section_idx: int, kind: str, language: str, model: str, force: bool):
    if force:
        return None
    hit = db.get_ai_cache(book_id, section_idx, kind, language, model)
    if not hit:
        return None
    return hit["payload"]


def _cache_put(book_id: str, section_idx: int, kind: str, language: str, model: str, payload):
    db.put_ai_cache(book_id, section_idx, kind, language, model, payload)


@router.post("/summary")
def english_summary(body: AiRequest):
    text = _section_text(body.book_id, body.section_idx)
    kind = "ll_summary"

    if body.stream:
        # Streaming bypasses cache (tokens can't be usefully cached mid-flight)
        prompt = build_ll_summary_prompt(text, body.language)
        return StreamingResponse(
            _sse_token_stream(stream_ollama(prompt, body.model, SYSTEM_PROMPT_LL_SUMMARY)),
            media_type="text/event-stream",
        )

    cached = _cache_get(body.book_id, body.section_idx, kind, body.language, body.model, body.force)
    if isinstance(cached, dict) and "summary" in cached:
        return cached

    prompt = build_ll_summary_prompt(text, body.language)
    summary = call_ollama(prompt, body.model, SYSTEM_PROMPT_LL_SUMMARY)
    if summary.startswith("ERROR:"):
        raise HTTPException(502, summary)
    result = {"summary": summary}
    _cache_put(body.book_id, body.section_idx, kind, body.language, body.model, result)
    return result


@router.post("/vocab")
def vocabulary(body: AiRequest):
    text = _section_text(body.book_id, body.section_idx)
    kind = "ll_vocab"
    cached = _cache_get(body.book_id, body.section_idx, kind, body.language, body.model, body.force)
    if isinstance(cached, dict) and "vocab" in cached:
        return cached

    prompt = build_ll_vocab_prompt(text, body.language)
    raw = call_ollama(prompt, body.model, SYSTEM_PROMPT_LL_VOCAB)
    if raw.startswith("ERROR:"):
        raise HTTPException(502, raw)
    entries = parse_vocab_response(raw)
    result = {
        "vocab": entries,
        "raw_preview": (raw[:400] if not entries else None),
        "parse_ok": bool(entries),
    }
    _cache_put(body.book_id, body.section_idx, kind, body.language, body.model, result)
    return result


@router.post("/chat")
def chat(body: ChatRequest):
    """Stream a chat reply (SSE) grounded in the current section."""
    if not (body.message or "").strip():
        raise HTTPException(400, "Empty message")
    text = _section_text(body.book_id, body.section_idx)
    history = [{"role": m.role, "content": m.content} for m in body.history[-6:]]
    prompt = build_chat_prompt(text, history, body.message.strip())

    return StreamingResponse(
        _sse_token_stream(stream_ollama(prompt, body.model, SYSTEM_PROMPT_CHAT)),
        media_type="text/event-stream",
    )


@router.get("/chat/{book_id}/{section_idx}")
def get_chat(book_id: str, section_idx: int):
    if not db.get_book_meta(book_id):
        raise HTTPException(404, "Book not found")
    return {"messages": db.get_chat_messages(book_id, section_idx)}


@router.put("/chat/{book_id}/{section_idx}")
def put_chat(book_id: str, section_idx: int, body: ChatHistoryBody):
    if not db.get_book_meta(book_id):
        raise HTTPException(404, "Book not found")
    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    saved = db.put_chat_messages(book_id, section_idx, msgs)
    return {"messages": saved}


@router.post("/commentary")
def commentary(body: SectionAiRequest):
    text = _section_text(body.book_id, body.section_idx)
    kind = "commentary"
    cached = _cache_get(body.book_id, body.section_idx, kind, "", body.model, body.force)
    if isinstance(cached, dict) and "commentary" in cached:
        return cached

    prompt = build_commentary_prompt(text)
    result_text = call_ollama(prompt, body.model, SYSTEM_PROMPT_COMMENTARY)
    if result_text.startswith("ERROR:"):
        raise HTTPException(502, result_text)
    result = {"commentary": result_text}
    _cache_put(body.book_id, body.section_idx, kind, "", body.model, result)
    return result


@router.post("/question")
def question(body: SectionAiRequest):
    text = _section_text(body.book_id, body.section_idx)
    kind = "question"
    cached = _cache_get(body.book_id, body.section_idx, kind, "", body.model, body.force)
    if isinstance(cached, dict) and "question" in cached:
        return cached

    prompt = build_question_prompt(text)
    result_text = call_ollama(prompt, body.model, SYSTEM_PROMPT_QUESTION)
    if result_text.startswith("ERROR:"):
        raise HTTPException(502, result_text)
    result = {"question": result_text}
    _cache_put(body.book_id, body.section_idx, kind, "", body.model, result)
    return result


@router.post("/feedback")
def feedback(body: FeedbackRequest):
    text = _section_text(body.book_id, body.section_idx)
    prompt = build_feedback_prompt(text, body.question, body.answer)
    result = call_ollama(prompt, body.model, num_predict=1024)
    if result.startswith("ERROR:"):
        raise HTTPException(502, result)
    return {"feedback": result}


@router.post("/section-summary")
def section_summary(body: SectionAiRequest):
    """Reading-mode section summarizer (not LL English gist)."""
    text = _section_text(body.book_id, body.section_idx)
    kind = "section_summary"
    cached = _cache_get(body.book_id, body.section_idx, kind, "", body.model, body.force)
    if isinstance(cached, dict) and "summary" in cached:
        return cached

    prompt = build_summary_prompt(text)
    result_text = call_ollama(prompt, body.model)
    if result_text.startswith("ERROR:"):
        raise HTTPException(502, result_text)
    result = {"summary": result_text}
    _cache_put(body.book_id, body.section_idx, kind, "", body.model, result)
    return result
