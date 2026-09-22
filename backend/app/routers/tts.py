"""Progressive section TTS, oneshot text, word pronunciation, audiobook export."""

from __future__ import annotations

import re
import threading
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from companion.constants import EDGE_LANG_VOICES, EDGE_VOICES, KOKORO_VOICES, MMS_ITALIAN_VOICES, XTTS_LANGUAGES
from companion.tts import (
    _engine_key,
    _generate_one,
    _preprocess_tts_text,
    concat_audio_bytes,
    edge_voice_for_language,
    estimate_audio_seconds,
    generate_vocab_word_audio,
    mms_italian_available,
    split_for_tts,
)

from backend.app.store import AudiobookJob, TtsJob, TtsSegment, store

router = APIRouter()


class SectionStartRequest(BaseModel):
    book_id: str
    section_idx: int
    engine: str = "Edge TTS"  # Edge TTS | Kokoro | XTTS | MMS Italian
    voice: str = "Aria (US, Female)"
    rate: float = 1.0
    speaker_key: Optional[str] = None


class TextStartRequest(BaseModel):
    """Progressive TTS for arbitrary text (chat reply, commentary, etc.)."""
    text: str
    engine: str = "Edge TTS"
    voice: str = "Aria (US, Female)"
    rate: float = 1.0
    speaker_key: Optional[str] = None


class OneshotRequest(BaseModel):
    text: str
    engine: str = "Edge TTS"
    voice: str = "Aria (US, Female)"
    rate: float = 1.0
    speaker_key: Optional[str] = None


class WordRequest(BaseModel):
    word: str
    language: str = "Italian"
    rate: float = 1.0
    speaker_key: Optional[str] = None
    xtts_lang: Optional[str] = None
    engine: Optional[str] = None  # when MMS Italian, word ▶ uses MMS too


class AudiobookStartRequest(BaseModel):
    book_id: str
    start_idx: int = Field(..., description="Inclusive section index")
    end_idx: int = Field(..., description="Inclusive section index")
    engine: str = "Edge TTS"
    voice: str = "Aria (US, Female)"
    rate: float = 1.0
    speaker_key: Optional[str] = None


def _resolve_voice_id(engine: str, voice_label: str) -> tuple[str, str]:
    """Return (engine_key, voice_id)."""
    ek = _engine_key(engine)
    if ek == "edge":
        return ek, EDGE_VOICES.get(voice_label, voice_label)
    if ek == "kokoro":
        return ek, KOKORO_VOICES.get(voice_label, voice_label)
    if ek == "mms_italian":
        return ek, MMS_ITALIAN_VOICES.get(voice_label, voice_label or "ita")
    if voice_label in XTTS_LANGUAGES:
        return ek, XTTS_LANGUAGES[voice_label]
    if voice_label in XTTS_LANGUAGES.values():
        return ek, voice_label
    return ek, voice_label or "en"


def _speaker_bytes(ek: str, speaker_key: Optional[str]) -> bytes | None:
    if ek != "xtts":
        return None
    if speaker_key and speaker_key in store.speaker_wavs:
        return store.speaker_wavs[speaker_key]
    raise HTTPException(400, "XTTS requires a speaker WAV (POST /api/tts/speaker first)")


def _run_job(job: TtsJob) -> None:
    try:
        for seg in job.segments:
            with job.lock:
                if job.cancelled:
                    job.done = True
                    return
            try:
                audio, mime = _generate_one(
                    seg.text,
                    job.voice,
                    job.rate,
                    job.engine,
                    speaker_wav_bytes=job.speaker_wav,
                )
                dur = estimate_audio_seconds(audio, mime)
                with job.lock:
                    if job.cancelled:
                        job.done = True
                        return
                    seg.audio_bytes = audio
                    seg.mime = mime
                    seg.duration_sec = dur
                    seg.ready = True
            except Exception as e:
                with job.lock:
                    seg.error = str(e)
                    seg.ready = True
        with job.lock:
            job.done = True
    except Exception as e:
        with job.lock:
            job.error = str(e)
            job.done = True


def _start_progressive(
    texts: list[str],
    ek: str,
    voice_id: str,
    rate: float,
    speaker: bytes | None,
) -> dict:
    job_id = store.new_job_id()
    segments = [TtsSegment(index=i, text=p) for i, p in enumerate(texts)]
    job = TtsJob(
        job_id=job_id,
        engine=ek,
        voice=voice_id,
        rate=rate,
        total=len(segments),
        segments=segments,
        speaker_wav=speaker,
    )
    store.tts_jobs[job_id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True, name=f"tts-{job_id}").start()
    return {"job_id": job_id, "total_segments": len(segments), "engine": ek, "voice": voice_id}


def _split_text(text: str, ek: str) -> list[str]:
    processed = _preprocess_tts_text(text)
    if not processed:
        raise HTTPException(400, "No speakable text")
    return split_for_tts(processed, ek) or [processed]


@router.post("/speaker")
async def upload_speaker(file: UploadFile = File(...)):
    """Upload an XTTS speaker WAV; returns speaker_key for later TTS calls."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty speaker file")
    key = store.new_job_id()
    store.speaker_wavs[key] = data
    return {"speaker_key": key, "bytes": len(data)}


@router.post("/section/start")
def start_section_tts(body: SectionStartRequest):
    book = store.books.get(body.book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    section = next((s for s in book.sections if s["index"] == body.section_idx), None)
    if not section:
        raise HTTPException(404, "Section not found")

    ek, voice_id = _resolve_voice_id(body.engine, body.voice)
    speaker = _speaker_bytes(ek, body.speaker_key)
    parts = _split_text(section["text"], ek)
    return _start_progressive(parts, ek, voice_id, body.rate, speaker)


@router.post("/text/start")
def start_text_tts(body: TextStartRequest):
    """Progressive TTS for arbitrary text (chat last response, commentary, …)."""
    ek, voice_id = _resolve_voice_id(body.engine, body.voice)
    speaker = _speaker_bytes(ek, body.speaker_key)
    parts = _split_text(body.text or "", ek)
    return _start_progressive(parts, ek, voice_id, body.rate, speaker)


@router.post("/oneshot")
def oneshot_tts(body: OneshotRequest):
    """Generate a single audio clip for short/medium text (returns bytes)."""
    ek, voice_id = _resolve_voice_id(body.engine, body.voice)
    speaker = _speaker_bytes(ek, body.speaker_key)
    parts = _split_text(body.text or "", ek)
    try:
        generated = [
            _generate_one(p, voice_id, body.rate, ek, speaker_wav_bytes=speaker)
            for p in parts
        ]
        audio, mime = concat_audio_bytes(generated)
    except Exception as e:
        raise HTTPException(502, f"TTS failed: {e}") from e
    return Response(content=audio, media_type=mime)


@router.get("/jobs/{job_id}/segments")
def list_segments(job_id: str):
    job = store.tts_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    with job.lock:
        ready = []
        loaded_sec = 0.0
        for seg in job.segments:
            if seg.ready and seg.audio_bytes and not seg.error:
                ready.append(
                    {
                        "index": seg.index,
                        "duration_sec": seg.duration_sec,
                        "mime": seg.mime,
                        "url": f"/api/tts/jobs/{job_id}/segments/{seg.index}.wav",
                        "text_preview": (seg.text[:80] + "…") if len(seg.text) > 80 else seg.text,
                    }
                )
                loaded_sec += seg.duration_sec
            elif seg.ready and seg.error:
                ready.append(
                    {
                        "index": seg.index,
                        "error": seg.error,
                        "duration_sec": 0,
                    }
                )
        return {
            "job_id": job_id,
            "total": job.total,
            "ready_count": sum(1 for s in job.segments if s.ready and s.audio_bytes),
            "done": job.done,
            "cancelled": job.cancelled,
            "error": job.error,
            "loaded_sec": loaded_sec,
            "segments": ready,
        }


@router.get("/jobs/{job_id}/segments/{n}.wav")
def get_segment_audio(job_id: str, n: int):
    job = store.tts_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    with job.lock:
        if n < 0 or n >= len(job.segments):
            raise HTTPException(404, "Segment not found")
        seg = job.segments[n]
        if not seg.ready or not seg.audio_bytes:
            raise HTTPException(404, "Segment not ready")
        mime = seg.mime or "audio/wav"
        return Response(content=seg.audio_bytes, media_type=mime)


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    job = store.tts_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    with job.lock:
        job.cancelled = True
        job.done = True
    return {"ok": True, "job_id": job_id}


@router.post("/word")
def pronounce_word(body: WordRequest):
    word = (body.word or "").strip()
    if not word:
        raise HTTPException(400, "Empty word")

    eng_tag = (body.engine or "").strip() or "default"
    cache_key = (word.lower(), body.language, eng_tag, round(body.rate, 3))
    if cache_key in store.word_cache:
        audio, mime = store.word_cache[cache_key]
        return Response(content=audio, media_type=mime)

    speaker = None
    if body.speaker_key and body.speaker_key in store.speaker_wavs:
        speaker = store.speaker_wavs[body.speaker_key]

    xtts_lang = body.xtts_lang
    if not xtts_lang:
        xtts_lang = XTTS_LANGUAGES.get(body.language)

    try:
        result = generate_vocab_word_audio(
            word,
            body.language,
            body.rate,
            xtts_voice=xtts_lang,
            speaker_wav_bytes=speaker,
            preferred_engine=body.engine,
        )
    except Exception as e:
        raise HTTPException(502, str(e)) from e
    if not result:
        edge = edge_voice_for_language(body.language)
        pref = (body.engine or "").strip()
        if pref in ("MMS Italian", "mms_italian") and not mms_italian_available():
            hint = (
                'MMS Italian failed: dependencies missing. '
                'Install: pip install ".[mms]" (or: pip install transformers torch torchaudio), '
                "then restart uvicorn. Word ▶ also falls back to Edge when Edge is available."
            )
        elif not edge:
            hint = "No Edge voice for this language and no speaker WAV for XTTS fallback."
        else:
            hint = "Generation failed (MMS/Edge/XTTS all unavailable for this word)."
        raise HTTPException(502, hint)

    audio, mime = result
    store.word_cache[cache_key] = (audio, mime)
    while len(store.word_cache) > 200:
        store.word_cache.pop(next(iter(store.word_cache)))
    return Response(content=audio, media_type=mime)


def _run_audiobook(job: AudiobookJob, sections: list[dict]) -> None:
    try:
        parts: list[tuple[bytes, str]] = []
        for i, sec in enumerate(sections):
            with job.lock:
                if job.cancelled:
                    job.done = True
                    return
            processed = _preprocess_tts_text(sec.get("text") or "")
            if not processed:
                with job.lock:
                    job.done_sections = i + 1
                continue
            seg_texts = split_for_tts(processed, job.engine) or [processed]
            for stxt in seg_texts:
                with job.lock:
                    if job.cancelled:
                        job.done = True
                        return
                audio, mime = _generate_one(
                    stxt,
                    job.voice,
                    job.rate,
                    job.engine,
                    speaker_wav_bytes=job.speaker_wav,
                )
                parts.append((audio, mime))
            with job.lock:
                job.done_sections = i + 1
        if not parts:
            with job.lock:
                job.error = "No speakable text in selected range"
                job.done = True
            return
        audio, mime = concat_audio_bytes(parts)
        ext = "wav" if "wav" in mime else "mp3"
        with job.lock:
            job.audio_bytes = audio
            job.mime = mime
            job.filename = f"audiobook_{job.start_idx}-{job.end_idx}.{ext}"
            job.done = True
    except Exception as e:
        with job.lock:
            job.error = str(e)
            job.done = True


@router.post("/audiobook/start")
def start_audiobook(body: AudiobookStartRequest):
    book = store.books.get(body.book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    if body.start_idx > body.end_idx:
        raise HTTPException(400, "start_idx must be ≤ end_idx")

    selected = [s for s in book.sections if body.start_idx <= s["index"] <= body.end_idx]
    if not selected:
        raise HTTPException(404, "No sections in range")

    ek, voice_id = _resolve_voice_id(body.engine, body.voice)
    speaker = _speaker_bytes(ek, body.speaker_key)

    base = re.sub(r"\.(pdf|epub)$", "", book.filename, flags=re.IGNORECASE)
    job_id = store.new_job_id()
    job = AudiobookJob(
        job_id=job_id,
        engine=ek,
        voice=voice_id,
        rate=body.rate,
        start_idx=body.start_idx,
        end_idx=body.end_idx,
        total_sections=len(selected),
        speaker_wav=speaker,
        filename=f"{base}_audiobook_{body.start_idx}-{body.end_idx}.mp3",
    )
    store.audiobook_jobs[job_id] = job
    threading.Thread(
        target=_run_audiobook,
        args=(job, selected),
        daemon=True,
        name=f"audiobook-{job_id}",
    ).start()
    return {
        "job_id": job_id,
        "total_sections": len(selected),
        "engine": ek,
        "voice": voice_id,
    }


@router.get("/audiobook/{job_id}")
def audiobook_status(job_id: str):
    job = store.audiobook_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Audiobook job not found")
    with job.lock:
        return {
            "job_id": job_id,
            "done": job.done,
            "cancelled": job.cancelled,
            "error": job.error,
            "done_sections": job.done_sections,
            "total_sections": job.total_sections,
            "ready": bool(job.audio_bytes),
            "mime": job.mime if job.audio_bytes else None,
            "filename": job.filename if job.audio_bytes else None,
            "bytes": len(job.audio_bytes) if job.audio_bytes else 0,
            "download_url": f"/api/tts/audiobook/{job_id}/download" if job.audio_bytes else None,
        }


@router.get("/audiobook/{job_id}/download")
def audiobook_download(job_id: str):
    job = store.audiobook_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Audiobook job not found")
    with job.lock:
        if not job.audio_bytes:
            raise HTTPException(404, "Audiobook not ready")
        headers = {
            "Content-Disposition": f'attachment; filename="{job.filename}"',
        }
        return Response(content=job.audio_bytes, media_type=job.mime, headers=headers)


@router.post("/audiobook/{job_id}/stop")
def stop_audiobook(job_id: str):
    job = store.audiobook_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Audiobook job not found")
    with job.lock:
        job.cancelled = True
        job.done = True
    return {"ok": True, "job_id": job_id}


@router.get("/voices")
def list_voices():
    return {
        "edge": EDGE_VOICES,
        "kokoro": KOKORO_VOICES,
        "xtts_languages": XTTS_LANGUAGES,
        "mms_italian": MMS_ITALIAN_VOICES,
        "edge_lang_voices": dict(EDGE_LANG_VOICES),
        "mms_available": mms_italian_available(),
    }


@router.get("/engines")
def list_engines():
    """Engine availability hints for the UI (deps may still need a model download)."""
    return {
        "engines": ["Edge TTS", "Kokoro", "XTTS", "MMS Italian"],
        "mms_available": mms_italian_available(),
        "mms_install": 'pip install ".[mms]"',
    }
