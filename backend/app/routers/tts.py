"""Progressive section TTS + single-word pronunciation."""

from __future__ import annotations

import threading
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from companion.constants import EDGE_VOICES, KOKORO_VOICES, XTTS_LANGUAGES
from companion.tts import (
    _engine_key,
    _generate_one,
    _preprocess_tts_text,
    edge_voice_for_language,
    estimate_audio_seconds,
    generate_vocab_word_audio,
    split_for_tts,
)

from backend.app.store import TtsJob, TtsSegment, store

router = APIRouter()


class SectionStartRequest(BaseModel):
    book_id: str
    section_idx: int
    engine: str = "Edge TTS"  # Edge TTS | Kokoro | XTTS
    voice: str = "Aria (US, Female)"
    rate: float = 1.0
    speaker_key: Optional[str] = None  # key from /api/tts/speaker upload


class WordRequest(BaseModel):
    word: str
    language: str = "Italian"
    rate: float = 1.0
    speaker_key: Optional[str] = None
    xtts_lang: Optional[str] = None


def _resolve_voice_id(engine: str, voice_label: str) -> tuple[str, str]:
    """Return (engine_key, voice_id)."""
    ek = _engine_key(engine)
    if ek == "edge":
        return ek, EDGE_VOICES.get(voice_label, voice_label)
    if ek == "kokoro":
        return ek, KOKORO_VOICES.get(voice_label, voice_label)
    # XTTS: voice_label is language name or code
    if voice_label in XTTS_LANGUAGES:
        return ek, XTTS_LANGUAGES[voice_label]
    if voice_label in XTTS_LANGUAGES.values():
        return ek, voice_label
    return ek, voice_label or "en"


def _run_job(job: TtsJob) -> None:
    try:
        for seg in job.segments:
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
    speaker = None
    if ek == "xtts":
        if body.speaker_key and body.speaker_key in store.speaker_wavs:
            speaker = store.speaker_wavs[body.speaker_key]
        if not speaker:
            raise HTTPException(400, "XTTS requires a speaker WAV (POST /api/tts/speaker first)")

    processed = _preprocess_tts_text(section["text"])
    if not processed:
        raise HTTPException(400, "Section has no speakable text")

    parts = split_for_tts(processed, ek)
    job_id = store.new_job_id()
    segments = [
        TtsSegment(index=i, text=p) for i, p in enumerate(parts)
    ]
    job = TtsJob(
        job_id=job_id,
        engine=ek,
        voice=voice_id,
        rate=body.rate,
        total=len(segments),
        segments=segments,
        speaker_wav=speaker,
    )
    store.tts_jobs[job_id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True, name=f"tts-{job_id}").start()
    return {"job_id": job_id, "total_segments": len(segments), "engine": ek, "voice": voice_id}


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
        # Keep .wav in path for simplicity; content-type reflects actual format
        return Response(content=seg.audio_bytes, media_type=mime)


@router.post("/word")
def pronounce_word(body: WordRequest):
    word = (body.word or "").strip()
    if not word:
        raise HTTPException(400, "Empty word")

    cache_key = (word.lower(), body.language)
    if cache_key in store.word_cache:
        audio, mime = store.word_cache[cache_key]
        return Response(content=audio, media_type=mime)

    speaker = None
    if body.speaker_key and body.speaker_key in store.speaker_wavs:
        speaker = store.speaker_wavs[body.speaker_key]

    xtts_lang = body.xtts_lang
    if not xtts_lang:
        xtts_lang = XTTS_LANGUAGES.get(body.language)

    result = generate_vocab_word_audio(
        word,
        body.language,
        body.rate,
        xtts_voice=xtts_lang,
        speaker_wav_bytes=speaker,
    )
    if not result:
        edge = edge_voice_for_language(body.language)
        hint = (
            "No Edge voice for this language and no speaker WAV for XTTS fallback."
            if not edge
            else "Generation failed."
        )
        raise HTTPException(502, hint)

    audio, mime = result
    store.word_cache[cache_key] = (audio, mime)
    # Bound cache size
    while len(store.word_cache) > 200:
        store.word_cache.pop(next(iter(store.word_cache)))
    return Response(content=audio, media_type=mime)


@router.get("/voices")
def list_voices():
    return {
        "edge": EDGE_VOICES,
        "kokoro": KOKORO_VOICES,
        "xtts_languages": XTTS_LANGUAGES,
        "edge_lang_voices": {
            k: v for k, v in __import__("companion.constants", fromlist=["EDGE_LANG_VOICES"]).EDGE_LANG_VOICES.items()
        },
    }
