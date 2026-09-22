"""Ephemeral TTS/audiobook/word/speaker stores; books live in SQLite (see db.py)."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class TtsSegment:
    index: int
    text: str
    ready: bool = False
    audio_bytes: bytes | None = None
    mime: str = "audio/wav"
    duration_sec: float = 0.0
    error: str | None = None


@dataclass
class TtsJob:
    job_id: str
    engine: str
    voice: str
    rate: float
    total: int
    segments: list[TtsSegment] = field(default_factory=list)
    done: bool = False
    cancelled: bool = False
    error: str | None = None
    speaker_wav: bytes | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class AudiobookJob:
    job_id: str
    engine: str
    voice: str
    rate: float
    start_idx: int
    end_idx: int
    total_sections: int
    done_sections: int = 0
    done: bool = False
    cancelled: bool = False
    error: str | None = None
    audio_bytes: bytes | None = None
    mime: str = "audio/mp3"
    filename: str = "audiobook.mp3"
    speaker_wav: bytes | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class Store:
    def __init__(self) -> None:
        # Books/sections/AI/chat are persisted via backend.app.db
        self.tts_jobs: dict[str, TtsJob] = {}
        self.audiobook_jobs: dict[str, AudiobookJob] = {}
        self.word_cache: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.speaker_wavs: dict[str, bytes] = {}  # optional session key → wav
        self._lock = threading.Lock()

    def new_book_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def new_job_id(self) -> str:
        return uuid.uuid4().hex[:12]


store = Store()
