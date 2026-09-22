import asyncio
import base64
import hashlib
import io
import re

import edge_tts
import streamlit as st

from .constants import (
    EDGE_VOICES,
    KOKORO_VOICES,
    TTS_CACHE_MAX_ENTRIES,
    TTS_PROGRESSIVE_MIN_SEGMENTS,
    XTTS_LANGUAGES,
)

DEFAULT_EN_EDGE_VOICE = "en-US-AriaNeural"
DEFAULT_EN_KOKORO_VOICE = "af_heart"


async def _edge_generate(text: str, voice: str, rate_str: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _preprocess_tts_text(text: str) -> str:
    # Rejoin words hyphenated across lines (e.g. "some-\nthing" → "something")
    text = re.sub(r'-\n', '', text)
    # Paragraph breaks: if the preceding sentence already ends with punctuation,
    # just use a space; otherwise insert a period so edge-tts doesn't over-pause.
    text = re.sub(r'([.!?])\n\n+', r'\1 ', text)
    text = re.sub(r'([^.!?])\n\n+', r'\1. ', text)
    # Remaining single line breaks → space
    text = re.sub(r'\n', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _speak_edge(text: str, voice: str, rate: float) -> bytes:
    rate_str = f"{int((rate - 1.0) * 100):+d}%"
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_edge_generate(text, voice, rate_str))
    finally:
        loop.close()


def _speak_kokoro(text: str, voice: str, speed: float) -> bytes:
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError as e:
        raise RuntimeError(
            f"Kokoro dependencies not installed: {e}. "
            "Run: pip install kokoro>=0.9.4 soundfile"
        )
    import warnings
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # American English voices start with 'a', British with 'b'
    lang_code = "b" if voice.startswith("b") else "a"
    cache_key = f"_kokoro_pipeline_{lang_code}_{device}"
    if cache_key not in st.session_state:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.session_state[cache_key] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=device)
    pipeline = st.session_state[cache_key]

    chunks = [
        audio.detach().cpu().numpy()
        for _, _, audio in pipeline(text, voice=voice, speed=speed)
        if audio is not None
    ]
    if not chunks:
        return b""
    full_audio = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, full_audio, 24000, format="WAV")
    buf.seek(0)
    return buf.read()


def convert_audio_to_wav(audio_bytes: bytes) -> bytes:
    """Convert browser-recorded audio (WebM/OGG/etc.) to WAV bytes."""
    import io
    import soundfile as sf
    import numpy as np
    try:
        buf = io.BytesIO(audio_bytes)
        data, sr = sf.read(buf)
        out = io.BytesIO()
        sf.write(out, data.astype(np.float32), sr, format="WAV")
        out.seek(0)
        return out.read()
    except Exception:
        pass
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        out = io.BytesIO()
        seg.export(out, format="wav")
        out.seek(0)
        return out.read()
    except ImportError:
        raise RuntimeError(
            "pydub is required to convert browser audio to WAV. "
            "Run: pip install pydub  (also needs ffmpeg on PATH)"
        )
    except Exception as e:
        raise RuntimeError(f"Could not convert recorded audio to WAV: {e}")


def split_for_xtts(t: str, max_chars: int = 200) -> list:
    """Split text into speakable chunks of at most max_chars (sentence/clause/word)."""
    result, current = [], ""
    for sent in re.split(r'(?<=[.!?])\s+', t.strip()):
        parts = [sent]
        if len(sent) > max_chars:
            parts = re.split(r'(?<=[,;:])\s+', sent) or [sent]
        for part in parts:
            if not part:
                continue
            if len(part) > max_chars:
                words = part.split()
                for w in words:
                    candidate = (current + " " + w).strip() if current else w
                    if len(candidate) > max_chars and current:
                        result.append(current)
                        current = w
                    else:
                        current = candidate
            else:
                candidate = (current + " " + part).strip() if current else part
                if len(candidate) > max_chars and current:
                    result.append(current)
                    current = part
                else:
                    current = candidate
    if current:
        result.append(current)
    return result or [t]


def split_for_tts(text: str, engine: str) -> list:
    """Split processed text into progressive segments. XTTS uses ~200-char chunks;
    Edge/Kokoro use longer sentence groups (~500 chars) for fewer players."""
    processed = text if not text else text
    if engine == "xtts":
        return split_for_xtts(processed, max_chars=200)
    # Group sentences into ~500-char segments for progressive Edge/Kokoro
    max_chars = 500
    result, current = [], ""
    for sent in re.split(r'(?<=[.!?])\s+', processed.strip()):
        if not sent:
            continue
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) > max_chars and current:
            result.append(current)
            current = sent
        else:
            current = candidate
    if current:
        result.append(current)
    return result or [processed]


def _load_xtts_model():
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from huggingface_hub import snapshot_download
        import torch
        import torchaudio
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(
            f"XTTS dependencies not installed: {e}. "
            "Run: pip install TTS huggingface_hub"
        )

    # torchaudio 2.9+ delegates to torchcodec which requires FFmpeg DLLs on Windows.
    # Patch it once to use soundfile instead — soundfile handles WAV natively.
    def _sf_load(path, normalize=True, frame_offset=0, num_frames=None, **kwargs):
        frames = num_frames if (num_frames is not None and num_frames >= 0) else -1
        data, sr = sf.read(str(path), always_2d=True, start=frame_offset or 0, frames=frames)
        tensor = __import__("torch").from_numpy(data.T.copy()).float()
        return tensor, sr
    torchaudio.load = _sf_load

    model_cache_key = "_xtts_model"
    if model_cache_key not in st.session_state:
        model_dir = snapshot_download("tartuNLP/XTTS-v2-multi")
        config = XttsConfig()
        config.load_json(f"{model_dir}/config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
        if torch.cuda.is_available():
            model.cuda()
        st.session_state[model_cache_key] = model

    model = st.session_state[model_cache_key]

    _orig_preprocess = model.tokenizer.preprocess_text
    def _patched_preprocess(txt, lang):
        try:
            return _orig_preprocess(txt, lang)
        except NotImplementedError:
            return _orig_preprocess(txt, "en")
    model.tokenizer.preprocess_text = _patched_preprocess
    return model


def _get_xtts_conditioning(model, speaker_wav_bytes: bytes):
    """Cache speaker conditioning latents by WAV hash."""
    import tempfile
    import os
    spk_hash = hashlib.md5(speaker_wav_bytes).hexdigest()
    cache_key = f"_xtts_cond_{spk_hash}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(speaker_wav_bytes)
        tmp.flush()
        tmp.close()
        latents = model.get_conditioning_latents(audio_path=[tmp.name])
        st.session_state[cache_key] = latents
        return latents
    finally:
        os.unlink(tmp.name)


def _speak_xtts_chunks(text_chunks: list, language: str, speaker_wav_bytes: bytes, speed: float) -> bytes:
    """Synthesize one or more XTTS text chunks and return a single WAV."""
    import numpy as np
    import soundfile as sf

    model = _load_xtts_model()
    gpt_cond_latent, speaker_embedding = _get_xtts_conditioning(model, speaker_wav_bytes)
    wav_parts = []
    for chunk in text_chunks:
        out = model.inference(
            chunk,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            speed=speed,
        )
        wav_parts.append(np.squeeze(np.array(out["wav"])))
    wav = np.concatenate(wav_parts) if len(wav_parts) > 1 else wav_parts[0]
    if wav.size == 0:
        raise RuntimeError("XTTS returned empty audio — check your speaker WAV and language setting.")
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), 24000, format="WAV")
    buf.seek(0)
    return buf.read()


def _speak_xtts(text: str, language: str, speaker_wav_bytes: bytes, speed: float) -> bytes:
    text_chunks = split_for_xtts(text)
    return _speak_xtts_chunks(text_chunks, language, speaker_wav_bytes, speed)


def _engine_key(tts_engine: str) -> str:
    if tts_engine == "Kokoro":
        return "kokoro"
    if tts_engine == "XTTS":
        return "xtts"
    return "edge"


def _make_tts_cache_key(processed: str, voice: str, rate: float, engine: str) -> str:
    if engine == "xtts":
        spk = st.session_state.get("xtts_speaker_wav") or b""
        spk_hash = hashlib.md5(spk).hexdigest()
        return hashlib.md5(f"{processed}|xtts|{voice}|{spk_hash}".encode()).hexdigest()
    return hashlib.md5(f"{processed}|{voice}|{rate}|{engine}".encode()).hexdigest()


def _cache_put(cache_key: str, audio_bytes: bytes, fmt: str):
    cache = st.session_state.tts_cache
    cache[cache_key] = (audio_bytes, fmt)
    while len(cache) > TTS_CACHE_MAX_ENTRIES:
        cache.pop(next(iter(cache)))


def _generate_one(processed: str, voice: str, rate: float, engine: str) -> tuple[bytes, str]:
    """Generate audio for a single (already-split) text segment. Returns (bytes, mime)."""
    cache_key = _make_tts_cache_key(processed, voice, rate, engine)
    if cache_key in st.session_state.tts_cache:
        return st.session_state.tts_cache[cache_key]
    if engine == "kokoro":
        audio_bytes = _speak_kokoro(processed, voice, rate)
        fmt = "audio/wav"
    elif engine == "xtts":
        if not st.session_state.xtts_speaker_wav:
            raise RuntimeError("Upload a speaker WAV file in the sidebar to use XTTS.")
        # Single segment — do not re-split
        audio_bytes = _speak_xtts_chunks(
            [processed], voice, st.session_state.xtts_speaker_wav, rate
        )
        fmt = "audio/wav"
    else:
        audio_bytes = _speak_edge(processed, voice, rate)
        fmt = "audio/mp3"
    _cache_put(cache_key, audio_bytes, fmt)
    return audio_bytes, fmt



def concat_audio_bytes(parts: list) -> tuple[bytes, str]:
    """Concatenate segment (bytes, mime) pairs into one audio buffer.

    WAV engines (XTTS/Kokoro): soundfile/numpy PCM concat.
    MP3 (Edge): pydub concat when available — never raw-byte-join MP3 frames.
    """
    if not parts:
        return b"", "audio/wav"
    if len(parts) == 1:
        return parts[0][0], parts[0][1]

    all_wav = all("wav" in (fmt or "") for _, fmt in parts)
    if all_wav:
        import numpy as np
        import soundfile as sf

        arrays = []
        sr = 24000
        for ab, _ in parts:
            data, sr = sf.read(io.BytesIO(ab))
            arrays.append(data)
        out = io.BytesIO()
        sf.write(out, np.concatenate(arrays), int(sr), format="WAV")
        return out.getvalue(), "audio/wav"

    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise RuntimeError(
            "pydub is required to concatenate Edge/MP3 progressive clips. "
            "Install pydub (and ffmpeg on PATH), or use Kokoro/XTTS for progressive TTS."
        ) from e

    combined = None
    for ab, fmt in parts:
        fmt_l = (fmt or "").lower()
        file_fmt = "wav" if "wav" in fmt_l else "mp3"
        seg = AudioSegment.from_file(io.BytesIO(ab), format=file_fmt)
        combined = seg if combined is None else combined + seg
    out = io.BytesIO()
    combined.export(out, format="mp3")
    return out.getvalue(), "audio/mp3"


def estimate_audio_seconds(audio_bytes: bytes, fmt: str) -> float:
    """Best-effort duration in seconds for caption display."""
    if not audio_bytes:
        return 0.0
    try:
        if "wav" in (fmt or ""):
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(audio_bytes))
            return float(len(data)) / float(sr or 24000)
        from pydub import AudioSegment

        file_fmt = "mp3" if "mp3" in (fmt or "") else "wav"
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_fmt)
        return len(seg) / 1000.0
    except Exception:
        if "wav" in (fmt or ""):
            return max(0.0, (len(audio_bytes) - 44) / 48000.0)
        return max(0.0, len(audio_bytes) / 16000.0)


def _rebuild_concat_audio() -> None:
    """Set tts_audio/tts_format to the concatenation of all segments so far."""
    audios = st.session_state.get("tts_segment_audios") or []
    if not audios:
        st.session_state.tts_audio = b""
        return
    audio_bytes, fmt = concat_audio_bytes(audios)
    st.session_state.tts_audio = audio_bytes
    st.session_state.tts_format = fmt


def _bump_tts_player_gen(source: str = "") -> None:
    st.session_state.tts_player_gen = int(st.session_state.get("tts_player_gen") or 0) + 1
    st.session_state.tts_player_source = source or st.session_state.get("tts_source") or ""


def _render_continuity_audio(audio_bytes: bytes, fmt: str, *, autoplay: bool) -> bool:
    """Render one <audio> element that restores playback position across remounts.

    Returns True if the HTML component rendered; False to fall back to st.audio.
    """
    try:
        import streamlit.components.v1 as components
    except Exception:
        return False
    if not audio_bytes:
        return False

    mime = fmt if "/" in (fmt or "") else f"audio/{fmt or 'wav'}"
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    gen = int(st.session_state.get("tts_player_gen") or 0)
    source = st.session_state.get("tts_source") or "tts"
    player_id = f"{source}_{gen}"
    auto_js = "true" if autoplay else "false"
    # base64 charset is URL/JS-safe; keep script compact for iframe height
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:transparent;">
<audio id="a" controls style="width:100%;height:40px;"></audio>
<script>
(function(){{
  const KEY="rc_tts_"+{player_id!r};
  const AUTO={auto_js};
  const audio=document.getElementById("a");
  let state={{t:0,playing:false}};
  try{{ const raw=sessionStorage.getItem(KEY); if(raw) state=JSON.parse(raw); }}catch(e){{}}
  function persist(){{
    try{{ sessionStorage.setItem(KEY, JSON.stringify({{t:audio.currentTime||0, playing:!audio.paused}})); }}catch(e){{}}
  }}
  audio.addEventListener("timeupdate", persist);
  audio.addEventListener("play", persist);
  audio.addEventListener("pause", persist);
  audio.src="data:{mime};base64,{b64}";
  audio.addEventListener("loadedmetadata", function(){{
    const dur=audio.duration||0;
    const resume=state.t>0.2 && (!dur || state.t<dur);
    if(resume){{ try{{ audio.currentTime=state.t; }}catch(e){{}} }}
    if(state.playing || (AUTO && !resume)){{
      audio.play().catch(function(){{}});
    }}
  }});
}})();
</script></body></html>"""
    components.html(html, height=52)
    return True


def clear_progressive_tts():
    st.session_state.tts_segments = []
    st.session_state.tts_segment_audios = []
    st.session_state.tts_segments_done = 0
    st.session_state.tts_progressive_active = False
    st.session_state.tts_progressive_meta = {}
    st.session_state.tts_progressive_just_added = -1


def stop_speech():
    st.session_state.tts_audio = b""
    st.session_state.tts_source = ""
    st.session_state.tts_voice_note = ""
    st.session_state.tts_player_gen = int(st.session_state.get("tts_player_gen") or 0) + 1
    clear_progressive_tts()


def resolve_english_voice(tts_engine: str, tts_voice: str) -> tuple[str, str, str]:
    """Pick an English-capable engine/voice for English content.

    Returns (display_engine, voice_id, engine_key).
    Prefers keeping Kokoro/Edge if already selected; swaps XTTS → Kokoro English
    (af_heart). Callers should fall back to Edge if Kokoro is unavailable.
    """
    if tts_engine == "Kokoro":
        return "Kokoro", tts_voice, "kokoro"
    if tts_engine == "Edge TTS":
        return "Edge TTS", tts_voice, "edge"
    # XTTS (or unknown): prefer local Kokoro English voice
    return "Kokoro", DEFAULT_EN_KOKORO_VOICE, "kokoro"


def speak_text(
    text: str,
    voice: str,
    rate: float,
    engine: str = "edge",
    source: str = "",
    progressive: bool = True,
    fallback_edge: bool = False,
):
    """Generate TTS. For long text, start progressive playback (first segment ASAP).

    If fallback_edge and Kokoro fails (missing deps), retry once with Edge English.
    """
    st.session_state.tts_error = ""
    processed = _preprocess_tts_text(text)
    if not processed:
        return

    segments = split_for_tts(processed, engine) if progressive else [processed]
    use_progressive = progressive and len(segments) >= TTS_PROGRESSIVE_MIN_SEGMENTS

    # Edge MP3 progressive needs pydub for safe concat — otherwise one-shot.
    if use_progressive and engine == "edge":
        try:
            from pydub import AudioSegment  # noqa: F401
        except ImportError:
            use_progressive = False

    if not use_progressive:
        # Full one-shot (short text or progressive disabled)
        cache_key = _make_tts_cache_key(processed, voice, rate, engine)
        if cache_key in st.session_state.tts_cache:
            audio_bytes, fmt = st.session_state.tts_cache[cache_key]
            st.session_state.tts_audio = audio_bytes
            st.session_state.tts_format = fmt
            st.session_state.tts_source = source
            clear_progressive_tts()
            _bump_tts_player_gen(source)
            return
        try:
            if engine == "kokoro":
                audio_bytes = _speak_kokoro(processed, voice, rate)
                fmt = "audio/wav"
            elif engine == "xtts":
                if not st.session_state.xtts_speaker_wav:
                    st.error("Upload a speaker WAV file in the sidebar to use XTTS.")
                    return
                audio_bytes = _speak_xtts(processed, voice, st.session_state.xtts_speaker_wav, rate)
                fmt = "audio/wav"
            else:
                audio_bytes = _speak_edge(processed, voice, rate)
                fmt = "audio/mp3"
            _cache_put(cache_key, audio_bytes, fmt)
            st.session_state.tts_audio = audio_bytes
            st.session_state.tts_format = fmt
            st.session_state.tts_source = source
            clear_progressive_tts()
            _bump_tts_player_gen(source)
        except Exception as e:
            if fallback_edge and engine == "kokoro":
                st.session_state.tts_voice_note = (
                    (st.session_state.get("tts_voice_note") or "")
                    + (" · " if st.session_state.get("tts_voice_note") else "")
                    + f"Kokoro unavailable; using Edge English ({e})"
                ).strip(" ·")
                speak_text(
                    text,
                    DEFAULT_EN_EDGE_VOICE,
                    rate,
                    "edge",
                    source=source,
                    progressive=progressive,
                    fallback_edge=False,
                )
                return
            import traceback
            st.session_state.tts_error = f"TTS error ({engine}): {e}\n\n```\n{traceback.format_exc()}\n```"
        return

    # Progressive path: generate segment 0 now; continue on later runs
    clear_progressive_tts()
    _bump_tts_player_gen(source)
    st.session_state.tts_segments = segments
    st.session_state.tts_segment_audios = []
    st.session_state.tts_segments_done = 0
    st.session_state.tts_progressive_meta = {
        "voice": voice,
        "rate": rate,
        "engine": engine,
        "source": source,
    }
    st.session_state.tts_source = source
    try:
        audio_bytes, fmt = _generate_one(segments[0], voice, rate, engine)
        st.session_state.tts_segment_audios = [(audio_bytes, fmt)]
        st.session_state.tts_segments_done = 1
        st.session_state.tts_audio = audio_bytes
        st.session_state.tts_format = fmt
        st.session_state.tts_progressive_just_added = 0
        st.session_state.tts_progressive_active = len(segments) > 1
    except Exception as e:
        if fallback_edge and engine == "kokoro":
            clear_progressive_tts()
            st.session_state.tts_voice_note = (
                (st.session_state.get("tts_voice_note") or "")
                + (" · " if st.session_state.get("tts_voice_note") else "")
                + f"Kokoro unavailable; using Edge English ({e})"
            ).strip(" ·")
            speak_text(
                text,
                DEFAULT_EN_EDGE_VOICE,
                rate,
                "edge",
                source=source,
                progressive=progressive,
                fallback_edge=False,
            )
            return
        import traceback
        st.session_state.tts_error = f"TTS error ({engine}): {e}\n\n```\n{traceback.format_exc()}\n```"
        clear_progressive_tts()


def continue_progressive_tts() -> bool:
    """Generate the next progressive segment if any remain.

    Returns True if more work may remain (caller should schedule another run).
    """
    if not st.session_state.get("tts_progressive_active"):
        return False
    segments = st.session_state.get("tts_segments") or []
    done = int(st.session_state.get("tts_segments_done") or 0)
    meta = st.session_state.get("tts_progressive_meta") or {}
    if done >= len(segments) or not meta:
        st.session_state.tts_progressive_active = False
        st.session_state.tts_progressive_just_added = -1
        return False
    voice = meta["voice"]
    rate = meta["rate"]
    engine = meta["engine"]
    try:
        audio_bytes, fmt = _generate_one(segments[done], voice, rate, engine)
        st.session_state.tts_segment_audios.append((audio_bytes, fmt))
        st.session_state.tts_segments_done = done + 1
        st.session_state.tts_progressive_just_added = done
        # Grow the single playback buffer (concat of all segments so far).
        _rebuild_concat_audio()
        if done + 1 >= len(segments):
            st.session_state.tts_progressive_active = False
            return False
        return True
    except Exception as e:
        import traceback
        st.session_state.tts_error = f"TTS error ({engine}): {e}\n\n```\n{traceback.format_exc()}\n```"
        st.session_state.tts_progressive_active = False
        return False


def _tts_cached(text: str, voice: str, rate: float, engine: str) -> bool:
    processed = _preprocess_tts_text(text)
    key = _make_tts_cache_key(processed, voice, rate, engine)
    if key in st.session_state.tts_cache:
        return True
    # Also treat as cached when every progressive segment is cached
    segments = split_for_tts(processed, engine)
    if len(segments) < TTS_PROGRESSIVE_MIN_SEGMENTS:
        return False
    return all(
        _make_tts_cache_key(seg, voice, rate, engine) in st.session_state.tts_cache
        for seg in segments
    )


def _get_or_generate_audio(text: str, voice: str, rate: float, engine: str) -> bytes:
    """Return audio bytes from cache or generate them, without touching tts_audio state."""
    processed = _preprocess_tts_text(text)
    cache_key = _make_tts_cache_key(processed, voice, rate, engine)
    if cache_key in st.session_state.tts_cache:
        return st.session_state.tts_cache[cache_key][0]
    if engine == "kokoro":
        audio_bytes = _speak_kokoro(processed, voice, rate)
        fmt = "audio/wav"
    elif engine == "xtts":
        audio_bytes = _speak_xtts(processed, voice, st.session_state.xtts_speaker_wav, rate)
        fmt = "audio/wav"
    else:
        audio_bytes = _speak_edge(processed, voice, rate)
        fmt = "audio/mp3"
    _cache_put(cache_key, audio_bytes, fmt)
    return audio_bytes


def speak_bilingual_vocab(
    entries: list,
    book_voice: str,
    book_engine: str,
    rate: float,
    source: str = "ll_vocab",
):
    """Speak vocab as alternating book-language word + English translation clips.

    Uses XTTS (or current book engine) for words and Edge English for translations.
    Progressive playlist so the first word plays ASAP.
    """
    st.session_state.tts_error = ""
    clear_progressive_tts()

    book_ek = _engine_key(book_engine) if book_engine in ("Edge TTS", "Kokoro", "XTTS") else book_engine
    if book_ek not in ("edge", "kokoro", "xtts"):
        book_ek = "xtts"

    # Build playlist: each entry → (word, book) then (translation, edge)
    jobs = []
    for v in entries:
        word = (v.get("word") or "").strip()
        translation = (v.get("translation") or "").strip()
        if word:
            jobs.append((_preprocess_tts_text(word), book_voice, book_ek))
        if translation:
            jobs.append((_preprocess_tts_text(translation), DEFAULT_EN_EDGE_VOICE, "edge"))

    if not jobs:
        return

    st.session_state.tts_segments = [j[0] for j in jobs]
    st.session_state.tts_segment_audios = []
    st.session_state.tts_segments_done = 0
    # Store per-segment voice/engine in meta
    st.session_state.tts_progressive_meta = {
        "jobs": [{"voice": j[1], "engine": j[2]} for j in jobs],
        "source": source,
        "bilingual": True,
        "voice": jobs[0][1],
        "rate": rate,
        "engine": jobs[0][2],
    }
    st.session_state.tts_source = source
    st.session_state.tts_voice_note = "Vocabulary: book-language words + English translations"
    _bump_tts_player_gen(source)

    try:
        text0, voice0, eng0 = jobs[0]
        audio_bytes, fmt = _generate_one(text0, voice0, rate, eng0)
        st.session_state.tts_segment_audios = [(audio_bytes, fmt)]
        st.session_state.tts_segments_done = 1
        st.session_state.tts_audio = audio_bytes
        st.session_state.tts_format = fmt
        st.session_state.tts_progressive_just_added = 0
        st.session_state.tts_progressive_active = len(jobs) > 1
    except Exception as e:
        import traceback
        st.session_state.tts_error = f"TTS error: {e}\n\n```\n{traceback.format_exc()}\n```"
        clear_progressive_tts()


def continue_progressive_tts_bilingual(rate: float) -> bool:
    """Continue bilingual vocab playlist using per-segment jobs in meta."""
    meta = st.session_state.get("tts_progressive_meta") or {}
    jobs = meta.get("jobs") or []
    done = int(st.session_state.get("tts_segments_done") or 0)
    segments = st.session_state.get("tts_segments") or []
    if done >= len(jobs) or done >= len(segments):
        st.session_state.tts_progressive_active = False
        return False
    job = jobs[done]
    try:
        audio_bytes, fmt = _generate_one(segments[done], job["voice"], rate, job["engine"])
        st.session_state.tts_segment_audios.append((audio_bytes, fmt))
        st.session_state.tts_segments_done = done + 1
        st.session_state.tts_progressive_just_added = done
        _rebuild_concat_audio()
        if done + 1 >= len(jobs):
            st.session_state.tts_progressive_active = False
            return False
        return True
    except Exception as e:
        import traceback
        st.session_state.tts_error = f"TTS error: {e}\n\n```\n{traceback.format_exc()}\n```"
        st.session_state.tts_progressive_active = False
        return False


def tts_button(
    label: str,
    text: str,
    source: str,
    tts_voice: str,
    tts_rate: float,
    tts_engine: str,
    full_width: bool = False,
    engine_override: str | None = None,
    voice_override: str | None = None,
    voice_note: str | None = None,
    progressive: bool = True,
    bilingual_vocab: list | None = None,
    fallback_edge: bool = False,
):
    """Render a read-aloud button.

    Optional overrides route English content away from a non-English XTTS setup.
    If bilingual_vocab is provided, uses alternating book/English clips.
    fallback_edge: if Kokoro fails, retry with Edge English (for summary TTS).
    """
    engine_label = engine_override or tts_engine
    voice = voice_override if voice_override is not None else tts_voice
    engine_key = _engine_key(engine_label)

    is_playing = bool(st.session_state.tts_audio) and st.session_state.tts_source == source
    btn_label = f"▶ {label}" if is_playing else label

    if st.button(btn_label, key=f"btn_read_{source}", use_container_width=full_width):
        st.session_state.tts_voice_note = voice_note or ""
        if bilingual_vocab is not None:
            with st.spinner("Generating first clip…"):
                speak_bilingual_vocab(
                    bilingual_vocab,
                    book_voice=tts_voice,
                    book_engine=tts_engine,
                    rate=tts_rate,
                    source=source,
                )
            st.rerun()
            return

        cached = _tts_cached(text, voice, tts_rate, engine_key)
        if cached:
            speak_text(
                text, voice, tts_rate, engine_key,
                source=source, progressive=progressive, fallback_edge=fallback_edge,
            )
        else:
            spinner_msg = (
                "Generating first audio…"
                if progressive and len(split_for_tts(_preprocess_tts_text(text), engine_key)) >= TTS_PROGRESSIVE_MIN_SEGMENTS
                else "Generating audio..."
            )
            with st.spinner(spinner_msg):
                speak_text(
                    text, voice, tts_rate, engine_key,
                    source=source, progressive=progressive, fallback_edge=fallback_edge,
                )
        st.rerun()




def _advance_progressive_one() -> bool:
    """Generate exactly one pending progressive segment. Returns True if more remain."""
    if not st.session_state.get("tts_progressive_active"):
        return False
    meta = st.session_state.get("tts_progressive_meta") or {}
    rate = float(meta.get("rate") or 1.0)
    if meta.get("bilingual"):
        return continue_progressive_tts_bilingual(rate)
    return continue_progressive_tts()


def render_tts_player():
    """Render a single growing TTS player (concat of segments so far)."""
    if st.session_state.tts_error:
        st.error(st.session_state.tts_error)
        if st.button("Dismiss error", key="btn_dismiss_tts_error"):
            st.session_state.tts_error = ""
            st.rerun()

    note = st.session_state.get("tts_voice_note") or ""
    if note:
        st.caption(note)

    audios = st.session_state.get("tts_segment_audios") or []
    total = len(st.session_state.get("tts_segments") or [])
    done = int(st.session_state.get("tts_segments_done") or 0)
    just = int(st.session_state.get("tts_progressive_just_added", -1))
    active = bool(st.session_state.get("tts_progressive_active"))
    audio_bytes = st.session_state.get("tts_audio") or b""
    fmt = st.session_state.get("tts_format") or "audio/wav"

    # Progressive / multi-segment: one bar that grows as chunks append
    if total > 1 and (audios or audio_bytes):
        secs = estimate_audio_seconds(audio_bytes, fmt)
        if active:
            st.caption(f"Loaded {done}/{total} · ~{int(round(secs))}s so far")
        elif done >= total > 0:
            st.caption(f"Ready · {done}/{total} · ~{int(round(secs))}s")
        else:
            st.caption(f"Loaded {done}/{total} · ~{int(round(secs))}s so far")

        autoplay = just == 0  # only auto-start on first segment
        col_audio, col_stop = st.columns([5, 1])
        with col_audio:
            used_html = _render_continuity_audio(audio_bytes, fmt, autoplay=autoplay)
            if not used_html:
                # Fallback: single st.audio (may reset seek on remount when buffer grows)
                st.audio(audio_bytes, format=fmt, autoplay=autoplay)
        with col_stop:
            if st.button("Stop", key="btn_stop"):
                stop_speech()
                st.rerun()
        if just >= 0:
            st.session_state.tts_progressive_just_added = -1
        return

    if audio_bytes:
        col_audio, col_stop = st.columns([5, 1])
        with col_audio:
            used_html = _render_continuity_audio(audio_bytes, fmt, autoplay=True)
            if not used_html:
                st.audio(audio_bytes, format=fmt, autoplay=True)
        with col_stop:
            if st.button("Stop", key="btn_stop"):
                stop_speech()
                st.rerun()


def _progressive_fragment_tick():
    """Show playlist and advance one segment; fragment run_every re-invokes us."""
    was_active = bool(st.session_state.get("tts_progressive_active"))
    render_tts_player()
    if st.session_state.get("tts_progressive_active"):
        _advance_progressive_one()
        # When the last segment finishes, full-rerun so the main script drops
        # the run_every polling fragment and shows the static playlist.
        if was_active and not st.session_state.get("tts_progressive_active"):
            st.rerun()


# Build fragment wrappers once at import (no-op host if st.fragment missing).
_HAS_FRAGMENT = hasattr(st, "fragment")
if _HAS_FRAGMENT:
    try:
        from datetime import timedelta
        _progressive_fragment_polling = st.fragment(run_every=timedelta(seconds=0.7))(
            _progressive_fragment_tick
        )
    except TypeError:
        _progressive_fragment_polling = st.fragment(_progressive_fragment_tick)
    _progressive_fragment_static = st.fragment(render_tts_player)
else:
    _progressive_fragment_polling = None
    _progressive_fragment_static = None


def maybe_continue_progressive():
    """Render TTS UI; when progressive, keep generating via fragment polling."""
    active = bool(st.session_state.get("tts_progressive_active"))
    has_playlist = len(st.session_state.get("tts_segment_audios") or []) > 0 and (
        len(st.session_state.get("tts_segments") or []) > 1
    )

    if _HAS_FRAGMENT and (active or has_playlist):
        if active and _progressive_fragment_polling is not None:
            _progressive_fragment_polling()
        elif _progressive_fragment_static is not None:
            _progressive_fragment_static()
        else:
            render_tts_player()
        return

    # No fragment support: show first clip, then finish remaining in this script run.
    render_tts_player()
    if not active:
        return
    safety = 0
    while st.session_state.get("tts_progressive_active") and safety < 200:
        _advance_progressive_one()
        safety += 1
