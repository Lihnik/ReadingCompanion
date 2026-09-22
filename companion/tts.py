import asyncio
import base64
import hashlib
import io
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import edge_tts
import streamlit as st

from .constants import (
    EDGE_LANG_VOICES,
    KOKORO_VOICES,
    TTS_CACHE_MAX_ENTRIES,
    TTS_PROGRESSIVE_MIN_SEGMENTS,
    XTTS_LANGUAGES,
)

DEFAULT_EN_EDGE_VOICE = "en-US-AriaNeural"
DEFAULT_EN_KOKORO_VOICE = "af_heart"


# Background vocab prefetch (threads write here; drained into session_state each run)
_VOCAB_PREFETCH_LOCK = threading.Lock()
_VOCAB_PREFETCH_RESULTS: dict = {}
_VOCAB_PREFETCH_PENDING: set = set()


def _vocab_cache_key(word: str, book_language: str, voice: str) -> tuple:
    return ((word or "").strip(), book_language or "", voice or "")


def drain_vocab_prefetch_into_session() -> None:
    """Merge background-prefetch results into session_state.vocab_audio_cache."""
    if "vocab_audio_cache" not in st.session_state:
        st.session_state.vocab_audio_cache = {}
    with _VOCAB_PREFETCH_LOCK:
        items = list(_VOCAB_PREFETCH_RESULTS.items())
    cache = st.session_state.vocab_audio_cache
    tts_cache = st.session_state.get("tts_cache")
    for key, (audio_bytes, fmt) in items:
        cache[key] = (audio_bytes, fmt)
        if tts_cache is not None:
            word, _lang, voice = key
            ck = _make_tts_cache_key(word, voice, 1.0, "edge")
            if ck not in tts_cache:
                tts_cache[ck] = (audio_bytes, fmt)


def get_vocab_audio_cached(word: str, book_language: str, tts_rate: float = 1.0):
    """Return (bytes, mime) if cached, else None."""
    drain_vocab_prefetch_into_session()
    word = (word or "").strip()
    if not word:
        return None
    edge_voice = edge_voice_for_language(book_language)
    if not edge_voice:
        return None
    key = _vocab_cache_key(word, book_language, edge_voice)
    hit = st.session_state.get("vocab_audio_cache", {}).get(key)
    if hit:
        return hit
    ck = _make_tts_cache_key(word, edge_voice, tts_rate, "edge")
    if ck in st.session_state.get("tts_cache", {}):
        return st.session_state.tts_cache[ck]
    return None


def generate_vocab_word_audio(
    word: str,
    book_language: str,
    tts_rate: float,
    xtts_voice: str | None = None,
) -> tuple[bytes, str] | None:
    """Generate one vocab word clip and cache it. Does NOT touch progressive player state."""
    word = (word or "").strip()
    if not word:
        return None
    if "vocab_audio_cache" not in st.session_state:
        st.session_state.vocab_audio_cache = {}

    edge_voice = edge_voice_for_language(book_language)
    if edge_voice:
        key = _vocab_cache_key(word, book_language, edge_voice)
        hit = st.session_state.vocab_audio_cache.get(key)
        if hit:
            return hit
        try:
            cache_key = _make_tts_cache_key(word, edge_voice, tts_rate, "edge")
            if cache_key in st.session_state.tts_cache:
                audio_bytes, fmt = st.session_state.tts_cache[cache_key]
            else:
                audio_bytes = _speak_edge(word, edge_voice, tts_rate)
                fmt = "audio/mp3"
                _cache_put(cache_key, audio_bytes, fmt)
            st.session_state.vocab_audio_cache[key] = (audio_bytes, fmt)
            return audio_bytes, fmt
        except Exception:
            pass

    lang = (xtts_voice or "").strip() or XTTS_LANGUAGES.get(book_language) or "en"
    if not st.session_state.get("xtts_speaker_wav"):
        return None
    prompt = word.rstrip(".!?…") + "."
    vkey = _vocab_cache_key(word, book_language, f"xtts:{lang}")
    hit = st.session_state.vocab_audio_cache.get(vkey)
    if hit:
        return hit
    cache_key = _make_tts_cache_key(f"vocab:{prompt}", lang, tts_rate, "xtts")
    try:
        if cache_key in st.session_state.tts_cache:
            audio_bytes, fmt = st.session_state.tts_cache[cache_key]
        else:
            audio_bytes = _speak_xtts_chunks(
                [prompt],
                lang,
                st.session_state.xtts_speaker_wav,
                tts_rate,
                word_mode=True,
            )
            fmt = "audio/wav"
            _cache_put(cache_key, audio_bytes, fmt)
        st.session_state.vocab_audio_cache[vkey] = (audio_bytes, fmt)
        return audio_bytes, fmt
    except Exception:
        return None


def _prefetch_one_edge_word(word: str, voice: str, rate: float, book_language: str) -> None:
    word = (word or "").strip()
    if not word or not voice:
        return
    key = _vocab_cache_key(word, book_language, voice)
    with _VOCAB_PREFETCH_LOCK:
        if key in _VOCAB_PREFETCH_RESULTS or key in _VOCAB_PREFETCH_PENDING:
            return
        _VOCAB_PREFETCH_PENDING.add(key)
    try:
        audio_bytes = _speak_edge(word, voice, rate)
        with _VOCAB_PREFETCH_LOCK:
            _VOCAB_PREFETCH_RESULTS[key] = (audio_bytes, "audio/mp3")
    except Exception:
        pass
    finally:
        with _VOCAB_PREFETCH_LOCK:
            _VOCAB_PREFETCH_PENDING.discard(key)


def prefetch_vocab_audio(entries: list, book_language: str, tts_rate: float = 1.0) -> None:
    """Background-prefetch Edge audio for vocab words (non-blocking)."""
    edge_voice = edge_voice_for_language(book_language)
    if not edge_voice:
        return
    words = []
    for v in entries or []:
        if isinstance(v, dict):
            w = (v.get("word") or "").strip()
        else:
            w = (str(v) if v is not None else "").strip()
        if w:
            words.append(w)
    if not words:
        return
    seen = set()
    uniq = []
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            uniq.append(w)

    drain_vocab_prefetch_into_session()
    pending = []
    for w in uniq:
        key = _vocab_cache_key(w, book_language, edge_voice)
        if key in st.session_state.get("vocab_audio_cache", {}):
            continue
        with _VOCAB_PREFETCH_LOCK:
            if key in _VOCAB_PREFETCH_RESULTS or key in _VOCAB_PREFETCH_PENDING:
                continue
        pending.append(w)
    if not pending:
        return

    def _run(batch):
        workers = min(4, len(batch))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(
                pool.map(
                    lambda w: _prefetch_one_edge_word(w, edge_voice, tts_rate, book_language),
                    batch,
                )
            )

    threading.Thread(target=_run, args=(pending,), daemon=True, name="vocab-prefetch").start()


def render_inline_vocab_audio_button(
    word: str,
    audio_bytes: bytes,
    fmt: str,
    *,
    btn_id: str,
    autoplay: bool = False,
) -> bool:
    """HTML ▶ + hidden <audio> — plays in-browser with zero Streamlit rerun."""
    try:
        import streamlit.components.v1 as components
    except Exception:
        return False
    if not audio_bytes:
        return False
    mime = fmt if "/" in (fmt or "") else f"audio/{fmt or 'mp3'}"
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    safe_label = (word or "play")[:40].replace("<", "").replace(">", "").replace('"', "")
    auto_js = "true" if autoplay else "false"
    _ = btn_id
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:transparent;">
<button id="b" title="Pronounce: {safe_label}" style="
  width:100%;height:32px;border-radius:8px;border:1px solid rgba(255,255,255,.25);
  background:rgba(255,220,130,.18);color:rgba(255,230,160,.95);cursor:pointer;
  font-size:14px;line-height:1;">▶</button>
<audio id="a" preload="auto" src="data:{mime};base64,{b64}"></audio>
<script>
(function(){{
  const btn=document.getElementById("b");
  const audio=document.getElementById("a");
  const AUTO={auto_js};
  function play(){{
    try{{ audio.currentTime=0; }}catch(e){{}}
    audio.play().catch(function(){{}});
  }}
  btn.addEventListener("click", function(ev){{ ev.preventDefault(); play(); }});
  if(AUTO){{ play(); }}
}})();
</script></body></html>"""
    components.html(html, height=40)
    return True



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


def _speak_xtts_chunks(
    text_chunks: list,
    language: str,
    speaker_wav_bytes: bytes,
    speed: float,
    *,
    word_mode: bool = False,
) -> bytes:
    """Synthesize one or more XTTS text chunks and return a single WAV.

    word_mode: tighter sampling for isolated words (less autoregressive ramble).
    """
    import numpy as np
    import soundfile as sf

    model = _load_xtts_model()
    gpt_cond_latent, speaker_embedding = _get_xtts_conditioning(model, speaker_wav_bytes)
    wav_parts = []
    for chunk in text_chunks:
        kwargs = {
            "speed": speed,
            "enable_text_splitting": False,
        }
        if word_mode:
            # Lower temperature + stronger repetition penalty curb short-prompt hallucination.
            kwargs.update(
                temperature=0.55,
                length_penalty=1.0,
                repetition_penalty=5.0,
                top_k=30,
                top_p=0.8,
            )
        out = model.inference(
            chunk,
            language,
            gpt_cond_latent,
            speaker_embedding,
            **kwargs,
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



def _segment_payload(segments: list) -> list:
    """Serialize segment (bytes, mime) pairs to [{mime,b64}, ...] for JS."""
    segs = []
    for ab, fmt in segments or []:
        if not ab:
            continue
        mime = fmt if "/" in (fmt or "") else f"audio/{fmt or 'wav'}"
        segs.append({"mime": mime, "b64": base64.b64encode(ab).decode("ascii")})
    return segs


def _progressive_player_id() -> str:
    gen = int(st.session_state.get("tts_player_gen") or 0)
    source = st.session_state.get("tts_source") or "tts"
    return f"{source}_{gen}"


def _push_segments_to_browser(segments: list) -> bool:
    """Write current segment list into parent/localStorage for the persistent player.

    Safe to remount every fragment tick — does not create or touch an <audio> element.
    """
    try:
        import json
        import streamlit.components.v1 as components
    except Exception:
        return False
    segs = _segment_payload(segments)
    if not segs:
        return False
    player_id = _progressive_player_id()
    segs_json = json.dumps(segs, separators=(",", ":"))
    html = f"""<!DOCTYPE html><html><body style="margin:0;height:0;overflow:hidden;">
<script>
(function(){{
  const ID={player_id!r};
  const SEGS={segs_json};
  try {{
    window.parent.__rc_tts = window.parent.__rc_tts || {{}};
    window.parent.__rc_tts[ID] = SEGS;
  }} catch(e) {{}}
  try {{
    localStorage.setItem("rc_tts_segs_"+ID, JSON.stringify(SEGS));
  }} catch(e) {{}}
}})();
</script></body></html>"""
    components.html(html, height=0)
    return True


def _render_persistent_queue_player(*, autoplay: bool) -> bool:
    """One persistent <audio> that drains segments from parent/localStorage.

    HTML depends only on player_id + autoplay (NOT segment count), so fragment
    reruns that remount a sibling pusher do not recreate this element when the
    player itself is mounted outside the polling fragment.
    """
    try:
        import streamlit.components.v1 as components
    except Exception:
        return False

    player_id = _progressive_player_id()
    auto_js = "true" if autoplay else "false"
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:transparent;">
<audio id="a" controls style="width:100%;height:40px;"></audio>
<script>
(function(){{
  const ID={player_id!r};
  const AUTO={auto_js};
  const STATE_KEY="rc_tts_q_"+ID;
  const SEGS_KEY="rc_tts_segs_"+ID;
  const audio=document.getElementById("a");
  let segs=[];
  let state={{i:0,t:0,playing:false}};
  let loading=false;
  let lastLen=-1;

  try{{ const raw=sessionStorage.getItem(STATE_KEY); if(raw) state=JSON.parse(raw); }}catch(e){{}}
  if(!Number.isFinite(state.i) || state.i<0) state.i=0;

  function persist(){{
    try{{
      sessionStorage.setItem(STATE_KEY, JSON.stringify({{
        i: Number(audio.dataset.seg||state.i||0),
        t: audio.currentTime||0,
        playing: !audio.paused
      }}));
    }}catch(e){{}}
  }}

  function readSegs(){{
    try {{
      if(window.parent.__rc_tts && window.parent.__rc_tts[ID]) {{
        return window.parent.__rc_tts[ID] || [];
      }}
    }} catch(e) {{}}
    try {{
      const raw=localStorage.getItem(SEGS_KEY);
      if(raw) return JSON.parse(raw);
    }} catch(e) {{}}
    return segs;
  }}

  function playSeg(i, seek, wantPlay){{
    segs = readSegs();
    if(i<0 || i>=segs.length) return;
    const s=segs[i];
    if(!s || !s.b64) return;
    // Same segment already loaded — never reload src mid-play.
    if(String(audio.dataset.seg)===String(i) && audio.src && audio.src.indexOf("base64,")>=0){{
      if(seek>0.2){{ try{{ audio.currentTime=seek; }}catch(e){{}} }}
      if(wantPlay && audio.paused){{ audio.play().catch(function(){{}}); }}
      return;
    }}
    loading=true;
    state.i=i;
    audio.dataset.seg=String(i);
    const onMeta=function(){{
      audio.removeEventListener("loadedmetadata", onMeta);
      loading=false;
      const dur=audio.duration||0;
      if(seek>0.2 && (!dur || seek<dur)){{
        try{{ audio.currentTime=seek; }}catch(e){{}}
      }}
      if(wantPlay){{ audio.play().catch(function(){{}}); }}
    }};
    audio.addEventListener("loadedmetadata", onMeta);
    audio.src="data:"+s.mime+";base64,"+s.b64;
  }}

  function maybeAdvanceQueue(){{
    segs = readSegs();
    if(segs.length === lastLen) return;
    lastLen = segs.length;
    if(audio.paused && !loading){{
      const i=Number(audio.dataset.seg||0);
      const endedish = audio.ended || (audio.duration && audio.currentTime >= audio.duration - 0.05);
      if(endedish && (i+1) < segs.length){{
        playSeg(i+1, 0, true);
      }} else if((!audio.src || audio.error) && segs.length){{
        playSeg(Math.min(i, segs.length-1), state.t||0, !!state.playing || AUTO);
      }}
    }}
  }}

  audio.addEventListener("timeupdate", persist);
  audio.addEventListener("play", function(){{ state.playing=true; persist(); }});
  audio.addEventListener("pause", persist);
  audio.addEventListener("ended", function(){{
    segs = readSegs();
    const next=Number(audio.dataset.seg||0)+1;
    if(next<segs.length){{
      playSeg(next, 0, true);
    }} else {{
      state.playing=true; // keep wanting more if progressive still loading
      persist();
    }}
  }});

  segs = readSegs();
  lastLen = segs.length;
  if(state.i>=segs.length && segs.length) state.i=Math.max(0, segs.length-1);
  const resume=state.t>0.2;
  const wantPlay=!!state.playing || (AUTO && !resume);
  if(segs.length){{
    playSeg(state.i||0, state.t||0, wantPlay);
  }} else if(AUTO){{
    state.playing=true;
  }}

  setInterval(maybeAdvanceQueue, 250);
}})();
</script></body></html>"""
    components.html(html, height=52)
    return True


def _render_segment_queue_audio(segments: list, *, autoplay: bool) -> bool:
    """Legacy one-shot queue player (fallback when persistent split unavailable)."""
    try:
        import json
        import streamlit.components.v1 as components
    except Exception:
        return False
    segs = _segment_payload(segments)
    if not segs:
        return False

    player_id = _progressive_player_id()
    auto_js = "true" if autoplay else "false"
    segs_json = json.dumps(segs, separators=(",", ":"))
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:transparent;">
<audio id="a" controls style="width:100%;height:40px;"></audio>
<script>
(function(){{
  const KEY="rc_tts_q_"+{player_id!r};
  const AUTO={auto_js};
  const SEGS={segs_json};
  const audio=document.getElementById("a");
  let state={{i:0,t:0,playing:false}};
  try{{ const raw=sessionStorage.getItem(KEY); if(raw) state=JSON.parse(raw); }}catch(e){{}}
  if(!Number.isFinite(state.i) || state.i<0) state.i=0;
  if(state.i>=SEGS.length) state.i=Math.max(0, SEGS.length-1);
  function persist(){{
    try{{
      sessionStorage.setItem(KEY, JSON.stringify({{
        i: Number(audio.dataset.seg||state.i||0),
        t: audio.currentTime||0,
        playing: !audio.paused
      }}));
    }}catch(e){{}}
  }}
  function playSeg(i, seek, wantPlay){{
    if(i<0 || i>=SEGS.length) return;
    const s=SEGS[i];
    state.i=i;
    audio.dataset.seg=String(i);
    const onMeta=function(){{
      audio.removeEventListener("loadedmetadata", onMeta);
      const dur=audio.duration||0;
      if(seek>0.2 && (!dur || seek<dur)){{
        try{{ audio.currentTime=seek; }}catch(e){{}}
      }}
      if(wantPlay){{ audio.play().catch(function(){{}}); }}
    }};
    audio.addEventListener("loadedmetadata", onMeta);
    audio.src="data:"+s.mime+";base64,"+s.b64;
  }}
  audio.addEventListener("timeupdate", persist);
  audio.addEventListener("play", persist);
  audio.addEventListener("pause", persist);
  audio.addEventListener("ended", function(){{
    const next=Number(audio.dataset.seg||0)+1;
    if(next<SEGS.length){{
      playSeg(next, 0, true);
    }} else {{
      persist();
    }}
  }});
  const resume=state.t>0.2;
  const wantPlay=!!state.playing || (AUTO && !resume);
  playSeg(state.i||0, state.t||0, wantPlay);
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


def edge_voice_for_language(book_language: str) -> str | None:
    """Return an Edge Neural voice id for the book language, or None if unsupported."""
    if not book_language:
        return None
    return EDGE_LANG_VOICES.get(book_language) or EDGE_LANG_VOICES.get(book_language.title())


def speak_vocab_word(
    word: str,
    book_language: str,
    tts_rate: float,
    xtts_voice: str | None = None,
    source: str = "",
):
    """Legacy shared-player vocab path.

    Prefer ``generate_vocab_word_audio`` + ``render_inline_vocab_audio_button`` so the
    progressive section player is never remounted. This path still clears progressive
    and writes ``tts_audio`` when a caller needs the shared bar.
    """
    word = (word or "").strip()
    if not word:
        return

    st.session_state.tts_error = ""
    result = generate_vocab_word_audio(word, book_language, tts_rate, xtts_voice=xtts_voice)
    if not result:
        if not edge_voice_for_language(book_language) and not st.session_state.get("xtts_speaker_wav"):
            st.session_state.tts_error = (
                "Upload a speaker WAV in the sidebar to pronounce words with XTTS "
                "(or use a language with an Edge voice)."
            )
        else:
            st.session_state.tts_error = f"Could not generate pronunciation for “{word}”."
        return

    audio_bytes, fmt = result
    clear_progressive_tts()
    st.session_state.tts_segment_audios = []
    st.session_state.tts_segments = []
    src = source or f"vocab_word:0:{word}"
    st.session_state.tts_source = src
    st.session_state.tts_voice_note = f"Pronouncing: {word}"
    st.session_state.tts_audio = audio_bytes
    st.session_state.tts_format = fmt
    _bump_tts_player_gen(src)


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
        # Queue player advances on ended — no live concat splice into the playing bar.
        if done + 1 >= len(segments):
            st.session_state.tts_progressive_active = False
            _rebuild_concat_audio()  # final full buffer once complete
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
        # Queue player — avoid mid-play concat remount stutter.
        if done + 1 >= len(jobs):
            st.session_state.tts_progressive_active = False
            _rebuild_concat_audio()
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


def render_tts_caption_and_push() -> bool:
    """Caption / errors / segment push — never mounts the progressive <audio> element.

    Returns True when in progressive / multi-segment mode.
    """
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
    active = bool(st.session_state.get("tts_progressive_active"))
    audio_bytes = st.session_state.get("tts_audio") or b""
    fmt = st.session_state.get("tts_format") or "audio/wav"

    if total > 1 and (audios or audio_bytes):
        if audios:
            secs = sum(estimate_audio_seconds(ab, f) for ab, f in audios)
        else:
            secs = estimate_audio_seconds(audio_bytes, fmt)
        if active:
            st.caption(f"Loaded {done}/{total} · ~{int(round(secs))}s so far")
        elif done >= total > 0:
            st.caption(f"Ready · {done}/{total} · ~{int(round(secs))}s")
        else:
            st.caption(f"Loaded {done}/{total} · ~{int(round(secs))}s so far")
        if audios:
            _push_segments_to_browser(audios)
        return True
    return False


def render_tts_player(*, mount_audio: bool = True):
    """Render TTS UI.

    When mount_audio is False (polling fragment), only caption + segment push update —
    the persistent <audio> stays mounted outside the fragment.
    """
    progressive = render_tts_caption_and_push()
    just = int(st.session_state.get("tts_progressive_just_added", -1))
    audios = st.session_state.get("tts_segment_audios") or []
    audio_bytes = st.session_state.get("tts_audio") or b""
    fmt = st.session_state.get("tts_format") or "audio/wav"

    if progressive:
        autoplay = just == 0
        if mount_audio:
            col_audio, col_stop = st.columns([5, 1])
            with col_audio:
                used_html = _render_persistent_queue_player(autoplay=autoplay)
                if not used_html:
                    used_html = _render_segment_queue_audio(audios, autoplay=autoplay)
                if not used_html:
                    if audios and not audio_bytes:
                        audio_bytes, fmt = concat_audio_bytes(audios)
                        st.session_state.tts_audio = audio_bytes
                        st.session_state.tts_format = fmt
                    used_html = _render_continuity_audio(audio_bytes, fmt, autoplay=autoplay)
                if not used_html:
                    st.audio(audio_bytes, format=fmt, autoplay=autoplay)
            with col_stop:
                if st.button("Stop", key="btn_stop"):
                    stop_speech()
                    st.rerun()
        else:
            if st.button("Stop", key="btn_stop_frag"):
                stop_speech()
                st.rerun()
        if just >= 0:
            st.session_state.tts_progressive_just_added = -1
        return

    if audio_bytes and mount_audio:
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
    """Advance generation + caption/push only — do NOT remount the audio element."""
    was_active = bool(st.session_state.get("tts_progressive_active"))
    render_tts_player(mount_audio=False)
    if st.session_state.get("tts_progressive_active"):
        _advance_progressive_one()
        audios = st.session_state.get("tts_segment_audios") or []
        if audios:
            _push_segments_to_browser(audios)
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
    _progressive_fragment_static = st.fragment(lambda: render_tts_player(mount_audio=False))
else:
    _progressive_fragment_polling = None
    _progressive_fragment_static = None


def maybe_continue_progressive():
    """Mount persistent audio once; fragment only advances segments + caption."""
    drain_vocab_prefetch_into_session()

    active = bool(st.session_state.get("tts_progressive_active"))
    has_playlist = len(st.session_state.get("tts_segment_audios") or []) > 0 and (
        len(st.session_state.get("tts_segments") or []) > 1
    )
    audio_bytes = st.session_state.get("tts_audio") or b""
    just = int(st.session_state.get("tts_progressive_just_added", -1))

    if _HAS_FRAGMENT and (active or has_playlist):
        audios = st.session_state.get("tts_segment_audios") or []
        if audios:
            _push_segments_to_browser(audios)
        autoplay = just == 0
        col_audio, col_stop = st.columns([5, 1])
        with col_audio:
            used = _render_persistent_queue_player(autoplay=autoplay)
            if not used:
                used = _render_segment_queue_audio(audios, autoplay=autoplay)
            if not used and audio_bytes:
                _render_continuity_audio(
                    audio_bytes,
                    st.session_state.get("tts_format") or "audio/wav",
                    autoplay=True,
                )
        with col_stop:
            if st.button("Stop", key="btn_stop"):
                stop_speech()
                st.rerun()

        if active and _progressive_fragment_polling is not None:
            _progressive_fragment_polling()
        elif _progressive_fragment_static is not None:
            _progressive_fragment_static()
        else:
            render_tts_caption_and_push()

        if just >= 0:
            st.session_state.tts_progressive_just_added = -1
        return

    render_tts_player(mount_audio=True)
    if not active:
        return
    safety = 0
    while st.session_state.get("tts_progressive_active") and safety < 200:
        _advance_progressive_one()
        safety += 1
