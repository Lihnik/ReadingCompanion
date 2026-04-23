import asyncio
import hashlib
import io
import re

import edge_tts
import streamlit as st

from .constants import EDGE_VOICES, KOKORO_VOICES, XTTS_LANGUAGES


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


def _speak_xtts(text: str, language: str, speaker_wav_bytes: bytes, speed: float) -> bytes:
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from huggingface_hub import snapshot_download
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as e:
        raise RuntimeError(
            f"XTTS dependencies not installed: {e}. "
            "Run: pip install TTS huggingface_hub"
        )
    import tempfile
    import os
    import torchaudio

    # torchaudio 2.9+ delegates to torchcodec which requires FFmpeg DLLs on Windows.
    # Patch it once to use soundfile instead — soundfile handles WAV natively.
    # Always re-apply the patch (session_state-cached model may have stale torchaudio.load).
    def _sf_load(path, normalize=True, frame_offset=0, num_frames=None, **kwargs):
        frames = num_frames if (num_frames is not None and num_frames >= 0) else -1
        data, sr = sf.read(str(path), always_2d=True, start=frame_offset or 0, frames=frames)
        tensor = torch.from_numpy(data.T.copy()).float()
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

    # The base Coqui TTS tokenizer only knows the original 16 XTTS languages.
    # tartuNLP adds Estonian and Finnish at the model level; patch the tokenizer
    # to fall back to English text preprocessing for any unrecognised language.
    _orig_preprocess = model.tokenizer.preprocess_text
    def _patched_preprocess(txt, lang):
        try:
            return _orig_preprocess(txt, lang)
        except NotImplementedError:
            return _orig_preprocess(txt, "en")
    model.tokenizer.preprocess_text = _patched_preprocess

    # XTTS hard limits: 400 tokens per inference call; warns and truncates above 250 chars.
    # Split into chunks of at most 200 characters, breaking on sentence then clause boundaries.
    def _split_for_xtts(t: str, max_chars: int = 200) -> list:
        import re as _re
        result, current = [], ""
        for sent in _re.split(r'(?<=[.!?])\s+', t.strip()):
            # If a single sentence still exceeds the limit, split on clause boundaries
            parts = [sent]
            if len(sent) > max_chars:
                parts = _re.split(r'(?<=[,;:])\s+', sent) or [sent]
            for part in parts:
                if not part:
                    continue
                # If even one clause is too long, hard-split by words
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

    text_chunks = _split_for_xtts(text)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(speaker_wav_bytes)
        tmp.flush()
        tmp.close()
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[tmp.name])
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
    finally:
        os.unlink(tmp.name)

    wav = np.concatenate(wav_parts) if len(wav_parts) > 1 else wav_parts[0]
    if wav.size == 0:
        raise RuntimeError("XTTS returned empty audio — check your speaker WAV and language setting.")
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), 24000, format="WAV")
    buf.seek(0)
    return buf.read()


def _make_tts_cache_key(processed: str, voice: str, rate: float, engine: str) -> str:
    if engine == "xtts":
        spk_hash = hashlib.md5(st.session_state.xtts_speaker_wav).hexdigest()
        return hashlib.md5(f"{processed}|xtts|{voice}|{spk_hash}".encode()).hexdigest()
    return hashlib.md5(f"{processed}|{voice}|{rate}|{engine}".encode()).hexdigest()


def speak_text(text: str, voice: str, rate: float, engine: str = "edge", source: str = ""):
    st.session_state.tts_error = ""
    processed = _preprocess_tts_text(text)
    cache_key = _make_tts_cache_key(processed, voice, rate, engine)
    if cache_key in st.session_state.tts_cache:
        audio_bytes, fmt = st.session_state.tts_cache[cache_key]
        st.session_state.tts_audio = audio_bytes
        st.session_state.tts_format = fmt
        st.session_state.tts_source = source
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
        st.session_state.tts_cache[cache_key] = (audio_bytes, fmt)
        st.session_state.tts_audio = audio_bytes
        st.session_state.tts_format = fmt
        st.session_state.tts_source = source
    except Exception as e:
        import traceback
        st.session_state.tts_error = f"TTS error ({engine}): {e}\n\n```\n{traceback.format_exc()}\n```"


def stop_speech():
    st.session_state.tts_audio = b""
    st.session_state.tts_source = ""


def _tts_cached(text: str, voice: str, rate: float, engine: str) -> bool:
    processed = _preprocess_tts_text(text)
    key = _make_tts_cache_key(processed, voice, rate, engine)
    return key in st.session_state.tts_cache


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
    st.session_state.tts_cache[cache_key] = (audio_bytes, fmt)
    return audio_bytes


def tts_button(label: str, text: str, source: str, tts_voice: str, tts_rate: float, tts_engine: str, full_width: bool = False):
    if tts_engine == "Kokoro":
        engine_key = "kokoro"
    elif tts_engine == "XTTS":
        engine_key = "xtts"
    else:
        engine_key = "edge"
    is_playing = bool(st.session_state.tts_audio) and st.session_state.tts_source == source
    btn_label = f"▶ {label}" if is_playing else label
    if st.button(btn_label, key=f"btn_read_{source}", use_container_width=full_width):
        if _tts_cached(text, tts_voice, tts_rate, engine_key):
            speak_text(text, tts_voice, tts_rate, engine_key, source=source)
        else:
            with st.spinner("Generating audio..."):
                speak_text(text, tts_voice, tts_rate, engine_key, source=source)
        st.rerun()
