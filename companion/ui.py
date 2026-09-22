import html
import io
import re
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from .constants import (
    EDGE_VOICES,
    KOKORO_VOICES,
    XTTS_LANGUAGES,
    PREFERRED_OLLAMA_MODELS,
    SYSTEM_PROMPT_COMMENTARY,
    SYSTEM_PROMPT_QUESTION,
    SYSTEM_PROMPT_CHAT,
    SYSTEM_PROMPT_LL_SUMMARY,
    SYSTEM_PROMPT_LL_VOCAB,
)
from .navigation import go_to_chunk, reset_session
from .ollama import (
    call_ollama,
    stream_ollama,
    build_commentary_prompt,
    build_question_prompt,
    build_feedback_prompt,
    build_chat_prompt,
    build_summary_prompt,
    build_ll_summary_prompt,
    build_ll_vocab_prompt,
    parse_vocab_response,
    fetch_ollama_models,
    order_ollama_models,
)
from .hero import render_landing_hero, render_ambient_bg_html
from .parsing import parse_epub, parse_pdf
from .tts import (
    _engine_key,
    _get_or_generate_audio,
    _tts_cached,
    maybe_continue_progressive,
    resolve_english_voice,
    speak_text,
    speak_vocab_word,
    tts_button,
)

# st.fragment landed in Streamlit 1.33; use when available for snappier LL/TTS updates.
_HAS_FRAGMENT = hasattr(st, "fragment")


# ---------------------------------------------------------------------------
# GLASS PANEL STYLE
# These values are reused for every hand-rolled HTML panel (book text,
# AI commentary).  Change them once here to restyle all panels together.
#
#   _PANEL_BG      – fill colour;  raise alpha (0.0–1.0) for a more opaque
#                    panel, lower it to let more video show through
#   _PANEL_BORDER  – white border opacity; 0.0 = invisible, 0.25 = obvious
#   _PANEL_RADIUS  – corner rounding in px
#   _PANEL_BLUR    – backdrop blur in px; higher = frostier glass
# ---------------------------------------------------------------------------
_PANEL_BG     = "rgba(6, 14, 34, 0.22)"
_PANEL_BORDER = "rgba(255, 255, 255, 0.13)"
_PANEL_RADIUS = "14px"
_PANEL_BLUR   = "8px"

# Shared inline-style fragment used by every glass panel.
_GLASS = (
    f"background:{_PANEL_BG};"
    f"backdrop-filter:blur({_PANEL_BLUR}) saturate(1.5);"
    f"-webkit-backdrop-filter:blur({_PANEL_BLUR}) saturate(1.5);"
    f"border:1px solid {_PANEL_BORDER};"
    f"border-radius:{_PANEL_RADIUS};"
    f"padding:.875rem 1.125rem;"
)


def _section_ai_cache_key(model: str) -> str:
    """Stable cache key for per-section commentary / questions."""
    return f"{st.session_state.pdf_name}|{st.session_state.current_chunk_idx}|{model}"


def _ll_ai_cache_key(model: str, language: str) -> str:
    """Stable cache key for per-section LL summary / vocabulary."""
    return f"{st.session_state.pdf_name}|{st.session_state.current_chunk_idx}|{language}|{model}"


def render_comprehension_question(chunk: dict, model: str, tts_voice: str, tts_rate: float, tts_engine: str):
    st.markdown("**Comprehension Check**")

    cache_key = _section_ai_cache_key(model)
    # Always sync display field from the cache for this (pdf, section, model).
    st.session_state.ai_question = st.session_state.question_cache.get(cache_key, "")
    # Drop answer/feedback UI when the active question identity changes (nav or model).
    if st.session_state.get("_question_cache_key") != cache_key:
        st.session_state._question_cache_key = cache_key
        st.session_state.question_answered = False
        st.session_state.question_feedback = ""

    if not st.session_state.ai_question:
        st.caption("Read the section first, then get a question when you're ready.")
        if st.button("Get a question", key="btn_gen_question"):
            with st.spinner("Generating comprehension question..."):
                q = call_ollama(build_question_prompt(chunk["text"]), model, SYSTEM_PROMPT_QUESTION)
                st.session_state.ai_question = q
                st.session_state.question_cache[cache_key] = q
            st.rerun()
        return

    st.markdown(f"> {st.session_state.ai_question}")
    tts_button("Read question", st.session_state.ai_question, "question", tts_voice, tts_rate, tts_engine)

    if not st.session_state.question_answered:
        # Form contains the answer input + Submit / Skip buttons.
        # The glass look on this form comes from the CSS in hero.py
        # ([data-testid="stForm"] rule), not from inline styles here.
        with st.form(key="answer_form"):
            user_answer = st.text_input("Your answer:")
            col1, col2 = st.columns([3, 1])
            with col1:
                submitted = st.form_submit_button("Submit Answer", type="primary")
            with col2:
                skipped = st.form_submit_button("Skip")

        if submitted and user_answer.strip():
            with st.spinner("AI is evaluating your answer..."):
                feedback = call_ollama(
                    build_feedback_prompt(chunk["text"], st.session_state.ai_question, user_answer),
                    model,
                    num_predict=1024,
                )
            st.session_state.question_feedback = feedback
            st.session_state.question_answered = True
            st.rerun()

        if skipped:
            st.session_state.question_answered = True
            st.rerun()
    else:
        if st.session_state.question_feedback:
            # Streamlit's st.success() — styled by the [data-testid="stAlert"]
            # rule in hero.py.  To change colour/style, edit that CSS rule.
            st.success(st.session_state.question_feedback)
            tts_button("Read feedback", st.session_state.question_feedback, "feedback", tts_voice, tts_rate, tts_engine)


def render_reading_panel(model: str, tts_voice: str = "en-US-AriaNeural", tts_rate: float = 1.0, tts_engine: str = "edge"):
    chunk = st.session_state.pdf_chunks[st.session_state.current_chunk_idx]

    st.subheader(chunk["title"])

    # --- Book text panel ---
    # This is a plain HTML div so we control its style completely.
    # Tweak height (default 300px) to show more/less text before scrolling.
    # Text colour: rgba(225, 232, 248, 0.95) — bright near-white.
    # Font size:   0.9rem (~14px).  Increase to 1rem for larger text.
    _BOOK_TEXT_HEIGHT = "300px"
    _BOOK_TEXT_COLOR  = "rgba(225, 232, 248, 0.95)"
    _BOOK_TEXT_SIZE   = "0.9rem"

    _paras = [p.strip() for p in chunk["text"].split("\n\n") if p.strip()] or [chunk["text"]]
    _body = "".join(
        f'<p style="margin:0 0 .8em 0;line-height:1.75;">{html.escape(p)}</p>'
        for p in _paras
    )
    st.markdown(
        f'<div style="width:100%;box-sizing:border-box;'
        f'height:{_BOOK_TEXT_HEIGHT};overflow-y:auto;{_GLASS}'
        f'color:{_BOOK_TEXT_COLOR};font-size:{_BOOK_TEXT_SIZE};'
        f'scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.15) transparent;">'
        f'{_body}</div>',
        unsafe_allow_html=True,
    )

    tts_button("Read section aloud", chunk["text"], "section", tts_voice, tts_rate, tts_engine, full_width=True)

    st.markdown("---")

    # --- AI Commentary panel ---
    # Same glass style as the book text panel (_GLASS constant above).
    # Slightly more muted text colour to visually separate it from the
    # book text.  Change _COMMENTARY_COLOR to make it brighter/dimmer.
    _COMMENTARY_COLOR = "rgba(210, 225, 248, 0.90)"
    _COMMENTARY_SIZE  = "0.9rem"

    st.markdown("**AI Commentary**")
    cache_key = _section_ai_cache_key(model)
    # Always sync display field from the cache for this (pdf, section, model).
    st.session_state.ai_commentary = st.session_state.commentary_cache.get(cache_key, "")

    if not st.session_state.ai_commentary:
        st.caption("Read the section first, then generate an insight when you're ready.")
        if st.button("Generate insight", key="btn_gen_commentary"):
            with st.spinner("AI is reading this section..."):
                commentary = call_ollama(build_commentary_prompt(chunk["text"]), model, SYSTEM_PROMPT_COMMENTARY)
                st.session_state.ai_commentary = commentary
                st.session_state.commentary_cache[cache_key] = commentary
            st.rerun()
    else:
        _paras = [p.strip() for p in st.session_state.ai_commentary.split("\n\n") if p.strip()] or [st.session_state.ai_commentary]
        _body = "".join(
            f'<p style="margin:0 0 .7em 0;line-height:1.7;">{html.escape(p)}</p>'
            for p in _paras
        )
        st.markdown(
            f'<div style="width:100%;box-sizing:border-box;{_GLASS}'
            f'color:{_COMMENTARY_COLOR};font-size:{_COMMENTARY_SIZE};line-height:1.7;">'
            f'{_body}</div>',
            unsafe_allow_html=True,
        )
        tts_button("Read commentary", st.session_state.ai_commentary, "commentary", tts_voice, tts_rate, tts_engine)

    st.markdown("---")
    render_comprehension_question(chunk, model, tts_voice, tts_rate, tts_engine)

    st.markdown("---")
    if st.button("Summarize This Section", key="btn_summarize"):
        with st.spinner("Summarizing..."):
            summary = call_ollama(build_summary_prompt(chunk["text"]), model)
            st.session_state.section_summary = summary

    if st.session_state.section_summary:
        # Expander glass style comes from [data-testid="stExpander"] in hero.py.
        with st.expander("Section Summary", expanded=True):
            st.write(st.session_state.section_summary)
            tts_button("Read summary", st.session_state.section_summary, "summary", tts_voice, tts_rate, tts_engine)


def render_chat_panel(model: str, tts_voice: str, tts_rate: float, tts_engine: str):
    st.subheader("Chat with AI")

    # Chat message container — glass styling comes from the CSS rule
    # [data-testid="stVerticalBlockBorderWrapper"] in hero.py.
    # Change height here to give more/less scroll space for messages.
    # border=True tells Streamlit to render the border wrapper that the
    # CSS targets; set to False to remove the container entirely.
    with st.container(height=400, border=True):
        if not st.session_state.chat_history:
            st.caption("No messages yet. Ask anything about what you're reading.")
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    # Chat input bar — glass style from [data-testid="stChatInputContainer"]
    # in hero.py.  The pill shape (border-radius: 28px) is set there too.
    user_input = st.chat_input(
        "Ask anything about this book...",
        disabled=not st.session_state.reading_started,
    )

    if user_input:
        chunk = st.session_state.pdf_chunks[st.session_state.current_chunk_idx]
        prompt = build_chat_prompt(chunk["text"], st.session_state.chat_history, user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            response = st.write_stream(stream_ollama(prompt, model, SYSTEM_PROMPT_CHAT))
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if (
        st.session_state.chat_history
        and st.session_state.chat_history[-1]["role"] == "assistant"
    ):
        tts_button("Read last response", st.session_state.chat_history[-1]["content"], "chat", tts_voice, tts_rate, tts_engine)


def render_audiobook_panel(tts_voice: str, tts_rate: float, tts_engine: str):
    chunks = st.session_state.pdf_chunks
    total = len(chunks)
    if tts_engine == "Kokoro":
        engine_key = "kokoro"
    elif tts_engine == "XTTS":
        engine_key = "xtts"
    else:
        engine_key = "edge"

    st.markdown(
        "Generate a single audio file for a range of sections. "
        "Use the start/end inputs to skip front matter or back matter. "
        "Sections already read aloud are pulled from cache instantly."
    )

    col_s, col_e = st.columns(2)
    with col_s:
        ab_start = int(st.number_input("Start section", min_value=1, max_value=total, value=1, step=1, key="ab_start"))
    with col_e:
        ab_end = int(st.number_input("End section", min_value=1, max_value=total, value=total, step=1, key="ab_end"))

    if ab_start > ab_end:
        st.warning("Start section must be ≤ end section.")
        return

    selected = chunks[ab_start - 1 : ab_end]
    cached_count = sum(1 for c in selected if _tts_cached(c["text"], tts_voice, tts_rate, engine_key))
    st.caption(f"{len(selected)} sections selected · {cached_count} already cached")

    if st.button("Generate Audiobook", type="primary", use_container_width=True, key="btn_gen_audiobook"):
        audio_parts = []
        progress = st.progress(0.0, text="Starting…")
        for i, chunk in enumerate(selected):
            progress.progress((i + 1) / len(selected), text=f"Section {ab_start + i} of {ab_end}…")
            audio_parts.append(_get_or_generate_audio(chunk["text"], tts_voice, tts_rate, engine_key))
        progress.progress(1.0, text="Combining sections…")

        if engine_key in ("kokoro", "xtts"):
            import numpy as np
            import soundfile as sf
            arrays = []
            for ab in audio_parts:
                data, _ = sf.read(io.BytesIO(ab))
                arrays.append(data)
            out_buf = io.BytesIO()
            sf.write(out_buf, np.concatenate(arrays), 24000, format="WAV")
            st.session_state.audiobook_bytes = out_buf.getvalue()
            st.session_state.audiobook_ext = "wav"
        else:
            st.session_state.audiobook_bytes = b"".join(audio_parts)
            st.session_state.audiobook_ext = "mp3"
        progress.empty()

    if st.session_state.audiobook_bytes:
        ext = st.session_state.audiobook_ext
        base = re.sub(r'\.(pdf|epub)$', '', st.session_state.pdf_name, flags=re.IGNORECASE)
        fname = f"{base}_audiobook.{ext}"
        size_mb = len(st.session_state.audiobook_bytes) / 1024 / 1024
        st.download_button(
            f"Download Audiobook ({size_mb:.1f} MB)",
            data=st.session_state.audiobook_bytes,
            file_name=fname,
            mime=f"audio/{ext}",
            use_container_width=True,
        )



def _render_ll_summary_body(summary: str):
    _sum_paras = [p.strip() for p in summary.split("\n\n") if p.strip()] or [summary]
    _sum_body = "".join(
        f'<p style="margin:0 0 .7em 0;line-height:1.7;">{html.escape(p)}</p>'
        for p in _sum_paras
    )
    st.markdown(
        f'<div style="width:100%;box-sizing:border-box;{_GLASS}'
        f'color:rgba(210,225,248,0.90);font-size:0.9rem;line-height:1.7;">'
        f'{_sum_body}</div>',
        unsafe_allow_html=True,
    )


def _render_ll_vocab_table(
    vocab: list,
    book_language: str,
    tts_voice: str | None = None,
    tts_rate: float = 1.0,
    tts_engine: str | None = None,
):
    """Render vocab rows; optional ▶ per Italian/book-language word for pronunciation."""
    # Header
    h_play, h_word, h_trans = st.columns([0.55, 2.2, 3.0])
    with h_play:
        st.markdown(
            f'<div style="font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;'
            f'color:rgba(180,180,200,0.7);padding:.35rem 0;">▶</div>',
            unsafe_allow_html=True,
        )
    with h_word:
        st.markdown(
            f'<div style="font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;'
            f'color:rgba(180,180,200,0.7);padding:.35rem 0;">{html.escape(book_language)}</div>',
            unsafe_allow_html=True,
        )
    with h_trans:
        st.markdown(
            '<div style="font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;'
            'color:rgba(180,180,200,0.7);padding:.35rem 0;">English</div>',
            unsafe_allow_html=True,
        )

    can_speak = bool(tts_voice is not None and tts_engine is not None)
    for i, v in enumerate(vocab):
        word = (v.get("word") or "").strip()
        translation = (v.get("translation") or "").strip()
        c_play, c_word, c_trans = st.columns([0.55, 2.2, 3.0])
        with c_play:
            if can_speak and word:
                if st.button("▶", key=f"btn_vocab_word_{i}", help=f"Pronounce: {word}"):
                    eng = _engine_key(tts_engine)
                    xtts_lang = tts_voice if eng == "xtts" else None
                    with st.spinner(f"Pronouncing “{word}”…"):
                        speak_vocab_word(
                            word,
                            book_language,
                            tts_rate,
                            xtts_voice=xtts_lang,
                            source=f"vocab_word:{i}:{word}",
                        )
                    st.rerun()
            else:
                st.write("")
        with c_word:
            st.markdown(
                f'<div style="padding:.35rem 0;font-weight:600;color:rgba(255,220,130,0.95);">'
                f'{html.escape(word)}</div>',
                unsafe_allow_html=True,
            )
        with c_trans:
            st.markdown(
                f'<div style="padding:.35rem 0;color:rgba(210,225,248,0.90);">'
                f'{html.escape(translation)}</div>',
                unsafe_allow_html=True,
            )



def _apply_vocab_result(cache_key: str, raw: str) -> list:
    """Parse vocab; cache only on success. On failure store preview for UI warning."""
    vocab = parse_vocab_response(raw)
    if vocab:
        st.session_state.ll_vocab = vocab
        st.session_state.ll_vocab_cache[cache_key] = vocab
        st.session_state.ll_vocab_parse_fail = None
        return vocab
    # Never cache empty list as success
    st.session_state.ll_vocab = []
    st.session_state.ll_vocab_cache.pop(cache_key, None)
    preview = (raw or "").strip()
    if len(preview) > 600:
        preview = preview[:600] + "…"
    st.session_state.ll_vocab_parse_fail = {"key": cache_key, "raw": preview or "(empty model output)"}
    return []


def _render_vocab_parse_fail(cache_key: str, book_language: str, model: str, chunk_text: str):
    fail = st.session_state.get("ll_vocab_parse_fail") or {}
    if fail.get("key") != cache_key:
        return False
    st.warning(
        "Could not parse vocabulary from the model response. "
        "Nothing was cached — try again (or use Prepare this section)."
    )
    with st.expander("Model output preview", expanded=False):
        st.code(fail.get("raw") or "", language=None)
    if st.button("Retry vocabulary", key="btn_ll_vocab_retry"):
        with st.spinner(f"Extracting {book_language} vocabulary..."):
            raw = call_ollama(
                build_ll_vocab_prompt(chunk_text, book_language),
                model,
                SYSTEM_PROMPT_LL_VOCAB,
                num_predict=512,
            )
            _apply_vocab_result(cache_key, raw)
        # Fall through: successful parse shows table in same run via session state
        if st.session_state.ll_vocab:
            st.session_state.ll_vocab_parse_fail = None
        st.rerun()
    return True


def render_language_learning_panel(model: str, tts_voice: str, tts_rate: float, tts_engine: str, book_language: str):
    chunk = st.session_state.pdf_chunks[st.session_state.current_chunk_idx]
    cache_key = _ll_ai_cache_key(model, book_language)

    st.subheader(chunk["title"])

    if tts_engine != "XTTS" and book_language != "English":
        st.info(
            f"Tip: Edge TTS and Kokoro are English-only — they can't read {book_language} well. "
            f"Switch to the XTTS engine to hear this section aloud."
        )

    _paras = [p.strip() for p in chunk["text"].split("\n\n") if p.strip()] or [chunk["text"]]
    _body = "".join(
        f'<p style="margin:0 0 .8em 0;line-height:1.75;">{html.escape(p)}</p>'
        for p in _paras
    )
    st.markdown(
        f'<div style="width:100%;box-sizing:border-box;'
        f'height:300px;overflow-y:auto;{_GLASS}'
        f'color:rgba(225,232,248,0.95);font-size:0.9rem;'
        f'scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.15) transparent;">'
        f'{_body}</div>',
        unsafe_allow_html=True,
    )
    # Section text stays on book language + sidebar engine (XTTS/Italian as configured)
    tts_button(
        f"Read aloud in {book_language}",
        chunk["text"],
        "section",
        tts_voice,
        tts_rate,
        tts_engine,
        full_width=True,
        progressive=True,
    )

    st.markdown("---")
    _render_ll_ai_controls(model, tts_voice, tts_rate, tts_engine, book_language, chunk, cache_key)


def _render_ll_ai_controls(
    model: str,
    tts_voice: str,
    tts_rate: float,
    tts_engine: str,
    book_language: str,
    chunk: dict,
    cache_key: str,
):
    """English summary + vocabulary controls (isolated for optional fragment wrapping)."""
    # Sync display fields from caches for this (pdf, section, language, model).
    # Do not overwrite in-run results with empty cache after a failed parse.
    cached_summary = st.session_state.ll_summary_cache.get(cache_key, "")
    if cached_summary:
        st.session_state.ll_summary = cached_summary
    elif cache_key not in st.session_state.ll_summary_cache:
        # Navigating to a section with no cache — clear stale display
        if not st.session_state.ll_summary or st.session_state.get("_ll_summary_key") != cache_key:
            st.session_state.ll_summary = ""
    st.session_state._ll_summary_key = cache_key

    cached_vocab = st.session_state.ll_vocab_cache.get(cache_key)
    if cached_vocab:
        st.session_state.ll_vocab = list(cached_vocab)
    else:
        # Keep in-run vocab if we just generated it; otherwise clear for this key
        if st.session_state.get("_ll_vocab_key") != cache_key:
            st.session_state.ll_vocab = []
    st.session_state._ll_vocab_key = cache_key

    prepare_col, _ = st.columns([1, 1])
    with prepare_col:
        if st.button(
            "Prepare this section",
            key="btn_ll_prepare",
            use_container_width=True,
            help="Generate English summary and vocabulary in parallel.",
        ):
            with st.spinner("Preparing summary and vocabulary…"):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_sum = pool.submit(
                        call_ollama,
                        build_ll_summary_prompt(chunk["text"], book_language),
                        model,
                        SYSTEM_PROMPT_LL_SUMMARY,
                        512,
                    )
                    fut_vocab = pool.submit(
                        call_ollama,
                        build_ll_vocab_prompt(chunk["text"], book_language),
                        model,
                        SYSTEM_PROMPT_LL_VOCAB,
                        512,
                    )
                    summary = fut_sum.result()
                    raw_vocab = fut_vocab.result()
                st.session_state.ll_summary = summary
                st.session_state.ll_summary_cache[cache_key] = summary
                _apply_vocab_result(cache_key, raw_vocab)
            # Prefer showing results in the same run (state already set); soft rerun
            # only if we need widgets to flip from button → content cleanly.
            st.rerun()

    st.markdown("**English Summary**")
    if not st.session_state.ll_summary:
        st.caption("Read the section first, then get an English gist when you're ready.")
        if st.button("English summary", key="btn_ll_summary"):
            with st.spinner("Generating English summary..."):
                summary = call_ollama(
                    build_ll_summary_prompt(chunk["text"], book_language),
                    model,
                    SYSTEM_PROMPT_LL_SUMMARY,
                    num_predict=512,
                )
                st.session_state.ll_summary = summary
                st.session_state.ll_summary_cache[cache_key] = summary
            st.rerun()
    else:
        _render_ll_summary_body(st.session_state.ll_summary)
        # English summary: Kokoro (fast) when sidebar is XTTS; never Italian XTTS
        eng_engine, eng_voice, _ = resolve_english_voice(tts_engine, tts_voice)
        voice_note = None
        engine_override = None
        voice_override = None
        use_fallback_edge = False
        if eng_engine != tts_engine or eng_voice != tts_voice:
            engine_override = eng_engine
            voice_override = eng_voice
            if eng_engine == "Kokoro":
                voice_note = "Summary uses English voice (Kokoro)"
                use_fallback_edge = True
            else:
                voice_note = f"Summary uses English voice ({eng_engine})"
            st.caption(voice_note)
        elif eng_engine == "Kokoro":
            voice_note = "Summary uses English voice (Kokoro)"
            st.caption(voice_note)
        tts_button(
            "Read summary",
            st.session_state.ll_summary,
            "ll_summary",
            tts_voice,
            tts_rate,
            tts_engine,
            engine_override=engine_override,
            voice_override=voice_override,
            voice_note=voice_note,
            progressive=True,
            fallback_edge=use_fallback_edge,
        )

    st.markdown("---")

    st.markdown("**Vocabulary**")
    if st.session_state.ll_vocab:
        _render_ll_vocab_table(
            st.session_state.ll_vocab,
            book_language,
            tts_voice=tts_voice,
            tts_rate=tts_rate,
            tts_engine=tts_engine,
        )
        st.caption(
            "▶ plays the word alone (book-language voice). "
            "Read vocabulary concatenates words + English translations into one track."
        )
        tts_button(
            "Read vocabulary",
            "",  # unused when bilingual_vocab set
            "ll_vocab",
            tts_voice,
            tts_rate,
            tts_engine,
            bilingual_vocab=st.session_state.ll_vocab,
            voice_note="Vocabulary: book-language words + English translations",
        )
    elif _render_vocab_parse_fail(cache_key, book_language, model, chunk["text"]):
        pass  # warning + retry already rendered
    else:
        st.caption(f"Pick useful {book_language} words from this section when you're ready.")
        if st.button("Vocabulary", key="btn_ll_vocab"):
            with st.spinner(f"Extracting {book_language} vocabulary..."):
                raw = call_ollama(
                    build_ll_vocab_prompt(chunk["text"], book_language),
                    model,
                    SYSTEM_PROMPT_LL_VOCAB,
                    num_predict=512,
                )
                vocab = _apply_vocab_result(cache_key, raw)
            if vocab:
                # Show table in the same run — avoid rerun-with-empty-cache flicker
                _render_ll_vocab_table(
                    vocab,
                    book_language,
                    tts_voice=tts_voice,
                    tts_rate=tts_rate,
                    tts_engine=tts_engine,
                )
                st.caption(
                    "▶ plays the word alone (book-language voice). "
                    "Read vocabulary concatenates words + English translations into one track."
                )
                tts_button(
                    "Read vocabulary",
                    "",
                    "ll_vocab",
                    tts_voice,
                    tts_rate,
                    tts_engine,
                    bilingual_vocab=vocab,
                    voice_note="Vocabulary: book-language words + English translations",
                )
            else:
                _render_vocab_parse_fail(cache_key, book_language, model, chunk["text"])


# Wrap LL AI controls in a fragment when Streamlit supports it (isolates reruns).
if _HAS_FRAGMENT:
    _render_ll_ai_controls = st.fragment(_render_ll_ai_controls)



def render_sidebar():
    with st.sidebar:
        st.title("📖 Reading Companion")
        st.markdown("---")

        st.subheader("Upload Book / Document")
        uploaded_file = st.file_uploader("Choose a PDF or EPUB file", type=["pdf", "epub"])

        if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
            with st.spinner(f"Parsing {uploaded_file.name}..."):
                try:
                    if uploaded_file.name.lower().endswith(".epub"):
                        chunks = parse_epub(uploaded_file.read())
                    else:
                        chunks = parse_pdf(uploaded_file.read())
                except Exception as e:
                    st.error(f"Failed to parse file: {e}")
                    chunks = []

            if chunks:
                st.session_state.pdf_chunks = chunks
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.current_chunk_idx = 0
                st.session_state.chat_history = []
                st.session_state.ai_commentary = ""
                st.session_state.ai_question = ""
                st.session_state.question_answered = False
                st.session_state.section_summary = ""
                st.session_state.reading_started = False
                st.session_state.commentary_cache = {}
                st.session_state.question_cache = {}
                st.session_state.ll_summary = ""
                st.session_state.ll_vocab = []
                st.session_state.ll_vocab_parse_fail = None
                st.session_state.ll_summary_cache = {}
                st.session_state.ll_vocab_cache = {}
                st.success(f"Loaded {len(chunks)} sections from '{uploaded_file.name}'")
            else:
                st.error("Could not extract text from this file. PDFs must be text-based (not scanned). EPUBs must contain HTML content.")

        st.markdown("---")
        st.subheader("AI Model")
        installed, tags_err = fetch_ollama_models()
        if installed is not None:
            st.session_state.ollama_models_last = installed
            if installed:
                model_options = order_ollama_models(installed)
            else:
                model_options = list(PREFERRED_OLLAMA_MODELS)
                st.warning("Ollama is running but no models are pulled yet.")
        elif st.session_state.ollama_models_last:
            model_options = order_ollama_models(st.session_state.ollama_models_last)
            st.warning(tags_err or "Ollama unreachable — showing last known models.")
        else:
            model_options = list(PREFERRED_OLLAMA_MODELS)
            st.warning(tags_err or "Ollama unreachable — showing recommended model names.")

        model = st.selectbox(
            "Ollama model",
            options=model_options,
            index=0,
            help="Models are loaded from Ollama `/api/tags`. Pull with e.g. `ollama pull llama3.1:8b`.",
        )

        st.markdown("---")
        st.subheader("Mode")
        prev_mode = st.session_state.get("_prev_app_mode", st.session_state.app_mode)
        app_mode_label = st.radio(
            "Reading mode",
            ["Reading", "Language Learning"],
            index=0 if st.session_state.app_mode == "reading" else 1,
            horizontal=True,
        )
        st.session_state.app_mode = "reading" if app_mode_label == "Reading" else "language_learning"

        if st.session_state.app_mode == "language_learning":
            ll_lang_keys = list(XTTS_LANGUAGES.keys())
            if "ll_lang_select" not in st.session_state:
                default_lang = st.session_state.ll_book_language
                st.session_state.ll_lang_select = (
                    default_lang if default_lang in ll_lang_keys else "Italian"
                )
            ll_lang_label = st.selectbox("Book language", ll_lang_keys, key="ll_lang_select")
            if ll_lang_label != st.session_state.ll_book_language:
                st.session_state.ll_book_language = ll_lang_label
                st.session_state.ll_summary = ""
                st.session_state.ll_vocab = []
                st.session_state.ll_vocab_parse_fail = None
                st.session_state["xtts_lang_select"] = ll_lang_label

        # Prefer XTTS when entering Language Learning with a non-English book language.
        if (
            st.session_state.app_mode == "language_learning"
            and prev_mode != "language_learning"
            and st.session_state.ll_book_language != "English"
        ):
            st.session_state["tts_engine_choice"] = "XTTS"
        st.session_state._prev_app_mode = st.session_state.app_mode

        st.markdown("---")
        st.subheader("Read Aloud")
        # Seed engine default once; prefer XTTS already handled above on mode switch.
        if "tts_engine_choice" not in st.session_state:
            if (
                st.session_state.app_mode == "language_learning"
                and st.session_state.ll_book_language != "English"
            ):
                st.session_state.tts_engine_choice = "XTTS"
            else:
                st.session_state.tts_engine_choice = "Edge TTS"
        tts_engine = st.radio(
            "Engine", ["Edge TTS", "Kokoro", "XTTS"], horizontal=True,
            key="tts_engine_choice",
            help=(
                "Edge TTS requires internet. "
                "Kokoro runs fully locally (~115 MB, English only). "
                "XTTS supports Estonian and 17 other languages via voice cloning (~5.8 GB, downloaded once)."
            ),
        )
        if tts_engine == "Edge TTS":
            tts_voice_label = st.selectbox("Voice", list(EDGE_VOICES.keys()))
            tts_voice = EDGE_VOICES[tts_voice_label]
            if (
                st.session_state.app_mode == "language_learning"
                and st.session_state.ll_book_language != "English"
            ):
                st.caption(
                    f"Edge TTS can't do {st.session_state.ll_book_language} — switch to XTTS for that language."
                )
        elif tts_engine == "Kokoro":
            tts_voice_label = st.selectbox("Voice", list(KOKORO_VOICES.keys()))
            tts_voice = KOKORO_VOICES[tts_voice_label]
            if (
                st.session_state.app_mode == "language_learning"
                and st.session_state.ll_book_language != "English"
            ):
                st.caption(
                    f"Kokoro is English-only — switch to XTTS to read {st.session_state.ll_book_language}."
                )
        else:  # XTTS
            lang_keys = list(XTTS_LANGUAGES.keys())
            if st.session_state.app_mode == "language_learning":
                # Single source of truth: book language. Avoid a second selectbox that can diverge.
                tts_lang_label = st.session_state.ll_book_language
                if tts_lang_label not in lang_keys:
                    tts_lang_label = "Italian"
                st.caption(
                    f"XTTS language: **{tts_lang_label}** "
                    "(synced with Book language above)"
                )
                st.session_state["xtts_lang_select"] = tts_lang_label
            else:
                if "xtts_lang_select" not in st.session_state:
                    st.session_state["xtts_lang_select"] = lang_keys[0]
                tts_lang_label = st.selectbox("Language", lang_keys, key="xtts_lang_select")
            tts_voice = XTTS_LANGUAGES[tts_lang_label]

            tab_upload, tab_record = st.tabs(["Upload WAV", "Record Voice"])

            with tab_upload:
                spk_file = st.file_uploader(
                    "Speaker voice (WAV, 6+ seconds)",
                    type=["wav"],
                    key="xtts_spk_upload",
                    help="A clean WAV clip of the voice you want to clone.",
                )
                if spk_file is not None:
                    st.session_state.xtts_speaker_wav = spk_file.read()
                    st.success("Speaker voice loaded.")
                elif not st.session_state.xtts_speaker_wav:
                    st.info("Upload a WAV recording to clone a speaker voice. Any 6+ second clip works.")

            with tab_record:
                if st.session_state.get("xtts_clip_recorded"):
                    st.success("Voice clip saved — ready to use.")
                    if st.button("Record new clip", key="xtts_rerecord_btn"):
                        st.session_state["xtts_clip_recorded"] = False
                        st.rerun()
                else:
                    st.caption("Record 6+ seconds of the voice to clone.")
                    recorded = st.audio_input("Record voice clip", key="xtts_mic_record")
                    if recorded is not None:
                        try:
                            recorded.seek(0)
                            raw = recorded.read()
                            if raw:
                                from companion.tts import convert_audio_to_wav
                                wav_bytes = convert_audio_to_wav(raw)
                                st.session_state.xtts_speaker_wav = wav_bytes
                                st.session_state["xtts_clip_recorded"] = True
                                st.rerun()
                        except Exception as e:
                            st.error(str(e))
        tts_rate = st.slider("Reading speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                             help="1.0 = normal speed. Drag left to slow down, right to speed up.")

        if st.button("Check Ollama Status", use_container_width=True):
            models, err = fetch_ollama_models(timeout=3.0)
            if models is not None:
                st.session_state.ollama_models_last = models
                if models:
                    st.success(f"Ollama running. Models: {', '.join(models)}")
                else:
                    st.warning("Ollama running but no models pulled yet.")
            else:
                st.error(err or "Ollama not reachable. Run: `ollama serve`")

        if st.session_state.pdf_chunks:
            st.markdown("---")
            st.subheader("Reading Controls")

            total = len(st.session_state.pdf_chunks)
            idx = st.session_state.current_chunk_idx
            st.caption(f"Section {idx + 1} of {total}")
            st.progress((idx + 1) / total)

            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("◀ Prev", disabled=(idx == 0), use_container_width=True):
                    go_to_chunk(idx - 1)
                    st.rerun()
            with col_next:
                if st.button("Next ▶", disabled=(idx == total - 1), use_container_width=True):
                    go_to_chunk(idx + 1)
                    st.rerun()

            jump = st.number_input("Jump to section", min_value=1, max_value=total, value=idx + 1, step=1)
            if st.button("Go", use_container_width=True):
                go_to_chunk(int(jump) - 1)
                st.rerun()

            st.markdown("---")
            if not st.session_state.reading_started:
                if st.button("▶ Start Reading", type="primary", use_container_width=True):
                    st.session_state.reading_started = True
                    st.rerun()
            else:
                if st.button("Reset Session", use_container_width=True):
                    reset_session()
                    st.rerun()

    return model, tts_voice, tts_rate, tts_engine, st.session_state.app_mode, st.session_state.ll_book_language


def render_app():
    model, tts_voice, tts_rate, tts_engine, app_mode, ll_book_language = render_sidebar()

    # ------------------------------------------------------------------
    # PATH 1 — no book loaded yet: full-screen cinematic hero.
    # To change the headline, pass a different headline_variant:
    #   "silence" | "quiet" | "pages" | "voice"  (defined in hero.py)
    # To change the background scrim darkness:
    #   scrim="soft" | "heavy" | "none"
    # show_hint=True shows "Upload a book in the sidebar to begin."
    # ------------------------------------------------------------------
    if not st.session_state.pdf_chunks:
        st.markdown("""
        <style>
          [data-testid="stMain"] .block-container {
            padding: 0 !important;
            max-width: 100% !important;
          }
          header[data-testid="stHeader"] { background: transparent; }
          [data-testid="stMain"] iframe {
            height: calc(100vh - 60px) !important;
            min-height: 640px !important;
            display: block !important;
          }
        </style>
        """, unsafe_allow_html=True)
        render_landing_hero(
            headline_variant="silence",
            scrim="soft",
            show_meta=True,
            show_hint=True,
        )
        st.stop()

    # ------------------------------------------------------------------
    # PATH 2 — book parsed, reading not started yet: ambient video +
    # a frosted "Ready" card.
    # To adjust the card appearance, edit the inline div below:
    #   max-width      — card width
    #   border-radius  — corner rounding
    #   background     — card fill (rgba, affects opacity)
    #   backdrop-filter: blur(Npx)  — frosted-glass blur amount
    # Color of the "READY" label: #b8995e (gold)
    # ------------------------------------------------------------------
    if not st.session_state.reading_started:
        st.markdown(render_ambient_bg_html(), unsafe_allow_html=True)
        book_name = html.escape(st.session_state.pdf_name)
        section_count = len(st.session_state.pdf_chunks)
        st.markdown(f"""
        <div style="margin:6rem auto;max-width:480px;padding:2rem 2.5rem;
                    border:1px solid rgba(255,255,255,0.10);border-radius:14px;
                    background:rgba(10,18,30,0.70);backdrop-filter:blur(8px);
                    -webkit-backdrop-filter:blur(8px);color:#d4d0c8;
                    font-family:'Inter',sans-serif;box-shadow:0 8px 32px rgba(0,0,0,0.45);">
          <p style="margin:0 0 .4rem;font-size:.75rem;letter-spacing:.12em;
                    text-transform:uppercase;color:#b8995e;">Ready</p>
          <p style="margin:0 0 1rem;font-size:1.1rem;font-weight:600;">{book_name}</p>
          <p style="margin:0;color:#8a8a9a;font-size:.95rem;">
            {section_count} sections loaded —
            click <strong style="color:#d4d0c8;">&#9654; Start Reading</strong> in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ------------------------------------------------------------------
    # PATH 3 — active reading session.
    #
    # render_ambient_bg_html() injects the background video + all glass CSS.
    # Key knobs (edit in hero.py → render_ambient_bg_html):
    #   scrim_opacity  — base dark tint over the video (0.0–1.0).
    #                    Higher = darker page, easier to read, less video.
    #                    Current: 0.22
    #   opacity        — video visibility (default 0.85 in the function).
    #                    Lower to make the video more subtle.
    #   glass_ui=True  — applies backdrop-filter to all Streamlit panels
    #                    (sidebar, header, bordered containers, alerts…).
    #                    Set False to turn off glass on Streamlit widgets.
    #
    # All CSS for Streamlit-native widgets (forms, alerts, chat input,
    # bordered containers) lives in companion/hero.py inside the
    # glass_css string.  Edit _PANEL_BG / _PANEL_BORDER etc. at the top
    # of THIS file to restyle the HTML-rendered panels (book text,
    # AI commentary).
    # ------------------------------------------------------------------
    st.markdown(render_ambient_bg_html(scrim_opacity=0.5, glass_ui=True), unsafe_allow_html=True)

    # Progressive TTS player (fragment continues generating remaining segments).
    maybe_continue_progressive()

    # Two-column layout: reading panel (left, wider) + chat panel (right).
    # Change the ratio [3, 2] to shift space between columns, e.g.:
    #   [2, 1]  — reading panel gets more room
    #   [1, 1]  — equal split
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        if app_mode == "language_learning":
            render_language_learning_panel(model, tts_voice, tts_rate, tts_engine, ll_book_language)
        else:
            render_reading_panel(model, tts_voice, tts_rate, tts_engine)

    with right_col:
        render_chat_panel(model, tts_voice, tts_rate, tts_engine)

    st.markdown("---")
    # Audiobook generator lives in a collapsible expander at the bottom.
    # Glass style: [data-testid="stExpander"] in hero.py.
    with st.expander("Audiobook Generator"):
        render_audiobook_panel(tts_voice, tts_rate, tts_engine)
