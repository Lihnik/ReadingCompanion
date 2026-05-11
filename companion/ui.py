import html
import io
import re

import requests
import streamlit as st

from .constants import (
    EDGE_VOICES,
    KOKORO_VOICES,
    XTTS_LANGUAGES,
    SYSTEM_PROMPT_COMMENTARY,
    SYSTEM_PROMPT_QUESTION,
    SYSTEM_PROMPT_CHAT,
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
)
from .hero import render_landing_hero, render_ambient_bg_html
from .parsing import parse_epub, parse_pdf
from .tts import (
    _get_or_generate_audio,
    _tts_cached,
    speak_text,
    stop_speech,
    tts_button,
)


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


def render_comprehension_question(chunk: dict, model: str, tts_voice: str, tts_rate: float, tts_engine: str):
    st.markdown("**Comprehension Check**")

    if not st.session_state.ai_question:
        with st.spinner("Generating comprehension question..."):
            q = call_ollama(build_question_prompt(chunk["text"]), model, SYSTEM_PROMPT_QUESTION)
            st.session_state.ai_question = q

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
    if not st.session_state.ai_commentary:
        with st.spinner("AI is reading this section..."):
            commentary = call_ollama(build_commentary_prompt(chunk["text"]), model, SYSTEM_PROMPT_COMMENTARY)
            st.session_state.ai_commentary = commentary
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
                st.success(f"Loaded {len(chunks)} sections from '{uploaded_file.name}'")
            else:
                st.error("Could not extract text from this file. PDFs must be text-based (not scanned). EPUBs must contain HTML content.")

        st.markdown("---")
        st.subheader("AI Model")
        model = st.selectbox(
            "Ollama model",
            options=[
                "llama3.1:8b",
                "llama3.2:3b",
                "mistral:7b",
                "gemma2:9b",
                "qwen2.5:7b",
                "qwen3.5:9b",
                "deepseek-r1:8b",
                "phi3:mini",
            ],
            index=0,
            help="Make sure this model is pulled in Ollama first (e.g. `ollama pull llama3.1:8b`).",
        )

        st.markdown("---")
        st.subheader("Read Aloud")
        tts_engine = st.radio(
            "Engine", ["Edge TTS", "Kokoro", "XTTS"], horizontal=True,
            help=(
                "Edge TTS requires internet. "
                "Kokoro runs fully locally (~115 MB, English only). "
                "XTTS supports Estonian and 17 other languages via voice cloning (~5.8 GB, downloaded once)."
            ),
        )
        if tts_engine == "Edge TTS":
            tts_voice_label = st.selectbox("Voice", list(EDGE_VOICES.keys()))
            tts_voice = EDGE_VOICES[tts_voice_label]
        elif tts_engine == "Kokoro":
            tts_voice_label = st.selectbox("Voice", list(KOKORO_VOICES.keys()))
            tts_voice = KOKORO_VOICES[tts_voice_label]
        else:  # XTTS
            tts_lang_label = st.selectbox("Language", list(XTTS_LANGUAGES.keys()), index=0)
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
            try:
                r = requests.get("http://localhost:11434/api/tags", timeout=3)
                models = [m["name"] for m in r.json().get("models", [])]
                if models:
                    st.success(f"Ollama running. Models: {', '.join(models)}")
                else:
                    st.warning("Ollama running but no models pulled yet.")
            except Exception:
                st.error("Ollama not reachable. Run: `ollama serve`")

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

    return model, tts_voice, tts_rate, tts_engine


def render_app():
    model, tts_voice, tts_rate, tts_engine = render_sidebar()

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

    if st.session_state.tts_error:
        st.error(st.session_state.tts_error)
        if st.button("Dismiss error", key="btn_dismiss_tts_error"):
            st.session_state.tts_error = ""
            st.rerun()

    if st.session_state.tts_audio:
        col_audio, col_stop = st.columns([5, 1])
        with col_audio:
            st.audio(st.session_state.tts_audio, format=st.session_state.tts_format, autoplay=True)
        with col_stop:
            if st.button("Stop", key="btn_stop"):
                stop_speech()
                st.rerun()

    # Two-column layout: reading panel (left, wider) + chat panel (right).
    # Change the ratio [3, 2] to shift space between columns, e.g.:
    #   [2, 1]  — reading panel gets more room
    #   [1, 1]  — equal split
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        render_reading_panel(model, tts_voice, tts_rate, tts_engine)

    with right_col:
        render_chat_panel(model, tts_voice, tts_rate, tts_engine)

    st.markdown("---")
    # Audiobook generator lives in a collapsible expander at the bottom.
    # Glass style: [data-testid="stExpander"] in hero.py.
    with st.expander("Audiobook Generator"):
        render_audiobook_panel(tts_voice, tts_rate, tts_engine)
