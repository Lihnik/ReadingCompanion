import asyncio
import hashlib
import io
import json
import re
import requests
import streamlit as st
import fitz  # PyMuPDF
import edge_tts

# ── Constants ────────────────────────────────────────────────────────────────

MAX_CHUNK_CHARS = 3000
OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT_COMMENTARY = (
    "You are an insightful reading companion. When given a passage from a book, "
    "provide a brief, engaging 2-3 sentence commentary that highlights the key ideas, "
    "interesting insights, or notable writing style. Be concise and thought-provoking. "
    "Do not summarize — add perspective."
)

SYSTEM_PROMPT_QUESTION = (
    "You are a thoughtful reading tutor. Generate ONE clear comprehension question "
    "about the given passage. The question should test genuine understanding, not just "
    "recall. Output only the question itself, no preamble."
)

SYSTEM_PROMPT_CHAT = (
    "You are a knowledgeable reading companion helping a user understand a book they "
    "are reading. You have access to the current section text. Answer questions "
    "accurately, encourage deeper thinking, and refer to specific parts of the text "
    "when relevant. Be conversational and helpful."
)

# ── Session State ─────────────────────────────────────────────────────────────

DEFAULTS = {
    "pdf_chunks": [],
    "current_chunk_idx": 0,
    "chat_history": [],
    "ai_commentary": "",
    "ai_question": "",
    "question_answered": False,
    "question_feedback": "",
    "pdf_name": "",
    "reading_started": False,
    "section_summary": "",
    "tts_audio": b"",
    "tts_format": "audio/mp3",
    "tts_source": "",
    "tts_cache": {},
    "audiobook_bytes": b"",
    "audiobook_ext": "wav",
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── PDF Functions ─────────────────────────────────────────────────────────────

def split_long_page(page_text: str, page_num: int, base_index: int) -> list:
    if len(page_text) <= MAX_CHUNK_CHARS:
        return [{"index": base_index, "title": f"Page {page_num}", "text": page_text, "page": page_num}]

    paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
    sub_chunks = []
    current = []
    current_len = 0
    sub_num = 0

    for para in paragraphs:
        if current_len + len(para) > MAX_CHUNK_CHARS and current:
            sub_chunks.append({
                "index": base_index + sub_num,
                "title": f"Page {page_num}, Part {sub_num + 1}",
                "text": "\n\n".join(current),
                "page": page_num,
            })
            sub_num += 1
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        sub_chunks.append({
            "index": base_index + sub_num,
            "title": f"Page {page_num}, Part {sub_num + 1}",
            "text": "\n\n".join(current),
            "page": page_num,
        })
    return sub_chunks


def parse_pdf(file_bytes: bytes) -> list:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    chunk_index = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_width = page.rect.width
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]

        if not text_blocks:
            continue

        # Detect multi-column layout: in a two-column PDF each block is roughly
        # half the page wide, so the median block width falls well below 55% of
        # the page width.  Single-column text typically spans 70-90%.
        median_block_width = sorted(b[2] - b[0] for b in text_blocks)[len(text_blocks) // 2]
        if median_block_width < page_width * 0.55:
            # Multi-column: assign blocks to left/right column by comparing
            # each block's x-centre to the page midpoint, then sort within
            # each column top-to-bottom so the full left column is read before
            # the full right column.
            mid_x = page_width / 2
            text_blocks = sorted(
                text_blocks,
                key=lambda b: (0 if (b[0] + b[2]) / 2 < mid_x else 1, b[1]),
            )
        else:
            # Single column: simple top-to-bottom sort.
            text_blocks = sorted(text_blocks, key=lambda b: b[1])

        page_text = "\n\n".join(b[4].strip() for b in text_blocks if b[4].strip())

        if not page_text.strip():
            continue

        page_chunks = split_long_page(page_text, page_num + 1, chunk_index)
        chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    doc.close()
    return chunks

# ── Ollama Functions ──────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models (qwen3, deepseek-r1, etc.)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_ollama(prompt: str, model: str, system_prompt: str = "", num_predict: int = 4000) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": num_predict},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return _strip_thinking(resp.json().get("response", ""))
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure `ollama serve` is running."
    except requests.exceptions.Timeout:
        return "ERROR: Ollama timed out. Try a smaller model or shorter section."
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return f"ERROR: Model '{model}' not found. Run `ollama pull {model}` first."
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


def stream_ollama(prompt: str, model: str, system_prompt: str = "", num_predict: int = 2048):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {"temperature": 0.7, "num_predict": num_predict},
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            in_thinking = False
            buf = ""
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    if token:
                        buf += token
                        # Suppress <think>...</think> blocks from reasoning models
                        while True:
                            if in_thinking:
                                end = buf.find("</think>")
                                if end == -1:
                                    buf = ""
                                    break
                                buf = buf[end + len("</think>"):]
                                in_thinking = False
                            else:
                                start = buf.find("<think>")
                                if start == -1:
                                    yield buf
                                    buf = ""
                                    break
                                yield buf[:start]
                                buf = buf[start + len("<think>"):]
                                in_thinking = True
                    if data.get("done", False):
                        if buf and not in_thinking:
                            yield buf
                        return
    except requests.exceptions.ConnectionError:
        yield "\n\nERROR: Ollama not reachable. Run `ollama serve` first."
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            yield f"\n\nERROR: Model '{model}' not found. Run `ollama pull {model}` first."
        else:
            yield f"\n\nERROR: {e}"
    except Exception as e:
        yield f"\n\nERROR: {e}"

# ── Text-to-Speech ────────────────────────────────────────────────────────────

EDGE_VOICES = {
    "Aria (US, Female)": "en-US-AriaNeural",
    "Guy (US, Male)": "en-US-GuyNeural",
    "Jenny (US, Female)": "en-US-JennyNeural",
    "Sonia (UK, Female)": "en-GB-SoniaNeural",
    "Ryan (UK, Male)": "en-GB-RyanNeural",
    "Natasha (AU, Female)": "en-AU-NatashaNeural",
}

# Top-graded Kokoro voices (A/B quality). Prefix af_/am_ = American, bf_/bm_ = British.
KOKORO_VOICES = {
    "Heart (US, Female) A": "af_heart",
    "Bella (US, Female) A-": "af_bella",
    "Nicole (US, Female) B-": "af_nicole",
    "Michael (US, Male) B": "am_michael",
    "Fenrir (US, Male) B": "am_fenrir",
    "Puck (US, Male) B": "am_puck",
    "Emma (UK, Female) B-": "bf_emma",
    "George (UK, Male) B": "bm_george",
    "Fable (UK, Male) B": "bm_fable",
}


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


def speak_text(text: str, voice: str, rate: float, engine: str = "edge", source: str = ""):
    processed = _preprocess_tts_text(text)
    cache_key = hashlib.md5(f"{processed}|{voice}|{rate}|{engine}".encode()).hexdigest()
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
        else:
            audio_bytes = _speak_edge(processed, voice, rate)
            fmt = "audio/mp3"
        st.session_state.tts_cache[cache_key] = (audio_bytes, fmt)
        st.session_state.tts_audio = audio_bytes
        st.session_state.tts_format = fmt
        st.session_state.tts_source = source
    except RuntimeError as e:
        st.error(str(e))


def stop_speech():
    st.session_state.tts_audio = b""
    st.session_state.tts_source = ""


def _tts_cached(text: str, voice: str, rate: float, engine: str) -> bool:
    processed = _preprocess_tts_text(text)
    key = hashlib.md5(f"{processed}|{voice}|{rate}|{engine}".encode()).hexdigest()
    return key in st.session_state.tts_cache


def _get_or_generate_audio(text: str, voice: str, rate: float, engine: str) -> bytes:
    """Return audio bytes from cache or generate them, without touching tts_audio state."""
    processed = _preprocess_tts_text(text)
    cache_key = hashlib.md5(f"{processed}|{voice}|{rate}|{engine}".encode()).hexdigest()
    if cache_key in st.session_state.tts_cache:
        return st.session_state.tts_cache[cache_key][0]
    if engine == "kokoro":
        audio_bytes = _speak_kokoro(processed, voice, rate)
        fmt = "audio/wav"
    else:
        audio_bytes = _speak_edge(processed, voice, rate)
        fmt = "audio/mp3"
    st.session_state.tts_cache[cache_key] = (audio_bytes, fmt)
    return audio_bytes


def tts_button(label: str, text: str, source: str, tts_voice: str, tts_rate: float, tts_engine: str, full_width: bool = False):
    engine_key = "kokoro" if tts_engine == "Kokoro" else "edge"
    is_playing = bool(st.session_state.tts_audio) and st.session_state.tts_source == source
    btn_label = f"▶ {label}" if is_playing else label
    if st.button(btn_label, key=f"btn_read_{source}", use_container_width=full_width):
        if _tts_cached(text, tts_voice, tts_rate, engine_key):
            speak_text(text, tts_voice, tts_rate, engine_key, source=source)
        else:
            with st.spinner("Generating audio..."):
                speak_text(text, tts_voice, tts_rate, engine_key, source=source)
        st.rerun()


# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_commentary_prompt(chunk_text: str) -> str:
    return f"Please provide your commentary on this passage:\n\n---\n{chunk_text[:MAX_CHUNK_CHARS]}\n---\n\nYour commentary:"


def build_question_prompt(chunk_text: str) -> str:
    return f"Based on this passage, generate ONE comprehension question:\n\n---\n{chunk_text[:MAX_CHUNK_CHARS]}\n---\n\nQuestion:"


def build_feedback_prompt(chunk_text: str, question: str, answer: str) -> str:
    return (
        f"A reader is answering a comprehension question about this passage.\n\n"
        f"PASSAGE:\n{chunk_text[:MAX_CHUNK_CHARS]}\n\n"
        f"QUESTION: {question}\n\n"
        f"READER'S ANSWER: {answer}\n\n"
        f"Evaluate the answer kindly and helpfully. Confirm what they got right, "
        f"gently correct any misunderstandings, and add one insight they may not have "
        f"considered. Keep your response to 3-4 sentences."
    )


def build_chat_prompt(chunk_text: str, history: list, user_message: str) -> str:
    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    return (
        f"Current reading section:\n---\n{chunk_text[:MAX_CHUNK_CHARS]}\n---\n\n"
        f"Conversation so far:\n{history_text}"
        f"User: {user_message}\nAssistant:"
    )


def build_summary_prompt(chunk_text: str) -> str:
    return (
        f"Summarize the following passage in 3-5 bullet points. "
        f"Focus on the most important ideas.\n\n{chunk_text[:MAX_CHUNK_CHARS]}\n\nSummary:"
    )

# ── Navigation Helpers ────────────────────────────────────────────────────────

def go_to_chunk(new_idx: int):
    total = len(st.session_state.pdf_chunks)
    st.session_state.current_chunk_idx = max(0, min(new_idx, total - 1))
    st.session_state.ai_commentary = ""
    st.session_state.ai_question = ""
    st.session_state.question_answered = False
    st.session_state.question_feedback = ""
    st.session_state.section_summary = ""
    st.session_state.tts_audio = b""
    st.session_state.tts_source = ""


def reset_session():
    for key, default in DEFAULTS.items():
        st.session_state[key] = default

# ── Render Functions ──────────────────────────────────────────────────────────

def render_comprehension_question(chunk: dict, model: str, tts_voice: str, tts_rate: float, tts_engine: str):
    st.markdown("**Comprehension Check**")

    if not st.session_state.ai_question:
        with st.spinner("Generating comprehension question..."):
            q = call_ollama(build_question_prompt(chunk["text"]), model, SYSTEM_PROMPT_QUESTION)
            st.session_state.ai_question = q

    st.markdown(f"> {st.session_state.ai_question}")
    tts_button("Read question", st.session_state.ai_question, "question", tts_voice, tts_rate, tts_engine)

    if not st.session_state.question_answered:
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
            st.success(st.session_state.question_feedback)
            tts_button("Read feedback", st.session_state.question_feedback, "feedback", tts_voice, tts_rate, tts_engine)


def render_reading_panel(model: str, tts_voice: str = "en-US-AriaNeural", tts_rate: float = 1.0, tts_engine: str = "edge"):
    chunk = st.session_state.pdf_chunks[st.session_state.current_chunk_idx]

    st.subheader(chunk["title"])

    with st.container(height=300, border=True):
        st.markdown(chunk["text"])

    tts_button("Read section aloud", chunk["text"], "section", tts_voice, tts_rate, tts_engine, full_width=True)

    st.markdown("---")

    st.markdown("**AI Commentary**")
    if not st.session_state.ai_commentary:
        with st.spinner("AI is reading this section..."):
            commentary = call_ollama(build_commentary_prompt(chunk["text"]), model, SYSTEM_PROMPT_COMMENTARY)
            st.session_state.ai_commentary = commentary
    st.info(st.session_state.ai_commentary)
    tts_button("Read commentary", st.session_state.ai_commentary, "commentary", tts_voice, tts_rate, tts_engine)

    st.markdown("---")
    render_comprehension_question(chunk, model, tts_voice, tts_rate, tts_engine)

    st.markdown("---")
    if st.button("Summarize This Section", key="btn_summarize"):
        with st.spinner("Summarizing..."):
            summary = call_ollama(build_summary_prompt(chunk["text"]), model)
            st.session_state.section_summary = summary

    if st.session_state.section_summary:
        with st.expander("Section Summary", expanded=True):
            st.write(st.session_state.section_summary)
            tts_button("Read summary", st.session_state.section_summary, "summary", tts_voice, tts_rate, tts_engine)


def render_chat_panel(model: str, tts_voice: str, tts_rate: float, tts_engine: str):
    st.subheader("Chat with AI")

    with st.container(height=400, border=True):
        if not st.session_state.chat_history:
            st.caption("No messages yet. Ask anything about what you're reading.")
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

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
    engine_key = "kokoro" if tts_engine == "Kokoro" else "edge"

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

        if engine_key == "kokoro":
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
        fname = st.session_state.pdf_name.replace(".pdf", f"_audiobook.{ext}")
        size_mb = len(st.session_state.audiobook_bytes) / 1024 / 1024
        st.download_button(
            f"Download Audiobook ({size_mb:.1f} MB)",
            data=st.session_state.audiobook_bytes,
            file_name=fname,
            mime=f"audio/{ext}",
            use_container_width=True,
        )


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Reading Companion",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📖 Reading Companion")
    st.markdown("---")

    st.subheader("Upload Book / Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner(f"Parsing {uploaded_file.name}..."):
            chunks = parse_pdf(uploaded_file.read())

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
            st.error("Could not extract text from this PDF. It may be image-based or encrypted.")

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
    tts_engine = st.radio("Engine", ["Edge TTS", "Kokoro"], horizontal=True,
                          help="Edge TTS requires internet. Kokoro runs fully locally (~115 MB model, downloaded once).")
    if tts_engine == "Edge TTS":
        tts_voice_label = st.selectbox("Voice", list(EDGE_VOICES.keys()))
        tts_voice = EDGE_VOICES[tts_voice_label]
    else:
        tts_voice_label = st.selectbox("Voice", list(KOKORO_VOICES.keys()))
        tts_voice = KOKORO_VOICES[tts_voice_label]
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

# ── Main Area ─────────────────────────────────────────────────────────────────

if not st.session_state.pdf_chunks:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

if not st.session_state.reading_started:
    st.info(f"**{st.session_state.pdf_name}** loaded with {len(st.session_state.pdf_chunks)} sections. Click **▶ Start Reading** in the sidebar.")
    st.stop()

if st.session_state.tts_audio:
    col_audio, col_stop = st.columns([5, 1])
    with col_audio:
        st.audio(st.session_state.tts_audio, format=st.session_state.tts_format, autoplay=True)
    with col_stop:
        if st.button("Stop", key="btn_stop"):
            stop_speech()
            st.rerun()

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    render_reading_panel(model, tts_voice, tts_rate, tts_engine)

with right_col:
    render_chat_panel(model, tts_voice, tts_rate, tts_engine)

st.markdown("---")
with st.expander("Audiobook Generator"):
    render_audiobook_panel(tts_voice, tts_rate, tts_engine)
