# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

AI-powered reading companion — upload a PDF or EPUB and read it section by section with
on-demand AI commentary, comprehension questions, text-to-speech, and a chat interface
powered by a local Ollama model.

## Environment

- Python 3.12 via `.venv` (kokoro>=0.9.4 requires Python >=3.10,<3.13)
- Activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (POSIX)

## Setup (first time)

```bash
# POSIX
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Windows
py -3.12 -m venv .venv && .venv\Scripts\pip install -r requirements.txt
```

Core deps are also listed in `pyproject.toml`. Optional XTTS extras: `pip install ".[xtts]"`.

Kokoro also requires espeak-ng for phoneme generation:
- Windows — download the installer from https://github.com/espeak-ng/espeak-ng/releases (`espeak-ng-X.X-x64.msi`)
- macOS — `brew install espeak-ng`
- Linux — `sudo apt install espeak-ng` (or equivalent)

`pydub` (mic → WAV for XTTS) also needs `ffmpeg` on `PATH`.

## Running

```bash
# Ollama must be running first:
ollama serve

# Then in a separate terminal:
python -m streamlit run reading_companion.py
# Windows without activating the venv:
# .venv\Scripts\python -m streamlit run reading_companion.py
```

Default Streamlit port is 8501. The app opens automatically in the browser.

## Key patterns

- **Modular package**: entry point is `reading_companion.py`; logic lives under `companion/`
  (`constants`, `navigation`, `ollama`, `parsing`, `tts`, `ui`, `hero`)
- **Session state**: all runtime state is in `st.session_state`. Defaults come from
  `fresh_defaults()` in `companion/constants.py` (deep-copied so mutables like
  `chat_history` / `tts_cache` are never shared with the module-level `DEFAULTS`)
- **On-demand AI**: section text renders first; commentary and comprehension questions
  are generated via “Generate insight” / “Get a question” and cached in
  `commentary_cache` / `question_cache` keyed by `pdf_name|chunk_idx|model`
- **Language Learning mode**: sidebar Mode radio; on-demand “English summary” /
  “Vocabulary” (plus optional “Prepare this section” via ThreadPoolExecutor);
  caches in `ll_summary_cache` / `ll_vocab_cache` keyed by
  `pdf_name|chunk_idx|language|model`. Non-English books prefer XTTS; XTTS
  language stays synced with `ll_book_language`
- **Ollama GPU**: model residency/GPU is via Ollama (`ollama ps`), not Streamlit.
  `call_ollama` uses short `num_predict` defaults (~512) and `keep_alive="10m"`
- **Ollama models**: sidebar selectbox is driven by `fetch_ollama_models()` (`/api/tags`);
  preferred names in `PREFERRED_OLLAMA_MODELS` are a soft sort order / offline fallback
- **TTS engines**: Edge TTS (internet, MP3), Kokoro-82M (local English WAV), XTTS (local
  multilingual WAV, voice cloning) — selected in sidebar
- **TTS cache**: generated audio is cached in session state; cache key =
  MD5(text|voice|rate|engine) for edge/kokoro, MD5(text|language|speaker_wav_hash|engine) for XTTS
- **XTTS**: uses `tartuNLP/XTTS-v2-multi` (~5.8 GB via `huggingface_hub.snapshot_download`);
  requires user-uploaded speaker WAV; imports `TTS` from the **`coqui-tts`** package;
  needs `coqui-tts[codec]`, `torchaudio`, `transformers>=4.33,<5.0`
- **PDF parsing**: PyMuPDF with multi-column detection; one section per page
  (split if >`MAX_CHUNK_CHARS`, currently 5000)
- **EPUB parsing**: stdlib zipfile + xml.etree + beautifulsoup4; spine order preserved,
  chapter titles from headings
- **AI model**: any Ollama model installed locally; default preference `llama3.1:8b` —
  pull with `ollama pull <model>`
