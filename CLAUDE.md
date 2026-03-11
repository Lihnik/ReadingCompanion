# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

AI-powered reading companion — upload a PDF and read it section by section with AI commentary,
comprehension questions, text-to-speech, and a chat interface powered by a local Ollama model.

## Environment

- Python 3.12 via `.venv\` (kokoro>=0.9.4 requires Python >=3.10,<3.13)
- Activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (bash)

## Setup (first time)

```bash
py -3.12 -m venv .venv && .venv/Scripts/pip install streamlit PyMuPDF requests edge-tts "numpy>=2.0" soundfile "kokoro>=0.9.4"
```

Kokoro also requires espeak-ng for phoneme generation on Windows — download the installer from
https://github.com/espeak-ng/espeak-ng/releases (espeak-ng-X.X-x64.msi).

## Running

```bash
# Ollama must be running first:
ollama serve

# Then in a separate terminal:
.venv\Scripts\python -m streamlit run reading_companion.py
```

Or with venv activated:
```bash
python -m streamlit run reading_companion.py
```

Default Streamlit port is 8501. The app opens automatically in the browser.

## Key patterns

- **Single-file app**: all logic lives in `reading_companion.py` — no modules or packages
- **Session state**: all runtime state is in `st.session_state` (defined in `DEFAULTS` dict at top)
- **TTS engines**: Edge TTS (internet, MP3) and Kokoro-82M (local, WAV) — selected in sidebar
- **TTS cache**: generated audio is cached in session state by hash of (text, voice, rate, engine)
- **PDF parsing**: PyMuPDF with multi-column detection; one section per page (split if >3000 chars)
- **AI model**: any Ollama model; default `llama3.1:8b` — pull with `ollama pull <model>`
