# Reading Companion — React + FastAPI (Language Learning MVP)

Additive React rewrite of the Streamlit app. The classic Streamlit entrypoint (`reading_companion.py`) is **unchanged** on this branch.

## Architecture

```
/backend          FastAPI app (uvicorn)
  app/main.py
  app/routers/    books, ai, tts, ollama
/companion/       Shared engines (PDF/EPUB, Ollama, Edge/Kokoro/XTTS)
/frontend         Vite + React + TypeScript (Language Learning UI)
```

Repo root is added to `PYTHONPATH` so FastAPI can `import companion…`.

## Prerequisites

- Python **3.10–3.12** preferred (Kokoro needs `<3.13`); 3.13 works for Edge TTS + API smoke tests
- Node.js 18+ (for Vite)
- [Ollama](https://ollama.com/) for summary/vocab (`ollama serve`, then `ollama pull llama3.1:8b`)
- Optional: espeak-ng (Kokoro), ffmpeg (Edge concat / mic convert), coqui-tts (XTTS)

## Setup

### Backend (POSIX)

```bash
cd ReadingCompanion
python3.12 -m venv .venv          # or python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
# Optional XTTS: pip install ".[xtts]"
```

### Backend (Windows PowerShell)

```powershell
cd ReadingCompanion
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-api.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Run

Terminal 1 — API (from **repo root**):

```bash
# POSIX
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

# Windows (cmd / PowerShell)
.\.venv\Scripts\python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
# Set PYTHONPATH to the repo root first, e.g.:
#   PowerShell:  $env:PYTHONPATH = (Get-Location).Path
#   cmd:         set PYTHONPATH=%CD%
```

Terminal 2 — Vite:

```bash
cd frontend
npm run dev
```

- Frontend: http://localhost:5173 (proxies `/api` → `:8000`)
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/api/health

Streamlit (still available):

```bash
python -m streamlit run reading_companion.py
```

## MVP features

- Upload PDF/EPUB → section list → read section text
- Ollama model picker (live `/api/tags` with fallback list)
- Book language (default Italian)
- **English summary**, **Vocabulary**, **Prepare** (parallel)
- Progressive **Read aloud** with one full-timeline scrubber (polls segment jobs)
- Vocab ▶ word pronunciation (Edge preferred; XTTS if speaker WAV uploaded)
- Chat stubbed as “coming soon”

## Known gaps (not in this MVP)

- Chat / reading-mode Q&A / commentary
- Audiobook export
- Hero cinematic UI
- Persistent disk store (books/jobs are in-memory)
- Polished XTTS speaker recording UI

## API sketch

| Method | Path | Notes |
|--------|------|--------|
| GET | `/api/health` | liveness |
| POST | `/api/books/upload` | PDF/EPUB → `book_id` + sections |
| GET | `/api/books/{id}/sections/{idx}` | section text |
| POST | `/api/ai/summary` | English gist (`stream` optional SSE) |
| POST | `/api/ai/vocab` | `[{word,translation},…]` |
| GET | `/api/ollama/models` | live tags |
| POST | `/api/tts/speaker` | upload XTTS speaker WAV |
| POST | `/api/tts/section/start` | start progressive job |
| GET | `/api/tts/jobs/{id}/segments` | ready segments + URLs |
| GET | `/api/tts/jobs/{id}/segments/{n}.wav` | audio bytes |
| POST | `/api/tts/word` | single-word clip |
