# Reading Companion — React + FastAPI

Additive React rewrite of the Streamlit app. The classic Streamlit entrypoint (`reading_companion.py`) is **unchanged** on this branch.

## Architecture

```
/backend          FastAPI app (uvicorn)
  app/main.py
  app/routers/    books, ai, tts, ollama
/companion/       Shared engines (PDF/EPUB, Ollama, Edge/Kokoro/XTTS)
/frontend         Vite + React + TypeScript
```

Repo root is added to `PYTHONPATH` so FastAPI can `import companion…`.

## Prerequisites

- Python **3.10–3.12** preferred (Kokoro needs `<3.13`); 3.13 works for Edge TTS + API smoke tests
- Node.js 18+ (for Vite)
- [Ollama](https://ollama.com/) for AI (`ollama serve`, then `ollama pull llama3.1:8b`)
- Optional: espeak-ng (Kokoro), ffmpeg (Edge concat / mic convert), coqui-tts (XTTS), pydub (Edge MP3 concat for audiobook)

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

## Features

### Modes

- **Language Learning** — English summary, vocabulary (+ ▶), Prepare, progressive read-aloud
- **Reading** — section text + read aloud, Generate insight (commentary), Get a question + answer feedback, optional section summarizer

### Shared

- Streaming **chat** (SSE) grounded in the current section; “Read last response” via progressive TTS
- Sidebar: Ollama model picker, TTS engine (Edge / Kokoro / XTTS), voice/lang, rate, XTTS speaker WAV, book language (LL)
- Prev / Next + jump-to-section; **Stop** TTS
- **Audiobook export** — section range → background job → download link
- Two-column desktop layout: reading/LL left, chat (+ LL cards) right

## Retest checklist

1. **Chat** — upload a book, open Chat, send a question; tokens should appear immediately. Click “Read last response”.
2. **Reading mode** — toggle Reading → Generate insight → Get a question → submit an answer → Summarize section. Cache should reuse on revisit with same model.
3. **LL mode** — Prepare / summary / vocab ▶ / Read aloud with rate change; Stop clears playback.
4. **Audiobook** — pick start/end indices, Generate, wait for download link (Edge/Kokoro first; XTTS needs speaker WAV).
5. **XTTS** — upload speaker WAV, select XTTS + language, Read aloud / audiobook.

## API sketch

| Method | Path | Notes |
|--------|------|--------|
| GET | `/api/health` | liveness |
| POST | `/api/books/upload` | PDF/EPUB → `book_id` + sections |
| GET | `/api/books/{id}/sections/{idx}` | section text |
| POST | `/api/ai/summary` | LL English gist (`stream` optional SSE) |
| POST | `/api/ai/vocab` | `[{word,translation},…]` |
| POST | `/api/ai/chat` | SSE streaming chat |
| POST | `/api/ai/commentary` | reading-mode insight |
| POST | `/api/ai/question` | comprehension question |
| POST | `/api/ai/feedback` | evaluate answer |
| POST | `/api/ai/section-summary` | reading-mode bullets |
| GET | `/api/ollama/models` | live tags |
| POST | `/api/tts/speaker` | upload XTTS speaker WAV |
| POST | `/api/tts/section/start` | progressive section job |
| POST | `/api/tts/text/start` | progressive arbitrary text |
| POST | `/api/tts/oneshot` | single audio clip |
| GET | `/api/tts/jobs/{id}/segments` | ready segments + URLs |
| GET | `/api/tts/jobs/{id}/segments/{n}.wav` | audio bytes |
| POST | `/api/tts/jobs/{id}/stop` | cancel progressive job |
| POST | `/api/tts/word` | single-word clip |
| POST | `/api/tts/audiobook/start` | range → combined audio job |
| GET | `/api/tts/audiobook/{id}` | status |
| GET | `/api/tts/audiobook/{id}/download` | file |
| POST | `/api/tts/audiobook/{id}/stop` | cancel |
| GET | `/api/tts/voices` | Edge / Kokoro / XTTS maps |

## Known gaps

- Cinematic hero / CloudFront video (out of scope)
- Persistent disk store (books/jobs are in-memory)
- Polished XTTS mic-record UI (file upload only)
- Edge audiobook concat needs `pydub` + ffmpeg; falls back to error if missing
