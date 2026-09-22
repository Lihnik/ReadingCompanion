# Reading Companion — React + FastAPI

Additive React rewrite of the Streamlit app. The classic Streamlit entrypoint (`reading_companion.py`) is **unchanged** on this branch.

## Architecture

```
/backend          FastAPI app (uvicorn)
  app/main.py
  app/db.py       SQLite persistence (books, sections, AI cache, chat)
  app/routers/    books, ai, tts, ollama
/companion/       Shared engines (PDF/EPUB, Ollama, Edge/Kokoro/XTTS/Piper Italian)
/frontend         Vite + React + TypeScript
.data/            SQLite DB (gitignored): reading_companion.sqlite3
```

Repo root is added to `PYTHONPATH` so FastAPI can `import companion…`.

## Data persistence

Books, sections, AI caches (summary / vocab / commentary / question / section-summary), and chat history are stored in SQLite:

- Default path: `.data/reading_companion.sqlite3` under the repo root
- Override directory with env `RC_DATA_DIR` (file is still `reading_companion.sqlite3` inside that dir)
- Survives uvicorn restarts; TTS jobs / word cache / speaker WAVs stay in-memory (ephemeral)
- To wipe: stop the API and delete `.data/reading_companion.sqlite3` (or the whole `.data/` folder)

The sidebar **Library** lists saved books; refresh restores the most recent book + `last_section_idx`.

## Prerequisites

- Python **3.10–3.12** preferred (Kokoro needs `<3.13`); 3.13 works for Edge TTS + API smoke tests
- Node.js 18+ (for Vite)
- [Ollama](https://ollama.com/) for AI (`ollama serve`, then `ollama pull llama3.1:8b`)
- Optional: espeak-ng (Kokoro), ffmpeg (Edge concat / mic convert), coqui-tts (XTTS), piper-tts + onnxruntime (Piper Italian), pydub (Edge MP3 concat for audiobook)

## Setup

### Backend (POSIX)

```bash
cd ReadingCompanion
python3.12 -m venv .venv          # or python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
# Optional XTTS: pip install ".[xtts]"
# Optional Piper Italian TTS: pip install ".[piper]"
#   (or: pip install piper-tts onnxruntime)
#   First run downloads it_IT-paola-medium (~60 MB) into .cache/piper/.
#   Note: facebook/mms-tts-ita does not exist on Hugging Face (Italian has MMS ASR only).
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

### Piper Italian TTS

Dedicated local Italian voice via **Piper** (`piper-tts`) and the Rhasspy voice `it_IT-paola-medium` (Paola).

> **Why not MMS?** `facebook/mms-tts-ita` does **not** exist on Hugging Face. Meta’s MMS project ships Italian **ASR**, not TTS ([HF forum](https://discuss.huggingface.co/t/why-is-mms-tts-ita-model-not-available/139990)). Piper Paola is the dedicated Italian engine used here instead.

```bash
pip install piper-tts onnxruntime
# or: pip install ".[piper]"
```

Then **restart uvicorn** so the API process picks up the new packages.

- Select **Piper Italian** in the sidebar TTS engine list (voice: **Paola (it_IT medium)**; no speaker WAV).
- First synthesis downloads the ONNX + JSON (~60 MB) into repo `.cache/piper/`.
- Speed slider maps to Piper `length_scale` (native; no crude resample).
- Long phrases are chunked (~280 chars) because Paola can glitch on very long unbroken text.
- `GET /api/tts/voices` includes `piper_italian_available: bool` so the UI can hint when deps are missing.
- Vocab word ▶ prefers Piper when that engine is selected, then **falls back to Edge** if Piper fails.

### Troubleshooting

**Word ▶ returns HTTP 502**

- If the JSON `detail` mentions `No module named 'piper'` (or onnxruntime), install Piper deps and restart:

  ```bash
  pip install ".[piper]"
  # restart: PYTHONPATH=. uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
  ```

- With Piper still unavailable, word ▶ should still play via **Edge** for languages that have an Edge voice (e.g. Italian). Section read-aloud with Piper Italian surfaces the real error on the progressive job (`seg.error`) instead of silent failure.

**Backend logs `missing ScriptRunContext!`**

- Shared `companion/tts.py` is used by both Streamlit and FastAPI. Cache helpers must not call `st.session_state` (or `get_script_run_ctx()` without `suppress_warning=True`) from AnyIO worker threads. If you still see these warnings on every `/api/tts/word`, pull the latest `grokbot-react-rewrite` fix.

### Modes

- **Language Learning** — English summary, vocabulary (+ ▶), Prepare, progressive read-aloud
- **Reading** — section text + read aloud, Generate insight (commentary), Get a question + answer feedback, optional section summarizer

### Shared

- Streaming **chat** (SSE) grounded in the current section; “Read last response” via progressive TTS
- Sidebar **Library** (restore / delete books) + Ollama model picker, TTS engine (Edge / Kokoro / XTTS / Piper Italian), voice/lang, rate, XTTS speaker WAV, book language (LL)
- Prev / Next + jump-to-section; **Stop** TTS
- **Audiobook export** — section range → background job → download link
- Two-column desktop layout: reading/LL left, chat (+ LL cards) right

## Retest checklist

1. **Chat** — upload a book, open Chat, send a question; tokens should appear immediately. Click “Read last response”.
2. **Reading mode** — toggle Reading → Generate insight → Get a question → submit an answer → Summarize section. Cache should reuse on revisit with same model.
3. **LL mode** — Prepare / summary / vocab ▶ / Read aloud with rate change; Stop clears playback.
4. **Audiobook** — pick start/end indices, Generate, wait for download link (Edge/Kokoro first; XTTS needs speaker WAV).
5. **XTTS** — upload speaker WAV, select XTTS + language, Read aloud / audiobook.
6. **Piper Italian** — select engine “Piper Italian” / voice “Paola (it_IT medium)” (no speaker upload). First synthesis downloads ~60 MB ONNX. Word ▶ uses Piper when this engine is selected.

## API sketch

| Method | Path | Notes |
|--------|------|--------|
| GET | `/api/health` | liveness |
| POST | `/api/books/upload` | PDF/EPUB → `book_id` + sections (persisted) |
| GET | `/api/books` | list books (by `updated_at` desc) |
| GET | `/api/books/{id}` | metadata + section list |
| GET | `/api/books/{id}/sections/{idx}` | section text |
| PATCH | `/api/books/{id}` | `{ last_section_idx?, language?, app_mode? }` |
| DELETE | `/api/books/{id}` | book + sections + AI cache + chat |
| POST | `/api/ai/summary` | LL English gist (`force` bypasses cache; `stream` SSE) |
| POST | `/api/ai/vocab` | `[{word,translation},…]` (cached; `force` optional) |
| POST | `/api/ai/chat` | SSE streaming chat |
| GET | `/api/ai/chat/{book_id}/{section_idx}` | saved chat thread |
| PUT | `/api/ai/chat/{book_id}/{section_idx}` | replace chat thread `{ messages }` |
| POST | `/api/ai/commentary` | reading-mode insight (cached; `force`) |
| POST | `/api/ai/question` | comprehension question (cached; `force`) |
| POST | `/api/ai/feedback` | evaluate answer (not cached) |
| POST | `/api/ai/section-summary` | reading-mode bullets (cached; `force`) |
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
| GET | `/api/tts/voices` | Edge / Kokoro / XTTS / Piper maps + `piper_italian_available` |
| GET | `/api/tts/engines` | Engine list + Piper install hint |

## Known gaps

- Cinematic hero / CloudFront video (out of scope)
- TTS / audiobook jobs remain in-memory (books + AI/chat are SQLite)
- Polished XTTS mic-record UI (file upload only)
- Edge audiobook concat needs `pydub` + ffmpeg; falls back to error if missing
