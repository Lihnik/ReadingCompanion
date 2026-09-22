# Reading Companion

An AI-powered reading companion that lets you upload a PDF or EPUB and read it section by section — with on-demand AI commentary, comprehension questions, text-to-speech, and a chat interface, all powered by a local Ollama model.

## Features

- **Section-by-section reading** — PDF or EPUB uploaded, split into sections (~5000 characters max); navigate with Prev/Next or jump directly to any section
- **AI commentary** — on-demand 2-3 sentence insight per section (not a summary — adds perspective); cached so revisiting a section does not regenerate
- **Comprehension questions** — on-demand question per section with answer submission and AI feedback; cached per section
- **Section summarizer** — on-demand bullet-point summary
- **Chat** — streaming conversation grounded in the current section; last 6 messages kept as context
- **Text-to-speech** — three engines:
  - **Edge TTS** (internet required) — 6 voices across US/UK/AU English, MP3 output
  - **Kokoro-82M** (fully local) — 9 neural voices, WAV output, ~115 MB model downloaded once; runs on GPU if CUDA is available
  - **XTTS** (fully local, voice cloning) — Estonian, Finnish, and 16 other languages; upload any 6+ second WAV to clone that voice; ~5.8 GB model downloaded once; GPU accelerated
- **Audiobook generator** — render a selectable range of sections to a single WAV/MP3 file and download it; useful for skipping front/back matter
- **Multi-column PDF support** — detects two-column layouts and reads left column before right
- **Reasoning model support** — `<think>` blocks from models like Qwen3 and DeepSeek-R1 are silently stripped; token budgets sized accordingly
- **Live model list** — sidebar model selectbox is filled from Ollama `/api/tags` (falls back to recommended names if Ollama is down)
- **Language Learning mode** — for foreign-language books (default book language: Italian): on-demand English summary and vocabulary, optional one-click “Prepare this section” (runs both in parallel), cached per section/language/model; prefers XTTS for non-English TTS

## Requirements

- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- Python 3.12 (kokoro requires `>=3.10,<3.13`)
- For Kokoro TTS phonemes: [espeak-ng](https://github.com/espeak-ng/espeak-ng)
  - Windows: download `espeak-ng-X.X-x64.msi` from the releases page
  - macOS: `brew install espeak-ng`
  - Linux: `sudo apt install espeak-ng` (or your distro equivalent)
- For mic → WAV conversion (XTTS record tab): `pydub` plus [ffmpeg](https://ffmpeg.org/) on `PATH`

## Setup

```bash
# Create a venv (POSIX)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create a venv (Windows PowerShell / cmd)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Or install from the minimal `pyproject.toml`:

```bash
pip install .
```

**GPU acceleration for Kokoro and XTTS (recommended for NVIDIA GPUs):**

```bash
# POSIX
pip install torch torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu128

# Windows
.venv\Scripts\pip install torch torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

Both engines automatically use the GPU if CUDA is detected; fall back to CPU otherwise.

LLM GPU usage is handled by **Ollama** (not Streamlit). Check loaded models with `ollama ps`.

**XTTS engine (Estonian/multilingual TTS):**

XTTS is imported as `from TTS...` — that module comes from the **`coqui-tts`** package (not a separate `TTS` PyPI name in current installs):

```bash
pip install "coqui-tts[codec]" huggingface_hub "transformers>=4.33.0,<5.0"
# or: pip install ".[xtts]"
```

The ~5.8 GB model is downloaded on first use and cached permanently.

Pull at least one Ollama model:

```bash
ollama pull llama3.1:8b
```

## Running

```bash
# Terminal 1
ollama serve

# Terminal 2 (POSIX)
python -m streamlit run reading_companion.py

# Terminal 2 (Windows, without activating the venv)
.venv\Scripts\python -m streamlit run reading_companion.py
```

Opens at `http://localhost:8501`.

## Supported Models

The sidebar selectbox lists models installed in Ollama (from `/api/tags`). Recommended names (used as soft preference order when present, and as fallback if Ollama is unreachable):

| Model | Notes |
|---|---|
| `llama3.1:8b` | Default preference, well-rounded |
| `llama3.2:3b` | Faster, lighter |
| `mistral:7b` | Good at instruction following |
| `gemma2:9b` | Strong comprehension |
| `qwen2.5:7b` | Multilingual capable |
| `qwen3.5:9b` | Reasoning model (thinking stripped) |
| `deepseek-r1:8b` | Reasoning model (thinking stripped) |
| `phi3:mini` | Very fast, small footprint |

Pull each with `ollama pull <name>` first.

## TTS Voices

**Edge TTS** (requires internet)

| Label | Voice ID |
|---|---|
| Aria (US, Female) | `en-US-AriaNeural` |
| Guy (US, Male) | `en-US-GuyNeural` |
| Jenny (US, Female) | `en-US-JennyNeural` |
| Sonia (UK, Female) | `en-GB-SoniaNeural` |
| Ryan (UK, Male) | `en-GB-RyanNeural` |
| Natasha (AU, Female) | `en-AU-NatashaNeural` |

**Kokoro** (local, Python 3.12 required)

| Label | Voice ID |
|---|---|
| Heart (US, Female) A | `af_heart` |
| Bella (US, Female) A- | `af_bella` |
| Nicole (US, Female) B- | `af_nicole` |
| Michael (US, Male) B | `am_michael` |
| Fenrir (US, Male) B | `am_fenrir` |
| Puck (US, Male) B | `am_puck` |
| Emma (UK, Female) B- | `bf_emma` |
| George (UK, Male) B | `bm_george` |
| Fable (UK, Male) B | `bm_fable` |

**XTTS** ([tartuNLP/XTTS-v2-multi](https://huggingface.co/tartuNLP/XTTS-v2-multi), local, voice cloning)

Voice is determined by a reference WAV file you upload — any 6+ second clean speech recording works. Supported languages: Estonian, Finnish, English, German, French, Spanish, Russian, Polish, Dutch, Italian, Portuguese, Czech, Turkish, Arabic, Chinese, Japanese, Korean, Hungarian.
