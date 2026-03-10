# Reading Companion

An AI-powered reading companion that lets you upload a PDF and read it section by section — with AI commentary, comprehension questions, text-to-speech, and a chat interface, all powered by a local Ollama model.

## Features

- **Section-by-section reading** — PDF is split into pages (long pages split further at ~3000 characters); navigate with Prev/Next or jump directly to any section
- **AI commentary** — automatic 2-3 sentence insight per section (not a summary — adds perspective)
- **Comprehension questions** — one question per section with answer submission and AI feedback
- **Section summarizer** — on-demand bullet-point summary
- **Chat** — streaming conversation grounded in the current section; last 6 messages kept as context
- **Text-to-speech** — two engines:
  - **Edge TTS** (internet required) — 6 voices across US/UK/AU English, MP3 output
  - **Kokoro-82M** (fully local) — 9 neural voices, WAV output, ~115 MB model downloaded once
- **Multi-column PDF support** — detects two-column layouts and reads left column before right
- **Reasoning model support** — `<think>` blocks from models like Qwen3 and DeepSeek-R1 are silently stripped; token budgets sized accordingly

## Requirements

- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- Python 3.12 (kokoro requires `>=3.10,<3.13`)
- For Kokoro TTS on Windows: [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) (download `espeak-ng-X.X-x64.msi`)

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install streamlit PyMuPDF requests edge-tts "numpy>=2.0" soundfile "kokoro>=0.9.4"
```

Pull at least one Ollama model:
```bash
ollama pull llama3.1:8b
```

## Running

```bash
# Terminal 1
ollama serve

# Terminal 2
.venv\Scripts\python -m streamlit run reading_companion.py
```

Opens at `http://localhost:8501`.

## Supported Models

Select in the sidebar — pull each with `ollama pull <name>` first:

| Model | Notes |
|---|---|
| `llama3.1:8b` | Default, well-rounded |
| `llama3.2:3b` | Faster, lighter |
| `mistral:7b` | Good at instruction following |
| `gemma2:9b` | Strong comprehension |
| `qwen2.5:7b` | Multilingual capable |
| `qwen3.5:9b` | Reasoning model (thinking stripped) |
| `deepseek-r1:8b` | Reasoning model (thinking stripped) |
| `phi3:mini` | Very fast, small footprint |

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
