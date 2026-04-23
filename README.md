# Reading Companion

An AI-powered reading companion that lets you upload a PDF and read it section by section — with AI commentary, comprehension questions, text-to-speech, and a chat interface, all powered by a local Ollama model.

## Features

- **Section-by-section reading** — PDF or EPUB uploaded, split into sections (~3000 characters max); navigate with Prev/Next or jump directly to any section
- **AI commentary** — automatic 2-3 sentence insight per section (not a summary — adds perspective)
- **Comprehension questions** — one question per section with answer submission and AI feedback
- **Section summarizer** — on-demand bullet-point summary
- **Chat** — streaming conversation grounded in the current section; last 6 messages kept as context
- **Text-to-speech** — three engines:
  - **Edge TTS** (internet required) — 6 voices across US/UK/AU English, MP3 output
  - **Kokoro-82M** (fully local) — 9 neural voices, WAV output, ~115 MB model downloaded once; runs on GPU if CUDA is available
  - **XTTS** (fully local, voice cloning) — Estonian, Finnish, and 16 other languages; upload any 6+ second WAV to clone that voice; ~5.8 GB model downloaded once; GPU accelerated
- **Audiobook generator** — render a selectable range of sections to a single WAV/MP3 file and download it; useful for skipping front/back matter
- **Multi-column PDF support** — detects two-column layouts and reads left column before right
- **Reasoning model support** — `<think>` blocks from models like Qwen3 and DeepSeek-R1 are silently stripped; token budgets sized accordingly

## Requirements

- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- Python 3.12 (kokoro requires `>=3.10,<3.13`)
- For Kokoro TTS on Windows: [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) (download `espeak-ng-X.X-x64.msi`)

## Setup

```bash
py -3.12 -m venv .venv && .venv/Scripts/pip install streamlit PyMuPDF requests edge-tts "numpy>=2.0" soundfile "kokoro>=0.9.4" beautifulsoup4
```

**GPU acceleration for Kokoro and XTTS (recommended for NVIDIA GPUs):**
```bash
.venv/Scripts/pip install torch torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```
Both engines automatically use the GPU if CUDA is detected; fall back to CPU otherwise.

**XTTS engine (Estonian/multilingual TTS):**
```bash
.venv/Scripts/pip install "coqui-tts[codec]" huggingface_hub "transformers>=4.33.0,<5.0"
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

**XTTS** ([tartuNLP/XTTS-v2-multi](https://huggingface.co/tartuNLP/XTTS-v2-multi), local, voice cloning)

Voice is determined by a reference WAV file you upload — any 6+ second clean speech recording works. Supported languages: Estonian, Finnish, English, German, French, Spanish, Russian, Polish, Dutch, Italian, Portuguese, Czech, Turkish, Arabic, Chinese, Japanese, Korean, Hungarian.
