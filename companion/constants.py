import copy

MAX_CHUNK_CHARS = 5000
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

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

SYSTEM_PROMPT_LL_SUMMARY = (
    "You are a language learning assistant. When given a passage in a foreign language, "
    "write a concise 2-3 sentence summary in English capturing the main idea. "
    "Do not translate word-for-word — give the gist clearly and simply."
)

SYSTEM_PROMPT_LL_VOCAB = (
    "You are a vocabulary tutor. When given a passage in a foreign language, "
    "select exactly 5 words or short phrases that are useful to learn. "
    "Avoid very common words (articles, basic prepositions). "
    "For each entry use EXACTLY this format on separate lines:\n"
    "WORD: <word>\nTRANSLATION: <English meaning>\n---"
)

# Soft preference order for the model selectbox when those models are installed.
PREFERRED_OLLAMA_MODELS = [
    "llama3.1:8b",
    "llama3.2:3b",
    "mistral:7b",
    "gemma2:9b",
    "qwen2.5:7b",
    "qwen3.5:9b",
    "deepseek-r1:8b",
    "phi3:mini",
]

# Mutable values (lists/dicts) must never be shared across sessions — always
# obtain a fresh copy via fresh_defaults() / apply_defaults().
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
    "tts_error": "",
    "audiobook_bytes": b"",
    "audiobook_ext": "wav",
    "xtts_speaker_wav": b"",
    "xtts_clip_recorded": False,
    # Per-section AI caches keyed by "pdf_name|chunk_idx|model"
    "commentary_cache": {},
    "question_cache": {},
    # Language learning mode
    "app_mode": "reading",
    "ll_book_language": "Italian",
    "ll_summary": "",
    "ll_vocab": [],
    # Per-section LL caches keyed by "pdf_name|chunk_idx|language|model"
    "ll_summary_cache": {},
    "ll_vocab_cache": {},
    # Last known Ollama model list (used when Ollama is temporarily unreachable)
    "ollama_models_last": [],
}


def fresh_defaults() -> dict:
    """Return a deep copy of DEFAULTS so mutables are never shared."""
    return copy.deepcopy(DEFAULTS)


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

XTTS_LANGUAGES = {
    "Estonian": "et",
    "Finnish": "fi",
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Russian": "ru",
    "Polish": "pl",
    "Dutch": "nl",
    "Italian": "it",
    "Portuguese": "pt",
    "Czech": "cs",
    "Turkish": "tr",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Hungarian": "hu",
}
