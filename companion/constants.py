import copy

MAX_CHUNK_CHARS = 5000
MAX_VOCAB_PASSAGE_CHARS = 1800  # shorter passage → better vocab format adherence
TTS_CACHE_MAX_ENTRIES = 40
TTS_PROGRESSIVE_MIN_SEGMENTS = 2  # use progressive path when split yields this many+
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
    "Respond with ONLY a JSON array of exactly 5 objects, each with keys "
    '"word" and "translation". Example: '
    '[{"word":"casa","translation":"house"},{"word":"andare","translation":"to go"}]. '
    "No markdown fences, no commentary, no thinking tags — JSON array only."
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
    # Progressive / playlist TTS
    "tts_segments": [],
    "tts_segment_audios": [],
    "tts_segments_done": 0,
    "tts_progressive_active": False,
    "tts_progressive_meta": {},
    "tts_progressive_just_added": -1,
    "tts_player_gen": 0,
    "tts_voice_note": "",
    # Per-word vocab pronunciation cache: (word, lang, voice) -> (bytes, mime)
    "vocab_audio_cache": {},
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
    "ll_vocab_parse_fail": None,  # {"key": cache_key, "raw": preview} on parse failure
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

# Edge Neural voices for single-word vocab pronunciation (book language → locale voice).
# Prefer when available; XTTS word-mode is the fallback for missing locales / offline.
EDGE_LANG_VOICES = {
    "English": "en-US-AriaNeural",
    "Italian": "it-IT-ElsaNeural",
    "German": "de-DE-KatjaNeural",
    "French": "fr-FR-DeniseNeural",
    "Spanish": "es-ES-ElviraNeural",
    "Portuguese": "pt-BR-FranciscaNeural",
    "Dutch": "nl-NL-ColetteNeural",
    "Polish": "pl-PL-AgnieszkaNeural",
    "Russian": "ru-RU-SvetlanaNeural",
    "Czech": "cs-CZ-VlastaNeural",
    "Turkish": "tr-TR-EmelNeural",
    "Arabic": "ar-SA-ZariyahNeural",
    "Chinese": "zh-CN-XiaoxiaoNeural",
    "Japanese": "ja-JP-NanamiNeural",
    "Korean": "ko-KR-SunHiNeural",
    "Hungarian": "hu-HU-NoemiNeural",
    "Finnish": "fi-FI-SelmaNeural",
    "Estonian": "et-EE-AnuNeural",
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

# Facebook MMS-TTS Italian (VITS). Single dedicated voice; no speaker cloning.
MMS_ITALIAN_VOICES = {
    "MMS Italian (ita)": "ita",
}

