MAX_CHUNK_CHARS = 5000
OLLAMA_URL = "http://localhost:11434/api/generate"

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
}

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
