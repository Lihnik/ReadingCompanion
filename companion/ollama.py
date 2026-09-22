import json
import re

import requests

from .constants import (
    MAX_CHUNK_CHARS,
    MAX_VOCAB_PASSAGE_CHARS,
    OLLAMA_URL,
    OLLAMA_TAGS_URL,
    PREFERRED_OLLAMA_MODELS,
    SYSTEM_PROMPT_COMMENTARY,
    SYSTEM_PROMPT_QUESTION,
    SYSTEM_PROMPT_CHAT,
    SYSTEM_PROMPT_LL_SUMMARY,
    SYSTEM_PROMPT_LL_VOCAB,
)

# Short tasks (commentary, question, LL summary/vocab) stay small; chat/stream keeps its own default.
DEFAULT_NUM_PREDICT = 512
DEFAULT_KEEP_ALIVE = "10m"


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models (qwen3, deepseek-r1, etc.)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_ollama(
    prompt: str,
    model: str,
    system_prompt: str = "",
    num_predict: int = DEFAULT_NUM_PREDICT,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": 0.7, "num_predict": num_predict},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return _strip_thinking(resp.json().get("response", ""))
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure `ollama serve` is running."
    except requests.exceptions.Timeout:
        return "ERROR: Ollama timed out. Try a smaller model or shorter section."
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return f"ERROR: Model '{model}' not found. Run `ollama pull {model}` first."
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


def stream_ollama(
    prompt: str,
    model: str,
    system_prompt: str = "",
    num_predict: int = 2048,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {"temperature": 0.7, "num_predict": num_predict},
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            in_thinking = False
            buf = ""
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    if token:
                        buf += token
                        # Suppress <think>...</think> blocks from reasoning models
                        while True:
                            if in_thinking:
                                end = buf.find("</think>")
                                if end == -1:
                                    buf = ""
                                    break
                                buf = buf[end + len("</think>"):]
                                in_thinking = False
                            else:
                                start = buf.find("<think>")
                                if start == -1:
                                    yield buf
                                    buf = ""
                                    break
                                yield buf[:start]
                                buf = buf[start + len("<think>"):]
                                in_thinking = True
                    if data.get("done", False):
                        if buf and not in_thinking:
                            yield buf
                        return
    except requests.exceptions.ConnectionError:
        yield "\n\nERROR: Ollama not reachable. Run `ollama serve` first."
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            yield f"\n\nERROR: Model '{model}' not found. Run `ollama pull {model}` first."
        else:
            yield f"\n\nERROR: {e}"
    except Exception as e:
        yield f"\n\nERROR: {e}"


def fetch_ollama_models(timeout: float = 2.0) -> tuple[list[str] | None, str | None]:
    """Return (model_names, error). model_names is None when Ollama is unreachable."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", []) if m.get("name")]
        return names, None
    except requests.exceptions.ConnectionError:
        return None, "Ollama not reachable. Run: `ollama serve`"
    except requests.exceptions.Timeout:
        return None, "Ollama tags request timed out."
    except Exception as e:
        return None, f"Could not list Ollama models: {e}"


def order_ollama_models(installed: list[str]) -> list[str]:
    """Preferred names first (when present), then remaining installed names sorted."""
    preferred = [m for m in PREFERRED_OLLAMA_MODELS if m in installed]
    rest = sorted(m for m in installed if m not in preferred)
    return preferred + rest


def build_commentary_prompt(chunk_text: str) -> str:
    return f"Please provide your commentary on this passage:\n\n---\n{chunk_text[:MAX_CHUNK_CHARS]}\n---\n\nYour commentary:"


def build_question_prompt(chunk_text: str) -> str:
    return f"Based on this passage, generate ONE comprehension question:\n\n---\n{chunk_text[:MAX_CHUNK_CHARS]}\n---\n\nQuestion:"


def build_feedback_prompt(chunk_text: str, question: str, answer: str) -> str:
    return (
        f"A reader is answering a comprehension question about this passage.\n\n"
        f"PASSAGE:\n{chunk_text[:MAX_CHUNK_CHARS]}\n\n"
        f"QUESTION: {question}\n\n"
        f"READER'S ANSWER: {answer}\n\n"
        f"Evaluate the answer kindly and helpfully. Confirm what they got right, "
        f"gently correct any misunderstandings, and add one insight they may not have "
        f"considered. Keep your response to 3-4 sentences."
    )


def build_chat_prompt(chunk_text: str, history: list, user_message: str) -> str:
    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    return (
        f"Current reading section:\n---\n{chunk_text[:MAX_CHUNK_CHARS]}\n---\n\n"
        f"Conversation so far:\n{history_text}"
        f"User: {user_message}\nAssistant:"
    )


def build_summary_prompt(chunk_text: str) -> str:
    return (
        f"Summarize the following passage in 3-5 bullet points. "
        f"Focus on the most important ideas.\n\n{chunk_text[:MAX_CHUNK_CHARS]}\n\nSummary:"
    )


def build_ll_summary_prompt(chunk_text: str, source_lang: str) -> str:
    return (
        f"The following passage is written in {source_lang}. "
        f"Summarize it in English in 2-3 sentences:\n\n"
        f"---\n{chunk_text[:MAX_CHUNK_CHARS]}\n---\n\nEnglish summary:"
    )


def build_ll_vocab_prompt(chunk_text: str, source_lang: str) -> str:
    passage = chunk_text[:MAX_VOCAB_PASSAGE_CHARS]
    return (
        f"From this {source_lang} passage, select exactly 5 vocabulary words to teach.\n\n"
        f"---\n{passage}\n---\n\n"
        f'Output ONLY a JSON array: [{{"word":"...","translation":"..."}}, ...] '
        f"with exactly 5 objects. No other text:"
    )


def _clean_vocab_pair(word: str, translation: str) -> dict | None:
    word = (word or "").strip().strip("*`\"'")
    translation = (translation or "").strip().strip("*`\"'")
    # Drop leftover labels if a model echoed them
    for prefix in ("WORD:", "Word:", "TRANSLATION:", "Translation:"):
        if word.upper().startswith(prefix.upper()):
            word = word.split(":", 1)[-1].strip()
        if translation.upper().startswith(prefix.upper()):
            translation = translation.split(":", 1)[-1].strip()
    if not word or not translation:
        return None
    if len(word) > 80 or len(translation) > 200:
        return None
    return {"word": word, "translation": translation}


def _parse_vocab_json(text: str) -> list:
    """Try to extract a JSON list of {word, translation} from model output."""
    candidates = []
    stripped = text.strip()
    # Fenced ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1))
    # First top-level array
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    for raw in candidates:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else data.get("vocabulary") or data.get("words") or data.get("entries")
        if not isinstance(items, list):
            continue
        entries = []
        for item in items:
            if isinstance(item, dict):
                word = item.get("word") or item.get("term") or item.get("lemma") or ""
                translation = (
                    item.get("translation")
                    or item.get("english")
                    or item.get("meaning")
                    or item.get("gloss")
                    or ""
                )
                pair = _clean_vocab_pair(str(word), str(translation))
                if pair:
                    entries.append(pair)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pair = _clean_vocab_pair(str(item[0]), str(item[1]))
                if pair:
                    entries.append(pair)
        if entries:
            return entries[:5]
    return []


def _parse_vocab_word_blocks(text: str) -> list:
    entries = []
    for block in re.split(r"\n\s*---\s*\n|\n---\n", text):
        block = block.strip()
        if not block:
            continue
        word, translation = "", ""
        for line in block.splitlines():
            upper = line.strip().upper()
            if upper.startswith("WORD:"):
                word = line.split(":", 1)[-1].strip()
            elif upper.startswith("TRANSLATION:") or upper.startswith("MEANING:"):
                translation = line.split(":", 1)[-1].strip()
        pair = _clean_vocab_pair(word, translation)
        if pair:
            entries.append(pair)
    return entries[:5]


def _parse_vocab_inline_lines(text: str) -> list:
    """word — translation / word: translation / bullets / numbered / markdown table rows."""
    entries = []
    # Em dash, en dash, or hyphen separators; also colon when not a WORD: label
    line_re = re.compile(
        r"^\s*(?:[-*•]\s+|\d+[.)]\s+)?"  # bullet / number
        r"[*`\"']?"
        r"(.+?)"
        r"[*`\"']?"
        r"\s*(?:—|–|\s-\s|:)\s*"
        r"[*`\"']?"
        r"(.+?)"
        r"[*`\"']?\s*$"
    )
    table_re = re.compile(
        r"^\s*\|?\s*(.+?)\s*\|\s*(.+?)\s*\|?\s*$"
    )
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("```"):
            continue
        upper = raw.upper()
        if upper.startswith("WORD:") or upper.startswith("TRANSLATION:"):
            continue
        # Skip markdown separator rows like |---|---|
        if re.match(r"^\|?\s*:?-{3,}", raw):
            continue
        # Skip header-ish rows
        if re.match(r"^\|?\s*(word|term|italian|vocabulary)\s*\|", raw, re.I):
            continue

        m = line_re.match(raw)
        if m:
            pair = _clean_vocab_pair(m.group(1), m.group(2))
            if pair and pair["word"].upper() not in ("WORD", "TERM"):
                entries.append(pair)
                continue

        if "|" in raw:
            tm = table_re.match(raw)
            if tm:
                left, right = tm.group(1).strip(), tm.group(2).strip()
                if left and right and not re.match(r"^-+$", left):
                    pair = _clean_vocab_pair(left, right)
                    if pair and pair["word"].upper() not in ("WORD", "TERM"):
                        entries.append(pair)
    return entries[:5]


def parse_vocab_response(text: str) -> list:
    """Parse model vocab output into up to 5 {word, translation} dicts.

    Accepts JSON arrays, WORD:/TRANSLATION: blocks, inline 'word — translation'
    / 'word: translation', bullets/numbers, and simple markdown table rows.
    Thinking leftovers are stripped first. Returns [] if nothing usable found.
    """
    if not text or text.startswith("ERROR:"):
        return []
    cleaned = _strip_thinking(text)
    # Drop obvious preambles before structured content
    cleaned = re.sub(r"^(?:here(?:'s| is)|sure[,!]?)[^\n]*\n+", "", cleaned, flags=re.I).strip()

    for parser in (_parse_vocab_json, _parse_vocab_word_blocks, _parse_vocab_inline_lines):
        entries = parser(cleaned)
        if entries:
            return entries[:5]
    return []
