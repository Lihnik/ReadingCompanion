import json
import re

import requests

from .constants import (
    MAX_CHUNK_CHARS,
    OLLAMA_URL,
    OLLAMA_TAGS_URL,
    PREFERRED_OLLAMA_MODELS,
    SYSTEM_PROMPT_COMMENTARY,
    SYSTEM_PROMPT_QUESTION,
    SYSTEM_PROMPT_CHAT,
)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models (qwen3, deepseek-r1, etc.)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_ollama(prompt: str, model: str, system_prompt: str = "", num_predict: int = 4000) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
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


def stream_ollama(prompt: str, model: str, system_prompt: str = "", num_predict: int = 2048):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
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
