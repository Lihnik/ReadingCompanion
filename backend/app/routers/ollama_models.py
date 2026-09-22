"""Ollama model listing."""

from __future__ import annotations

from fastapi import APIRouter

from companion.constants import PREFERRED_OLLAMA_MODELS
from companion.ollama import fetch_ollama_models, order_ollama_models

router = APIRouter()


@router.get("/models")
def list_models():
    names, err = fetch_ollama_models(timeout=3.0)
    if names is None:
        return {
            "models": list(PREFERRED_OLLAMA_MODELS),
            "live": False,
            "error": err,
        }
    return {
        "models": order_ollama_models(names),
        "live": True,
        "error": None,
    }
