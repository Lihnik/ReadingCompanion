"""FastAPI entrypoint for Reading Companion React rewrite."""

from __future__ import annotations

import sys
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Repo root on PYTHONPATH so `companion` package imports work.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import db  # noqa: E402
from backend.app.routers import ai, books, ollama_models, tts  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Reading Companion API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(books.router, prefix="/api/books", tags=["books"])
app.include_router(ai.router, prefix="/api/ai", tags=["ai"])
app.include_router(tts.router, prefix="/api/tts", tags=["tts"])
app.include_router(ollama_models.router, prefix="/api/ollama", tags=["ollama"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
