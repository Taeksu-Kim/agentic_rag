"""Endpoints from .env (gitignored; localhost defaults).

WSL note: loopback can be broken -- put the WSL IP (e.g. ``10.5.0.2``) in .env.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

EMBEDDER_URL = os.environ.get("EMBEDDER_URL", "http://localhost:8001/v1")
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "qwen3-emb")
RERANKER_URL = os.environ.get("RERANKER_URL", "http://localhost:8002")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "qwen3-reranker")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen35-9b")
