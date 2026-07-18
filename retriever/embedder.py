"""Embedders behind small interfaces (tests use fakes -- no server/downloads).

* Dense: vLLM Qwen3-Embedding server (OpenAI ``/v1/embeddings``).
* Sparse: BM25 via fastembed.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from qdrant_client import models

from retriever import config


@runtime_checkable
class DenseEmbedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[models.SparseVector]: ...

    def encode_query(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Query-side sparse (BM25 uses IDF weighting for queries)."""
        ...


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


class FakeDenseEmbedder:
    """Deterministic unit-norm vectors from the text hash (tests only)."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.random.default_rng(_seed(t)).standard_normal(self.dim)
            out.append((v / (np.linalg.norm(v) or 1.0)).tolist())
        return out


class FakeSparseEmbedder:
    """Deterministic sparse vectors from token hashes (tests only)."""

    def encode(self, texts: Sequence[str]) -> list[models.SparseVector]:
        out = []
        for t in texts:
            toks = sorted({w for w in t.split() if w})
            idx = sorted(int(hashlib.sha256(w.encode()).hexdigest()[:6], 16) % 100000 for w in toks)
            out.append(models.SparseVector(indices=idx or [0], values=[1.0] * max(len(idx), 1)))
        return out

    def encode_query(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return self.encode(texts)


class VLLMDenseEmbedder:
    """Dense embeddings from the vLLM embedding server (OpenAI-compatible)."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 dim: int = 1024, timeout: float = 120.0, batch: int = 64,
                 max_chars: int = 3500) -> None:
        self.base_url = (base_url or config.EMBEDDER_URL).rstrip("/")
        self.model = model or config.EMBEDDER_MODEL
        self.dim = dim
        self.timeout = timeout
        self.batch = batch
        # 서빙 max-model-len(4096토큰) 초과 시 400. 서버측 truncate_prompt_tokens는
        # pooling 엔드포인트에서 무한 대기(/score와 동일 실패 모드, 실측) —
        # 클라이언트 절단만 신뢰한다. 3,500자 초과 조문은 코퍼스에 극소수.
        self.max_chars = max_chars

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        import requests  # lazy

        texts = [t[:self.max_chars] for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            r = requests.post(f"{self.base_url}/embeddings",
                              json={"model": self.model, "input": texts[i:i + self.batch]},
                              timeout=self.timeout)
            r.raise_for_status()
            out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        return out


class BM25SparseEmbedder:
    """BM25 sparse vectors via fastembed (lazy import so tests stay offline)."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name)

    def encode(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return self._to_sparse(self._model.embed(list(texts)))

    def encode_query(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return self._to_sparse(self._model.query_embed(list(texts)))

    @staticmethod
    def _to_sparse(embs) -> list[models.SparseVector]:
        return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embs]
