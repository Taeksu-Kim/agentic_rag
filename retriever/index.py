"""Qdrant hybrid collection (named dense + BM25 sparse), one point per 조문."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient, models

DENSE_DIM = 1024
DENSE_NAME = "dense"
SPARSE_NAME = "bm25"


def ensure_collection(client: QdrantClient, name: str, *, dim: int = DENSE_DIM) -> None:
    """Create the hybrid collection if it does not already exist."""
    if client.collection_exists(name):
        return
    client.create_collection(
        name,
        vectors_config={DENSE_NAME: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_NAME: models.SparseVectorParams()},
    )


def make_point(pid: str, dense_vec: Sequence[float], sparse_vec: models.SparseVector,
               payload: dict[str, Any]) -> models.PointStruct:
    return models.PointStruct(
        id=pid,
        vector={DENSE_NAME: list(dense_vec), SPARSE_NAME: sparse_vec},
        payload=payload,
    )


def upsert_points(client: QdrantClient, name: str,
                  points: Iterable[models.PointStruct], *, batch: int = 256) -> int:
    points = list(points)
    for i in range(0, len(points), batch):
        client.upsert(name, points[i:i + batch])
    return len(points)
