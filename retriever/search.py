"""조문 검색: 하이브리드(dense+BM25, RRF) 1단계 + 선택적 크로스인코더 2단계.

교훈 승계(금융 프로젝트에서 검증): fusion 쿼리에서 필터는 **각 Prefetch 안**에
넣어야 한다 — top-level ``query_filter``는 prefetch 하위 쿼리에 적용되지 않는다.

리랭커는 파이프라인 내부 단계다(에이전트가 결정할 일이 아님): 1단계 상위
``prefetch_limit``개를 리랭커 점수로 재정렬해 상위 k만 반환하고, hit.score를
리랭커 점수로 바꿔 단다. ``reranker=None``이면 1단계 RRF 결과 그대로 (ablation).
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from qdrant_client import QdrantClient, models

from retriever.index import DENSE_NAME, SPARSE_NAME
from retriever.reranker import Reranker

SearchMode = Literal["hybrid", "dense", "sparse"]


def build_filter(law_names: Optional[Sequence[str]] = None,
                 law_types: Optional[Sequence[str]] = None) -> Optional[models.Filter]:
    must: list[models.FieldCondition] = []
    if law_names:
        must.append(models.FieldCondition(key="law_name", match=models.MatchAny(any=list(law_names))))
    if law_types:
        must.append(models.FieldCondition(key="law_type", match=models.MatchAny(any=list(law_types))))
    return models.Filter(must=must) if must else None


def search_statutes(
    client: QdrantClient,
    collection: str,
    *,
    dense_vec: Sequence[float],
    sparse_vec: models.SparseVector,
    k: int = 8,
    prefetch_limit: int = 30,
    flt: Optional[models.Filter] = None,
    reranker: Optional[Reranker] = None,
    query_text: Optional[str] = None,
    mode: SearchMode = "hybrid",
) -> list[models.ScoredPoint]:
    """1단계(``mode``: hybrid RRF / dense 단독 / sparse 단독) top-``prefetch_limit``
    -> (리랭커 있으면 재정렬) -> top-``k``. 단독 모드는 ablation용."""
    if reranker is not None and not query_text:
        raise ValueError("reranker requires query_text")

    first_k = prefetch_limit if reranker is not None else k
    if mode == "hybrid":
        res = client.query_points(
            collection,
            prefetch=[
                # 필터는 반드시 각 prefetch 안에 (top-level은 prefetch에 미적용).
                models.Prefetch(query=list(dense_vec), using=DENSE_NAME, limit=prefetch_limit, filter=flt),
                models.Prefetch(query=sparse_vec, using=SPARSE_NAME, limit=prefetch_limit, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=first_k,
            with_payload=True,
        )
    else:  # 단독 벡터 쿼리는 top-level query_filter가 정상 적용된다
        query = list(dense_vec) if mode == "dense" else sparse_vec
        res = client.query_points(
            collection,
            query=query,
            using=DENSE_NAME if mode == "dense" else SPARSE_NAME,
            limit=first_k,
            query_filter=flt,
            with_payload=True,
        )
    hits = res.points
    if reranker is None or not hits:
        return hits[:k]

    scores = reranker.rerank(query_text, [h.payload.get("text", "") for h in hits])
    for h, s in zip(hits, scores):
        h.score = float(s)
    return sorted(hits, key=lambda h: h.score, reverse=True)[:k]


def search_statutes_union(
    client: QdrantClient,
    collection: str,
    *,
    queries: Sequence[tuple[Sequence[float], models.SparseVector]],
    rerank_query: str,
    reranker: Reranker,
    k: int = 8,
    prefetch_limit: int = 30,
    flt: Optional[models.Filter] = None,
) -> list[models.ScoredPoint]:
    """합집합 멀티쿼리: 쿼리별 1단계 풀을 합치고(중복 제거) 원 질문으로 리랭크.

    실측 근거(일상어 101쿼리): 원문 단독 R@8 0.543 / 리라이팅 **대체** 0.520 /
    **합집합 0.567** — 리라이팅은 대체가 아니라 풀 확장으로 써야 값을 한다.
    합집합 병합의 점수 기준이 리랭커라서 reranker가 필수다.
    """
    seen: set = set()
    cands: list[models.ScoredPoint] = []
    for dense_vec, sparse_vec in queries:
        hits = search_statutes(
            client, collection, dense_vec=dense_vec, sparse_vec=sparse_vec,
            k=prefetch_limit, prefetch_limit=prefetch_limit, flt=flt,
        )
        for h in hits:
            cid = h.payload.get("cid")
            if cid not in seen:
                seen.add(cid)
                cands.append(h)
    if not cands:
        return []
    scores = reranker.rerank(rerank_query, [h.payload.get("text", "") for h in cands])
    for h, s in zip(cands, scores):
        h.score = float(s)
    return sorted(cands, key=lambda h: h.score, reverse=True)[:k]
