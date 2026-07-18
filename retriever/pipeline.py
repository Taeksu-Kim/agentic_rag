"""구조 확장 검색 파이프라인: union 검색 + 조문 그래프 동반 확장.

    union(원문+리라이팅) -> CE 리랭크 -> 상위 expand_top건의 동반 조문(위임·벌칙·
    교차참조)을 엣지 인덱스로 부착 -> 동반 조문도 CE로 채점 -> 병합 top-k

동기(실측): 1단계 풀 미스의 34%가 "적중 조문의 같은-법-계열 동반 조문" — 질문
텍스트에 단서가 없어 임베딩으로는 못 찾지만, 그래프로는 공짜로 따라갈 수 있다.
동반 조문도 CE 점수로 경쟁시키므로(자동 편입 아님) 무관한 동반은 잘려 나간다.
"""

from __future__ import annotations

from typing import Any, Optional

from retriever.edges import EdgeIndex
from retriever.payload import point_id
from retriever.rewrite import rewrite_query
from retriever.search import search_statutes_union


def retrieve_structured(
    question: str,
    *,
    client: Any,
    dense: Any,
    sparse: Any,
    reranker: Any,
    edge_index: EdgeIndex,
    llm: Any = None,
    collection: str = "statutes",
    k: int = 8,
    prefetch_limit: int = 30,
    expand_top: int = 4,
) -> list[dict[str, Any]]:
    """[{...payload, score}] 상위 k — llm이 있으면 union 리라이팅 포함."""
    def _vec(q: str):
        return dense.encode([q])[0], sparse.encode_query([q])[0]

    queries = [_vec(question)]
    if llm is not None:
        rew = rewrite_query(llm, question)
        if rew:
            queries.append(_vec(rew))

    hits = search_statutes_union(
        client, collection, queries=queries, rerank_query=question,
        reranker=reranker, k=k, prefetch_limit=prefetch_limit,
    )
    ranked = [{**h.payload, "score": float(h.score)} for h in hits]
    seen = {r["cid"] for r in ranked}

    comp_cids: list[str] = []
    for r in ranked[:expand_top]:
        for cid, _kind in edge_index.companions(r["cid"]):
            if cid not in seen:
                seen.add(cid)
                comp_cids.append(cid)

    if comp_cids:
        recs = client.retrieve(collection, ids=[point_id(c) for c in comp_cids],
                               with_payload=True)
        payloads = [rec.payload for rec in recs if rec.payload]
        if payloads:
            scores = reranker.rerank(question, [p.get("text", "") for p in payloads])
            ranked += [{**p, "score": float(s)} for p, s in zip(payloads, scores)]

    return sorted(ranked, key=lambda r: r["score"], reverse=True)[:k]
