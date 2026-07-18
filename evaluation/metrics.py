"""검색 평가 지표 (조문 단위): recall@k, MRR.

qrels 라벨과 검색 결과 모두 cid("법령명|조문번호") 문자열 — payload.py가 이 표기를
보장하므로 문자열 동등 비교로 조인된다.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """상위 k 안에 든 정답 비율. 정답이 없으면 0.0 (라벨 없는 쿼리는 상류에서 제외)."""
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


def mrr(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """첫 정답의 역순위 (정답이 리스트에 없으면 0.0)."""
    rel = set(relevant)
    for i, cid in enumerate(ranked, 1):
        if cid in rel:
            return 1.0 / i
    return 0.0
