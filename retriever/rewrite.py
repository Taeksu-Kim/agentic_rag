"""일상어 질문 -> 법률 용어 검색 쿼리 리라이팅 (합집합 멀티쿼리의 확장 쿼리용).

리라이팅은 원문 검색을 **대체하지 않는다** — search_statutes_union이 원문 풀과
합쳐 쓴다 (대체형은 실측 손해, 합집합형만 이득: docs/ablation_results.md 부록).
"""

from __future__ import annotations

import json

from agent.core.llm import LLMClient

REWRITE_SYSTEM = "너는 한국 노동법 조문 검색 쿼리를 만드는 도우미다."
REWRITE_PROMPT = (
    "다음 질문을 조문이 쓸 법한 법률 용어의 자연어 명사구 검색 쿼리 1개로 "
    "리라이팅하라.\n\n질문: {q}"
)
_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}},
           "required": ["query"]}


def rewrite_query(llm: LLMClient, question: str, *, max_chars: int = 2000) -> str:
    """리라이팅 쿼리 1개. 실패하면 빈 문자열 (호출측이 원문만으로 진행)."""
    try:
        out = json.loads(llm.complete(
            REWRITE_SYSTEM, REWRITE_PROMPT.format(q=question[:max_chars]),
            schema=_SCHEMA))
        rew = str(out.get("query", "")).strip()
        return "" if rew == question.strip() else rew
    except Exception:
        return ""
