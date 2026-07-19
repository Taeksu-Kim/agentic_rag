"""일상어 질문 -> 법률 용어 검색 쿼리 리라이팅 (합집합 멀티쿼리의 확장 쿼리용).

리라이팅은 원문 검색을 **대체하지 않는다** — search_statutes_union이 원문 풀과
합쳐 쓴다 (대체형은 실측 손해, 합집합형만 이득: docs/ablation_results.md 부록).
"""

from __future__ import annotations

import json

from agent.core.llm import LLMClient

REWRITE_SYSTEM = "너는 한국 노동법 조문 검색 쿼리를 만드는 도우미다."

# 퓨샷: 일상어 -> 조문이 쓸 법률 용어 명사구 변환 시범.
# 평가셋 표현을 베끼지 않은 창작 예시 — 형식(구어 상황어 -> 규범어 명사구)만
# 학습시켜 테스트셋 과적합을 피한다. 실제 데이터가 쌓이면 이 예시를
# 도메인 로그에서 마이닝한 쌍으로 교체하는 것이 다음 고도화 단계.
_FEWSHOT = [
    ("회사가 갑자기 문 닫는대요 밀린 월급은 어떻게 받나요", "도산 사업장 임금채권 우선변제"),
    ("아파서 며칠 못 나갔는데 이걸로 잘릴 수도 있나요", "업무외 부상 요양 기간 해고 제한"),
    ("주말에 일 시키면서 수당을 안 주던데요", "휴일근로 가산임금 지급 의무"),
]
REWRITE_PROMPT = (
    "일상어 질문을 조문이 쓸 법한 법률 용어의 자연어 명사구 검색 쿼리 1개로 "
    "리라이팅하라. 아래 예시의 변환 방식을 따르되, 예시 내용을 그대로 쓰지 말고 "
    "주어진 질문에 맞춰라.\n\n"
    + "\n".join(f"질문: {q}\n쿼리: {r}" for q, r in _FEWSHOT)
    + "\n\n질문: {q}\n쿼리:"
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
