"""조문 -> 일상어 예상 질문 생성 (doc2query) — 인덱스 텍스트 부착용.

STATUS(2026-07-19): 실장·TDD 완료했으나 **실행 보류(셸빙)**. 인덱싱 비용이 LLM
추론에 비례해 정적 코퍼스에선 값이 안 맞는다는 판단 — 실서비스 고도화(운영
로그 기반 표적 생성) 시점의 카드로 남긴다. 근거: docs/design_and_plan.md §10.


동기(실측, docs/postmortem.md): 남은 최대 진성 실패 = 추론형 어휘 갭 — 질문은
일상어·상황어("KCs 표시된 제품만 사용 가능?"), 조문은 규범어("제조·수입·양도
금지")라 임베딩/BM25 어느 축도 못 잇는다. 조문마다 예상 질문 2~3개를 미리
생성해 **문서 쪽 어휘를 질문 쪽으로** 확장한다(docT5query 방식). 질문은
인덱스 텍스트에만 부착하고 payload ``text``(리랭커·에이전트 입력)는 원문
그대로 둔다 — A/B가 검색 1단계 효과만 재도록.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from agent.core.llm import LLMClient
from retriever.text import build_embedding_text

DOC2QUERY_SYSTEM = "너는 한국 노동법 조문을 읽고 검색 색인용 예상 질문을 만드는 도우미다."
DOC2QUERY_PROMPT = (
    "다음 조문에 대한 답이 되는 **예상 질문 2~3개**를 만들어라.\n"
    "- 법률 용어가 아니라 근로자·인사담당자가 실제로 물을 법한 일상어로 쓴다.\n"
    "- 조문 문구를 되풀이하지 말고, 이 조문이 답이 되는 구체적 상황을 물어라.\n"
    "- 각 질문은 한 문장.\n\n조문:\n{clause}"
)
_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"},
                                 "minItems": 2, "maxItems": 3}},
    "required": ["questions"],
}


def generate_questions(llm: LLMClient, clause: Mapping[str, Any], *,
                       max_chars: int = 2000) -> list[str]:
    """일상어 예상 질문 목록. 실패하면 빈 목록 (호출측이 부착 없이 진행)."""
    text = build_embedding_text(clause)[:max_chars]
    try:
        out = json.loads(llm.complete(
            DOC2QUERY_SYSTEM, DOC2QUERY_PROMPT.format(clause=text), schema=_SCHEMA))
        seen: list[str] = []
        for q in out.get("questions", []):
            q = str(q).strip()
            if q and q not in seen:
                seen.append(q)
        return seen
    except Exception:
        return []
