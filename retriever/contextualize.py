"""멀티턴 질문 응축 (conversational RAG) — 후속 질문을 독립형 질문으로.

동기: react 루프는 ``query`` 하나만 프롬프트에 넣고 대화 history는 무시한다
(agent/react/policy.py::build_user_prompt — 프레임워크 계약). 멀티턴을
프레임워크가 아니라 **리트리버 레벨**에서 푼다: 이전 턴 + 후속 질문을 9B가
읽고 대명사·생략("그 기간은?", "그럼 그건?")을 풀어 자기완결적 질문으로
다시 쓴 뒤, 기존 단발 파이프라인을 그대로 태운다(표준 condense-question 패턴).
history가 없으면 LLM을 호출하지 않는다(첫 턴은 응축 불필요).
"""

from __future__ import annotations

import json
from typing import Sequence

from agent.core.llm import LLMClient

CONDENSE_SYSTEM = "너는 대화 맥락을 반영해 후속 질문을 자기완결적 질문으로 다시 쓰는 도우미다."
CONDENSE_PROMPT = (
    "아래 대화에 이어 사용자가 후속 질문을 했다. 대명사·생략을 이전 맥락으로 풀어 "
    "**그 자체로 이해되는 독립형 질문 1개**로 다시 써라. 맥락이 필요 없으면 그대로 두어라.\n\n"
    "대화:\n{history}\n\n후속 질문: {q}"
)
_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}},
           "required": ["question"]}


def condense_question(llm: LLMClient, history: Sequence[tuple[str, str]],
                      question: str, *, max_turns: int = 4) -> str:
    """대화 history(=[(사용자, 어시스턴트), ...]) + 후속 질문 -> 독립형 질문.

    history가 비면 원문 그대로(호출 없음). 실패/빈 결과면 원문으로 폴백.
    """
    if not history:
        return question
    convo = "\n".join(f"Q: {u}\nA: {a}" for u, a in history[-max_turns:])
    try:
        out = json.loads(llm.complete(
            CONDENSE_SYSTEM, CONDENSE_PROMPT.format(history=convo, q=question),
            schema=_SCHEMA))
        rewritten = str(out.get("question", "")).strip()
        return rewritten or question
    except Exception:
        return question
