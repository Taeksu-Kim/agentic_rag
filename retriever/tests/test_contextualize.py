"""멀티턴 질문 응축: 대화 맥락 + 후속 질문 -> 독립형 질문 (fake LLM)."""

from agent.core.llm import FakeLLM
from retriever.contextualize import condense_question

HISTORY = [
    ("육아휴직 중에도 연차가 발생하나요?", "네, 출근한 것으로 보아 발생합니다(근기법 60조)."),
]


def test_no_history_returns_question_unchanged():
    # 첫 턴은 응축 불필요 — LLM 호출 없이 원문 그대로
    llm = FakeLLM([])  # 응답 큐가 비어도 호출되지 않아야 통과
    assert condense_question(llm, [], "육아휴직 중 연차가 발생하나요?") == "육아휴직 중 연차가 발생하나요?"
    assert llm.calls == []


def test_followup_condensed_into_standalone():
    llm = FakeLLM.json({"question": "육아휴직 기간이 퇴직금 산정에 포함되나요?"})
    out = condense_question(llm, HISTORY, "그럼 그 기간은 퇴직금 계산에 들어가나요?")
    assert out == "육아휴직 기간이 퇴직금 산정에 포함되나요?"
    _sys, user, _schema = llm.calls[0]
    assert "육아휴직 중에도 연차" in user  # 이전 턴이 프롬프트에 포함
    assert "그 기간은 퇴직금" in user       # 후속 질문도 포함


def test_condense_failure_falls_back_to_raw_question():
    assert condense_question(FakeLLM(["not json"]), HISTORY, "그 기간은요?") == "그 기간은요?"


def test_empty_condensed_falls_back_to_raw_question():
    llm = FakeLLM.json({"question": "   "})
    assert condense_question(llm, HISTORY, "그 기간은요?") == "그 기간은요?"
