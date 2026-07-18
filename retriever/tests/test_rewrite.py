"""리라이팅 헬퍼: 합집합 멀티쿼리의 확장 쿼리 생성 (fake LLM)."""

from agent.core.llm import FakeLLM
from retriever.rewrite import rewrite_query


def test_rewrite_returns_legal_query():
    llm = FakeLLM.json({"query": "해고 정당한 이유"})
    assert rewrite_query(llm, "잘렸어요 어떡하죠") == "해고 정당한 이유"
    assert "잘렸어요" in llm.calls[0][1]


def test_rewrite_identical_to_question_dropped():
    llm = FakeLLM.json({"query": "연차 유급휴가"})
    assert rewrite_query(llm, "연차 유급휴가") == ""  # 원문과 같으면 확장 가치 없음


def test_rewrite_failure_returns_empty():
    assert rewrite_query(FakeLLM(["not json"]), "질문") == ""
