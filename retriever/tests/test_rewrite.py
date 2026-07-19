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


def test_rewrite_prompt_carries_fewshot_examples():
    # 프롬프트에 변환 시범(퓨샷)이 실려 있어야 한다 — 형식 학습용, 정답 누수 아님
    llm = FakeLLM.json({"query": "임금 체불 지연이자"})
    rewrite_query(llm, "월급이 밀렸는데")
    _sys, user, _schema = llm.calls[0]
    assert user.count("질문:") >= 3  # 퓨샷 예시 여러 개 + 실제 질문
    assert user.count("쿼리:") >= 2  # 예시마다 변환 결과 시범
