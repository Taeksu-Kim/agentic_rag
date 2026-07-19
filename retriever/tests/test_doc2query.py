"""doc2query 역질문 생성: 조문 -> 일상어 예상 질문 (fake LLM)."""

from agent.core.llm import FakeLLM
from retriever.doc2query import generate_questions

CLAUSE = {
    "law_name": "근로기준법",
    "clause_no": "60",
    "clause_title": "연차 유급휴가",
    "clause_content": "① 사용자는 1년간 80퍼센트 이상 출근한 근로자에게 15일의 유급휴가를 주어야 한다.",
}


def test_generate_returns_questions_and_prompts_with_clause():
    llm = FakeLLM.json({"questions": ["연차는 며칠 받을 수 있나요?", "1년 일하면 휴가가 생기나요?"]})
    qs = generate_questions(llm, CLAUSE)
    assert qs == ["연차는 며칠 받을 수 있나요?", "1년 일하면 휴가가 생기나요?"]
    system, user, schema = llm.calls[0]
    assert "근로기준법 제60조(연차 유급휴가)" in user  # 헤더 포함 원문 제시
    assert "80퍼센트" in user
    assert schema is not None  # 구조화 출력 강제


def test_generate_strips_blanks_and_duplicates():
    llm = FakeLLM.json({"questions": ["  연차 며칠?  ", "", "연차 며칠?", "휴가 언제?"]})
    assert generate_questions(llm, CLAUSE) == ["연차 며칠?", "휴가 언제?"]


def test_generate_failure_returns_empty():
    assert generate_questions(FakeLLM(["not json"]), CLAUSE) == []


def test_generate_truncates_long_content():
    llm = FakeLLM.json({"questions": ["질문?", "질문2?"]})
    generate_questions(llm, dict(CLAUSE, clause_content="가" * 5000), max_chars=100)
    assert len(llm.calls[0][1]) < 400  # 본문이 max_chars로 잘려 프롬프트가 폭발하지 않음
