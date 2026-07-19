"""조문 -> 임베딩 텍스트 / Qdrant payload·point id."""

import uuid

from retriever.payload import build_payload, point_id
from retriever.text import build_embedding_text

CLAUSE = {
    "id": "1872-60",
    "law_name": "근로기준법",
    "law_type": "법률",
    "clause_no": "60",
    "clause_title": "연차 유급휴가",
    "clause_content": "제60조(연차 유급휴가)\n① 사용자는 1년간 80퍼센트 이상 출근한 근로자에게 15일의 유급휴가를 주어야 한다.",
    "effective_date": "2025-02-23",
}


def test_embedding_text_has_law_article_header_and_content():
    text = build_embedding_text(CLAUSE)
    assert text.startswith("근로기준법 제60조(연차 유급휴가)")
    assert "80퍼센트" in text


def test_embedding_text_appends_doc2query_questions():
    text = build_embedding_text(CLAUSE, questions=["연차 며칠 받나요?", "1년 일하면 휴가 생기나요?"])
    assert text.startswith("근로기준법 제60조(연차 유급휴가)")
    assert "80퍼센트" in text
    assert "예상 질문:" in text and "연차 며칠 받나요?" in text
    assert build_embedding_text(CLAUSE, questions=[]) == build_embedding_text(CLAUSE)


def test_payload_text_stays_clean_of_questions():
    # doc2query는 인덱스 텍스트 전용 — payload text(리랭커/에이전트 입력)는 원문 그대로
    assert "예상 질문" not in build_payload(CLAUSE)["text"]


def test_embedding_text_branch_article_formats_uijo():
    c = dict(CLAUSE, clause_no="74-2", clause_title="태아검진 시간의 허용 등")
    assert build_embedding_text(c).startswith("근로기준법 제74조의2(태아검진 시간의 허용 등)")


def test_payload_carries_metadata_and_cid():
    p = build_payload(CLAUSE)
    assert p["cid"] == "근로기준법|60"  # qrels 라벨과 같은 표기
    assert p["law_name"] == "근로기준법" and p["law_type"] == "법률"
    assert p["clause_no"] == "60" and p["clause_title"] == "연차 유급휴가"
    assert "80퍼센트" in p["text"]


def test_point_id_is_stable_uuid():
    a, b = point_id("근로기준법|60"), point_id("근로기준법|60")
    assert a == b and uuid.UUID(a)
    assert point_id("근로기준법|61") != a
