"""리랭커: 공식 Qwen3-Reranker 템플릿 포맷 + fake 스코어러 + LLM 리랭커."""

import json

from agent.core.llm import FakeLLM
from retriever.reranker import FakeReranker, LLMReranker, format_score_inputs


def test_fake_reranker_scores_token_overlap():
    scores = FakeReranker().rerank("연차 휴가", ["연차 유급 휴가 조문", "최저임금 조문"])
    assert scores[0] > scores[1]


def test_format_score_inputs_official_template():
    t1, t2 = format_score_inputs("연차휴가?", ["문서A", "문서B"], instruction="inst")
    # 쿼리측: system 프롬프트 + Instruct/Query 마커
    assert t1.startswith("<|im_start|>system\n")
    assert "<Instruct>: inst" in t1 and "<Query>: 연차휴가?" in t1
    # 문서측: Document 마커 + assistant/think suffix (뒤에 와야 함)
    assert len(t2) == 2
    assert t2[0].startswith("<Document>: 문서A")
    assert t2[0].endswith("<think>\n\n</think>\n\n")


def test_llm_reranker_batches_and_collects_scores():
    llm = FakeLLM.json({"scores": [1, 9]}, {"scores": [5]})
    rr = LLMReranker(llm, batch=2)
    assert rr.rerank("q", ["a", "b", "c"]) == [1.0, 9.0, 5.0]
    assert len(llm.calls) == 2  # 3문서, batch=2 -> 2호출
    # 프롬프트에 번호 매긴 문서 목록이 들어간다
    assert "[1] a" in llm.calls[0][1] and "[2] b" in llm.calls[0][1]


def test_llm_reranker_truncates_docs():
    llm = FakeLLM.json({"scores": [1]})
    rr = LLMReranker(llm, doc_chars=3)
    rr.rerank("q", ["가나다라마바사"])
    assert "가나다" in llm.calls[0][1] and "라마바사" not in llm.calls[0][1]


def test_llm_reranker_bad_batch_falls_back_preserving_order():
    # 배치1은 길이 불일치(폴백), 배치2는 정상
    llm = FakeLLM([json.dumps({"scores": [1, 2, 3]}), json.dumps({"scores": [7, 8]})])
    rr = LLMReranker(llm, batch=2)
    scores = rr.rerank("q", ["a", "b", "c", "d"])
    assert scores[2:] == [7.0, 8.0]
    assert scores[0] > scores[1]  # 폴백 배치는 1단계 순서 보존 (미세 감쇠)
    assert all(s < 7 for s in scores[:2])  # 정상 점수보다 아래


def test_llm_reranker_unparseable_falls_back():
    llm = FakeLLM(["not json"])
    scores = LLMReranker(llm, batch=10).rerank("q", ["a", "b"])
    assert len(scores) == 2 and scores[0] > scores[1]
