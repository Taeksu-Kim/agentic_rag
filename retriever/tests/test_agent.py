"""리트리버 에이전트 조립: react + statute_search, finish 계약(cids) 해소. 전부 fake."""

import pytest
from qdrant_client import QdrantClient

from agent.core.llm import FakeLLM
from retriever.agent import build_statute_agent, run_statute_agent
from retriever.embedder import FakeDenseEmbedder, FakeSparseEmbedder
from retriever.index import ensure_collection, make_point, upsert_points
from retriever.payload import build_payload, point_id
from retriever.text import build_embedding_text

CLAUSES = [
    {"id": "1-60", "law_name": "근로기준법", "law_type": "법률", "clause_no": "60",
     "clause_title": "연차 유급휴가", "clause_content": "출근율 산정 시 육아휴직 기간은 출근한 것으로 본다.", "effective_date": "2025-01-01"},
    {"id": "3-19", "law_name": "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률", "law_type": "법률", "clause_no": "19",
     "clause_title": "육아휴직", "clause_content": "사업주는 육아휴직을 허용하여야 한다.", "effective_date": "2025-01-01"},
]


def _index():
    client = QdrantClient(":memory:")
    dense, sparse = FakeDenseEmbedder(dim=32), FakeSparseEmbedder()
    ensure_collection(client, "statutes", dim=32)
    texts = [build_embedding_text(c) for c in CLAUSES]
    pts = [make_point(point_id(f"{c['law_name']}|{c['clause_no']}"), dv, sv, build_payload(c))
           for c, dv, sv in zip(CLAUSES, dense.encode(texts), sparse.encode(texts))]
    upsert_points(client, "statutes", pts)
    return client, dense, sparse


async def test_agent_searches_then_finishes_with_resolved_evidence():
    client, dense, sparse = _index()
    q60 = build_embedding_text(CLAUSES[0])
    llm = FakeLLM.json(
        {"action": "tool", "tool": "statute_search", "args": {"query": q60, "k": 1}},
        {"action": "finish", "final": "육아휴직 기간은 출근으로 간주되어 연차가 발생합니다.",
         "result": {"cids": ["근로기준법|60", "환각된법|99"]}},
    )
    graph, search_tool = build_statute_agent(llm=llm, client=client, dense=dense, sparse=sparse)
    out = await run_statute_agent(graph, search_tool, "육아휴직 연차 발생?", seed_raw_search=False)

    assert "출근" in out["answer"]
    assert [e["cid"] for e in out["evidence"]] == ["근로기준법|60"]  # 환각 cid 제거
    assert out["evidence"][0]["text"]           # 전문은 코드가 세션에서 해소
    assert out["steps"][0]["tool"] == "statute_search"


async def test_agent_without_cids_falls_back_to_top_session():
    client, dense, sparse = _index()
    llm = FakeLLM.json(
        {"action": "tool", "tool": "statute_search",
         "args": {"query": build_embedding_text(CLAUSES[1]), "k": 1}},
        {"action": "finish", "final": "답"},  # result 없음
    )
    graph, search_tool = build_statute_agent(llm=llm, client=client, dense=dense, sparse=sparse)
    out = await run_statute_agent(graph, search_tool, "질문", seed_raw_search=False)
    assert out["evidence"] and out["evidence"][0]["cid"].startswith("남녀고용평등")


async def test_seed_raw_search_guarantees_evidence_floor():
    # LLM이 검색 한 번 없이 finish해도(최악의 루프) 원 질문 시딩이 근거를 보장
    client, dense, sparse = _index()
    llm = FakeLLM.json({"action": "finish", "final": "답", "result": None})
    graph, search_tool = build_statute_agent(llm=llm, client=client, dense=dense, sparse=sparse)
    out = await run_statute_agent(graph, search_tool, build_embedding_text(CLAUSES[0]))
    assert out["evidence"]  # 시딩된 세션에서 채워짐
    assert out["steps"] == []  # 시딩은 LLM 관측(scratchpad)에 안 들어감


async def test_evidence_fills_from_session_beyond_llm_picks():
    # LLM이 한 조문만 골라도(큐레이션 실패) 세션 고점수 조문이 max_evidence까지 채워진다
    client, dense, sparse = _index()
    llm = FakeLLM.json(
        {"action": "tool", "tool": "statute_search",
         "args": {"query": build_embedding_text(CLAUSES[0]), "k": 2}},  # 세션에 2건
        {"action": "finish", "final": "답",
         "result": {"cids": ["남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률|19"]}},
    )
    graph, search_tool = build_statute_agent(llm=llm, client=client, dense=dense, sparse=sparse)
    out = await run_statute_agent(graph, search_tool, "질문", seed_raw_search=False)
    cids = [e["cid"] for e in out["evidence"]]
    assert cids[0] == "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률|19"  # LLM 선택 우선
    assert "근로기준법|60" in cids                                        # 세션 채움


async def test_llm_picks_capped():
    client, dense, sparse = _index()
    llm = FakeLLM.json(
        {"action": "tool", "tool": "statute_search",
         "args": {"query": build_embedding_text(CLAUSES[0]), "k": 2}},
        {"action": "finish", "final": "답",
         "result": {"cids": ["근로기준법|60", "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률|19"]}},
    )
    graph, search_tool = build_statute_agent(llm=llm, client=client, dense=dense, sparse=sparse)
    out = await run_statute_agent(graph, search_tool, "질문", max_llm_picks=1, max_evidence=2, seed_raw_search=False)
    cids = [e["cid"] for e in out["evidence"]]
    assert cids[0] == "근로기준법|60"   # 캡 이후는 LLM 선택이 아니라 세션 점수순
    assert len(cids) == 2


async def test_system_prompt_carries_rewriting_guidance():
    client, dense, sparse = _index()
    llm = FakeLLM.json({"action": "finish", "final": "x"})
    graph, search_tool = build_statute_agent(llm=llm, client=client, dense=dense, sparse=sparse)
    await run_statute_agent(graph, search_tool, "질문", seed_raw_search=False)
    assert "법률 용어" in llm.calls[0][0]  # 리라이팅 지침이 system에 포함


async def test_session_reset_between_runs():
    client, dense, sparse = _index()
    llm = FakeLLM.json(
        {"action": "tool", "tool": "statute_search",
         "args": {"query": build_embedding_text(CLAUSES[0]), "k": 1}},
        {"action": "finish", "final": "a", "result": {"cids": ["근로기준법|60"]}},
        {"action": "finish", "final": "b"},  # 2번째 run: 검색 안 함 -> evidence 없어야
    )
    graph, search_tool = build_statute_agent(llm=llm, client=client, dense=dense, sparse=sparse)
    out1 = await run_statute_agent(graph, search_tool, "q1", seed_raw_search=False)
    out2 = await run_statute_agent(graph, search_tool, "q2", seed_raw_search=False)
    assert out1["evidence"] and out2["evidence"] == []  # 이전 세션이 새 run에 새면 안 됨


async def test_answer_synthesized_from_evidence_when_final_empty():
    # 루프가 finish 없이 캡에 걸려도(final 빔) 근거 조문으로 답을 합성한다
    client, dense, sparse = _index()
    llm = FakeLLM.json(
        {"action": "tool", "tool": "statute_search",
         "args": {"query": build_embedding_text(CLAUSES[0]), "k": 1}},
        {"action": "finish", "final": ""},                    # 빈 final
    )
    synth = FakeLLM(["육아휴직 기간은 출근으로 간주되어 연차가 발생합니다."])
    graph, search_tool = build_statute_agent(llm=llm, client=client, dense=dense, sparse=sparse)
    out = await run_statute_agent(graph, search_tool, "육아휴직 연차?", synth_llm=synth, seed_raw_search=False)

    assert "연차가 발생" in out["answer"]          # 합성 답변
    assert "근로기준법" in synth.calls[0][1]        # 조문 전문이 합성 프롬프트에 포함
    assert out["evidence"]


async def test_agent_final_kept_when_present():
    client, dense, sparse = _index()
    llm = FakeLLM.json(
        {"action": "finish", "final": "이미 좋은 답", "result": None},
    )
    synth = FakeLLM(["안 쓰임"])
    graph, search_tool = build_statute_agent(llm=llm, client=client, dense=dense, sparse=sparse)
    out = await run_statute_agent(graph, search_tool, "q", synth_llm=synth)
    assert out["answer"] == "이미 좋은 답"
    assert synth.calls == []  # final 있으면 합성 생략
