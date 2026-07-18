"""에이전트 툴: statute_search (세션 누적 + cid 해소), web_search (백엔드 교체형)."""

import pytest
from qdrant_client import QdrantClient

from retriever.embedder import FakeDenseEmbedder, FakeSparseEmbedder
from retriever.index import ensure_collection, make_point, upsert_points
from retriever.payload import build_payload, point_id
from retriever.text import build_embedding_text
from retriever.tool import StatuteSearchTool
from retriever.web import FakeSearchBackend, WebSearchTool

CLAUSES = [
    {"id": "1-60", "law_name": "근로기준법", "law_type": "법률", "clause_no": "60",
     "clause_title": "연차 유급휴가", "clause_content": "연차 유급휴가 출근율 산정 시 육아휴직 기간은 출근한 것으로 본다.", "effective_date": "2025-01-01"},
    {"id": "3-19", "law_name": "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률", "law_type": "법률", "clause_no": "19",
     "clause_title": "육아휴직", "clause_content": "사업주는 근로자가 신청하면 육아휴직을 허용하여야 한다.", "effective_date": "2025-01-01"},
]


@pytest.fixture()
def tool():
    client = QdrantClient(":memory:")
    dense, sparse = FakeDenseEmbedder(dim=32), FakeSparseEmbedder()
    ensure_collection(client, "statutes", dim=32)
    texts = [build_embedding_text(c) for c in CLAUSES]
    pts = [make_point(point_id(f"{c['law_name']}|{c['clause_no']}"), dv, sv, build_payload(c))
           for c, dv, sv in zip(CLAUSES, dense.encode(texts), sparse.encode(texts))]
    upsert_points(client, "statutes", pts)
    return StatuteSearchTool(client=client, collection="statutes", dense=dense, sparse=sparse)


def test_search_returns_compact_results_for_llm(tool):
    out = tool.run(query=build_embedding_text(CLAUSES[0]), k=2)
    assert out[0]["cid"] == "근로기준법|60"
    assert "제60조" in out[0]["ref"] and "연차 유급휴가" in out[0]["ref"]
    assert "snippet" in out[0] and "score" in out[0]
    assert "text" not in out[0]  # LLM에는 전문 대신 snippet만


def test_session_accumulates_across_calls_and_resolves_cids(tool):
    tool.run(query=build_embedding_text(CLAUSES[0]), k=1)
    tool.run(query=build_embedding_text(CLAUSES[1]), k=1)
    ev = tool.resolve(["근로기준법|60", "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률|19"])
    assert [e["cid"] for e in ev] == ["근로기준법|60", "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률|19"]
    assert all("text" in e and e["text"] for e in ev)  # 해소본에는 전문 포함


def test_resolve_drops_unknown_cids(tool):
    tool.run(query=build_embedding_text(CLAUSES[0]), k=1)
    ev = tool.resolve(["근로기준법|60", "존재하지않는법|1"])  # 환각 cid는 무시
    assert [e["cid"] for e in ev] == ["근로기준법|60"]


def test_top_session_fallback_and_reset(tool):
    tool.run(query=build_embedding_text(CLAUSES[0]), k=2)
    top = tool.top_session(k=1)
    assert top and top[0]["cid"]
    tool.reset()
    assert tool.top_session(k=5) == [] and tool.resolve(["근로기준법|60"]) == []


def test_filter_zero_hits_retries_without_filter(tool):
    out = tool.run(query="육아휴직 허용", law_names=["존재하지 않는 법"])
    assert "note" in out[0] and "필터" in out[0]["note"]   # LLM에게 필터 실패 고지
    results = [o for o in out if "cid" in o]
    assert results                                          # 필터 해제 결과는 돌려준다
    assert tool.top_session(k=1)                            # 세션에도 누적


def test_valid_filter_not_dropped(tool):
    out = tool.run(query="휴가", law_names=["근로기준법"])
    assert all("note" not in o for o in out)
    assert all(o["cid"].startswith("근로기준법") for o in out)


LAWS = ["근로기준법", "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률"]


@pytest.fixture()
def tool_with_laws(tool):
    tool._valid_laws = LAWS
    tool._law_norm = {n.replace(" ", ""): n for n in LAWS}
    return tool


def test_valid_laws_listed_in_description():
    t = StatuteSearchTool(client=QdrantClient(":memory:"), collection="s",
                          dense=FakeDenseEmbedder(dim=8), sparse=FakeSparseEmbedder(),
                          valid_laws=LAWS)
    assert all(l in t.description for l in LAWS)
    assert "여러 법령을 한 번에" in t.description


def test_law_name_space_variants_normalized(tool_with_laws):
    # "남녀고용평등과일ㆍ가정양립..." (공백 누락)도 정확 표기로 교정되어 필터 적중
    out = tool_with_laws.run(query="육아휴직",
                             law_names=["남녀고용평등과일ㆍ가정양립지원에관한법률"])
    hits = [o for o in out if "cid" in o]
    assert hits and all(o["cid"].startswith("남녀고용평등") for o in hits)
    assert all("note" not in o for o in out)  # 폴백이 아니라 정규화로 해결


def test_k_clamped(tool):
    out = tool.run(query="휴가", k=999)  # 상한 초과 요청도 에러 없이 동작
    assert len([o for o in out if "cid" in o]) <= 10


def test_web_search_normalizes_backend_results():
    backend = FakeSearchBackend(results=[{"title": "연차휴가 안내", "url": "http://x", "snippet": "설명"}])
    tool = WebSearchTool(backend=backend)
    out = tool.run(query="연차")
    assert out == [{"title": "연차휴가 안내", "url": "http://x", "snippet": "설명"}]
    assert backend.queries == ["연차"]
    assert tool.name == "web_search" and tool.description
