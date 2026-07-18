"""구조 확장 파이프라인: 동반 조문 부착 + CE 경쟁 (오프라인 fakes)."""

from qdrant_client import QdrantClient

from agent.core.llm import FakeLLM
from retriever.edges import EdgeIndex
from retriever.embedder import FakeDenseEmbedder, FakeSparseEmbedder
from retriever.index import ensure_collection, make_point, upsert_points
from retriever.payload import build_payload, point_id
from retriever.pipeline import retrieve_structured
from retriever.reranker import FakeReranker
from retriever.text import build_embedding_text

CLAUSES = [
    {"id": "1", "law_name": "산업안전보건법", "law_type": "법률", "clause_no": "17",
     "clause_title": "안전관리자", "clause_content": "안전관리자의 수는 대통령령으로 정한다.", "effective_date": ""},
    {"id": "2", "law_name": "산업안전보건법 시행령", "law_type": "대통령령", "clause_no": "16",
     "clause_title": "안전관리자의 선임", "clause_content": "공동 안전관리자 합산 근로자수 300명 기준.", "effective_date": ""},
    {"id": "3", "law_name": "근로기준법", "law_type": "법률", "clause_no": "60",
     "clause_title": "연차 유급휴가", "clause_content": "연차 유급휴가 조문.", "effective_date": ""},
]

EDGES = [("산업안전보건법 시행령|16", "산업안전보건법|17", "위임")]


def _setup():
    client = QdrantClient(":memory:")
    dense, sparse = FakeDenseEmbedder(dim=32), FakeSparseEmbedder()
    ensure_collection(client, "statutes", dim=32)
    texts = [build_embedding_text(c) for c in CLAUSES]
    pts = [make_point(point_id(f"{c['law_name']}|{c['clause_no']}"), dv, sv, build_payload(c))
           for c, dv, sv in zip(CLAUSES, dense.encode(texts), sparse.encode(texts))]
    upsert_points(client, "statutes", pts)
    return client, dense, sparse


def test_companion_attached_and_competes_by_score():
    client, dense, sparse = _setup()
    q = build_embedding_text(CLAUSES[0]) + " 합산 근로자수"  # 17조가 1위가 되는 쿼리 + 동반이 이길 토큰
    out = retrieve_structured(
        q, client=client, dense=dense, sparse=sparse,
        reranker=FakeReranker(), edge_index=EdgeIndex(EDGES), k=3,
    )
    cids = [r["cid"] for r in out]
    assert "산업안전보건법 시행령|16" in cids  # 쿼리 임베딩과 무관하게 엣지로 편입
    assert all("score" in r and "text" in r for r in out)


def test_no_edges_equals_union_only():
    client, dense, sparse = _setup()
    out = retrieve_structured(
        build_embedding_text(CLAUSES[2]), client=client, dense=dense, sparse=sparse,
        reranker=FakeReranker(), edge_index=EdgeIndex([]), k=2,
    )
    assert out and out[0]["cid"] == "근로기준법|60"


def test_rewrite_used_when_llm_given():
    client, dense, sparse = _setup()
    rew = build_embedding_text(CLAUSES[2])  # 리라이팅이 정확히 60조 텍스트를 만들면
    llm = FakeLLM.json({"query": rew})
    out = retrieve_structured(
        "연차 며칠?", client=client, dense=dense, sparse=sparse, llm=llm,
        reranker=FakeReranker(), edge_index=EdgeIndex([]), k=3,
    )
    assert "근로기준법|60" in [r["cid"] for r in out]  # 리라이팅 쿼리 풀이 합집합에 포함
    assert llm.calls  # 리라이팅 호출 발생
