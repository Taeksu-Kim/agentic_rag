"""하이브리드 검색 + 메타데이터 필터 + 2단계 리랭킹 (전부 오프라인: :memory: Qdrant, fakes)."""

import pytest
from qdrant_client import QdrantClient

from retriever.embedder import FakeDenseEmbedder, FakeSparseEmbedder
from retriever.index import DENSE_DIM, ensure_collection, make_point, upsert_points
from retriever.payload import build_payload, point_id
from retriever.reranker import FakeReranker
from retriever.search import build_filter, search_statutes
from retriever.text import build_embedding_text

CLAUSES = [
    {"id": "1-60", "law_name": "근로기준법", "law_type": "법률", "clause_no": "60",
     "clause_title": "연차 유급휴가", "clause_content": "연차 유급휴가는 1년간 80퍼센트 이상 출근한 근로자에게 준다.", "effective_date": "2025-01-01"},
    {"id": "1-61", "law_name": "근로기준법", "law_type": "법률", "clause_no": "61",
     "clause_title": "연차 유급휴가의 사용 촉진", "clause_content": "연차 유급휴가의 사용을 촉진하기 위한 조치.", "effective_date": "2025-01-01"},
    {"id": "2-30", "law_name": "근로기준법 시행령", "law_type": "대통령령", "clause_no": "30",
     "clause_title": "휴일", "clause_content": "주휴일은 1주 동안의 소정근로일을 개근한 자에게 준다.", "effective_date": "2025-01-01"},
]


@pytest.fixture()
def indexed():
    client = QdrantClient(":memory:")
    dense, sparse = FakeDenseEmbedder(dim=32), FakeSparseEmbedder()
    ensure_collection(client, "statutes", dim=32)
    texts = [build_embedding_text(c) for c in CLAUSES]
    dvs, svs = dense.encode(texts), sparse.encode(texts)
    pts = [make_point(point_id(f"{c['law_name']}|{c['clause_no']}"), dv, sv, build_payload(c))
           for c, dv, sv in zip(CLAUSES, dvs, svs)]
    upsert_points(client, "statutes", pts)
    return client, dense, sparse


def test_ensure_collection_idempotent(indexed):
    client, *_ = indexed
    ensure_collection(client, "statutes", dim=32)  # second call must not raise
    assert client.count("statutes").count == 3


def test_hybrid_search_returns_relevant_clause(indexed):
    client, dense, sparse = indexed
    # fake dense는 해시 기반이라 인덱스 텍스트와 동일한 쿼리만 dense 일치가 생긴다
    q = build_embedding_text(CLAUSES[0])  # 60조 전체 텍스트
    hits = search_statutes(
        client, "statutes",
        dense_vec=dense.encode([q])[0], sparse_vec=sparse.encode_query([q])[0], k=2,
    )
    assert hits and hits[0].payload["cid"] == "근로기준법|60"


def test_metadata_filter_restricts_law(indexed):
    client, dense, sparse = indexed
    q = "휴가"
    hits = search_statutes(
        client, "statutes",
        dense_vec=dense.encode([q])[0], sparse_vec=sparse.encode_query([q])[0],
        k=5, flt=build_filter(law_names=["근로기준법 시행령"]),
    )
    assert hits and all(h.payload["law_name"] == "근로기준법 시행령" for h in hits)


def test_build_filter_none_when_no_criteria():
    assert build_filter() is None


def test_rerank_stage_reorders_and_truncates(indexed):
    client, dense, sparse = indexed
    q = "사용 촉진 조치"
    hits = search_statutes(
        client, "statutes",
        dense_vec=dense.encode([q])[0], sparse_vec=sparse.encode_query([q])[0],
        k=2, reranker=FakeReranker(), query_text=q, prefetch_limit=10,
    )
    assert len(hits) == 2
    assert hits[0].payload["cid"] == "근로기준법|61"  # '촉진' 토큰 겹침 최다
    assert hits[0].score >= hits[1].score  # 점수가 리랭커 점수로 대체됨


def test_dense_only_mode(indexed):
    client, dense, sparse = indexed
    q = build_embedding_text(CLAUSES[2])  # 시행령 30조 전체 텍스트 = dense 완전 일치
    hits = search_statutes(
        client, "statutes",
        dense_vec=dense.encode([q])[0], sparse_vec=sparse.encode_query([q])[0],
        k=1, mode="dense",
    )
    assert hits and hits[0].payload["cid"] == "근로기준법 시행령|30"


def test_sparse_only_mode_ranks_by_token_overlap(indexed):
    client, dense, sparse = indexed
    q = "연차 유급휴가의 사용을 촉진하기 위한 조치."
    hits = search_statutes(
        client, "statutes",
        dense_vec=dense.encode([q])[0], sparse_vec=sparse.encode_query([q])[0],
        k=1, mode="sparse",
    )
    assert hits and hits[0].payload["cid"] == "근로기준법|61"


def test_single_vector_mode_honours_filter(indexed):
    client, dense, sparse = indexed
    q = "휴가"
    hits = search_statutes(
        client, "statutes",
        dense_vec=dense.encode([q])[0], sparse_vec=sparse.encode_query([q])[0],
        k=5, mode="dense", flt=build_filter(law_names=["근로기준법 시행령"]),
    )
    assert hits and all(h.payload["law_name"] == "근로기준법 시행령" for h in hits)


def test_union_merges_pools_and_reranks(indexed):
    from retriever.search import search_statutes_union
    client, dense, sparse = indexed
    q1, q2 = build_embedding_text(CLAUSES[0]), build_embedding_text(CLAUSES[2])
    hits = search_statutes_union(
        client, "statutes",
        queries=[(dense.encode([q1])[0], sparse.encode_query([q1])[0]),
                 (dense.encode([q2])[0], sparse.encode_query([q2])[0])],
        rerank_query="주휴일 개근", reranker=FakeReranker(),
        k=8, prefetch_limit=1,  # 쿼리당 풀 1개 -> 합집합이 두 쿼리 결과를 다 담아야 함
    )
    cids = [h.payload["cid"] for h in hits]
    assert set(cids) == {"근로기준법|60", "근로기준법 시행령|30"}
    assert cids[0] == "근로기준법 시행령|30"  # rerank_query('주휴일') 기준 재정렬


def test_union_empty_queries_returns_empty(indexed):
    from retriever.search import search_statutes_union
    client, *_ = indexed
    assert search_statutes_union(client, "statutes", queries=[],
                                 rerank_query="q", reranker=FakeReranker()) == []


def test_rerank_requires_query_text(indexed):
    client, dense, sparse = indexed
    with pytest.raises(ValueError):
        search_statutes(
            client, "statutes",
            dense_vec=dense.encode(["q"])[0], sparse_vec=sparse.encode_query(["q"])[0],
            reranker=FakeReranker(),
        )
