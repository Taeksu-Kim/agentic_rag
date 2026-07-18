"""kiwi BM25: fit/encode 로직은 fake 토크나이저로, 실 kiwi는 스모크 1개."""

import pytest

from retriever.kiwi_bm25 import KiwiBM25SparseEmbedder, _tok_index

CORPUS = ["연차 유급 휴가 조문", "해고 예고 조문", "휴가 사용 촉진"]


def fake_tok(text):
    return [w for w in text.split() if w]


@pytest.fixture()
def emb():
    return KiwiBM25SparseEmbedder(tokenizer=fake_tok).fit(CORPUS)


def test_idf_rare_term_weighs_more(emb):
    assert emb.idf["해고"] > emb.idf["조문"]  # 조문은 2/3 문서 등장 -> 낮은 IDF


def test_doc_query_dot_product_matches(emb):
    dv = emb.encode(["연차 유급 휴가 조문"])[0]
    qv = emb.encode_query(["휴가 신청"])[0]  # '신청'은 코퍼스 밖 -> 제외
    assert list(qv.indices) == [_tok_index("휴가")]
    d = dict(zip(dv.indices, dv.values))
    assert d[_tok_index("휴가")] > 0  # 내적 > 0 (매칭 성립)


def test_unknown_query_terms_dropped(emb):
    qv = emb.encode_query(["없는말 뿐"])[0]
    assert list(qv.values) == [0.0]  # 전부 미등장 -> 빈 벡터 폴백


def test_save_load_roundtrip(tmp_path, emb):
    p = tmp_path / "stats.json"
    emb.save(p)
    emb2 = KiwiBM25SparseEmbedder(tokenizer=fake_tok).load(p)
    assert emb2.idf == emb.idf and emb2.avgdl == emb.avgdl


def test_real_kiwi_normalizes_particles():
    emb = KiwiBM25SparseEmbedder().fit(["연차 유급휴가를 주어야 한다"])
    q1 = emb.encode_query(["휴가를"])[0]
    q2 = emb.encode_query(["휴가는"])[0]
    assert list(q1.indices) == list(q2.indices) != [0]  # 조사 무관 동일 토큰
