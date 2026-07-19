"""Index the statute corpus into Qdrant (hybrid dense + BM25).

Needs: Qdrant (:6333) + the vLLM embedder (:8001). 1,787 clauses -> a few
minutes. Idempotent: point ids are uuid5(cid), so re-running upserts in place.

    PYTHONPATH=. python scripts/build_index.py [--collection statutes]
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from qdrant_client import QdrantClient

from retriever import config
from retriever.embedder import BM25SparseEmbedder, VLLMDenseEmbedder
from retriever.index import ensure_collection, make_point, upsert_points
from retriever.payload import build_payload, point_id
from retriever.text import build_embedding_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus", "labor_statutes.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="statutes")
    ap.add_argument("--sparse", choices=["fastembed", "kiwi"], default="fastembed",
                    help="kiwi = 형태소 BM25 (통계를 --stats-out에 저장)")
    ap.add_argument("--stats-out", default=os.path.join(ROOT, "data", "corpus", "kiwi_bm25_stats.json"),
                    help="kiwi BM25 통계 저장 경로 — 컬렉션별로 분리할 것 (쿼리 시 같은 통계 필요)")
    ap.add_argument("--doc2query",
                    help="doc2query parquet(cid, questions) — 역질문을 인덱스 텍스트에 부착 (payload는 원문)")
    args = ap.parse_args()

    df = pd.read_parquet(CORPUS)
    clauses = df.to_dict("records")
    d2q: dict[str, list] = {}
    if args.doc2query:
        qdf = pd.read_parquet(args.doc2query)
        d2q = {r.cid: list(r.questions) for r in qdf.itertuples()}
    texts = [build_embedding_text(c, questions=d2q.get(f"{c['law_name']}|{c['clause_no']}"))
             for c in clauses]
    print(f"{len(clauses)} clauses" + (f", doc2query 부착 {sum(bool(v) for v in d2q.values())}" if d2q else ""))

    t0 = time.time()
    dense = VLLMDenseEmbedder()
    if args.sparse == "kiwi":
        from retriever.kiwi_bm25 import KiwiBM25SparseEmbedder
        sparse = KiwiBM25SparseEmbedder().fit(texts)
        sparse.save(args.stats_out)
    else:
        sparse = BM25SparseEmbedder()
    dvs = dense.encode(texts)
    print(f"dense encoded in {time.time()-t0:.1f}s")
    svs = sparse.encode(texts)
    print(f"sparse encoded in {time.time()-t0:.1f}s (cumulative)")

    client = QdrantClient(url=config.QDRANT_URL)
    ensure_collection(client, args.collection, dim=dense.dim)
    pts = [make_point(point_id(f"{c['law_name']}|{c['clause_no']}"), dv, sv, build_payload(c))
           for c, dv, sv in zip(clauses, dvs, svs)]
    n = upsert_points(client, args.collection, pts)
    print(f"upserted {n} points -> '{args.collection}' "
          f"(total {client.count(args.collection).count}) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
