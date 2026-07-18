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
    args = ap.parse_args()

    df = pd.read_parquet(CORPUS)
    clauses = df.to_dict("records")
    texts = [build_embedding_text(c) for c in clauses]
    print(f"{len(clauses)} clauses")

    t0 = time.time()
    dense = VLLMDenseEmbedder()
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
