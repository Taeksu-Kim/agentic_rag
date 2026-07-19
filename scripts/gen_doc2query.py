"""조문별 doc2query 역질문 오프라인 생성 (9B, 1,787 x 1호출).

Needs: 9B (:8000). 재개 가능 — 진행분은 JSONL append, 완료 시 parquet 스냅샷.

    PYTHONPATH=. python scripts/gen_doc2query.py [--workers 6]
    -> data/corpus/doc2query.parquet (cid, questions)
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "labor_statutes.parquet"
JSONL = ROOT / "data" / "corpus" / "doc2query.jsonl"
OUT = ROOT / "data" / "corpus" / "doc2query.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    from agent.core.llm import OpenAICompatLLM
    from retriever import config
    from retriever.doc2query import generate_questions

    clauses = pd.read_parquet(CORPUS).to_dict("records")
    done: set[str] = set()
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["cid"])
    todo = [c for c in clauses if f"{c['law_name']}|{c['clause_no']}" not in done]
    print(f"{len(clauses)} clauses, {len(done)} done, {len(todo)} todo")

    # temperature 0.7: 질문 다양성용 (결과는 JSONL/parquet에 고정되므로 재현성은 파일이 담보)
    llm = OpenAICompatLLM(base_url=config.LLM_URL, model=config.LLM_MODEL,
                          timeout=300.0,
                          sampling={"temperature": 0.7, "max_tokens": 1024})
    lock = Lock()
    n_done, n_empty, t0 = 0, 0, time.time()

    def gen(clause) -> None:
        nonlocal n_done, n_empty
        cid = f"{clause['law_name']}|{clause['clause_no']}"
        qs = generate_questions(llm, clause)
        with lock:
            with JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"cid": cid, "questions": qs}, ensure_ascii=False) + "\n")
            n_done += 1
            n_empty += not qs
            if n_done % 50 == 0:
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(todo)} ({rate:.1f}/s, empty {n_empty})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(gen, todo))

    rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates("cid", keep="last")
    df.to_parquet(OUT, index=False)
    empty = int((df.questions.str.len() == 0).sum())
    print(f"-> {OUT}: {len(df)} rows, empty {empty}, "
          f"평균 질문 수 {df.questions.str.len().mean():.2f}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
