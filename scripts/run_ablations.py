"""Ablation 실행 (docs/design_and_plan.md §6 매트릭스).

    PYTHONPATH=. python scripts/run_ablations.py retrieval            # 3모드 x {none, ce}
    PYTHONPATH=. python scripts/run_ablations.py retrieval --arms hybrid+llm --limit 200
    PYTHONPATH=. python scripts/run_ablations.py agent --steps 2,4,6 --sample 120
    PYTHONPATH=. python scripts/run_ablations.py report               # 표 -> docs/ablation_results.md

arm 이름 = ``{dense|sparse|hybrid}[+ce|+llm]`` 또는 ``react-s{N}``.
결과는 data/eval/ablation/<arm>.jsonl (재개 가능 — 러너가 기록된 쿼리는 건너뜀).
에이전트 arm은 고정 샘플(seed 42)로 돌고, report가 같은 샘플 부분집합에 대한
검색 arm 지표도 함께 내 공정 비교 표를 만든다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "eval" / "ablation"
QRELS = ROOT / "data" / "eval" / "qrels.parquet"
SAMPLE_FILE = OUT_DIR / "sample_ids.json"
K, PREFETCH = 8, 30

RETRIEVAL_ARMS = ["dense", "sparse", "hybrid",
                  "dense+ce", "sparse+ce", "hybrid+ce",
                  "dense+llm", "sparse+llm", "hybrid+llm"]


def _live():
    """라이브 컴포넌트 (임베더/Qdrant/리랭커/9B) 초기화 — 명령별 지연 임포트."""
    from qdrant_client import QdrantClient

    from agent.core.llm import OpenAICompatLLM
    from retriever import config
    from retriever.embedder import BM25SparseEmbedder, VLLMDenseEmbedder

    return {
        "client": QdrantClient(url=config.QDRANT_URL),
        "dense": VLLMDenseEmbedder(),
        "sparse": BM25SparseEmbedder(),
        # timeout 300s: 9B thinking이 2048토큰을 다 쓰면 ~65s+, 120s 기본은 빠듯 (실측 타임아웃)
        "llm": OpenAICompatLLM(base_url=config.LLM_URL, model=config.LLM_MODEL,
                               timeout=300.0,
                               sampling={"temperature": 0.2, "frequency_penalty": 0.5,
                                         "max_tokens": 2048}),
    }


def _make_reranker(kind: str, live):
    from retriever.reranker import LLMReranker, VLLMReranker

    if kind == "ce":
        return VLLMReranker()
    if kind == "llm":
        return LLMReranker(live["llm"])
    return None


def _retrieval_fn(arm: str, live):
    from retriever.search import search_statutes

    mode, _, rr = arm.partition("+")
    reranker = _make_reranker(rr, live)

    def retrieve(question: str) -> list[str]:
        last_err = None
        for attempt in range(3):  # 간헐 GPU 스톨/타임아웃은 재시도 (실측: dense+ce 313번째서 사망)
            try:
                hits = search_statutes(
                    live["client"], "statutes",
                    dense_vec=live["dense"].encode([question])[0],
                    sparse_vec=live["sparse"].encode_query([question])[0],
                    k=K, prefetch_limit=PREFETCH, mode=mode,
                    reranker=reranker, query_text=question if reranker else None,
                )
                return [h.payload["cid"] for h in hits]
            except Exception as e:
                last_err = e
                print(f"  retry {attempt + 1}/3 after {type(e).__name__}")
                time.sleep(10)
        raise last_err

    return retrieve


def _agent_fn(max_steps: int, live):
    from retriever.agent import build_statute_agent, run_statute_agent
    from retriever.reranker import VLLMReranker

    graph, tool = build_statute_agent(
        llm=live["llm"], client=live["client"], dense=live["dense"],
        sparse=live["sparse"], reranker=VLLMReranker(), max_steps=max_steps,
    )

    def retrieve(question: str):
        try:
            out = asyncio.run(run_statute_agent(graph, tool, question))
        except Exception as e:  # 쿼리 하나가 arm 전체 재개를 막지 않게 기록하고 진행
            return [], {"error": f"{type(e).__name__}: {e}"[:200]}
        ranked = [e["cid"] for e in out["evidence"]]
        for r in tool.top_session(k=K):  # 나머지는 세션 최고점으로 채워 top-8 리스트화
            if r["cid"] not in ranked:
                ranked.append(r["cid"])
        n_searches = sum(1 for s in out["steps"] if s.get("tool") == "statute_search")
        return ranked[:K], {"iterations": out["iterations"], "n_searches": n_searches}

    return retrieve


def _sample_queries(queries, n: int) -> list[dict]:
    """고정 샘플 (파일로 고정 — arm 간/재실행 간 동일 보장)."""
    if SAMPLE_FILE.exists():
        ids = set(json.loads(SAMPLE_FILE.read_text()))
        return [q for q in queries if q["query_id"] in ids]
    picked = random.Random(42).sample(queries, min(n, len(queries)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_FILE.write_text(json.dumps(sorted(q["query_id"] for q in picked)))
    return picked


def cmd_retrieval(args) -> None:
    from evaluation.runner import load_queries, run_arm, run_arm_parallel, summarize

    queries = load_queries(QRELS)
    if args.limit:
        queries = queries[:args.limit]
    arms = args.arms.split(",") if args.arms else [a for a in RETRIEVAL_ARMS if "+llm" not in a]
    live = None
    for arm in arms:
        assert arm in RETRIEVAL_ARMS, f"unknown arm: {arm}"
        if "+llm" in arm and args.workers > 1:  # 느린 arm만 병렬 (vLLM이 배칭)
            rows = run_arm_parallel(arm, lambda arm=arm: _retrieval_fn(arm, _live()),
                                    queries, OUT_DIR, workers=args.workers, progress_every=50)
        else:
            live = live or _live()
            rows = run_arm(arm, _retrieval_fn(arm, live), queries, OUT_DIR, progress_every=50)
        print(summarize(arm, rows))


def cmd_agent(args) -> None:
    from evaluation.runner import load_queries, run_arm_parallel, summarize

    queries = _sample_queries(load_queries(QRELS), args.sample)
    if args.limit:
        queries = queries[:args.limit]
    for steps in (int(s) for s in args.steps.split(",")):
        arm = f"react-s{steps}"
        # 팩토리 = 워커별 graph+tool 격리 (StatuteSearchTool 세션은 스레드 비안전)
        rows = run_arm_parallel(arm, lambda steps=steps: _agent_fn(steps, _live()),
                                queries, OUT_DIR, workers=args.workers, progress_every=10)
        print(summarize(arm, rows))


def cmd_report(args) -> None:
    from evaluation.runner import format_matrix, summarize

    def rows_of(path: Path):
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    all_rows = {p.stem: rows_of(p) for p in sorted(OUT_DIR.glob("*.jsonl")) if rows_of(p)}
    order = [a for a in RETRIEVAL_ARMS if a in all_rows] + \
            sorted(a for a in all_rows if a.startswith("react-"))

    md = ["# Ablation 결과", "",
          f"평가셋: 고용노동부 질의회시+법제처 해석례 qrels ({QRELS.name}), 조문(cid) 단위 정답.",
          "", "## 검색 파이프라인 (전체 평가셋)", "",
          format_matrix([summarize(a, all_rows[a]) for a in order if not a.startswith("react-")])]

    react_arms = [a for a in order if a.startswith("react-")]
    if react_arms and SAMPLE_FILE.exists():
        ids = set(json.loads(SAMPLE_FILE.read_text()))
        sub = []
        for a in order:
            rows = [r for r in all_rows[a] if r["query_id"] in ids]
            if rows:
                sub.append(summarize(a, rows))
        agent_meta = []
        for a in react_arms:
            iters = [r.get("meta", {}).get("iterations", 0) for r in all_rows[a]]
            searches = [r.get("meta", {}).get("n_searches", 0) for r in all_rows[a]]
            agent_meta.append(f"- `{a}`: 평균 반복 {sum(iters)/len(iters):.1f}, "
                              f"평균 검색 호출 {sum(searches)/len(searches):.1f}")
        md += ["", f"## 에이전트 비교 (고정 샘플 {len(ids)}개 — 동일 부분집합 공정 비교)", "",
               format_matrix(sub), "", *agent_meta]

    text = "\n".join(md) + "\n"
    out = ROOT / "docs" / "ablation_results.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("retrieval", help="검색 arm 실행 (기본: llm 리랭크 제외 6종)")
    p.add_argument("--arms", help="쉼표 구분 arm 목록 (예: hybrid+llm)")
    p.add_argument("--limit", type=int, help="쿼리 수 제한 (스모크용)")
    p.add_argument("--workers", type=int, default=3, help="+llm arm 동시 쿼리 수")
    p.set_defaults(fn=cmd_retrieval)

    p = sub.add_parser("agent", help="react 에이전트 arm 실행 (고정 샘플)")
    p.add_argument("--steps", default="2,4,6")
    p.add_argument("--sample", type=int, default=120)
    p.add_argument("--limit", type=int, help="샘플 내 쿼리 수 제한 (스모크용)")
    p.add_argument("--workers", type=int, default=3, help="동시 쿼리 수 (워커별 그래프 격리)")
    p.set_defaults(fn=cmd_agent)

    p = sub.add_parser("report", help="JSONL -> 매트릭스 표 (docs/ablation_results.md)")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
