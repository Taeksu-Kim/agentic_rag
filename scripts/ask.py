"""단일 질문 라이브 실행: 리트리버 에이전트 (9B + Qdrant + 임베더 [+리랭커]).

    PYTHONPATH=. python scripts/ask.py "육아휴직 기간에도 연차휴가가 발생하나요?"
    옵션: --no-rerank  --max-steps 6  --trace
"""

from __future__ import annotations

import argparse
import asyncio
import json

from qdrant_client import QdrantClient

from agent.core.llm import OpenAICompatLLM
from retriever import config
from retriever.agent import build_statute_agent, run_statute_agent
from retriever.embedder import BM25SparseEmbedder, VLLMDenseEmbedder
from retriever.reranker import VLLMReranker


async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--trace", action="store_true", help="스텝 트레이스 출력")
    args = ap.parse_args()

    llm = OpenAICompatLLM(base_url=config.LLM_URL, model=config.LLM_MODEL, timeout=300.0,
                          sampling={"temperature": 0.2, "frequency_penalty": 0.5, "max_tokens": 2048})
    import pandas as pd
    laws = sorted(pd.read_parquet("data/corpus/labor_statutes.parquet").law_name.unique())
    graph, search_tool = build_statute_agent(
        llm=llm,
        client=QdrantClient(url=config.QDRANT_URL),
        dense=VLLMDenseEmbedder(),
        sparse=BM25SparseEmbedder(),
        reranker=None if args.no_rerank else VLLMReranker(),
        max_steps=args.max_steps,
        valid_laws=laws,
    )
    out = await run_statute_agent(graph, search_tool, args.question, synth_llm=llm)

    print(f"\nQ: {args.question}")
    print("=" * 70)
    if args.trace:
        for i, s in enumerate(out["steps"], 1):
            print(f"[step {i}] {s['tool']}({json.dumps(s.get('args', {}), ensure_ascii=False)})")
            obs = s.get("observation")
            if isinstance(obs, list):
                for o in obs[:3]:
                    if isinstance(o, dict) and "ref" in o:
                        print(f"    {o.get('score')}  {o['ref']}")
            print()
    print(f"A: {out['answer']}\n")
    print("근거 조문:")
    for e in out["evidence"]:
        print(f"  - {e['law_name']} 제{e['clause_no']}조({e['clause_title']})  score={e['score']:.3f}")


if __name__ == "__main__":
    asyncio.run(amain())
