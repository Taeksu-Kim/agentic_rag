"""확답형 쿼리의 '핵심 근거 조문' 재라벨링 (LLM이 회답을 읽고 선택).

동기(실측): qrels의 labels[0](회답의 첫 인용)은 관행상 목적·적용범위 같은
보일러플레이트거나 교차 인용 앵커인 경우가 있어(11.2%), 순서 기반 '주인용'
채점이 검색 실패가 아닌 것을 실패로 센다. 회답의 결론을 직접 뒷받침하는 조문
1개를 9B가 고르게 해 채점 기준을 의미 기반으로 교정한다.

    PYTHONPATH=. python scripts/relabel_primary.py
    -> data/eval/primary_labels.parquet (query_id, primary_cid)
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "eval" / "primary_labels.parquet"

SYS = "너는 노동부 행정해석 회답의 근거 조문을 분석하는 도우미다."
PROMPT = """질의와 회답, 그리고 회답이 인용한 조문 목록이 있다.
회답의 **결론을 직접 뒷받침하는 핵심 근거 조문 1개**를 목록에서 골라라.
목적·정의·적용범위처럼 배경으로만 인용된 조문은 핵심이 아니다.

질의: {q}

회답: {a}

조문 목록:
{cands}"""


def answer_text(qid: str) -> str:
    src, num = qid.rsplit("-", 1)
    xml = (ROOT / f"data/eval/raw/{src}/{num}.xml").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<회답>.*?CDATA\[(.*?)\]\].*?</회답>", xml, re.DOTALL)
    return re.sub(r"<[^>]+>", " ", m.group(1))[:1200] if m else ""


def main() -> None:
    from agent.core.llm import OpenAICompatLLM
    from retriever import config

    corpus = pd.read_parquet(ROOT / "data/corpus/labor_statutes.parquet")
    corpus["cid"] = corpus.law_name + "|" + corpus.clause_no.astype(str)
    titles = corpus.set_index("cid").clause_title.fillna("")

    qrels = pd.read_parquet(ROOT / "data/eval/qrels.parquet")
    types = pd.read_parquet(ROOT / "data/eval/answer_types.parquet") \
        .set_index("query_id").answer_type
    firm = qrels[qrels.query_id.map(types) == "확답"]
    print(f"{len(firm)}개 확답형 재라벨링")

    llm = OpenAICompatLLM(base_url=config.LLM_URL, model=config.LLM_MODEL,
                          timeout=300.0, sampling={"temperature": 0.0, "max_tokens": 512})

    def pick(row) -> tuple[str, str]:
        labels = [str(x) for x in row.labels]
        if len(labels) == 1:
            return row.query_id, labels[0]
        cands = "\n".join(f"- {c} ({titles.get(c, '?')})" for c in labels)
        schema = {"type": "object", "properties": {"cid": {"type": "string", "enum": labels}},
                  "required": ["cid"]}
        try:
            out = json.loads(llm.complete(
                SYS, PROMPT.format(q=str(row.question)[:1000], a=answer_text(row.query_id),
                                   cands=cands), schema=schema))
            cid = str(out["cid"])
            return row.query_id, cid if cid in labels else labels[0]
        except Exception:
            return row.query_id, labels[0]

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(pick, (r for _, r in firm.iterrows())))
    df = pd.DataFrame(rows, columns=["query_id", "primary_cid"])
    df.to_parquet(OUT, index=False)
    changed = sum(df.set_index("query_id").primary_cid[r.query_id] != str(r.labels[0])
                  for _, r in firm.iterrows())
    print(f"labels[0]과 달라진 재라벨: {changed}/{len(df)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
