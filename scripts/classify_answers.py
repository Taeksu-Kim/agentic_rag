"""회답 유형 분류 (확답/조건부/판단불가) — 스코프 라우팅의 오프라인 라벨.

동기: 질의회시 회답의 71%는 "구체적 사실관계에 따라"류 유보가 핵심인 해석
자문이라 조문 검색만으로 완결 답변이 불가능하다. 확답형(~26%)이 "사내 챗봇이
마땅히 답해야 할 질문" 스코프이고, 평가 헤드라인도 이 서브셋으로 병기한다.

    PYTHONPATH=. python scripts/classify_answers.py
    -> data/eval/answer_types.parquet (query_id, answer_type)
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "eval" / "answer_types.parquet"

SYS = "너는 노동부 행정해석 회답의 유형을 분류하는 도우미다."
PROMPT = ("아래 회답이 질의에 대해 어떤 유형인지 분류하라.\n"
          "- 확답: 조문을 근거로 명확한 결론을 제시\n"
          "- 조건부: 결론은 있으나 '구체적 사실관계에 따라' 등 유보가 핵심\n"
          "- 판단불가: 사실상 결론 없이 일반 원칙만 안내\n\n회답:\n{a}")
SCHEMA = {"type": "object",
          "properties": {"type": {"type": "string", "enum": ["확답", "조건부", "판단불가"]}},
          "required": ["type"]}


def answer_text(qid: str) -> str:
    src, num = qid.rsplit("-", 1)
    xml = (ROOT / f"data/eval/raw/{src}/{num}.xml").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<회답>.*?CDATA\[(.*?)\]\].*?</회답>", xml, re.DOTALL)
    return re.sub(r"<[^>]+>", " ", m.group(1))[:800] if m else ""


def main() -> None:
    from agent.core.llm import OpenAICompatLLM
    from retriever import config

    llm = OpenAICompatLLM(base_url=config.LLM_URL, model=config.LLM_MODEL,
                          timeout=300.0, sampling={"temperature": 0.0, "max_tokens": 512})
    qids = pd.read_parquet(ROOT / "data/eval/qrels.parquet").query_id.tolist()

    def cls(qid: str) -> tuple[str, str]:
        a = answer_text(qid)
        if not a:
            return qid, "?"
        try:
            return qid, json.loads(llm.complete(SYS, PROMPT.format(a=a), schema=SCHEMA))["type"]
        except Exception:
            return qid, "?"

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(cls, qids))
    df = pd.DataFrame(rows, columns=["query_id", "answer_type"])
    df.to_parquet(OUT, index=False)
    print(df.answer_type.value_counts().to_dict())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
