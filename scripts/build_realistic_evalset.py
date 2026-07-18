"""질의회시 질문 -> 사내 챗봇에 물어볼 법한 짧은 구어체 질문으로 변형 (라벨 유지).

동기 (ablation 발견): 질의회시 질문은 이미 법률 용어로 정제된 수백~2,000자
행정 문서라 ① 실제 챗봇 입력과 동떨어져 있고 ② 에이전트의 핵심 무기인
리라이팅이 발휘될 여지가 없다. 같은 정답 라벨을 유지한 채 질문만 현실화해
"현실 조건에서 에이전틱 반복이 값을 하는가"를 측정한다.

    PYTHONPATH=. python scripts/build_realistic_evalset.py
    -> data/eval/qrels_realistic.parquet (120 고정 샘플)
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QRELS = ROOT / "data" / "eval" / "qrels.parquet"
SAMPLE_FILE = ROOT / "data" / "eval" / "ablation" / "sample_ids.json"
OUT = ROOT / "data" / "eval" / "qrels_realistic.parquet"

SYSTEM = (
    "너는 사내 HR 챗봇의 사용성 테스트용 질문을 만드는 도우미다. "
    "노동부 행정해석 질의를, 직원이 사내 챗봇에 실제로 입력할 법한 질문으로 바꾼다."
)

PROMPT = """아래 질의를 직원이 사내 HR 챗봇에 물어볼 법한 **1~2문장 구어체 질문**으로 재작성하라.

규칙:
- 질의가 담고 있는 법적 쟁점은 **모두** 유지하라 (쟁점을 빠뜨리면 안 된다).
- 회사 내부 사정, 수치, 경위 설명 등 배경은 과감히 줄여라.
- 법령명·조문 번호 인용(「…법」 제N조 같은 것)은 **모두 제거**하라 — 일반 직원은 조문을 모른다.
- 법률 용어 대신 일상어를 써라 (예: "상시근로자" 대신 "직원", "사용자" 대신 "회사").

질의:
{question}"""

SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}},
          "required": ["question"]}


def main() -> None:
    from agent.core.llm import OpenAICompatLLM
    from retriever import config

    llm = OpenAICompatLLM(base_url=config.LLM_URL, model=config.LLM_MODEL,
                          timeout=300.0,
                          sampling={"temperature": 0.3, "max_tokens": 1024})

    df = pd.read_parquet(QRELS)
    ids = set(json.loads(SAMPLE_FILE.read_text()))
    df = df[df["query_id"].isin(ids)].reset_index(drop=True)
    print(f"{len(df)}개 질문 변형 시작")

    def rewrite(q: str) -> str:
        text = llm.complete(SYSTEM, PROMPT.format(question=q[:3000]), schema=SCHEMA)
        try:
            return str(json.loads(text)["question"]).strip() or q
        except (ValueError, KeyError):
            return q

    with ThreadPoolExecutor(max_workers=3) as ex:
        rewritten = list(ex.map(rewrite, df["question"]))

    out = df.copy()
    out["orig_question"] = out["question"]
    out["question"] = rewritten
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT}")
    for _, r in out.head(5).iterrows():
        print(f"\n[{r.query_id}] {r.question}")


if __name__ == "__main__":
    main()
