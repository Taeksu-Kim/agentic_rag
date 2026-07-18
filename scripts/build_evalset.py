"""Collect 법령해석 sources and build qrels for retrieval evaluation.

Sources (law.go.kr DRF Open API; see docs/design_and_plan.md §3):
* ``moelCgmExpc`` -- 고용노동부 행정해석(질의회시): 질의요지 = query, 회답 = citations
* ``expc``        -- 법제처 법령해석례: harder multi-hop subset (질의요지/회답/이유)

Flow: keyword list-search (paged, deduped by serial id) -> per-id detail XML,
cached under data/eval/raw/<target>/<id>.xml (resumable; API hit only on first
run) -> parse citations (evaluation/citations.py) against the corpus laws ->
data/eval/qrels.parquet + coverage report.

    python scripts/build_evalset.py [--oc test] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "eval", "raw")
OUT_DIR = os.path.join(ROOT, "data", "eval")
CORPUS = os.path.join(ROOT, "data", "corpus", "labor_statutes.parquet")

BASE = "http://www.law.go.kr/DRF"
DELAY = 0.3  # politeness

# moelCgmExpc는 키워드 검색만 되므로 노동 도메인을 넓게 커버하는 키워드로 긁고
# 일련번호로 dedupe한다. expc는 법령명 검색.
MOEL_KEYWORDS = [
    "근로기준법", "기간제", "단시간", "산업안전", "고용보험", "최저임금", "퇴직급여",
    "남녀고용평등", "연차", "임금", "해고", "휴가", "휴게", "근로시간", "육아휴직",
    "퇴직금", "출산", "연장근로", "취업규칙", "휴일",
]
EXPC_KEYWORDS = [
    "근로기준법", "남녀고용평등", "최저임금법", "근로자퇴직급여", "기간제",
    "산업안전보건법", "고용보험법", "근로자참여",
]

# 전부개정(조문 재배열) 이전의 해석은 옛 조문번호를 인용하므로 현행 코퍼스와
# 매칭하면 오답 라벨이 된다 (예: 2003년 회시의 "근로기준법 제49조" = 현행 제50조).
# 해석일자가 기준일 이전이면 그 법령(모법 기준)의 라벨을 버린다.
# 기본값 20120726(근퇴법 전부개정 시행 — 근기법 2007, 고보법 2007 등을 모두 커버하는
# 보수적 컷), 산업안전보건법은 2019 전부개정 시행일.
DATE_CUTOFF_DEFAULT = 20120726
DATE_CUTOFFS = {"산업안전보건법": 20200116}


def _base_law(law: str) -> str:
    return re.sub(r"\s*시행(?:령|규칙)$", "", law)


def _label_ok(law: str, date_str: str) -> bool:
    digits = re.sub(r"\D", "", date_str or "")
    if len(digits) < 8:
        return False  # 날짜 불명 -> 보수적으로 제외
    return int(digits[:8]) >= DATE_CUTOFFS.get(_base_law(law), DATE_CUTOFF_DEFAULT)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (eval-builder)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def search_ids(oc: str, target: str, query: str) -> list[str]:
    """List-search one keyword, all pages -> serial ids."""
    ids, page = [], 1
    while True:
        url = (f"{BASE}/lawSearch.do?OC={oc}&target={target}&type=XML"
               f"&query={urllib.parse.quote(query)}&display=100&page={page}")
        root = ET.fromstring(_get(url))
        total = int(root.findtext("totalCnt") or 0)
        serials = [e.text for e in root.iter() if e.tag in ("법령해석례일련번호", "법령해석일련번호") and e.text]
        ids.extend(serials)
        if page * 100 >= total or not serials:
            return ids
        page += 1
        time.sleep(DELAY)


def fetch_detail(oc: str, target: str, sid: str) -> str:
    """Detail XML, cached on disk (resumable)."""
    path = os.path.join(RAW_DIR, target, f"{sid}.xml")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xml = _get(f"{BASE}/lawService.do?OC={oc}&target={target}&ID={sid}&type=XML").decode("utf-8")
    open(path, "w", encoding="utf-8").write(xml)
    time.sleep(DELAY)
    return xml


def parse_detail(xml: str) -> dict | None:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    get = lambda tag: (root.findtext(tag) or "").strip()
    return {
        "title": get("안건명"),
        "case_no": get("안건번호"),
        "date": get("해석일자"),
        "question": get("질의요지"),
        "answer": get("회답"),
        "reason": get("이유"),  # expc only
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oc", default="test")
    ap.add_argument("--limit", type=int, default=0, help="cap detail fetches per target (0=all)")
    args = ap.parse_args()

    corpus = pd.read_parquet(CORPUS)
    known_laws = sorted(corpus["law_name"].unique())
    valid_ids = set(corpus["law_name"] + "|" + corpus["clause_no"].astype(str))
    from evaluation.citations import extract_citations

    rows, coverage = [], {"docs": 0, "no_question": 0, "no_label": 0, "labeled": 0,
                          "cited_out_of_corpus": 0, "date_filtered": 0}
    for target, keywords in (("moelCgmExpc", MOEL_KEYWORDS), ("expc", EXPC_KEYWORDS)):
        ids: dict[str, None] = {}
        for kw in keywords:
            found = search_ids(args.oc, target, kw)
            for i in found:
                ids.setdefault(i)
            print(f"[{target}] '{kw}': {len(found)} (unique so far {len(ids)})", flush=True)
            time.sleep(DELAY)
        todo = list(ids)
        if args.limit:
            todo = todo[: args.limit]
        print(f"[{target}] fetching {len(todo)} details ...", flush=True)

        for n, sid in enumerate(todo, 1):
            d = parse_detail(fetch_detail(args.oc, target, sid))
            if n % 200 == 0:
                print(f"  [{target}] {n}/{len(todo)}", flush=True)
            if not d:
                continue
            coverage["docs"] += 1
            if not d["question"]:
                coverage["no_question"] += 1
                continue
            cite_text = " ".join(filter(None, [d["answer"], d["reason"]]))
            cites = extract_citations(cite_text, known_laws)
            in_corpus = [(law, no) for law, no in cites if f"{law}|{no}" in valid_ids]
            labels = [f"{law}|{no}" for law, no in in_corpus if _label_ok(law, d["date"])]
            if not labels:
                coverage["no_label"] += 1
                if in_corpus:  # 라벨은 있었는데 개정 컷오프로 전부 탈락
                    coverage["date_filtered"] += 1
                elif cites:  # 인용은 있는데 전부 코퍼스 밖 조문
                    coverage["cited_out_of_corpus"] += 1
                continue
            coverage["labeled"] += 1
            rows.append({
                "query_id": f"{target}-{sid}", "source": target, "title": d["title"],
                "case_no": d["case_no"], "date": d["date"], "question": d["question"],
                "labels": labels, "n_labels": len(labels),
            })

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(os.path.join(OUT_DIR, "qrels.parquet"), index=False)
    print(f"\nwrote {len(df)} queries -> data/eval/qrels.parquet")
    print("coverage:", json.dumps(coverage, ensure_ascii=False))
    if len(df):
        print("\nby source:\n", df.groupby("source").agg(n=("query_id", "size"), labels=("n_labels", "mean")).to_string())
        print("\nlabel count dist:\n", df.n_labels.value_counts().sort_index().head(10).to_string())


if __name__ == "__main__":
    main()
