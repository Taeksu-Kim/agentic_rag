"""Build the labor-law corpus from HF `ducut91/korean-statutes` (MIT).

Downloads the full statutes CSV (231MB, cached), filters it down to the 8-law
labor package (법률 + 시행령 + 시행규칙), and writes clause-level parquet/csv to
data/corpus/. One clause = one chunk = one future Qdrant point.

    python scripts/build_corpus.py [--csv <cached-csv-path>]

Reproducibility: deterministic given the dataset revision; ~1,787 clauses.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import pandas as pd

URL = "https://huggingface.co/datasets/ducut91/korean-statutes/resolve/main/korean-statutes-v2.csv"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "corpus")

# 노동법 8종 패키지. 시행규칙의 LawTypeName은 "고용노동부령"/"노동부령"(옛 명칭)이
# 섞여 있으므로 법령명(정확 일치)으로 필터한다.
BASE_LAWS = [
    "근로기준법",
    "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률",
    "최저임금법",
    "근로자퇴직급여 보장법",
    "기간제 및 단시간근로자 보호 등에 관한 법률",
    "산업안전보건법",
    "고용보험법",
    "근로자참여 및 협력증진에 관한 법률",
]
LAW_NAMES = BASE_LAWS + [b + " 시행령" for b in BASE_LAWS] + [b + " 시행규칙" for b in BASE_LAWS]

KEEP_COLS = {
    "LawNameKor": "law_name",
    "LawTypeName": "law_type",
    "DepartmentName": "department",
    "ClauseNumber": "clause_no",
    "ClauseTitle": "clause_title",
    "ClauseContent": "clause_content",
    "EffectiveDate": "effective_date",
    "PromulgationDate": "promulgation_date",
    "LawID": "law_id",
}


def download(csv_path: str) -> str:
    if os.path.exists(csv_path):
        print(f"using cached {csv_path}")
        return csv_path
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print(f"downloading {URL} -> {csv_path} (231MB)")
    urllib.request.urlretrieve(URL, csv_path)
    return csv_path


def build(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    core = df[df["LawNameKor"].isin(LAW_NAMES)].copy()
    missing = set(LAW_NAMES) - set(core["LawNameKor"].unique())
    if missing:
        print(f"WARNING: {len(missing)} law names not found: {sorted(missing)}", file=sys.stderr)
    core = core[list(KEEP_COLS)].rename(columns=KEEP_COLS)
    core = core.dropna(subset=["clause_content"]).reset_index(drop=True)
    # stable id: 법령ID-조문번호 (조문번호는 법령 내 유일)
    core["id"] = core["law_id"].astype(str) + "-" + core["clause_no"].astype(str)
    assert core["id"].is_unique, "clause ids must be unique"
    return core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "raw", "korean-statutes-v2.csv"),
                    help="path to (or cache location for) the full statutes CSV")
    args = ap.parse_args()

    core = build(download(args.csv))
    os.makedirs(OUT_DIR, exist_ok=True)
    core.to_parquet(os.path.join(OUT_DIR, "labor_statutes.parquet"), index=False)
    core.to_csv(os.path.join(OUT_DIR, "labor_statutes.csv"), index=False)

    print(f"\nwrote {len(core)} clauses -> {OUT_DIR}/labor_statutes.parquet|csv")
    print(core.groupby(["law_name", "law_type"]).size().to_string())


if __name__ == "__main__":
    main()
