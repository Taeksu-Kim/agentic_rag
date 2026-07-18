"""Ablation 러너: arm(설정) 하나 = retrieve 콜러블 하나를 평가셋에 돌려 지표를 낸다.

* arm별 per-query 결과를 JSONL로 기록 — **재개 가능** (이미 기록된 query_id 건너뜀).
  긴 arm(LLM 리랭크, react 에이전트)이 중간에 죽어도 이어서 돌린다.
* retrieve는 ``question -> ranked cid 리스트`` 또는 ``(ranked, meta)`` —
  에이전트 arm은 meta에 반복 횟수 등을 실어 보낸다. latency는 러너가 잰다.
* 지표(recall@k, MRR)는 기록된 rows에서 사후 계산 — 라이브 컴포넌트와 무관하게
  전부 fake로 테스트한다 (이 레포의 테스트 규칙).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence, Union

from evaluation.metrics import mrr, recall_at_k

Retrieve = Callable[[str], Union[list, tuple]]

DEFAULT_KS = (1, 3, 5, 8)


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                done.add(row["query_id"])
    return rows, done


def _make_row(q: dict[str, Any], got, latency: float) -> dict[str, Any]:
    ranked, meta = got if isinstance(got, tuple) else (got, {})
    row = {"query_id": q["query_id"], "ranked": list(ranked),
           "labels": q["labels"], "latency": latency}
    if meta:
        row["meta"] = dict(meta)
    return row


def load_queries(qrels_path: str | Path) -> list[dict[str, Any]]:
    """qrels parquet -> [{query_id, question, labels}] (라벨 없는 쿼리는 제외)."""
    import pandas as pd

    df = pd.read_parquet(qrels_path)
    out = []
    for r in df.itertuples(index=False):
        labels = [str(x) for x in r.labels]
        if labels:
            out.append({"query_id": str(r.query_id), "question": str(r.question),
                        "labels": labels})
    return out


def run_arm(
    name: str,
    retrieve: Retrieve,
    queries: Sequence[dict[str, Any]],
    out_dir: str | Path,
    *,
    progress_every: int = 0,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """arm 하나 실행 -> per-query rows (JSONL ``<out_dir>/<name>.jsonl``에 append).

    이미 파일에 있는 query_id는 retrieve를 호출하지 않는다 (재개).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    rows, done = _load_existing(path)

    with path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(queries, 1):
            if q["query_id"] in done:
                continue
            t0 = time.perf_counter()
            got = retrieve(q["question"])
            latency = time.perf_counter() - t0
            row = _make_row(q, got, latency)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            if progress_every and i % progress_every == 0:
                log(f"[{name}] {i}/{len(queries)}")
    return rows


def run_arm_parallel(
    name: str,
    make_retrieve: Callable[[], Retrieve],
    queries: Sequence[dict[str, Any]],
    out_dir: str | Path,
    *,
    workers: int = 3,
    progress_every: int = 0,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """run_arm의 스레드 병렬판 — LLM이 낀 느린 arm용 (vLLM이 동시 요청을 배칭).

    ``make_retrieve``는 **스레드당 한 번** 호출되는 팩토리: 에이전트 arm처럼
    상태(검색 세션)가 있는 retrieve를 워커별로 격리한다. JSONL append는 락으로
    직렬화하며 완료 순서대로 기록된다 (재개는 query_id 기준이라 순서 무관).
    주의: 이렇게 잰 per-query latency는 서버 큐잉이 섞인 값 — 처리량은 늘지만
    개별 수치는 순차 실행보다 부풀려진다 (참고치로만).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    rows, done = _load_existing(path)
    pending = [q for q in queries if q["query_id"] not in done]
    if not pending:
        return rows

    lock = threading.Lock()
    tls = threading.local()
    n_done = 0

    def work(q: dict[str, Any]) -> None:
        nonlocal n_done
        if not hasattr(tls, "retrieve"):
            tls.retrieve = make_retrieve()
        t0 = time.perf_counter()
        got = tls.retrieve(q["question"])
        latency = time.perf_counter() - t0
        row = _make_row(q, got, latency)
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            n_done += 1
            if progress_every and n_done % progress_every == 0:
                log(f"[{name}] {n_done}/{len(pending)}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, pending))  # list()로 예외 전파
    return rows


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, round(p * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def summarize(name: str, rows: Sequence[dict[str, Any]],
              ks: Sequence[int] = DEFAULT_KS) -> dict[str, Any]:
    """rows -> {name, n, recall@k..., mrr, lat_p50, lat_p95}."""
    lats = sorted(r["latency"] for r in rows)
    out: dict[str, Any] = {"name": name, "n": len(rows)}
    for k in ks:
        out[f"recall@{k}"] = mean(recall_at_k(r["ranked"], r["labels"], k) for r in rows) if rows else 0.0
    out["mrr"] = mean(mrr(r["ranked"], r["labels"]) for r in rows) if rows else 0.0
    out["lat_p50"] = _pct(lats, 0.50)
    out["lat_p95"] = _pct(lats, 0.95)
    return out


def format_matrix(summaries: Sequence[dict[str, Any]],
                  ks: Sequence[int] = DEFAULT_KS) -> str:
    """summaries -> 마크다운 표 (README/보고서용)."""
    heads = ["arm", "n"] + [f"R@{k}" for k in ks] + ["MRR", "p50(s)", "p95(s)"]
    lines = ["| " + " | ".join(heads) + " |",
             "|" + "|".join("---" for _ in heads) + "|"]
    for s in summaries:
        cells = [s["name"], str(s["n"])]
        cells += [f"{s[f'recall@{k}']:.3f}" for k in ks]
        cells += [f"{s['mrr']:.3f}", f"{s['lat_p50']:.2f}", f"{s['lat_p95']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
