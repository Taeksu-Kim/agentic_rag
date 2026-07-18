"""러너: JSONL 기록/재개, 요약 지표, 매트릭스 표 (전부 오프라인 fake)."""

import json
import time

from evaluation.runner import format_matrix, run_arm, run_arm_parallel, summarize

QUERIES = [
    {"query_id": "q1", "question": "연차?", "labels": ["근로기준법|60"]},
    {"query_id": "q2", "question": "해고?", "labels": ["근로기준법|23", "근로기준법|27"]},
]


def fake_retrieve(question):
    return {"연차?": ["근로기준법|60", "x"], "해고?": ["x", "근로기준법|23"]}[question]


def test_run_arm_writes_rows_and_jsonl(tmp_path):
    rows = run_arm("a", fake_retrieve, QUERIES, tmp_path)
    assert [r["query_id"] for r in rows] == ["q1", "q2"]
    assert rows[0]["ranked"] == ["근로기준법|60", "x"]
    assert rows[0]["latency"] >= 0
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["query_id"] == "q1"


def test_run_arm_resumes_without_recalling(tmp_path):
    calls = []

    def counting(question):
        calls.append(question)
        return ["x"]

    run_arm("a", counting, QUERIES, tmp_path)
    assert len(calls) == 2
    rows = run_arm("a", counting, QUERIES, tmp_path)  # 재실행: 전부 스킵
    assert len(calls) == 2 and len(rows) == 2


def test_run_arm_records_meta_from_tuple(tmp_path):
    rows = run_arm("a", lambda q: (["x"], {"iters": 3}), QUERIES[:1], tmp_path)
    assert rows[0]["meta"] == {"iters": 3}


def test_run_arm_parallel_processes_all_once(tmp_path):
    made, seen = [], []

    def make_retrieve():
        made.append(1)

        def retrieve(question):
            seen.append(question)
            return fake_retrieve(question)

        return retrieve

    rows = run_arm_parallel("p", make_retrieve, QUERIES, tmp_path, workers=2)
    assert sorted(r["query_id"] for r in rows) == ["q1", "q2"]
    assert sorted(seen) == ["연차?", "해고?"]  # 각 쿼리 정확히 1회
    assert 1 <= len(made) <= 2  # 팩토리는 스레드당 최대 1회
    by_id = {r["query_id"]: r for r in rows}
    assert by_id["q1"]["ranked"] == ["근로기준법|60", "x"]


def test_run_arm_parallel_resumes_without_factory(tmp_path):
    made = []

    def make_retrieve():
        made.append(1)
        return fake_retrieve

    run_arm_parallel("p", make_retrieve, QUERIES, tmp_path, workers=2)
    n = len(made)
    rows = run_arm_parallel("p", make_retrieve, QUERIES, tmp_path, workers=2)
    assert len(rows) == 2 and len(made) == n  # 전부 완료 -> 팩토리 재호출 없음


def test_run_arm_parallel_actually_concurrent(tmp_path):
    def make_retrieve():
        def retrieve(question):
            time.sleep(0.2)
            return ["x"]

        return retrieve

    t0 = time.perf_counter()
    run_arm_parallel("p", make_retrieve, QUERIES, tmp_path, workers=2)
    assert time.perf_counter() - t0 < 0.35  # 순차면 0.4s+


def test_summarize_metrics():
    rows = [
        {"query_id": "q1", "ranked": ["근로기준법|60", "x"], "labels": ["근로기준법|60"], "latency": 0.1},
        {"query_id": "q2", "ranked": ["x", "근로기준법|23"], "labels": ["근로기준법|23", "근로기준법|27"], "latency": 0.3},
    ]
    s = summarize("a", rows, ks=(1, 2))
    assert s["n"] == 2
    assert s["recall@1"] == 0.5          # q1 hit, q2 miss
    assert s["recall@2"] == (1.0 + 0.5) / 2
    assert s["mrr"] == (1.0 + 0.5) / 2
    assert s["lat_p50"] == 0.1 and s["lat_p95"] == 0.3


def test_format_matrix_markdown():
    s = summarize("hybrid+ce", [
        {"query_id": "q", "ranked": ["a"], "labels": ["a"], "latency": 0.05},
    ], ks=(1,))
    md = format_matrix([s], ks=(1,))
    assert md.splitlines()[0].startswith("| arm | n | R@1 | MRR")
    assert "| hybrid+ce | 1 | 1.000 | 1.000 |" in md
