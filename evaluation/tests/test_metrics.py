from evaluation.metrics import mrr, recall_at_k


def test_recall_at_k_counts_hits_within_k():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a", "d"}, k=2) == 0.5
    assert recall_at_k(ranked, {"a", "d"}, k=4) == 1.0


def test_recall_at_k_no_relevant_is_zero():
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_recall_at_k_miss_is_zero():
    assert recall_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_mrr_reciprocal_rank_of_first_hit():
    assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5
    assert mrr(["a"], {"a"}) == 1.0


def test_mrr_zero_when_absent():
    assert mrr(["x", "y"], {"a"}) == 0.0
    assert mrr([], {"a"}) == 0.0
