import math
import random

from twindex.eval import mrr, ndcg, permutation_test, recall
from twindex.evalset import cohen_kappa, split, stratified_sample


def test_ndcg_hand_computed():
    qrel = {"a": 2, "b": 1, "c": 0}
    assert ndcg(["a", "b"], qrel) == 1.0
    # [b, a]: dcg = 1/log2(2) + 3/log2(3); ideal = 3 + 1/log2(3)
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert math.isclose(ndcg(["b", "a"], qrel), expected)
    assert ndcg(["x", "y"], qrel) == 0.0 and ndcg(["a"], {"c": 0}) == 0.0


def test_recall_and_mrr_use_threshold_1():
    qrel = {"a": 2, "b": 1, "c": 0}
    assert recall(["c", "b"], qrel) == 0.5
    assert mrr(["c", "x", "b"], qrel) == 1 / 3
    assert mrr(["c"], qrel) == 0.0


def test_permutation_test_detects_consistent_difference():
    a = [0.9] * 30
    b = [0.1] * 30
    assert permutation_test(a, b, n=2000) < 0.01
    assert permutation_test(a, a, n=200) == 1.0


def test_kappa():
    a = [0, 1, 2, 2, 0, 1]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, a, "quadratic") == 1.0
    # Disagreeing by one step costs less under quadratic weights than by two.
    near, far = [1, 1, 2, 1, 0, 1], [2, 1, 2, 0, 0, 1]
    assert cohen_kappa(a, near, "quadratic") > cohen_kappa(a, far, "quadratic")


def test_split_is_stratified_seeded_and_complete():
    qs = [{"qid": f"q{i}", "type": t} for i, t in enumerate(["exact"] * 10 + ["slang"] * 20)]
    s = split(qs)
    assert s == split(qs) and len(s) == 30
    assert sum(1 for q in qs if q["type"] == "exact" and s[q["qid"]] == "dev") == 3
    assert sum(1 for q in qs if q["type"] == "slang" and s[q["qid"]] == "dev") == 6


def test_stratified_sample_balances_channels_and_activity():
    ws = [{"id": f"{c}:{i}", "channel": c, "activity": i} for c, n in (("a", 300), ("b", 60)) for i in range(n)]
    picked = stratified_sample(ws, 30, random.Random(0))
    assert sum(w["channel"] == "a" for w in picked) == 15 and sum(w["channel"] == "b" for w in picked) == 15
    a_acts = sorted(w["activity"] for w in picked if w["channel"] == "a")
    assert sum(x < 100 for x in a_acts) == 5 and sum(x >= 200 for x in a_acts) == 5
