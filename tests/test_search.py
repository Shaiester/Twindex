from twindex.search import boost, rrf


def test_rrf_rewards_agreement_and_uses_ranks_not_scores():
    a = [("x", 100.0), ("y", 50.0), ("z", 1.0)]
    b = [("y", 0.9), ("w", 0.8)]
    fused = [d for d, _ in rrf(a, b)]
    assert fused[0] == "y"  # rank 2 + rank 1 beats rank 1 alone
    assert set(fused) == {"x", "y", "z", "w"}
    # Rescaling one list's scores changes nothing.
    assert rrf(a, b) == rrf([(d, s * 1e6) for d, s in a], b)


def test_rrf_score_formula():
    assert dict(rrf([("x", 1)], k=60))["x"] == 1 / 61


def test_boost_reorders_close_calls_but_not_big_gaps():
    hits = [("quiet", 10.0), ("loud", 9.9), ("offtopic_loud", 1.0)]
    act = {"quiet": -1.0, "loud": 3.0, "offtopic_loud": 3.0}
    ranked = [d for d, _ in boost(hits, act, weight=0.3)]
    assert ranked[0] == "loud"  # near-tie on relevance, much louder chat
    assert ranked[-1] == "offtopic_loud"  # loudness alone can't overcome a big relevance gap
    assert [d for d, _ in boost(hits, act, weight=0.0)] == ["quiet", "loud", "offtopic_loud"]


def test_boost_handles_constant_and_empty():
    assert boost([], {}, 0.3) == []
    assert [d for d, _ in boost([("a", 1.0), ("b", 1.0)], {"a": 0.0, "b": 0.0}, 0.3)] == ["a", "b"]
