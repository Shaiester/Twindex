"""Retrieval methods behind one entry point: bm25, dense, hybrid (RRF), each optionally activity-boosted."""
from . import dense, index
from .config import load_config

RRF_K = 60  # Cormack et al. 2009; damps the gap between rank 1 and rank 2 so neither list dominates
DEPTH = 100  # candidates taken from each list before fusing / reranking


def rrf(*rankings: list[tuple[str, float]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: uses ranks only, so BM25 and cosine scores need no calibration."""
    scores = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def boost(hits: list[tuple[str, float]], activity: dict[str, float], weight: float) -> list[tuple[str, float]]:
    """Rerank candidates by relevance + weight * chat activity, both min-max scaled within the candidate set.

    Additive and only over already-retrieved candidates, so a loud but off-topic window can't enter the
    results; it can only move up among windows that matched the query.
    """
    if not hits:
        return hits
    rel = _minmax([s for _, s in hits])
    act = _minmax([activity[d] for d, _ in hits])
    return sorted(((d, r + weight * a) for (d, _), r, a in zip(hits, rel, act)), key=lambda x: -x[1])


def search(con, q: str, method: str = "bm25", boost_on: bool = False, k: int = 10, weight: float | None = None,
           **filters) -> list[tuple[str, float]]:
    depth = max(k, DEPTH) if (boost_on or method == "hybrid") else k
    if method == "bm25":
        hits = index.bm25(con, q, depth, **filters)
    elif method == "dense":
        hits = dense.search(con, q, depth, **filters)
    elif method == "hybrid":
        hits = rrf(index.bm25(con, q, depth, **filters), dense.search(con, q, depth, **filters))[:depth]
    else:
        raise ValueError(method)
    if boost_on and hits:
        if weight is None:
            weight = load_config().get("ranking", {}).get("boost_weight", 0.3)
        ids = [d for d, _ in hits]
        activity = dict(con.execute(f"SELECT id, activity FROM windows WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall())
        hits = boost(hits, activity, weight)
    return hits[:k]
