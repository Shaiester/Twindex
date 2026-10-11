"""Dense retrieval: bge-small-en-v1.5 window embeddings, exact inner-product search with FAISS."""
import json
import os
import time
from collections import Counter

import faiss
import numpy as np

from pathlib import Path

from .index import filter_sql

MODEL = "BAAI/bge-small-en-v1.5"
# bge-v1.5 recommends this instruction on short queries (not documents) for query->passage retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_model = None
_stores = {}


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL, device="cpu")
    return _model


def doc_text(msgs: list[list]) -> str:
    """Unique messages, most repeated first. bge truncates at 512 tokens, and a busy 30 s window can
    hold 200 messages; deduping keeps both what chat spammed and the variety of what was said."""
    return "\n".join(t for t, _ in Counter(m[2].strip() for m in msgs).most_common())


def vec_path(con) -> Path:
    """Vectors live beside their SQLite file (index.db -> index.npz), so each index has its own."""
    return Path(con.execute("PRAGMA database_list").fetchone()[2]).with_suffix(".npz")


def build(con, since: int | None = None) -> dict:
    """Embed windows missing from the store; with `since`, also re-embed windows starting at/after it."""
    t0, path = time.time(), vec_path(con)
    ids, vecs = np.array([], dtype=str), np.zeros((0, 384), dtype=np.float32)
    if since is not None and path.exists():
        with np.load(path) as z:  # closes the file; Windows can't replace an open file
            keep = np.array([int(i.rsplit(":", 1)[1]) < since for i in z["ids"]], dtype=bool)
            ids, vecs = z["ids"][keep], z["vecs"][keep]
    have = set(ids.tolist())
    todo = [(r[0], json.loads(r[1])) for r in con.execute("SELECT id, msgs FROM windows ORDER BY rowid") if r[0] not in have]
    if todo:
        new = model().encode([doc_text(m) for _, m in todo], batch_size=32, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=len(todo) > 500).astype(np.float32)
        ids, vecs = np.concatenate([ids, [i for i, _ in todo]]), np.vstack([vecs, new])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, ids=ids, vecs=vecs)
    os.replace(tmp, path)  # atomic, like the SQLite rebuild
    _stores.pop(path, None)
    return {"embedded": len(todo), "total": len(ids), "seconds": round(time.time() - t0, 2), "bytes": path.stat().st_size}


def _load(path: Path):
    if path not in _stores:
        with np.load(path) as npz:
            z = {"ids": npz["ids"], "vecs": npz["vecs"]}
        # shortcut: exact flat search; switch to HNSW/IVF past ~1M windows when brute force gets slow.
        idx = faiss.IndexFlatIP(z["vecs"].shape[1])
        idx.add(z["vecs"])
        _stores[path] = (idx, z["ids"].tolist(), {w: i for i, w in enumerate(z["ids"].tolist())})
    return _stores[path]


def encode_query(q: str) -> np.ndarray:
    return model().encode([QUERY_PREFIX + q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def search(con, q: str, k: int = 10, **filters) -> list[tuple[str, float]]:
    path = vec_path(con)
    if not path.exists():
        raise SystemExit("No dense vectors. Run: python -m twindex index")
    idx, ids, pos = _load(path)
    params = None
    if any(v is not None for v in filters.values()):
        where, args = filter_sql(**filters)
        allowed = [pos[r[0]] for r in con.execute(f"SELECT w.id FROM windows w WHERE 1{where}", args) if r[0] in pos]
        if not allowed:
            return []
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.array(allowed, dtype=np.int64)))
    scores, hits = idx.search(encode_query(q), k, params=params)
    return [(ids[h], float(s)) for h, s in zip(hits[0], scores[0]) if h >= 0]
