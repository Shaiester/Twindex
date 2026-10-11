"""Build the pooled, frozen eval set.

python -m twindex.evalset sample    # freeze corpus cutoff, build eval index, stratified sample, drafting sheet
  (LLM, in-session: write eval/queries.jsonl from data/eval/drafting.md)
python -m twindex.evalset pool      # top 5 from every method config -> data/eval/pool.md for pre-labeling
  (LLM, in-session: write eval/llm_labels.tsv)
python -m twindex.evalset label     # blind human check: every LLM 2 + a seeded 20% of the rest
python -m twindex.evalset freeze    # agreement report, 30/70 dev/test split, TREC qrels, checksums

Files under eval/ hold ids, queries and labels only (safe to commit). Raw chat stays under data/ (gitignored).
"""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

from . import index
from .config import DATA, ROOT
from .windows import top_repeated, utc

EVAL = ROOT / "eval"
WORK = DATA / "eval"
DB = WORK / "index.db"
META = EVAL / "meta.json"
SEED = 13
CONFIGS = [(m, b) for m in ("bm25", "dense", "hybrid") for b in (False, True)]
POOL_DEPTH = 5


def _meta() -> dict:
    return json.loads(META.read_text()) if META.exists() else {}


def _save_meta(**kw):
    EVAL.mkdir(exist_ok=True)
    META.write_text(json.dumps(_meta() | kw, indent=1))


def _require_unfrozen():
    if _meta().get("frozen_at") and "--force" not in sys.argv:
        sys.exit("Eval set is frozen; refusing to modify it (pass --force if you really mean to).")


def read_queries() -> list[dict]:
    """Published queries, with any locally kept private text swapped in for searching.

    Query text that names a chatter is redacted in eval/queries.jsonl; the real text lives in the gitignored
    data/eval/query_overrides.json so local reruns still reproduce the scored results. `public_query` keeps the
    published text for anything written to results/.
    """
    queries = [json.loads(l) for l in open(EVAL / "queries.jsonl", encoding="utf-8") if l.strip()]
    overrides = json.loads((WORK / "query_overrides.json").read_text()) if (WORK / "query_overrides.json").exists() else {}
    for q in queries:
        q["public_query"] = q["query"]
        q["query"] = overrides.get(q["qid"], q["query"])
    return queries


def build_eval_index():
    """The eval corpus is all chat before the frozen cutoff, rebuilt from raw files every time."""
    from . import dense
    WORK.mkdir(parents=True, exist_ok=True)
    fts = index.rebuild(DB, until=_meta()["cutoff"])
    vec = dense.build(index.connect(DB))
    return {"fts": fts, "dense": vec}


def stratified_sample(windows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Equal share per channel, split evenly across within-channel activity terciles (quiet/normal/loud)."""
    by_channel = defaultdict(list)
    for w in windows:
        by_channel[w["channel"]].append(w)
    per_channel = n // len(by_channel)
    out = []
    for ws in by_channel.values():
        ws = sorted(ws, key=lambda w: w["activity"])
        terciles = [ws[i * len(ws) // 3:(i + 1) * len(ws) // 3] for i in range(3)]
        for i, t in enumerate(terciles):
            take = per_channel // 3 + (1 if i < per_channel % 3 else 0)
            out += rng.sample(t, min(take, len(t)))
    return out


def sample(n: int = 150):
    _require_unfrozen()
    cutoff = int(datetime.now(timezone.utc).timestamp()) // 30 * 30
    _save_meta(cutoff=cutoff, cutoff_utc=utc(cutoff), seed=SEED)
    print(build_eval_index())
    con = index.connect(DB)
    windows = [dict(r) for r in con.execute("SELECT id, channel, activity FROM windows")]
    picked = stratified_sample(windows, n, random.Random(SEED))
    full = index.get_windows(con, [w["id"] for w in picked])
    lines = ["# Drafting sheet (raw chat: do not commit)\n"]
    for w in sorted(full.values(), key=lambda w: w["id"]):
        uniq = list(dict.fromkeys(m[2] for m in w["msgs"]))
        lines += [f"## {w['id']}", f"{w['channel']} · {utc(w['start'])} · {w['game']} · activity {w['activity']:+.2f} · "
                  f"{w['n_msgs']} msgs · events {w['events'] or '-'}",
                  "spam: " + " · ".join(f"{t!r}×{c}" for t, c in top_repeated(w["msgs"], 5)),
                  *[f"- {m[:160]}" for m in uniq[:30]], ""]
    (WORK / "drafting.md").write_text("\n".join(lines), encoding="utf-8")
    (EVAL / "sample.txt").write_text("\n".join(sorted(full)) + "\n")
    print(f"sampled {len(full)} windows -> {WORK / 'drafting.md'}")


def run_all(con, q: dict, k: int) -> dict[str, list[tuple[str, float]]]:
    from .search import search
    filt = {"channel": q.get("channel")}
    return {f"{m}{'+boost' if b else ''}": search(con, q["query"], method=m, boost_on=b, k=k, **filt) for m, b in CONFIGS}


def pool():
    _require_unfrozen()
    con = index.connect(DB)
    out, n_pairs = ["# Pool for pre-labeling (raw chat: do not commit)\n"], 0
    pool_rows = []
    for q in read_queries():
        docs = list(dict.fromkeys([q["source"]] + [d for run in run_all(con, q, POOL_DEPTH).values() for d, _ in run]))
        pool_rows += [(q["qid"], d) for d in docs]
        out.append(f"## {q['qid']} [{q['type']}] {q['query']}  (channel: {q.get('channel') or 'any'})")
        for w in index.get_windows(con, docs).values():
            uniq = list(dict.fromkeys(m[2][:100] for m in w["msgs"]))
            out.append(f"- {w['id']} | {w['game']} | spam: " + "; ".join(f"{t}×{c}" for t, c in top_repeated(w["msgs"], 3))
                       + " | " + " / ".join(uniq[:12]))
        out.append("")
        n_pairs += len(docs)
    (WORK / "pool.md").write_text("\n".join(out), encoding="utf-8")
    (EVAL / "pool.tsv").write_text("".join(f"{q}\t{d}\n" for q, d in pool_rows))
    print(f"pooled {n_pairs} (query, window) pairs -> {WORK / 'pool.md'}")


def read_labels(path) -> dict[tuple[str, str], int]:
    if not path.exists():
        return {}
    return {(q, d): int(r) for q, d, r in (l.split("\t")[:3] for l in path.read_text().splitlines() if l.strip())}


def to_verify(llm: dict, rate: float = 0.2) -> list[tuple[str, str]]:
    """All LLM 2s, plus a seeded random `rate` of the 0s and 1s."""
    rng = random.Random(SEED)
    return [p for p in sorted(llm) if llm[p] == 2 or rng.random() < rate]


GUIDE = """How relevant is this chat window to the query? Judge only what you see on screen.

  2  This window IS the moment the query describes.
     query "chat celebrating a viewer getting unbanned" + window with "HE UNBANNED", "PagMan I'M UNBANNED"  -> 2
  1  Partly: same topic but weaker, or the build-up/aftermath of the moment.
     same query + window with "request unban", "im banned too"  -> 1
  0  Not it: unrelated chat, even if it shares a word.
     same query + window about a champion being banned in draft  -> 0

Most windows are a 0. Expect a few 2s per query.
"""


def label():
    """Blind: the LLM's label is never shown, so the agreement number isn't inflated by anchoring."""
    sys.stdout.reconfigure(encoding="utf-8")
    llm, path = read_labels(EVAL / "llm_labels.tsv"), EVAL / "human_labels.tsv"
    done = read_labels(path)
    todo = [p for p in to_verify(llm) if p not in done]
    queries = {q["qid"]: q for q in read_queries()}
    con = index.connect(DB)
    print(GUIDE)
    print(f"{len(done)} done, {len(todo)} to go.\n")
    with open(path, "a") as f:
        for n, (qid, doc) in enumerate(todo, 1):
            q, w = queries[qid], index.get_windows(con, [doc])[doc]
            print(f"\n[{n}/{len(todo)}] QUERY: {q['query']}   (channel: {q.get('channel') or 'any'})")
            print(f"  {w['channel']} · {utc(w['start'])} · {w['game']} · {w['n_msgs']} msgs · events {w['events'] or '-'}")
            print("  spam: " + " · ".join(f"{t!r}×{c}" for t, c in top_repeated(w["msgs"], 4)))
            for m in list(dict.fromkeys(m[2] for m in w["msgs"]))[:20]:
                print(f"   - {m[:140]}")
            while (ans := input("  2 = this IS the moment · 1 = partly/nearby · 0 = not it · q = quit > ").strip()) not in ("0", "1", "2", "q"):
                pass
            if ans == "q":
                break
            f.write(f"{qid}\t{doc}\t{ans}\n")
            f.flush()


def cohen_kappa(a: list[int], b: list[int], weights: str | None = None) -> float:
    """Cohen's kappa over labels {0,1,2}; weights='quadratic' treats 0 vs 2 as worse than 0 vs 1."""
    cats = [0, 1, 2]
    w = {(i, j): ((i - j) ** 2 / 4 if weights == "quadratic" else float(i != j)) for i in cats for j in cats}
    n = len(a)
    obs = sum(w[x, y] for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    exp = sum(w[i, j] * pa[i] * pb[j] for i in cats for j in cats) / n ** 2
    return 1 - obs / exp if exp else 1.0


def agreement(llm: dict, human: dict) -> dict:
    pairs = [p for p in human if p in llm]
    a, b = [llm[p] for p in pairs], [human[p] for p in pairs]
    by_llm = {lab: round(sum(1 for x, y in zip(a, b) if x == lab and x == y) / max(1, a.count(lab)), 3) for lab in (0, 1, 2)}
    return {"n": len(pairs), "exact": round(sum(x == y for x, y in zip(a, b)) / len(pairs), 3),
            "binary_rel_ge1": round(sum((x > 0) == (y > 0) for x, y in zip(a, b)) / len(pairs), 3),
            "kappa": round(cohen_kappa(a, b), 3), "kappa_quadratic": round(cohen_kappa(a, b, "quadratic"), 3),
            "exact_by_llm_label": by_llm,
            "confusion_llm_x_human": {f"{i}->{j}": sum(1 for x, y in zip(a, b) if (x, y) == (i, j)) for i in (0, 1, 2) for j in (0, 1, 2)}}


def split(queries: list[dict], dev_frac: float = 0.3) -> dict[str, str]:
    """Seeded 30/70 split, stratified by query type so dev and test have the same mix."""
    rng, out = random.Random(SEED), {}
    by_type = defaultdict(list)
    for q in queries:
        by_type[q["type"]].append(q["qid"])
    for qids in by_type.values():
        qids = sorted(qids)
        rng.shuffle(qids)
        n_dev = round(len(qids) * dev_frac)
        out |= {q: ("dev" if i < n_dev else "test") for i, q in enumerate(qids)}
    return out


def freeze():
    _require_unfrozen()
    llm, human = read_labels(EVAL / "llm_labels.tsv"), read_labels(EVAL / "human_labels.tsv")
    missing = [p for p in to_verify(llm) if p not in human]
    if missing and "--force" not in sys.argv:
        sys.exit(f"{len(missing)} pairs still need human labels. Run: python -m twindex.evalset label")
    final = llm | human  # the human label wins wherever one exists
    queries = read_queries()
    splits = split(queries)
    for name in ("dev", "test"):
        rows = sorted((q, d, r) for (q, d), r in final.items() if splits[q] == name)
        (EVAL / f"qrels.{name}.txt").write_text("".join(f"{q} 0 {d} {r}\n" for q, d, r in rows))
    (EVAL / "splits.json").write_text(json.dumps(splits, indent=0, sort_keys=True))
    agree = agreement(llm, human)
    (EVAL / "agreement.json").write_text(json.dumps(agree, indent=1))
    digest = eval_digest()
    _save_meta(frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), sha256=digest)
    print(json.dumps(agree, indent=1))
    print(f"frozen: {Counter(splits.values())}")


def eval_digest() -> str:
    """SHA-256 of queries + qrels with line endings normalized, so a git checkout on another OS still matches."""
    data = b"".join((EVAL / f).read_bytes() for f in ("queries.jsonl", "qrels.dev.txt", "qrels.test.txt"))
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def read_qrels(name: str) -> dict[str, dict[str, int]]:
    qrels = defaultdict(dict)
    for line in (EVAL / f"qrels.{name}.txt").read_text().splitlines():
        q, _, d, r = line.split()
        qrels[q][d] = int(r)
    return dict(qrels)


def check_frozen():
    meta = _meta()
    digest = eval_digest()
    if not meta.get("frozen_at") or digest != meta["sha256"]:
        sys.exit("Eval set is not frozen or was modified after freezing.")


if __name__ == "__main__":
    if sys.argv[1] == "sample" and len(sys.argv) > 2 and sys.argv[2].isdigit():
        sample(int(sys.argv[2]))
    else:
        {"sample": sample, "pool": pool, "label": label, "freeze": freeze}[sys.argv[1]]()
