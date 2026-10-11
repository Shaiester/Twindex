"""Reproduce every eval number: python -m twindex.eval  (tune the boost on dev: python -m twindex.eval tune)

Rebuilds the frozen eval corpus from raw chat, runs all six method configs on the test queries, and writes
results/results.json, results/results.md, results/per_query.csv, results/runs/*.trec and results/ndcg.png.
"""
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict

from . import index
from .config import ROOT
from .evalset import CONFIGS, DB, build_eval_index, check_frozen, read_qrels, read_queries
from .search import search
from .windows import load_metadata

RESULTS = ROOT / "results"
REL = 1  # a window counts as relevant for Recall/MRR at label >= 1


def ndcg(ranked: list[str], qrel: dict[str, int], k: int = 10) -> float:
    """Graded gain 2^rel - 1; the ideal ranking is built from every judged window for the query."""
    dcg = sum((2 ** qrel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall(ranked: list[str], qrel: dict[str, int], k: int = 20) -> float:
    rel = {d for d, r in qrel.items() if r >= REL}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def mrr(ranked: list[str], qrel: dict[str, int]) -> float:
    return next((1 / (i + 1) for i, d in enumerate(ranked) if qrel.get(d, 0) >= REL), 0.0)


def judged(ranked: list[str], qrel: dict[str, int], k: int = 10) -> float:
    """Share of the top k that has a label: low values mean the pool under-covers this method."""
    return sum(d in qrel for d in ranked[:k]) / max(1, min(k, len(ranked)))


def permutation_test(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> float:
    """Two-sided paired randomization test on the mean difference (randomly swap each query's pair)."""
    rng, diffs = random.Random(seed), [x - y for x, y in zip(a, b)]
    observed = abs(sum(diffs))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def run(con, queries: list[dict], weight: float | None = None, configs=CONFIGS):
    """-> {config: {qid: ranked ids}}, {config: [latency ms]}"""
    runs, lat = defaultdict(dict), defaultdict(list)
    search(con, "warmup", method="hybrid", boost_on=True)  # load model + vectors outside the timings
    for m, b in configs:
        name = f"{m}{'+boost' if b else ''}"
        for q in queries:
            t0 = time.perf_counter()
            hits = search(con, q["query"], method=m, boost_on=b, k=20, weight=weight, channel=q.get("channel"))
            lat[name].append((time.perf_counter() - t0) * 1000)
            runs[name][q["qid"]] = [d for d, _ in hits]
    return runs, lat


def score(runs, qrels) -> dict[str, dict[str, list[float]]]:
    """-> {config: {metric: per-query values, in qid order}}"""
    qids = sorted(qrels)
    return {name: {"ndcg@10": [ndcg(r.get(q, []), qrels[q]) for q in qids],
                   "recall@20": [recall(r.get(q, []), qrels[q]) for q in qids],
                   "mrr": [mrr(r.get(q, []), qrels[q]) for q in qids],
                   "judged@10": [judged(r.get(q, []), qrels[q]) for q in qids]} for name, r in runs.items()}


def pct(xs: list[float], p: float) -> float:
    return statistics.quantiles(xs, n=100, method="inclusive")[p - 1]


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def clip_alignment(con, runs: dict, slack: int = 30) -> dict:
    """Do loud windows and top results coincide with moments viewers clipped?

    A clip covers [vod start + vod_offset, + duration]; chat reacts a few seconds late (stream delay), so a
    window aligns if it overlaps the clip span extended by `slack` seconds. Compared against the base rate
    of all windows in the same channels, so 'lift' > 1 means better than chance.
    """
    videos, _ = load_metadata()
    spans = defaultdict(list)
    for p in (ROOT / "data" / "helix").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        vstart = {v[2]: v[0] for v in videos.get(p.stem, [])}
        for c in d["clips"]:
            if c.get("vod_offset") is not None and c["video_id"] in vstart:
                s = vstart[c["video_id"]] + c["vod_offset"]
                spans[p.stem].append((s, s + math.ceil(c["duration"]) + slack))
    rows = con.execute("SELECT id, channel, start, end, activity FROM windows").fetchall()
    lo = {ch: min(s for s, _ in sp) for ch, sp in spans.items()}
    hi = {ch: max(e for _, e in sp) for ch, sp in spans.items()}
    # Only windows inside the period the fetched clips cover, or misses would just mean 'no clips fetched yet'.
    rows = [r for r in rows if r["channel"] in spans and lo[r["channel"]] <= r["start"] <= hi[r["channel"]]]
    if not rows:
        return {"note": "no clips overlap the eval corpus yet"}
    aligned = {r["id"]: any(s < r["end"] and r["start"] < e for s, e in spans[r["channel"]]) for r in rows}
    base = statistics.fmean(aligned.values())
    top = sorted(rows, key=lambda r: -r["activity"])[: max(1, len(rows) // 50)]
    out = {"windows_in_clip_period": len(rows), "clips": sum(map(len, spans.values())), "base_rate": round(base, 3),
           "top2pct_activity": round(statistics.fmean(aligned[r["id"]] for r in top), 3)}
    for name, r in runs.items():
        tops = [ids[0] for ids in r.values() if ids and ids[0] in aligned]
        if tops:
            out[f"top1_{name}"] = round(statistics.fmean(aligned[t] for t in tops), 3)
    out["lift_top2pct_activity"] = round(out["top2pct_activity"] / base, 2) if base else None
    return out


def chart(summary: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)[::-1]  # first config on top
    vals = [summary[n]["ndcg@10"] for n in names]
    lo = [summary[n]["ndcg@10"] - summary[n]["ndcg@10_ci"][0] for n in names]
    hi = [summary[n]["ndcg@10_ci"][1] - summary[n]["ndcg@10"] for n in names]
    surface, ink, ink2, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.barh(names, vals, height=0.55, color=blue)
    ax.errorbar(vals, names, xerr=[lo, hi], fmt="none", ecolor=ink2, elinewidth=1, capsize=3)
    for y, (v, h) in enumerate(zip(vals, hi)):
        ax.text(v + h + 0.01, y, f"{v:.3f}", va="center", fontsize=9, color=ink)
    ax.set_xlim(0, min(1, max(v + h for v, h in zip(vals, hi)) + 0.12))
    ax.set_title("nDCG@10 on the test split (95% bootstrap CI)", loc="left", fontsize=11, color=ink)
    ax.tick_params(colors=ink2, labelsize=9, length=0)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def tune():
    """Grid-search the boost weight on the dev split only (test is never looked at here)."""
    check_frozen()
    con = index.connect(DB)
    qrels = read_qrels("dev")
    queries = [q for q in read_queries() if q["qid"] in qrels]
    grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    table = {}
    for w in grid:
        runs, _ = run(con, queries, weight=w, configs=[(m, True) for m in ("bm25", "dense", "hybrid")])
        table[w] = {n: round(statistics.fmean(v["ndcg@10"]), 4) for n, v in score(runs, qrels).items()}
        print(w, table[w])
    best = max(grid, key=lambda w: statistics.fmean(table[w].values()))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tuning.json").write_text(json.dumps({"dev_ndcg@10_by_weight": table, "best": best}, indent=1))
    print(f"best boost_weight on dev: {best}  -> set [ranking] boost_weight in config.toml")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    check_frozen()
    build = build_eval_index()
    con = index.connect(DB)
    qrels = read_qrels("test")
    queries = [q for q in read_queries() if q["qid"] in qrels]
    runs, lat = run(con, queries)
    per = score(runs, qrels)
    summary = {}
    for name, m in per.items():
        summary[name] = {k: round(statistics.fmean(v), 4) for k, v in m.items()}
        summary[name]["ndcg@10_ci"] = [round(x, 4) for x in bootstrap_ci(m["ndcg@10"])]
        summary[name]["p_vs_bm25_ndcg"] = round(permutation_test(m["ndcg@10"], per["bm25"]["ndcg@10"]), 4) if name != "bm25" else None
        summary[name]["latency_ms_p50"] = round(pct(lat[name], 50), 2)
        summary[name]["latency_ms_p95"] = round(pct(lat[name], 95), 2)
    qtype = {q["qid"]: q["type"] for q in queries}
    qids = sorted(qrels)
    by_type = {name: {t: round(statistics.fmean(v for q, v in zip(qids, m["ndcg@10"]) if qtype[q] == t), 4)
                      for t in sorted(set(qtype.values()))} for name, m in per.items()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "runs").mkdir(exist_ok=True)
    for name, r in runs.items():  # TREC run format, so results can be checked with trec_eval
        (RESULTS / "runs" / f"{name}.trec").write_text("".join(
            f"{q} Q0 {d} {i + 1} {20 - i} {name}\n" for q in sorted(r) for i, d in enumerate(r[q])))
    with open(RESULTS / "per_query.csv", "w", encoding="utf-8") as f:
        f.write("qid,type,query," + ",".join(f"{n}_ndcg10" for n in per) + "\n")
        qtext = {q["qid"]: q["public_query"] for q in queries}  # never write private query text
        for i, q in enumerate(qids):
            f.write(f"{q},{qtype[q]},\"{qtext[q].replace(chr(34), chr(39))}\"," + ",".join(f"{per[n]['ndcg@10'][i]:.4f}" for n in per) + "\n")
    out = {"n_test_queries": len(queries), "build": build, "summary": summary, "ndcg@10_by_query_type": by_type,
           "clip_alignment": clip_alignment(con, runs)}
    (RESULTS / "results.json").write_text(json.dumps(out, indent=1))
    lines = ["| method | nDCG@10 (95% CI) | Recall@20 | MRR | judged@10 | p vs BM25 | p50 ms | p95 ms |", "|" + "---|" * 8]
    for n, s in summary.items():
        lines.append(f"| {n} | {s['ndcg@10']:.3f} ({s['ndcg@10_ci'][0]:.3f}–{s['ndcg@10_ci'][1]:.3f}) | {s['recall@20']:.3f} | "
                     f"{s['mrr']:.3f} | {s['judged@10']:.2f} | {s['p_vs_bm25_ndcg'] if s['p_vs_bm25_ndcg'] is not None else '—'} | "
                     f"{s['latency_ms_p50']:.1f} | {s['latency_ms_p95']:.1f} |")
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    chart(summary, RESULTS / "ndcg.png")
    print("\n".join(lines))
    print(json.dumps({"build": build, "by_type": by_type, "clip_alignment": out["clip_alignment"]}, indent=1))


if __name__ == "__main__":
    tune() if sys.argv[1:] == ["tune"] else main()
