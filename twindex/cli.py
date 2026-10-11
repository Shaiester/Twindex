"""python -m twindex {index,search,get,describe}"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone

from . import index
from .windows import top_repeated, utc

CARD_SAMPLES = 5
MSG_CHARS = 120


def _epoch(s: str | None) -> int | None:
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()) if s else None


def card(w: dict, terms: list[str]) -> dict:
    """Compact result: what chat spammed, plus messages containing query terms."""
    repeated = [{"text": t[:MSG_CHARS], "count": c} for t, c in top_repeated(w["msgs"]) if c > 1]
    seen = {r["text"] for r in repeated}
    matching = [m[2][:MSG_CHARS] for m in w["msgs"]
                if set(terms) & set(re.findall(r"\w+", m[2].lower())) and m[2].strip().lower() not in seen]
    samples = (matching + [m[2][:MSG_CHARS] for m in w["msgs"] if m[2].strip().lower() not in seen])
    samples = list(dict.fromkeys(samples))[:CARD_SAMPLES - len(repeated)]
    return {"id": w["id"], "channel": w["channel"], "time": utc(w["start"]), "game": w["game"], "vod_url": w["vod_url"],
            "activity": w["activity"], "events": w["events"], "repeated": repeated, "samples": samples,
            "shown": sum(r["count"] for r in repeated) + len(samples), "total": w["n_msgs"]}


def render(c: dict, rank: int, score: float) -> str:
    head = f"[{rank}] {c['channel']} · {c['time']} · {c['game'] or 'unknown game'} · activity {c['activity']:+.1f} · score {score:.3f}"
    lines = [head, f"    id: {c['id']}"]
    if c["vod_url"]:
        lines.append(f"    vod: {c['vod_url']}")
    if c["events"]:
        lines.append(f"    events: {', '.join(sorted(set(c['events'])))}")
    if c["repeated"]:
        lines.append("    spam: " + " · ".join(f'"{r["text"]}" ×{r["count"]}' for r in c["repeated"]))
    lines += [f"    > {s}" for s in c["samples"]]
    lines.append(f"    shown {c['shown']} of {c['total']} messages")
    return "\n".join(lines)


def cmd_search(a):
    from .search import search
    con = index.connect()
    hits = search(con, a.query, method=a.method, boost_on=a.boost, k=a.k,
                  channel=a.channel, game=a.game, since=_epoch(a.since), until=_epoch(a.until))
    terms = index.query_terms(a.query)
    wins = index.get_windows(con, [h[0] for h in hits]) if hits else {}
    cards = [(card(wins[i], terms), s) for i, s in hits]
    if a.json:
        print(json.dumps([c | {"score": round(s, 4)} for c, s in cards], ensure_ascii=False, indent=1))
        return
    out, used = [], 0
    for rank, (c, s) in enumerate(cards, 1):
        text = render(c, rank, s)
        if used + len(text) > a.max_chars and out:  # budget for agents with small contexts (~4 chars/token)
            out.append(f"... {len(cards) - rank + 1} more results omitted (--max-chars {a.max_chars})")
            break
        out.append(text)
        used += len(text)
    print("\n\n".join(out) if out else "No results.")


def cmd_index(a):
    from . import dense
    if a.incremental:
        fts = index.update()
        since = _epoch(fts.get("since_day"))  # absent when update() fell back to a full rebuild
    else:
        fts, since = index.rebuild(), None
    print(json.dumps({"fts": fts, "dense": dense.build(index.connect(), since=since)}))


def cmd_get(a):
    w = index.get_windows(index.connect(), [a.id]).get(a.id)
    if not w:
        sys.exit(f"No window {a.id}")
    if a.json:
        print(json.dumps(w, ensure_ascii=False, indent=1))
        return
    print(f"{w['channel']} · {utc(w['start'])} · {w['game'] or 'unknown game'} · {w['n_msgs']} msgs, {w['n_users']} chatters, "
          f"{w['n_emotes']} emotes · activity {w['activity']:+.2f}")
    if w["vod_url"]:
        print(f"vod: {w['vod_url']}")
    for ts, user, text in w["msgs"]:
        print(f"  {datetime.fromtimestamp(ts / 1000, timezone.utc):%H:%M:%S} {user[:6]}  {text}")


def cmd_describe(a):
    print(json.dumps(describe(index.connect()), indent=1))


def describe(con) -> dict:
    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    rows = con.execute("""SELECT channel, count(*), sum(n_msgs), min(start), max(start) FROM windows GROUP BY channel""").fetchall()
    games = con.execute("SELECT game, count(*) FROM windows GROUP BY game ORDER BY 2 DESC").fetchall()
    info = {"built_at": meta.get("built_at"), "build_seconds": meta.get("build_seconds"),
            "record": "30-second chat window per channel (>= 3 messages)",
            "fields": {"id": "channel:window_start_epoch", "text": "window messages (FTS5, porter)", "group": "channel",
                       "date": "window start", "filters": "game, since/until"},
            "channels": [{"channel": r[0], "windows": r[1], "messages": r[2], "from": utc(r[3]), "to": utc(r[4])} for r in rows],
            "games": {g or "unknown": n for g, n in games}}
    return info


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252; chat is full of emoji
    p = argparse.ArgumentParser(prog="twindex", description="Search Twitch chat moments.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index", help="build the index (atomic full rebuild by default)")
    s.add_argument("--incremental", action="store_true", help="upsert windows from the newest indexed day onward")
    s = sub.add_parser("search", help="ranked result cards")
    s.add_argument("query")
    s.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25")
    s.add_argument("--boost", action="store_true", help="rank chat-activity spikes higher")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--channel")
    s.add_argument("--game")
    s.add_argument("--since", help="UTC ISO date/time")
    s.add_argument("--until", help="UTC ISO date/time")
    s.add_argument("--max-chars", type=int, default=6000)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("get", help="every message in one window")
    s.add_argument("id")
    s.add_argument("--json", action="store_true")
    sub.add_parser("describe", help="index stats and field mapping")
    a = p.parse_args(argv)
    if a.cmd == "index":
        cmd_index(a)
    else:
        {"search": cmd_search, "get": cmd_get, "describe": cmd_describe}[a.cmd](a)
