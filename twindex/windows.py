"""Group chat records into fixed 30 s windows per channel and attach activity, game and VOD metadata."""
import json
import math
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA

MIN_MESSAGES = 3  # 1-2 message windows are mostly noise and would dominate the corpus by count


def read_records(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # partial last line if the collector was mid-write
                    pass
    return out


def build_windows(records: list[dict], size_s: int = 30) -> list[dict]:
    """Tumbling windows aligned to the epoch, so ids (channel:start) are stable across rebuilds."""
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["channel"], r["ts"] // 1000 // size_s * size_s)].append(r)
    windows = []
    for (channel, start), recs in sorted(buckets.items()):
        recs.sort(key=lambda r: r["ts"])
        msgs = [r for r in recs if r["type"] == "msg"]
        if len(msgs) < MIN_MESSAGES:
            continue
        events = [r for r in recs if r["type"] != "msg"]
        windows.append({
            "id": f"{channel}:{start}",
            "channel": channel,
            "start": start,
            "end": start + size_s,
            "n_msgs": len(msgs),
            "n_users": len({m["user"] for m in msgs}),
            "n_emotes": sum(len(m["emotes"]) for m in msgs),
            "events": [e["type"] for e in events],
            # Event system text ("X is raiding with 500 viewers") is searchable alongside chat.
            "text": "\n".join([m["text"] for m in msgs] + [e["text"] for e in events if e["text"]]),
            "msgs": [[m["ts"], m["user"], m["text"]] for m in msgs],
        })
    add_activity(windows)
    return windows


def add_activity(windows: list[dict]):
    """activity = mean robust z-score of log(msgs, users, emotes) within the window's channel-day.

    Per channel-day because raw volume differs ~10x between channels and across days; a 'reaction'
    is volume relative to that stream's own baseline.
    """
    groups = defaultdict(list)
    for w in windows:
        groups[(w["channel"], w["start"] // 86400)].append(w)
    for ws in groups.values():
        zs = []
        for key in ("n_msgs", "n_users", "n_emotes"):
            vals = [math.log1p(w[key]) for w in ws]
            med = statistics.median(vals)
            mad = statistics.median(abs(v - med) for v in vals) or 1.0
            zs.append([(v - med) / (1.4826 * mad) for v in vals])
        for w, *z in zip(ws, *zs):
            w["activity"] = round(sum(z) / len(z), 3)


def _parse_duration(d: str) -> int:
    """Helix '1h2m3s' -> seconds."""
    total, num = 0, ""
    for ch in d:
        if ch.isdigit():
            num += ch
        else:
            total += int(num) * {"h": 3600, "m": 60, "s": 1}[ch]
            num = ""
    return total


def _iso(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())


def vod_link(start: int, videos: list[tuple[int, int, str]]) -> str | None:
    """videos: sorted (created_epoch, duration_s, id). Link to the VOD covering `start`, at that offset."""
    i = bisect_right(videos, (start, math.inf)) - 1
    if i < 0:
        return None
    created, dur, vid = videos[i]
    off = start - created
    if off >= dur:
        return None
    return f"https://www.twitch.tv/videos/{vid}?t={off // 3600}h{off // 60 % 60}m{off % 60}s"


def load_metadata() -> tuple[dict, dict]:
    """Per channel: sorted VOD spans, and sorted (poll_epoch, game_name) from the stream poller."""
    games, last_live = defaultdict(list), {}
    path = DATA / "streams.jsonl"
    if path.exists():
        for poll in read_records([path]):
            for s in poll["live"]:
                games[s["user_login"]].append((poll["ts"] // 1000, s["game_name"]))
                last_live[s["id"]] = poll["ts"] // 1000
    videos = {}
    for p in (DATA / "helix").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        spans = []
        for v in d["videos"]:
            start, dur = _iso(v["created_at"]), _parse_duration(v["duration"])
            # A live stream's VOD keeps growing after we cached its duration; extend it to the last poll that saw it live (+1 poll).
            if v.get("stream_id") in last_live:
                dur = max(dur, last_live[v["stream_id"]] - start + 300)
            spans.append((start, dur, v["id"]))
        videos[p.stem] = sorted(spans)
    return videos, games


def game_at(start: int, polls: list[tuple[int, str]], max_gap: int = 600) -> str | None:
    """Category from the nearest poll at or before `start` (polls run every 5 min)."""
    i = bisect_right(polls, (start, "￿")) - 1
    return polls[i][1] if i >= 0 and start - polls[i][0] <= max_gap else None


def attach_metadata(windows: list[dict]):
    videos, games = load_metadata()
    for w in windows:
        w["vod_url"] = vod_link(w["start"], videos.get(w["channel"], []))
        w["game"] = game_at(w["start"], games.get(w["channel"], []))


def chat_files(since_day: str | None = None) -> list[Path]:
    return sorted(p for p in (DATA / "chat").glob("*.jsonl") if since_day is None or p.stem >= since_day)


def top_repeated(msgs: list[list], n: int = 3) -> list[tuple[str, int]]:
    """Most repeated messages (case-folded): what chat was spamming."""
    return Counter(m[2].strip().lower() for m in msgs).most_common(n)


def utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
