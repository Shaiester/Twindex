"""Helix metadata: VODs, clips, games (cached), and a stream poller for category/VOD timelines.

python -m twindex.helix fetch   # videos + clips for every configured channel
python -m twindex.helix poll    # every 5 min, log live stream state to data/streams.jsonl
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from .config import DATA, load_config, load_env

API = "https://api.twitch.tv/helix/"
CACHE = DATA / "helix_cache"
_token = None


def _app_token() -> str:
    """Client-credentials app token: Helix read endpoints need no user login."""
    global _token
    if _token is None:
        env = load_env()
        body = urllib.parse.urlencode({"client_id": env["TWITCH_CLIENT_ID"], "client_secret": env["TWITCH_CLIENT_SECRET"],
                                       "grant_type": "client_credentials"}).encode()
        with urllib.request.urlopen("https://id.twitch.tv/oauth2/token", body) as r:
            _token = json.load(r)["access_token"]
    return _token


def get(path: str, params: list[tuple], max_age: float = 6 * 3600) -> dict:
    """GET a Helix endpoint, caching the JSON on disk and pausing when the rate-limit bucket runs low."""
    url = API + path + "?" + urllib.parse.urlencode(params)
    cached = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if cached.exists() and time.time() - cached.stat().st_mtime < max_age:
        return json.loads(cached.read_text(encoding="utf-8"))
    for _ in range(5):
        token = _app_token()  # also loads .env
        req = urllib.request.Request(url, headers={"Client-Id": os.environ["TWITCH_CLIENT_ID"],
                                                   "Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req) as r:
                data, headers = json.load(r), r.headers
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            headers = e.headers
            time.sleep(max(1, int(headers.get("Ratelimit-Reset", time.time() + 5)) - time.time()))
    else:
        raise RuntimeError(f"rate limited 5 times: {url}")
    if int(headers.get("Ratelimit-Remaining", 100)) < 5:
        time.sleep(max(1, int(headers["Ratelimit-Reset"]) - time.time()))
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(data), encoding="utf-8")
    return data


def paginate(path: str, params: list[tuple], max_age: float = 6 * 3600, limit: int = 2000) -> list[dict]:
    out, cursor = [], None
    while len(out) < limit:
        page = get(path, params + ([("after", cursor)] if cursor else []), max_age)
        out += page["data"]
        cursor = page.get("pagination", {}).get("cursor")
        if not cursor or not page["data"]:
            break
    return out


def user_ids(logins: list[str]) -> dict[str, str]:
    data = get("users", [("login", l) for l in logins], max_age=float("inf"))["data"]
    return {u["login"]: u["id"] for u in data}


def fetch(days: int = 30):
    """Save VODs and clips per channel to data/helix/<channel>.json for the indexer."""
    out_dir = DATA / "helix"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    # One query per day: a range query returns ~1000 clips max sorted by views, so a 30-day range drops recent ones.
    # Always pass ended_at: without it Helix ends the range one week after started_at.
    days_ = [(now - timedelta(days=d + 1), now - timedelta(days=d)) for d in range(days)]
    for login, bid in user_ids(load_config()["channels"]).items():
        videos = paginate("videos", [("user_id", bid), ("type", "archive"), ("first", 100)], limit=300)
        clips = {}
        for s, e in days_:
            # Round to whole hours (end rounded up) so the cached URL is stable within an hour and covers "now".
            s = s.replace(minute=0, second=0, microsecond=0)
            e = e.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            for c in paginate("clips", [("broadcaster_id", bid), ("started_at", s.strftime(fmt)), ("ended_at", e.strftime(fmt)), ("first", 100)]):
                clips[c["id"]] = c
        clips = list(clips.values())
        game_ids = sorted({c["game_id"] for c in clips if c["game_id"]})
        games = {}
        for i in range(0, len(game_ids), 100):
            games |= {g["id"]: g["name"] for g in get("games", [("id", g) for g in game_ids[i:i + 100]], float("inf"))["data"]}
        (out_dir / f"{login}.json").write_text(json.dumps({"videos": videos, "clips": clips, "games": games}), encoding="utf-8")
        print(f"{login}: {len(videos)} videos, {len(clips)} clips")


def poll(every: int = 300):
    """Log which channels are live and in what category. Clips alone can't give a window's game."""
    ids = user_ids(load_config()["channels"])
    while True:
        try:
            live = get("streams", [("user_id", i) for i in ids.values()], max_age=0)["data"]
            with open(DATA / "streams.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": int(time.time() * 1000), "live": [
                    {k: s[k] for k in ("user_login", "id", "game_id", "game_name", "title", "started_at", "viewer_count")}
                    for s in live]}) + "\n")
        except (OSError, urllib.error.URLError) as e:
            print("poll error:", repr(e), file=sys.stderr)
        time.sleep(every)


if __name__ == "__main__":
    {"fetch": fetch, "poll": poll}[sys.argv[1]]()
