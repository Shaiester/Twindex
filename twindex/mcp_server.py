"""MCP server (stdio) exposing search, get_window and describe over the local index.

Run: python -m twindex.mcp_server   (see README for the client config)
"""
import json

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import index
from .cli import _epoch, card, describe as describe_index
from .search import search as run_search
from .windows import utc

mcp = MCPServer("twitch-indexer", log_level="WARNING", instructions=(
    "Search 30-second Twitch chat windows. Call describe first to see channels, dates and games. "
    "search returns ranked cards with a VOD link at the moment; get_window returns every message in one window."))
READ_ONLY = ToolAnnotations(read_only_hint=True)
MAX_WINDOW_MSGS = 200


@mcp.tool(annotations=READ_ONLY)
def search(query: str, method: str = "hybrid", channel: str | None = None, game: str | None = None,
           since: str | None = None, until: str | None = None, k: int = 10, boost: bool = False,
           max_chars: int = 6000) -> dict:
    """Find chat moments matching a plain-language query.

    method: hybrid (best in eval), bm25 (exact words, emotes) or dense (paraphrase, slang).
    boost: rank moments where chat reacted strongly higher; helps 'when did chat go crazy' queries, hurts topical ones.
    since/until: UTC ISO date or datetime. Cards are cut off once their JSON exceeds max_chars.
    """
    if method not in ("bm25", "dense", "hybrid"):
        raise ValueError("method must be bm25, dense or hybrid")
    con = index.connect()
    hits = run_search(con, query, method=method, boost_on=boost, k=max(1, min(k, 50)), channel=channel, game=game,
                      since=_epoch(since), until=_epoch(until))
    wins = index.get_windows(con, [d for d, _ in hits]) if hits else {}
    terms = index.query_terms(query)
    cards, used = [], 0
    for d, s in hits:
        c = card(wins[d], terms) | {"score": round(s, 4)}
        size = len(json.dumps(c, ensure_ascii=False))
        if cards and used + size > max_chars:
            break
        cards.append(c)
        used += size
    return {"results": cards, "shown": len(cards), "matched": len(hits)}


@mcp.tool(annotations=READ_ONLY)
def get_window(id: str) -> dict:
    """Every message in one window (id like 'jynxzi:1791675420'), oldest first, capped at 200 messages."""
    w = index.get_windows(index.connect(), [id]).get(id)
    if not w:
        raise ValueError(f"no window {id!r}")
    msgs = [{"time": utc(ts // 1000), "user": user[:6], "text": text} for ts, user, text in w["msgs"][:MAX_WINDOW_MSGS]]
    return {k: w[k] for k in ("id", "channel", "game", "vod_url", "n_msgs", "n_users", "n_emotes", "activity", "events")} | {
        "time": utc(w["start"]), "messages": msgs, "shown": len(msgs), "total": w["n_msgs"]}


@mcp.tool(annotations=READ_ONLY)
def describe() -> dict:
    """Index coverage: channels and date ranges, games, field mapping, build time."""
    return describe_index(index.connect())


if __name__ == "__main__":
    mcp.run(transport="stdio")
