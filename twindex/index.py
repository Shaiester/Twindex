"""SQLite FTS5 index over chat windows: atomic full rebuilds, incremental upserts, BM25 search."""
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

from .config import DATA
from .windows import attach_metadata, build_windows, chat_files, read_records

DB_PATH = DATA / "index.db"
COLS = ["id", "channel", "start", "end", "game", "n_msgs", "n_users", "n_emotes", "activity", "events", "vod_url", "msgs"]
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS windows (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, channel TEXT, start INTEGER, end INTEGER,
    game TEXT, n_msgs INTEGER, n_users INTEGER, n_emotes INTEGER, activity REAL, events TEXT, vod_url TEXT, msgs TEXT);
CREATE INDEX IF NOT EXISTS windows_channel_start ON windows(channel, start);
-- fts.rowid == windows.rowid. Porter stemming so 'clutched'/'clutch' match.
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# Generic question words and meta words ('chat', 'react') appear in most queries but say nothing about the moment.
STOPWORDS = set("""a an the and or of to in on at for with from by is are was were be been it this that what when
where who how did does do why which chat chats reacted react reaction reacting moment moments stream streamer
time during about after before""".split())


def _upsert(con: sqlite3.Connection, windows: list[dict]):
    for w in windows:
        row = con.execute("SELECT rowid FROM windows WHERE id = ?", (w["id"],)).fetchone()
        if row:
            con.execute("DELETE FROM fts WHERE rowid = ?", row)
        vals = [json.dumps(w[c]) if c in ("events", "msgs") else w[c] for c in COLS]
        cur = con.execute(f"INSERT OR REPLACE INTO windows (rowid, {', '.join(COLS)}) VALUES (?, {', '.join('?' * len(COLS))})",
                          [row[0] if row else None, *vals])
        con.execute("INSERT INTO fts (rowid, text) VALUES (?, ?)", (cur.lastrowid, w["text"]))


def _windows_from(files, until: int | None = None) -> list[dict]:
    records = read_records(files)
    if until is not None:
        records = [r for r in records if r["ts"] < until * 1000]
    windows = build_windows(records)
    attach_metadata(windows)
    return windows


def rebuild(db_path=DB_PATH, until: int | None = None) -> dict:
    """Build into a temp file and swap it in, so readers never see a half-built index.

    `until` (epoch s) drops later chat, so the eval can rebuild its frozen corpus exactly.
    """
    t0 = time.time()
    tmp = db_path.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    files = chat_files()
    windows = _windows_from(files, until)
    con = sqlite3.connect(tmp)
    with con:
        con.executescript(SCHEMA)
        _upsert(con, windows)
        con.execute("INSERT INTO fts(fts) VALUES ('optimize')")
    secs = time.time() - t0
    with con:
        con.execute("INSERT OR REPLACE INTO meta VALUES ('built_at', ?), ('build_seconds', ?)",
                    (datetime.now(timezone.utc).isoformat(timespec="seconds"), f"{secs:.2f}"))
    con.close()
    os.replace(tmp, db_path)
    return {"windows": len(windows), "seconds": round(secs, 2), "bytes": db_path.stat().st_size}


def update(db_path=DB_PATH) -> dict:
    """Re-window every chat file from the newest indexed UTC day onward and upsert.

    Whole days, because activity baselines are per channel-day and the last window may have been partial.
    """
    if not db_path.exists():
        return rebuild(db_path)
    con = sqlite3.connect(db_path)
    last = con.execute("SELECT max(start) FROM windows").fetchone()[0] or 0
    day = datetime.fromtimestamp(last, timezone.utc).strftime("%Y-%m-%d")
    windows = _windows_from(chat_files(since_day=day))
    with con:
        _upsert(con, windows)
    con.close()
    return {"windows_upserted": len(windows), "since_day": day}


def connect(db_path=DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"No index at {db_path}. Run: python -m twindex index")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def query_terms(q: str) -> list[str]:
    return [t for t in re.findall(r"\w+", q.lower()) if t not in STOPWORDS]


def filter_sql(channel=None, game=None, since=None, until=None) -> tuple[str, list]:
    clauses, args = [], []
    for sql, val in (("w.channel = ?", channel), ("w.game = ? COLLATE NOCASE", game), ("w.start >= ?", since), ("w.start < ?", until)):
        if val is not None:
            clauses.append(sql)
            args.append(val)
    return "".join(" AND " + c for c in clauses), args


def bm25(con, q: str, k: int = 10, **filters) -> list[tuple[str, float]]:
    """Terms are OR'ed: natural-language queries rarely have every word in one window; BM25 ranks by overlap."""
    terms = query_terms(q)
    if not terms:
        return []
    where, args = filter_sql(**filters)
    rows = con.execute(f"""SELECT w.id, -bm25(fts) AS score FROM fts JOIN windows w ON w.rowid = fts.rowid
                           WHERE fts MATCH ?{where} ORDER BY bm25(fts) LIMIT ?""",
                       [" OR ".join(f'"{t}"' for t in terms), *args, k]).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_windows(con, ids: list[str]) -> dict[str, dict]:
    rows = con.execute(f"SELECT * FROM windows WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
    return {r["id"]: dict(r, msgs=json.loads(r["msgs"]), events=json.loads(r["events"])) for r in rows}
