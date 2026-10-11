"""Collect live Twitch chat to daily JSONL files via anonymous read-only IRC.

Run: python -m twindex.collector
"""
import hashlib
import hmac
import json
import logging
import socket
import ssl
import time
from datetime import datetime, timezone

from .config import DATA, hash_salt, load_config

HOST, PORT = "irc.chat.twitch.tv", 6697
CHAT_DIR = DATA / "chat"
log = logging.getLogger("collector")

_UNESCAPE = {"s": " ", ":": ";", "\\": "\\", "r": "\r", "n": "\n"}


def _unescape(v: str) -> str:
    out, i = [], 0
    while i < len(v):
        if v[i] == "\\" and i + 1 < len(v):
            out.append(_UNESCAPE.get(v[i + 1], v[i + 1]))
            i += 2
        else:
            out.append(v[i])
            i += 1
    return "".join(out)


def _emote_names(text: str, tag: str) -> list[str]:
    """'25:0-4,6-10/1902:12-16' -> one name per occurrence (repeats count toward emote bursts)."""
    names = []
    for part in filter(None, tag.split("/")):
        for rng in part.partition(":")[2].split(","):
            start, _, end = rng.partition("-")
            names.append(text[int(start) : int(end) + 1])
    return names


def parse_line(line: str, salt: bytes) -> dict | None:
    """Turn one IRC line into a record, or None for lines we don't keep."""
    tags = {}
    if line.startswith("@"):
        raw, _, line = line.partition(" ")
        for kv in raw[1:].split(";"):
            k, _, v = kv.partition("=")
            tags[k] = _unescape(v)
    if line.startswith(":"):
        line = line.partition(" ")[2]
    head, _, trailing = line.partition(" :")
    parts = head.split()
    if len(parts) < 2 or parts[0] not in ("PRIVMSG", "USERNOTICE"):
        return None
    user_id = tags.get("user-id", "")
    rec = {
        "ts": int(tags.get("tmi-sent-ts") or time.time() * 1000),
        "channel": parts[1].lstrip("#"),
        "room_id": tags.get("room-id"),
        "id": tags.get("id"),
        # Salted so hashes can't be reversed by enumerating public user IDs.
        "user": hmac.new(salt, user_id.encode(), hashlib.sha256).hexdigest()[:16] if user_id else None,
    }
    if parts[0] == "PRIVMSG":
        rec |= {"type": "msg", "text": trailing, "emotes": _emote_names(trailing, tags.get("emotes", ""))}
    else:  # sub, resub, raid, subgift, announcement, ...
        rec |= {"type": tags.get("msg-id", "usernotice"), "text": tags.get("system-msg", ""), "user_text": trailing}
        if "msg-param-viewerCount" in tags:
            rec["viewers"] = int(tags["msg-param-viewerCount"])
    return rec


class DailyWriter:
    """Appends records to data/chat/YYYY-MM-DD.jsonl, by each record's UTC date."""

    def __init__(self):
        CHAT_DIR.mkdir(parents=True, exist_ok=True)
        self.day, self.f = None, None

    def write(self, rec: dict):
        day = datetime.fromtimestamp(rec["ts"] / 1000, timezone.utc).strftime("%Y-%m-%d")
        if day != self.day:
            if self.f:
                self.f.close()
            self.day, self.f = day, open(CHAT_DIR / f"{day}.jsonl", "a", encoding="utf-8")
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()


def run_once(channels: list[str], salt: bytes, writer: DailyWriter):
    sock = ssl.create_default_context().wrap_socket(socket.create_connection((HOST, PORT), timeout=30), server_hostname=HOST)
    # Twitch PINGs about every 5 minutes; silence longer than this means a dead connection.
    sock.settimeout(400)
    joins = ",".join("#" + c.lower() for c in channels)
    sock.sendall(f"CAP REQ :twitch.tv/tags twitch.tv/commands\r\nNICK justinfan{int(time.time()) % 100000}\r\nJOIN {joins}\r\n".encode())
    log.info("connected, joined %s", joins)
    with sock.makefile("r", encoding="utf-8", errors="replace", newline="\r\n") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("PING"):
                sock.sendall(("PONG" + line[4:] + "\r\n").encode())
            elif " RECONNECT" in line[:40]:
                log.info("server asked us to reconnect")
                return
            elif rec := parse_line(line, salt):
                writer.write(rec)
    log.info("connection closed by server")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(DATA / "collector.log"), logging.StreamHandler()])
    channels, salt, writer = load_config()["channels"], hash_salt(), DailyWriter()
    backoff = 1
    while True:
        started = time.time()
        try:
            run_once(channels, salt, writer)
        except (OSError, ssl.SSLError) as e:
            log.warning("connection error: %r", e)
        backoff = 1 if time.time() - started > 60 else min(backoff * 2, 300)
        time.sleep(backoff)


if __name__ == "__main__":
    main()
