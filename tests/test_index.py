import sqlite3

from twindex import index


def make_db(tmp_path, texts):
    con = sqlite3.connect(tmp_path / "t.db")
    con.executescript(index.SCHEMA)
    ws = [{"id": f"c:{i}", "channel": "c", "start": i * 30, "end": i * 30 + 30, "game": "G", "n_msgs": 3, "n_users": 3,
           "n_emotes": 0, "activity": 0.0, "events": [], "vod_url": None, "msgs": [], "text": t} for i, t in enumerate(texts)]
    index._upsert(con, ws)
    return con


def test_bm25_ranks_overlap_and_ignores_stopwords(tmp_path):
    con = make_db(tmp_path, ["boss fight boss down", "nothing here", "the fight was close"])
    hits = index.bm25(con, "when did chat react to the boss fight?")
    assert [h[0] for h in hits] == ["c:0", "c:2"]
    assert hits[0][1] > hits[1][1]


def test_query_syntax_is_escaped(tmp_path):
    con = make_db(tmp_path, ['he said "NOT" AND left'])
    assert index.bm25(con, 'NOT AND "') and index.bm25(con, "???") == []


def test_upsert_replaces_without_duplicating_fts(tmp_path):
    con = make_db(tmp_path, ["old words"])
    index._upsert(con, [{"id": "c:0", "channel": "c", "start": 0, "end": 30, "game": None, "n_msgs": 3, "n_users": 1,
                         "n_emotes": 0, "activity": 0.0, "events": [], "vod_url": None, "msgs": [], "text": "new words"}])
    assert index.bm25(con, "old") == [] and len(index.bm25(con, "words")) == 1
    assert con.execute("SELECT count(*) FROM fts").fetchone()[0] == 1


def test_filters(tmp_path):
    con = make_db(tmp_path, ["clutch", "clutch"])
    assert [h[0] for h in index.bm25(con, "clutch", since=30)] == ["c:1"]
    assert index.bm25(con, "clutch", channel="other") == []
