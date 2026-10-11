from twindex.windows import _parse_duration, build_windows, game_at, vod_link


def msg(ts_s, user="u", text="hi", channel="c", emotes=()):
    return {"ts": int(ts_s * 1000), "channel": channel, "user": user, "type": "msg", "text": text, "emotes": list(emotes)}


def test_tumbling_windows_aligned_and_min_size():
    recs = [msg(60), msg(61), msg(89.9), msg(90), msg(95)]  # 3 in [60,90), 2 in [90,120)
    ws = build_windows(recs)
    assert [w["id"] for w in ws] == ["c:60"]
    assert ws[0]["end"] == 90 and ws[0]["n_msgs"] == 3


def test_channels_window_separately_and_events_join_text():
    recs = [msg(0, channel="a")] * 3 + [msg(0, channel="b")] * 3
    recs.append({"ts": 1000, "channel": "a", "user": None, "type": "raid", "text": "500 raiders from X have joined!"})
    ws = {w["id"]: w for w in build_windows(recs)}
    assert set(ws) == {"a:0", "b:0"}
    assert ws["a:0"]["events"] == ["raid"] and "raiders" in ws["a:0"]["text"] and ws["a:0"]["n_msgs"] == 3


def test_activity_is_relative_to_channel_day():
    quiet = [msg(t * 30 + i, user=f"u{i}") for t in range(10) for i in range(3)]
    spike = [msg(300 + i * 0.1, user=f"s{i}", emotes=["KEKW"]) for i in range(40)]
    ws = {w["start"]: w for w in build_windows(quiet + spike)}
    assert ws[300]["activity"] > 1.5 and abs(ws[0]["activity"]) < 0.5
    assert ws[300]["n_users"] == 40 and ws[300]["n_emotes"] == 40


def test_vod_link_offset_and_bounds():
    videos = [(1000, 7200, "v1"), (10000, 60, "v2")]
    assert vod_link(1000 + 3723, videos) == "https://www.twitch.tv/videos/v1?t=1h2m3s"
    assert vod_link(999, videos) is None
    assert vod_link(1000 + 7200, videos) is None  # past the end of v1, before v2
    assert vod_link(10030, videos).endswith("v2?t=0h0m30s")


def test_duration_and_game_lookup():
    assert _parse_duration("5h31m13s") == 19873 and _parse_duration("45s") == 45
    polls = [(0, "Just Chatting"), (300, "Elden Ring")]
    assert game_at(299, polls) == "Just Chatting" and game_at(400, polls) == "Elden Ring"
    assert game_at(2000, polls) is None  # stale poll: channel probably offline
