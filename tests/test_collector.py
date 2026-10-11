from twindex.collector import _emote_names, parse_line

SALT = b"test"


def test_privmsg_with_tags_and_emotes():
    line = ("@emotes=25:0-4,12-16;id=abc;room-id=1;tmi-sent-ts=1700000000123;user-id=42 "
            ":u!u@u.tmi.twitch.tv PRIVMSG #jynxzi :Kappa hello Kappa")
    rec = parse_line(line, SALT)
    assert rec["type"] == "msg" and rec["channel"] == "jynxzi" and rec["ts"] == 1700000000123
    assert rec["text"] == "Kappa hello Kappa" and rec["emotes"] == ["Kappa", "Kappa"]
    assert rec["user"] != "42" and len(rec["user"]) == 16
    assert parse_line(line, SALT)["user"] == rec["user"]  # stable
    assert parse_line(line, b"other")["user"] != rec["user"]  # salted


def test_message_containing_space_colon():
    rec = parse_line("@user-id=1;tmi-sent-ts=1 :u!u@u PRIVMSG #c :ratio :) lol", SALT)
    assert rec["text"] == "ratio :) lol"


def test_raid_usernotice_unescapes_system_msg():
    line = (r"@msg-id=raid;msg-param-viewerCount=512;system-msg=512\sraiders\sfrom\sFoo\shave\sjoined!;"
            "tmi-sent-ts=5;user-id=9;room-id=1 :tmi.twitch.tv USERNOTICE #dantes")
    rec = parse_line(line, SALT)
    assert rec["type"] == "raid" and rec["viewers"] == 512 and rec["text"] == "512 raiders from Foo have joined!"


def test_ignored_lines():
    assert parse_line("PING :tmi.twitch.tv", SALT) is None
    assert parse_line(":tmi.twitch.tv 001 justinfan1 :Welcome, GLHF!", SALT) is None
    assert parse_line("@room-id=1 :tmi.twitch.tv ROOMSTATE #c", SALT) is None


def test_emote_names_counts_unicode_positions():
    # Twitch positions are code points; the emoji must not shift the slice.
    assert _emote_names("🔥 LUL", "425618:2-4") == ["LUL"]
    assert _emote_names("x", "") == []
