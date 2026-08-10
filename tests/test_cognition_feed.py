"""Tests for reachy_nova.harness.cognition_feed.CognitionFeed.

Covers the wire contract documented in reachy-mini-cli's
``docs/export-schema.md`` (cognition feed section): one compact JSON object
per line, ``t`` then ``ts`` first, ``ensure_ascii=False``, and exactly the
three block shapes ``thinking`` / ``message`` / ``emotion``.

Acceptance is nailed down two ways:

1. **Byte-exact golden lines** for fixed ``ts`` inputs — this IS the
   bridge-parses-unmodified contract: if reachy-export-bridge.py's ``Feed``
   accepts reachy-mini-cli's own example lines (it does, by construction of
   the schema), and our lines are byte-identical in shape/field-order/
   formatting, the bridge accepts ours too.
2. **A live parse against the actual bridge script**, when it's present on
   this machine (``~/.claude/skills/reterminal/scripts/reachy-export-bridge.py``).
   Skipped (not failed) elsewhere, since that script lives outside this repo.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import threading
from pathlib import Path

import pytest

from reachy_nova.harness.cognition_feed import CognitionFeed

BRIDGE_SCRIPT = Path(
    os.path.expanduser("~/.claude/skills/reterminal/scripts/reachy-export-bridge.py")
)


# ---------------------------------------------------------------------------
# Golden / byte-exact line shape
# ---------------------------------------------------------------------------


def test_thinking_line_is_byte_exact():
    buf = io.StringIO()
    feed = CognitionFeed(buf)
    feed.thinking(
        'apply_pose({"emoji": "🤔"}) speak({"text": "I heard something."})',
        cues=["speech from the left"],
        ts=1718362800.1,
    )
    assert buf.getvalue() == (
        '{"t":"thinking","ts":1718362800.1,"cues":["speech from the left"],'
        '"text":"apply_pose({\\"emoji\\": \\"\U0001f914\\"}) '
        'speak({\\"text\\": \\"I heard something.\\"})"}\n'
    )


def test_message_line_is_byte_exact():
    buf = io.StringIO()
    feed = CognitionFeed(buf)
    feed.message("I heard something.", ts=1718362800.5)
    assert buf.getvalue() == '{"t":"message","ts":1718362800.5,"text":"I heard something."}\n'


def test_emotion_line_is_byte_exact_with_pose():
    buf = io.StringIO()
    feed = CognitionFeed(buf)
    feed.emotion(
        "🤔",
        pose={"head_pitch": -5.0, "antenna_l": 30.0, "antenna_r": -30.0},
        ts=1718362800.2,
    )
    assert buf.getvalue() == (
        '{"t":"emotion","ts":1718362800.2,"emoji":"\U0001f914",'
        '"pose":{"head_pitch":-5.0,"antenna_l":30.0,"antenna_r":-30.0}}\n'
    )


def test_emotion_line_null_pose_is_byte_exact():
    buf = io.StringIO()
    feed = CognitionFeed(buf)
    feed.emotion("🙂", ts=1718362800.2)
    assert buf.getvalue() == '{"t":"emotion","ts":1718362800.2,"emoji":"\U0001f642","pose":null}\n'


# ---------------------------------------------------------------------------
# Field/format details
# ---------------------------------------------------------------------------


def test_key_order_is_t_then_ts():
    buf = io.StringIO()
    CognitionFeed(buf).message("hi", ts=1.0)
    line = buf.getvalue().rstrip("\n")
    assert line.startswith('{"t":"message","ts":1.0,')


def test_compact_separators_no_spaces():
    buf = io.StringIO()
    CognitionFeed(buf).thinking("x", cues=["a", "b"], ts=1.0)
    line = buf.getvalue().rstrip("\n")
    assert " " not in line


def test_emoji_kept_literal_not_escaped():
    buf = io.StringIO()
    CognitionFeed(buf).emotion("🤔", ts=1.0)
    line = buf.getvalue()
    assert "🤔" in line
    assert "\\u" not in line


def test_thinking_default_cues_is_empty_list():
    buf = io.StringIO()
    CognitionFeed(buf).thinking("no cues this turn", ts=1.0)
    obj = json.loads(buf.getvalue())
    assert obj["cues"] == []


def test_ts_defaults_to_time_time(monkeypatch):
    monkeypatch.setattr("reachy_nova.harness.cognition_feed.time.time", lambda: 42.5)
    buf = io.StringIO()
    CognitionFeed(buf).message("hi")
    obj = json.loads(buf.getvalue())
    assert obj["ts"] == 42.5


def test_each_call_writes_exactly_one_line():
    buf = io.StringIO()
    feed = CognitionFeed(buf)
    feed.thinking("t", ts=1.0)
    feed.message("m", ts=2.0)
    feed.emotion("🙂", ts=3.0)
    lines = buf.getvalue().splitlines()
    assert len(lines) == 3
    assert [json.loads(l)["t"] for l in lines] == ["thinking", "message", "emotion"]


def test_only_documented_block_types_emitted():
    buf = io.StringIO()
    feed = CognitionFeed(buf)
    feed.thinking("t", ts=1.0)
    feed.message("m", ts=2.0)
    feed.emotion("e", ts=3.0)
    for line in buf.getvalue().splitlines():
        assert json.loads(line)["t"] in {"thinking", "message", "emotion"}


# ---------------------------------------------------------------------------
# Target handling
# ---------------------------------------------------------------------------


def test_writes_to_path_target(tmp_path):
    p = tmp_path / "feed.ndjson"
    feed = CognitionFeed(p)
    feed.message("hello", ts=1.0)
    feed.close()
    assert p.read_text(encoding="utf-8") == '{"t":"message","ts":1.0,"text":"hello"}\n'


def test_path_target_appends_across_instances(tmp_path):
    p = tmp_path / "feed.ndjson"
    CognitionFeed(p).message("first", ts=1.0)
    CognitionFeed(p).message("second", ts=2.0)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_defaults_to_stdout(capsys):
    feed = CognitionFeed()
    feed.message("hi", ts=1.0)
    captured = capsys.readouterr()
    assert captured.out == '{"t":"message","ts":1.0,"text":"hi"}\n'


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_concurrent_writes_are_not_interleaved():
    buf = io.StringIO()
    feed = CognitionFeed(buf)
    n_threads = 16
    per_thread = 25

    def worker(idx: int) -> None:
        for i in range(per_thread):
            feed.message(f"thread-{idx}-{i}", ts=float(idx * 1000 + i))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = buf.getvalue().splitlines()
    assert len(lines) == n_threads * per_thread
    seen = set()
    for line in lines:
        obj = json.loads(line)  # would raise on any interleaved/corrupted line
        assert obj["t"] == "message"
        seen.add(obj["text"])
    assert len(seen) == n_threads * per_thread


# ---------------------------------------------------------------------------
# Bridge acceptance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not BRIDGE_SCRIPT.exists(),
    reason="reterminal bridge script not present on this machine",
)
def test_bridge_parser_accepts_sample_output_unmodified():
    spec = importlib.util.spec_from_file_location("reachy_export_bridge", BRIDGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    buf = io.StringIO()
    feed = CognitionFeed(buf)
    feed.emotion("🙂", ts=1718362800.2)
    feed.message("I heard something.", ts=1718362800.5)
    feed.thinking("turn complete", cues=["speech from the left"], ts=1718362800.6)

    bridge_feed = module.Feed(max_events=5)
    accepted = [bridge_feed.apply(line) for line in buf.getvalue().splitlines()]

    # emotion -> False (updates icon, no bullet), message -> True, thinking -> False (not a bullet)
    assert accepted == [False, True, False]
    assert bridge_feed.events == [("🙂", "I heard something.", 1718362800.5)]
