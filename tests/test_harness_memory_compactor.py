"""Memory compactor (t10) — ``reachy_nova/harness/memory_compactor.py``.

``MemoryCompactor`` is the distilled half of c11's memory story (the raw
half is ``Ledger``, t4): a background thread that periodically asks a fake
Nova 2 Lite what the last 24 h of the ledger were about, keeps the result at
a JSON file with per-entry timestamps, and expires anything older than
``max_age_s``. Four properties are load-bearing here and each has a test
group below:

1. **Off the hot threads (c28).** :meth:`compact` only ever runs on the
   compactor's own daemon thread once :meth:`start` is called.
2. **24 h expiry, bounded compaction cadence, and a final compaction at
   shutdown.**
3. **``history()``** renders the surviving memory as a USER-role context
   block plus the last few ledger exchanges, for Sonic's session-replay hook
   (c12) — capped in size, empty when nothing is remembered.
4. **A chatty model and a stale boot clock.** The parser tolerates trailing
   prose after the JSON object; the stale-clock guard (no RTC on the robot)
   skips expiry rather than wiping memory written under a clock NTP has not
   yet corrected.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time

import pytest

from reachy_nova.harness.ledger import Ledger
from reachy_nova.harness.memory_compactor import (
    MemoryCompactor,
    _extract_json_object,
    _merge_entries,
)

MODEL_ID = "us.amazon.nova-2-lite-v1:0"

_SENSE_LINE_RE = re.compile(
    r"^\[SENSE stage=(?P<stage>\S+) source=(?P<source>\S+) event=(?P<event>\S+)\] (?P<detail>.*)$"
)


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class FakeClock:
    """Injectable wall clock (epoch seconds)."""

    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeMonotonic:
    """Injectable monotonic clock the compactor's wait loop reads."""

    def __init__(self, t: float = 0.0):
        self.t = t
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.t

    def advance(self, dt: float) -> None:
        with self._lock:
            self.t += dt


class _FakeBody:
    def __init__(self, payload: dict | str):
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        self._raw = raw.encode("utf-8")

    def read(self) -> bytes:
        return self._raw


def _response(text: str) -> dict:
    return {"body": _FakeBody({"output": {"message": {"content": [{"text": text}]}}})}


class FakeBedrock:
    """A bedrock-runtime double: answers with fixed text, records call threads."""

    def __init__(self, text: str | None = None, fail: bool = False):
        self.text = text if text is not None else '{"topics": [], "items": []}'
        self.fail = fail
        self.calls: list[dict] = []
        self.call_threads: list[int] = []

    def invoke_model(self, modelId: str, body: str, **kwargs):  # noqa: N803 - boto3 casing
        self.call_threads.append(threading.get_ident())
        self.calls.append({"modelId": modelId, "body": json.loads(body)})
        if self.fail:
            raise RuntimeError("forced Lite failure")
        return _response(self.text)


def sense_lines(caplog) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if "[SENSE" in rec.getMessage()]


def compactor_lines(caplog) -> list[str]:
    return [line for line in sense_lines(caplog) if "event=compact]" in line]


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    assert predicate(), "condition never became true within the timeout"


# --------------------------------------------------------------------------- #
# 1. compact() runs on its own thread; writes the memory file atomically;    #
#    truncates the ledger.                                                   #
# --------------------------------------------------------------------------- #


def test_compact_runs_on_the_compactors_own_thread_when_started(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    ledger.append("USER", "let's talk about gardening", ts=1_000.0)
    ledger.append("ASSISTANT", "sure, what are you growing", ts=1_001.0)

    bedrock = FakeBedrock()
    compactor = MemoryCompactor(
        ledger,
        path=tmp_path / "memory.json",
        client=bedrock,
        model_id=MODEL_ID,
        interval_s=0.02,
        clock=FakeClock(2_000.0),
    )

    compactor.start(threading.Event())
    try:
        _wait_until(lambda: len(bedrock.call_threads) >= 1)
    finally:
        compactor.stop()

    caller_thread = threading.get_ident()
    assert bedrock.call_threads[0] != caller_thread
    assert bedrock.call_threads[0] == compactor._thread.ident  # type: ignore[attr-defined]


def test_compact_writes_memory_file_atomically_and_truncates_ledger(tmp_path):
    path = tmp_path / "memory.json"
    ledger_path = tmp_path / "ledger.jsonl"
    clock = FakeClock(2_000_000.0)
    ledger = Ledger(path=ledger_path, clock=clock)
    ledger.append("USER", "let's talk about the garden", ts=clock.t - 3600.0)
    ledger.append("ASSISTANT", "sure, what's growing", ts=clock.t - 3500.0)
    # old enough to be dropped by the 24h truncation compact() triggers.
    ledger.append("USER", "ancient small talk", ts=clock.t - 90_000.0)

    bedrock = FakeBedrock(
        text=json.dumps(
            {
                "topics": [{"text": "gardening"}],
                "items": [{"text": "wants tomato tips", "kind": "preference"}],
            }
        )
    )
    compactor = MemoryCompactor(
        ledger, path=path, client=bedrock, model_id=MODEL_ID, clock=clock
    )

    assert compactor.compact() is True

    # no leftover temp file from the atomic rewrite
    leftovers = [p for p in tmp_path.iterdir() if p.name not in {path.name, ledger_path.name}]
    assert leftovers == []

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["topics"] == [{"text": "gardening", "ts": clock.t}]
    assert data["items"] == [
        {"text": "wants tomato tips", "kind": "preference", "ts": clock.t}
    ]

    # the ledger was truncated to the last 24h by the same compaction.
    remaining = [json.loads(line)["text"] for line in ledger_path.read_text().splitlines()]
    assert remaining == ["let's talk about the garden", "sure, what's growing"]
    assert compactor.compactions == 1
    assert compactor.failures == 0


def test_compact_is_a_noop_on_an_empty_or_unchanged_ledger(tmp_path):
    clock = FakeClock(1_000.0)
    ledger = Ledger(path=tmp_path / "ledger.jsonl", clock=clock)
    bedrock = FakeBedrock()
    compactor = MemoryCompactor(ledger, path=tmp_path / "memory.json", client=bedrock, clock=clock)

    assert compactor.compact() is False
    assert bedrock.calls == []

    ledger.append("USER", "hello")
    assert compactor.compact() is True
    assert len(bedrock.calls) == 1

    # nothing new landed in the ledger since the last successful compaction.
    assert compactor.compact() is False
    assert len(bedrock.calls) == 1


# --------------------------------------------------------------------------- #
# 2. 24h expiry, Lite-failure safety, and compaction cadence.                #
# --------------------------------------------------------------------------- #


def test_entries_older_than_24h_are_absent_after_the_next_compaction(tmp_path):
    path = tmp_path / "memory.json"
    clock = FakeClock(1_000_000.0)
    ledger = Ledger(path=tmp_path / "ledger.jsonl", clock=clock)

    ledger.append("USER", "day one topic", ts=clock.t)
    bedrock = FakeBedrock(text=json.dumps({"topics": [{"text": "old-topic"}], "items": []}))
    compactor = MemoryCompactor(ledger, path=path, client=bedrock, clock=clock)
    assert compactor.compact() is True
    assert [t["text"] for t in compactor.memory()["topics"]] == ["old-topic"]

    # a day passes; new conversation happens, but the old topic must expire.
    clock.advance(86_401.0)
    ledger.append("USER", "day two topic", ts=clock.t)
    bedrock.text = json.dumps({"topics": [{"text": "new-topic"}], "items": []})
    assert compactor.compact() is True

    topics = [t["text"] for t in compactor.memory()["topics"]]
    assert "old-topic" not in topics
    assert "new-topic" in topics


def test_lite_failure_leaves_previous_file_intact_and_emits_one_drop_line(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    path = tmp_path / "memory.json"
    clock = FakeClock(500_000.0)
    ledger = Ledger(path=tmp_path / "ledger.jsonl", clock=clock)
    ledger.append("USER", "first thing", ts=clock.t)

    good_bedrock = FakeBedrock(text=json.dumps({"topics": [{"text": "first"}], "items": []}))
    compactor = MemoryCompactor(ledger, path=path, client=good_bedrock, clock=clock)
    assert compactor.compact() is True
    before = path.read_text(encoding="utf-8")

    clock.advance(10.0)
    ledger.append("USER", "second thing", ts=clock.t)
    compactor._client = FakeBedrock(fail=True)
    caplog.clear()

    assert compactor.compact() is False
    assert path.read_text(encoding="utf-8") == before
    assert compactor.failures == 1

    drop_lines = [
        line for line in compactor_lines(caplog) if "dropped reason=memory-compaction-failed" in line
    ]
    assert len(drop_lines) == 1


def test_unparseable_reply_leaves_previous_file_intact(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    path = tmp_path / "memory.json"
    clock = FakeClock(600_000.0)
    ledger = Ledger(path=tmp_path / "ledger.jsonl", clock=clock)
    ledger.append("USER", "first thing", ts=clock.t)

    good_bedrock = FakeBedrock(text=json.dumps({"topics": [{"text": "first"}], "items": []}))
    compactor = MemoryCompactor(ledger, path=path, client=good_bedrock, clock=clock)
    assert compactor.compact() is True
    before = path.read_text(encoding="utf-8")

    clock.advance(10.0)
    ledger.append("USER", "second thing", ts=clock.t)
    compactor._client = FakeBedrock(text="I refuse to answer in JSON today.")
    caplog.clear()

    assert compactor.compact() is False
    assert path.read_text(encoding="utf-8") == before
    assert compactor.failures == 1
    assert len(compactor_lines(caplog)) == 1


def test_compaction_runs_at_most_every_interval_and_once_at_shutdown(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    ledger.append("USER", "hi", ts=999_999.0)
    bedrock = FakeBedrock()
    compactor = MemoryCompactor(
        ledger,
        path=tmp_path / "memory.json",
        client=bedrock,
        interval_s=1000.0,
        clock=lambda: 1_000_000.0,
        monotonic=FakeMonotonic(0.0),
    )

    stop_event = threading.Event()
    compactor.start(stop_event)
    try:
        # no compaction has fired yet: the interval (fake monotonic time) has
        # not elapsed, no matter how much real wall-clock time passes.
        time.sleep(0.2)
        assert compactor.compactions == 0
        assert len(bedrock.calls) == 0
    finally:
        stop_event.set()
        compactor.stop()

    # exactly one compaction — the final one at shutdown.
    assert compactor.compactions == 1
    assert len(bedrock.calls) == 1


def test_compaction_fires_once_the_injected_interval_elapses(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    ledger.append("USER", "hi", ts=999_999.0)
    bedrock = FakeBedrock()
    monotonic = FakeMonotonic(0.0)
    compactor = MemoryCompactor(
        ledger,
        path=tmp_path / "memory.json",
        client=bedrock,
        interval_s=5.0,
        clock=lambda: 1_000_000.0,
        monotonic=monotonic,
    )

    compactor.start(threading.Event())
    try:
        time.sleep(0.1)
        assert compactor.compactions == 0

        monotonic.advance(5.1)
        _wait_until(lambda: compactor.compactions >= 1)
    finally:
        compactor.stop()


# --------------------------------------------------------------------------- #
# 3. history() — the session-replay view.                                    #
# --------------------------------------------------------------------------- #


def test_history_is_empty_with_no_memory_and_no_ledger(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    compactor = MemoryCompactor(ledger, path=tmp_path / "memory.json", client=FakeBedrock())
    assert compactor.history() == []


def test_history_returns_context_block_then_recent_exchanges_under_the_cap(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            {
                "topics": [{"text": "gardening", "ts": 1.0}],
                "items": [{"text": "wants tomato tips", "kind": "preference", "ts": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    ledger.append("USER", "what should I plant", ts=10.0)
    ledger.append("ASSISTANT", "tomatoes, obviously", ts=11.0)
    ledger.append("sense", "a pat", ts=12.0, source="touch")  # must be skipped
    ledger.append("USER", "anything else", ts=13.0)
    ledger.append("ASSISTANT", "basil, if you're feeling ambitious", ts=14.0)

    compactor = MemoryCompactor(ledger, path=path, client=FakeBedrock())
    blocks = compactor.history(max_chars=2000)

    # Bedrock's shape: USER first, roles alternating, ends on the assistant.
    assert [b["role"] for b in blocks] == ["USER", "ASSISTANT", "USER", "ASSISTANT"]
    assert blocks[0]["text"].startswith("(earlier today")
    assert "gardening" in blocks[0]["text"] and "wants tomato tips" in blocks[0]["text"]
    # only the last three exchanges are kept, so the context block stands alone
    assert blocks[0]["text"].endswith("spoken to.)")
    assert blocks[1]["text"] == "tomatoes, obviously"
    assert blocks[2]["text"] == "anything else"
    assert blocks[3]["text"] == "basil, if you're feeling ambitious"
    assert not any("a pat" in b["text"] for b in blocks)


def test_history_drops_oldest_exchanges_to_fit_a_tight_cap(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"topics": [{"text": "a topic", "ts": 1.0}], "items": []}), "utf-8")
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    ledger.append("USER", "x" * 50, ts=1.0)
    ledger.append("ASSISTANT", "y" * 50, ts=2.0)

    compactor = MemoryCompactor(ledger, path=path, client=FakeBedrock())
    blocks = compactor.history(max_chars=40)

    assert sum(len(b["text"]) for b in blocks) <= 40
    assert blocks  # the context block itself still survives, truncated


def test_history_without_distilled_memory_still_returns_recent_exchanges(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    ledger.append("USER", "hello there", ts=1.0)
    compactor = MemoryCompactor(ledger, path=tmp_path / "memory.json", client=FakeBedrock())

    blocks = compactor.history()

    # Even with nothing distilled the history opens with a USER context block
    # (a history that opens with the assistant kills the stream); the lone
    # exchange merges into it.
    assert len(blocks) == 1
    assert blocks[0]["role"] == "USER"
    assert blocks[0]["text"].startswith("(earlier today")
    assert blocks[0]["text"].endswith("hello there")


def test_history_never_opens_with_the_assistant(tmp_path):
    """The live incident: the ledger's first line was one of Nova's own reactions."""
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    ledger.append("ASSISTANT", "Thank you!", ts=1.0)
    ledger.append("ASSISTANT", "Lovely.", ts=2.0)
    compactor = MemoryCompactor(ledger, path=tmp_path / "memory.json", client=FakeBedrock())

    blocks = compactor.history()

    assert blocks and blocks[0]["role"] == "USER"
    assert [b["role"] for b in blocks] == ["USER", "ASSISTANT"]
    assert blocks[1]["text"] == "Thank you!\nLovely."


# --------------------------------------------------------------------------- #
# 4. Trailing prose after JSON; the stale-boot-clock guard.                  #
# --------------------------------------------------------------------------- #


def test_extract_json_object_ignores_trailing_prose():
    payload = {"topics": [{"text": "a"}], "items": []}
    text = json.dumps(payload) + "\n\n*Reasoning:* the user seemed happy about it."
    assert _extract_json_object(text) == payload


def test_extract_json_object_handles_braces_inside_strings():
    payload = {"topics": [{"text": "a {tricky} phrase"}], "items": []}
    text = json.dumps(payload) + " -- trailing note {not json"
    assert _extract_json_object(text) == payload


def test_extract_json_object_returns_none_for_no_object():
    assert _extract_json_object("no JSON here at all") is None


def test_compact_tolerates_a_reply_with_trailing_prose(tmp_path):
    path = tmp_path / "memory.json"
    clock = FakeClock(700_000.0)
    ledger = Ledger(path=tmp_path / "ledger.jsonl", clock=clock)
    ledger.append("USER", "let's talk shop", ts=clock.t)

    reply = (
        json.dumps({"topics": [{"text": "shop talk"}], "items": []})
        + "\n\n*Reasoning:* the conversation was about work."
    )
    bedrock = FakeBedrock(text=reply)
    compactor = MemoryCompactor(ledger, path=path, client=bedrock, clock=clock)

    assert compactor.compact() is True
    assert [t["text"] for t in compactor.memory()["topics"]] == ["shop talk"]


def test_stale_clock_guard_skips_expiry_when_clock_is_behind(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    path = tmp_path / "memory.json"
    # Memory already holds an entry timestamped in the "future" relative to
    # the clock this run observes — the no-RTC boot-clock case.
    path.write_text(
        json.dumps({"topics": [{"text": "future-topic", "ts": 2_000_000.0}], "items": []}),
        encoding="utf-8",
    )
    clock = FakeClock(1_000_000.0)  # far behind the newest memory entry
    ledger = Ledger(path=tmp_path / "ledger.jsonl", clock=clock)
    ledger.append("USER", "hello", ts=clock.t)

    bedrock = FakeBedrock(text=json.dumps({"topics": [{"text": "new-topic"}], "items": []}))
    compactor = MemoryCompactor(ledger, path=path, client=bedrock, clock=clock, max_age_s=100.0)

    assert compactor.compact() is True

    topics = [t["text"] for t in compactor.memory()["topics"]]
    # the pre-existing (far-future-timestamped) entry survives: expiry was
    # skipped for this run rather than wiping memory under a bad clock.
    assert "future-topic" in topics
    assert "new-topic" in topics
    assert compactor.expired == 0

    stale_lines = [line for line in compactor_lines(caplog) if "clock behind" in line]
    assert len(stale_lines) == 1

    # a second compaction under the SAME stale condition logs no further line.
    caplog.clear()
    clock.advance(1.0)
    ledger.append("USER", "hello again", ts=clock.t)
    bedrock.text = json.dumps({"topics": [{"text": "another-topic"}], "items": []})
    compactor.compact()
    assert [line for line in compactor_lines(caplog) if "clock behind" in line] == []


# --------------------------------------------------------------------------- #
# Pure-function coverage: merge/dedupe.                                      #
# --------------------------------------------------------------------------- #


def test_merge_entries_dedupes_by_normalised_text_and_keeps_earlier_ts():
    existing = [{"text": "Gardening", "ts": 10.0}]
    new = [{"text": "gardening  ", "ts": 20.0}]
    merged = _merge_entries(existing, new)
    assert len(merged) == 1
    assert merged[0]["ts"] == 10.0


def test_merge_entries_appends_genuinely_new_text():
    existing = [{"text": "gardening", "ts": 10.0}]
    new = [{"text": "cooking", "ts": 20.0}]
    merged = _merge_entries(existing, new)
    texts = {e["text"] for e in merged}
    assert texts == {"gardening", "cooking"}


@pytest.mark.parametrize("kind", ["request", "preference", "joke", "stop", "fact", "nonsense", ""])
def test_construction_accepts_all_documented_item_kinds_or_falls_back_to_fact(tmp_path, kind):
    from reachy_nova.harness.memory_compactor import _coerce_entries

    now = 42.0
    out = _coerce_entries([{"text": "a thing", "kind": kind}], now, with_kind=True)
    assert out[0]["kind"] in {"request", "preference", "joke", "stop", "fact"}



def test_malformed_persisted_entries_never_disable_replay(tmp_path):
    """Review thread 6: one junk element must not raise inside history()."""
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps({"topics": ["not-a-dict", {"text": ""}, {"text": "gardening", "ts": "bad"}, {"nope": 1}],
                    "items": [None, {"text": "  wants tips ", "ts": 5.0}]}),
        encoding="utf-8",
    )
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    compactor = MemoryCompactor(ledger, path=path, client=FakeBedrock())
    memory = compactor.memory()
    assert [e["text"] for e in memory["topics"]] == ["gardening"]
    assert memory["topics"][0]["ts"] == 0.0
    assert [e["text"] for e in memory["items"]] == ["wants tips"]
    blocks = compactor.history()
    assert blocks and "gardening" in blocks[0]["text"]
