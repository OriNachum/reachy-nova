"""SenseHistory: the bounded, queryable record behind ``recall_senses`` (t8).

Ordering convention under test: :meth:`SenseHistory.recent` returns entries
NEWEST FIRST — index 0 is always the most recently recorded sense.
"""

from __future__ import annotations

import threading

from reachy_nova.harness.sense_history import SenseHistory


def make_clock(start: float = 0.0):
    state = {"t": start}

    def clock() -> float:
        return state["t"]

    def advance(delta: float) -> None:
        state["t"] += delta

    return clock, advance


def test_recent_is_empty_when_nothing_recorded():
    history = SenseHistory()
    assert history.recent() == []


def test_record_and_recent_round_trip_the_fields():
    clock, _advance = make_clock()
    history = SenseHistory(clock=clock)
    history.record("pat", "level1", "pat-acknowledge", "someone is petting you", "touch", "brief")
    (entry,) = history.recent()
    assert entry["source"] == "pat"
    assert entry["type"] == "level1"
    assert entry["rule"] == "pat-acknowledge"
    assert entry["text"] == "someone is petting you"
    assert entry["sense_class"] == "touch"
    assert entry["voice"] == "brief"
    assert entry["t"] == 0.0
    assert entry["age_s"] == 0.0


def test_recent_returns_newest_first():
    clock, advance = make_clock()
    history = SenseHistory(clock=clock)
    history.record("pat", "level1", None, "first", None, None)
    advance(1.0)
    history.record("face", "recognized", None, "second", None, None)
    advance(1.0)
    history.record("rule", "fire", None, "third", None, None)

    texts = [e["text"] for e in history.recent()]
    assert texts == ["third", "second", "first"]


def test_recent_timestamps_are_monotonic_in_recording_order():
    clock, advance = make_clock()
    history = SenseHistory(clock=clock)
    history.record("a", "x", None, "one", None, None)
    advance(0.5)
    history.record("b", "y", None, "two", None, None)
    advance(0.5)
    history.record("c", "z", None, "three", None, None)

    # newest-first, so timestamps DEscend as you walk the list.
    ts = [e["t"] for e in history.recent()]
    assert ts == sorted(ts, reverse=True)


def test_recent_n_limits_the_count():
    clock, advance = make_clock()
    history = SenseHistory(clock=clock)
    for i in range(5):
        history.record("s", "t", None, f"text-{i}", None, None)
        advance(1.0)
    texts = [e["text"] for e in history.recent(2)]
    assert texts == ["text-4", "text-3"]


def test_age_s_is_computed_at_read_time():
    clock, advance = make_clock()
    history = SenseHistory(clock=clock)
    history.record("s", "t", None, "hello", None, None)
    advance(3.5)
    (entry,) = history.recent()
    assert entry["age_s"] == 3.5
    advance(1.5)
    (entry2,) = history.recent()
    assert entry2["age_s"] == 5.0


def test_bounded_at_maxlen_keeps_only_the_last_entries():
    history = SenseHistory(maxlen=20)
    for i in range(25):
        history.record("s", "t", None, f"text-{i}", None, None)
    entries = history.recent(20)
    assert len(entries) == 20
    # The oldest 5 (text-0..text-4) were evicted; the newest is text-24.
    texts = [e["text"] for e in entries]
    assert texts[0] == "text-24"
    assert "text-4" not in texts
    assert "text-0" not in texts
    assert "text-5" in texts


def test_recent_default_n_is_five():
    history = SenseHistory(maxlen=20)
    for i in range(10):
        history.record("s", "t", None, f"text-{i}", None, None)
    assert len(history.recent()) == 5


def test_record_is_thread_safe():
    history = SenseHistory(maxlen=200)

    def worker(offset: int) -> None:
        for i in range(50):
            history.record("s", "t", None, f"{offset}-{i}", None, None)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(history.recent(200)) == 200
