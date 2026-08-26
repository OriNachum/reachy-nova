"""F6: a failed inject must never poison dedupe.

Before this fix, ``NovaBus._handle_message`` committed the dedupe timestamp
and the SenseHistory record BEFORE calling ``self._on_inject(...)``. If the
callback raised, the same key was suppressed for the rest of the dedupe
window (a real sense silently lost) and ``recall_senses`` would report a cue
Nova never actually received.

The fix reserves the dedupe key under the lock, calls ``on_inject`` outside
the lock, and only *commits* the reservation (and records history) once
``on_inject`` returns without raising; a raise rolls the reservation back so
the same key can retry immediately, and logs one
``[SENSE stage=inject ... event=inject-failed]`` line. Two concurrent
same-key events must still result in exactly one delivery.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from reachy_nova.harness import bus
from reachy_nova.harness.sense_history import SenseHistory

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "config" / "nervous-system" / "rules.yaml"


def fake_msg(topic: str, payload) -> SimpleNamespace:
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, separators=(",", ":"))
    if isinstance(payload, str):
        payload = payload.encode()
    return SimpleNamespace(topic=topic, payload=payload)


def pat_msg(rule_name: str, ts: float = 0.0):
    return fake_msg(
        "reachy/events/rule/fire",
        {"t": "rule", "ts": ts, "rule": rule_name},
    )


class FlakyRecorder:
    """Raises on the first N calls, then behaves like a normal recorder."""

    def __init__(self, fail_times: int = 1) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.injects: list[str] = []

    def on_inject(self, text: str) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("sonic is not ready")
        self.injects.append(text)


def make_bus(on_inject, **kwargs) -> bus.NovaBus:
    return bus.NovaBus(on_inject=on_inject, **kwargs)


def make_clock(times: list[float]):
    state = {"i": 0}

    def clock() -> float:
        i = min(state["i"], len(times) - 1)
        state["i"] += 1
        return times[i]

    return clock


# --------------------------------------------------------------------------- #
# A failing on_inject must not poison history or the dedupe window           #
# --------------------------------------------------------------------------- #


def test_failed_inject_records_no_history_entry():
    flaky = FlakyRecorder(fail_times=1)
    history = SenseHistory()
    nb = make_bus(flaky.on_inject, sources="rule", rules_path=RULES_PATH, history=history)

    nb.on_message(None, None, pat_msg("pat-acknowledge"))

    assert history.recent() == []


def test_failed_inject_allows_the_same_key_to_retry_immediately():
    flaky = FlakyRecorder(fail_times=1)
    clock = make_clock([0.0, 0.01])  # both calls land well inside the 10s window
    nb = make_bus(flaky.on_inject, sources="rule", rules_path=RULES_PATH, clock=clock)

    nb.on_message(None, None, pat_msg("pat-acknowledge"))  # raises, rolled back
    nb.on_message(None, None, pat_msg("pat-acknowledge"))  # must NOT be suppressed

    assert flaky.calls == 2
    assert flaky.injects == ["(someone is petting you) (react briefly if at all)"]


def test_successful_inject_after_a_prior_failure_records_history_once():
    flaky = FlakyRecorder(fail_times=1)
    history = SenseHistory()
    clock = make_clock([0.0, 0.01])
    nb = make_bus(
        flaky.on_inject, sources="rule", rules_path=RULES_PATH, history=history, clock=clock
    )

    nb.on_message(None, None, pat_msg("pat-acknowledge"))
    nb.on_message(None, None, pat_msg("pat-acknowledge"))

    assert len(history.recent()) == 1


def test_failed_inject_still_dedupes_a_genuine_duplicate_afterward():
    """Rollback only undoes THIS reservation — once a later call succeeds it
    reserves its own timestamp, and dedupe behaves normally from there."""
    flaky = FlakyRecorder(fail_times=1)
    clock = make_clock([0.0, 0.01, 0.02])
    nb = make_bus(flaky.on_inject, sources="rule", rules_path=RULES_PATH, clock=clock)

    nb.on_message(None, None, pat_msg("pat-acknowledge"))  # fails, rolled back
    nb.on_message(None, None, pat_msg("pat-acknowledge"))  # succeeds, reserves t=0.01
    nb.on_message(None, None, pat_msg("pat-acknowledge"))  # within window -> suppressed

    assert flaky.calls == 2
    assert len(flaky.injects) == 1


def test_failed_inject_logs_one_inject_failed_line(caplog):
    flaky = FlakyRecorder(fail_times=1)
    nb = make_bus(flaky.on_inject, sources="rule", rules_path=RULES_PATH)

    with caplog.at_level("INFO", logger="nova.sensory"):
        nb.on_message(None, None, pat_msg("pat-acknowledge"))

    failed_lines = [
        r.getMessage()
        for r in caplog.records
        if "[SENSE stage=inject source=nova event=rule/fire]" in r.getMessage()
        and "inject-failed" in r.getMessage()
    ]
    assert len(failed_lines) == 1


# --------------------------------------------------------------------------- #
# Concurrency: two same-key events racing in must deliver exactly once       #
# --------------------------------------------------------------------------- #


def test_two_concurrent_same_key_events_deliver_exactly_once():
    lock = threading.Lock()
    delivered: list[str] = []

    def on_inject(text: str) -> None:
        # Simulate real work so both threads are genuinely racing.
        with lock:
            delivered.append(text)

    nb = make_bus(on_inject, sources="rule", rules_path=RULES_PATH)

    barrier = threading.Barrier(2)

    def fire():
        barrier.wait(timeout=5)
        nb.on_message(None, None, pat_msg("pat-acknowledge"))

    threads = [threading.Thread(target=fire) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(delivered) == 1
