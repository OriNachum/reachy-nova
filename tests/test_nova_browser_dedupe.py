"""Tests for queue_task duplicate-task dedupe (issue #8, t2).

Live (2026-09-06), a fresh Sonic session that received a mid-flight
"Working on: X..." progress cue read it as a request and called the browse
tool again — a duplicate act triggered by a session restart. queue_task now
normalises the instruction and drops a repeat that matches the currently
running task, or any queued/recently-enqueued task within a 300 s window.
"""

from __future__ import annotations

import logging
import threading
import time

from reachy_nova import nova_browser
from reachy_nova.nova_browser import NovaBrowser


def _sensory_records(caplog):
    return [r for r in caplog.records if r.name == "nova.sensory"]


def _make_browser(monkeypatch, clock):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    return NovaBrowser(clock=clock)


def test_same_instruction_queued_twice_within_window_enqueues_once(monkeypatch, caplog):
    fake_now = [0.0]
    browser = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        first = browser.queue_task("search for the weather")
        fake_now[0] = 5.0
        second = browser.queue_task("search for the weather")

    assert first == {
        "ok": True,
        "queued": True,
        "instruction": "search for the weather",
        "url": None,
    }
    assert second == {
        "ok": True,
        "queued": False,
        "duplicate": True,
        "instruction": "search for the weather",
        "url": None,
    }
    assert browser._task_queue.qsize() == 1

    records = _sensory_records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "dropped reason=duplicate" in message
    assert "window=300s" in message


def test_whitespace_and_case_variant_is_a_duplicate(monkeypatch, caplog):
    fake_now = [0.0]
    browser = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        browser.queue_task("  Search FOR   the Weather  ")
        second = browser.queue_task("search for the weather")

    assert second["duplicate"] is True
    assert browser._task_queue.qsize() == 1


def test_after_window_elapses_it_enqueues_again(monkeypatch):
    fake_now = [0.0]
    browser = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    browser.queue_task("search for the weather")
    fake_now[0] = 300.1  # past DEDUPE_WINDOW_S
    second = browser.queue_task("search for the weather")

    assert second == {
        "ok": True,
        "queued": True,
        "instruction": "search for the weather",
        "url": None,
    }
    assert browser._task_queue.qsize() == 2


def test_different_instruction_is_not_a_duplicate(monkeypatch):
    fake_now = [0.0]
    browser = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    browser.queue_task("search for the weather")
    second = browser.queue_task("search for restaurants nearby")

    assert second["queued"] is True
    assert "duplicate" not in second
    assert browser._task_queue.qsize() == 2


def test_matches_currently_running_task_regardless_of_time_elapsed(monkeypatch, caplog):
    fake_now = [0.0]
    browser = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    # Simulate a task that's been running well past the dedupe window.
    browser.current_task = "search for the weather"
    browser.state = "busy"
    fake_now[0] = 10_000.0

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        result = browser.queue_task("search for the weather")

    assert result["duplicate"] is True
    assert browser._task_queue.qsize() == 0
    assert len(_sensory_records(caplog)) == 1


def test_concurrent_same_instruction_enqueues_exactly_once(monkeypatch):
    """PR #26 review, finding 6: queue_task must be one synchronized step.

    20 threads racing queue_task with the identical instruction must not
    both pass the duplicate test before either records it — exactly one
    task lands on the queue and exactly one caller sees ``queued: True``.
    """
    browser = _make_browser(monkeypatch, clock=time.monotonic)
    n = 20
    barrier = threading.Barrier(n)
    results: list[dict] = [None] * n  # type: ignore[list-item]

    def worker(i: int) -> None:
        barrier.wait()
        results[i] = browser.queue_task("search for the weather")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert browser._task_queue.qsize() == 1
    queued_results = [r for r in results if r.get("queued") is True]
    duplicate_results = [r for r in results if r.get("duplicate") is True]
    assert len(queued_results) == 1
    assert len(duplicate_results) == n - 1


def test_concurrent_different_instruction_still_queues(monkeypatch):
    """A different instruction racing alongside duplicates of another must
    still queue — the lock serializes access, it doesn't block distinct
    instructions from ever landing."""
    browser = _make_browser(monkeypatch, clock=time.monotonic)
    n = 10
    barrier = threading.Barrier(n)
    results: list[dict] = [None] * n  # type: ignore[list-item]

    def worker(i: int) -> None:
        instruction = "search for restaurants nearby" if i == 0 else "search for the weather"
        barrier.wait()
        results[i] = browser.queue_task(instruction)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert browser._task_queue.qsize() == 2
    assert results[0]["queued"] is True
    assert sum(1 for r in results[1:] if r.get("queued") is True) == 1


def test_queue_task_while_disabled_returns_ok_false_without_enqueueing(monkeypatch, caplog):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "0")
    browser = NovaBrowser()

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        result = browser.queue_task("search for the weather")

    assert result == {"ok": False, "queued": False, "reason": "nova-act-disabled"}
    assert browser._task_queue.empty()
    records = _sensory_records(caplog)
    assert len(records) == 1
    assert "dropped reason=nova-act-disabled" in records[0].getMessage()
