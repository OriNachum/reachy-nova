"""LockState: the harness's own belief about the runtime's gaze lock (t13).

The runtime does not mirror lock state into state.json the way it mirrors
``intents.inhibitions``, so the harness keeps a small in-process belief
instead, updated from (a) confirmed lock_face/release_face tool results and
(b) the runtime's own motion/lock-released bus events. It is read-only color
for supervisor.status() and never gates a tool call — an engine restart just
clears the belief so the next lock_face is not blocked by anything stale.
"""

from __future__ import annotations

import logging

from reachy_nova.harness.lock_state import LockState


def test_starts_unknown():
    assert LockState().locked is None


def test_mark_locked_then_released():
    state = LockState()
    state.mark_locked()
    assert state.locked is True
    state.mark_released()
    assert state.locked is False


def test_on_bus_event_marks_released_on_lock_released():
    state = LockState()
    state.mark_locked()

    state.on_bus_event({"source": "motion", "type": "lock-released", "reason": "max-hold"})

    assert state.locked is False


def test_on_bus_event_ignores_unrelated_events():
    state = LockState()
    state.mark_locked()

    state.on_bus_event({"source": "motion", "type": "goto"})
    state.on_bus_event({"source": "rule", "type": "fire"})
    state.on_bus_event({})
    state.on_bus_event(None)  # type: ignore[arg-type]

    assert state.locked is True


def test_on_engine_dropped_logs_once_and_clears_belief_when_locked(caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    state = LockState()
    state.mark_locked()

    state.on_engine_dropped()

    assert state.locked is None
    assert (
        "[SENSE stage=supervise source=nova event=lock] released reason=engine-restart"
        in caplog.text
    )


def test_on_engine_dropped_is_silent_when_nothing_was_locked(caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    state = LockState()

    state.on_engine_dropped()

    assert state.locked is None
    assert "event=lock" not in caplog.text


def test_on_engine_dropped_logs_only_once_across_repeated_drops(caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    state = LockState()
    state.mark_locked()

    state.on_engine_dropped()
    state.on_engine_dropped()
    state.on_engine_dropped()

    assert caplog.text.count("event=lock") == 1
