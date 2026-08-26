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

import pytest

from reachy_nova.harness.lock_state import LockState


class FakeClock:
    """Injectable monotonic clock — no sleeping, no wall time."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


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
    """Finding L6 changed WHEN this clears: the drop arms, the grace fires it.
    With the grace explicitly at zero this is the original edge behaviour."""
    caplog.set_level(logging.INFO, logger="nova.sensory")
    state = LockState(drop_grace_s=0.0)
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
    state = LockState(drop_grace_s=0.0)
    state.mark_locked()

    state.on_engine_dropped()
    state.on_engine_dropped()
    state.on_engine_dropped()

    assert caplog.text.count("event=lock") == 1


# --------------------------------------------------------------------------- #
# Live finding L6 (on-device run 2026-08-26 18:26–18:31)                      #
#                                                                             #
# On the loaded CM4 the engine heartbeat flapped live/lost about every 2 s,   #
# so the edge-triggered clear above threw the lock belief away two seconds    #
# after every lock — while the RUNTIME lock was held perfectly well. The      #
# flapping is a pre-existing load problem on the device; the belief must stop #
# believing the flap.                                                         #
# --------------------------------------------------------------------------- #


def test_l6_a_drop_within_the_grace_leaves_the_belief_intact(caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    clock = FakeClock()
    state = LockState(clock=clock, drop_grace_s=5.0)
    state.mark_locked()

    state.on_engine_dropped()
    clock.advance(1.0)
    state.on_engine_live()  # the heartbeat came back — it only flickered
    clock.advance(60.0)

    assert state.locked is True
    assert "event=lock" not in caplog.text


def test_l6_a_drop_past_the_grace_clears_the_belief_exactly_once(caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    clock = FakeClock()
    state = LockState(clock=clock, drop_grace_s=5.0)
    state.mark_locked()

    state.on_engine_dropped()
    assert state.locked is True  # still believed inside the grace
    clock.advance(6.0)

    assert state.locked is None
    assert state.settle() is False  # already settled; no second line
    assert caplog.text.count("event=lock") == 1
    assert "released reason=engine-restart" in caplog.text


def test_l6_settle_is_what_fires_the_drop_so_a_poll_loop_can_drive_it(caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    clock = FakeClock()
    state = LockState(clock=clock, drop_grace_s=5.0)
    state.mark_locked()
    state.on_engine_dropped()

    clock.advance(4.9)
    assert state.settle() is False
    clock.advance(0.2)
    assert state.settle() is True


def test_l6_repeated_flapping_never_shortens_the_grace():
    """Each live edge cancels; the NEXT drop starts the grace over."""
    clock = FakeClock()
    state = LockState(clock=clock, drop_grace_s=5.0)
    state.mark_locked()

    for _ in range(10):
        state.on_engine_dropped()
        clock.advance(2.0)
        state.on_engine_live()

    assert state.locked is True


def test_l6_a_relock_cancels_a_pending_drop():
    clock = FakeClock()
    state = LockState(clock=clock, drop_grace_s=5.0)
    state.mark_locked()
    state.on_engine_dropped()
    clock.advance(4.0)

    state.mark_locked()
    clock.advance(60.0)

    assert state.locked is True


def test_l6_the_grace_defaults_to_five_seconds():
    assert LockState().drop_grace_s == pytest.approx(5.0)


def test_l6_the_grace_is_env_overridable(monkeypatch):
    monkeypatch.setenv("NOVA_LOCK_DROP_GRACE_S", "12.5")
    assert LockState().drop_grace_s == pytest.approx(12.5)


@pytest.mark.parametrize("raw", ["", "later", "-1", "nan"])
def test_l6_a_bad_grace_env_falls_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv("NOVA_LOCK_DROP_GRACE_S", raw)
    assert LockState().drop_grace_s == pytest.approx(5.0)
