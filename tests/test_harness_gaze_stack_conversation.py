"""Tests for the gaze stack's CONVERSATION layer (t9).

t8 proved the posture layering and the single-writer invariant; this file is
about the one layer t8 left as a seam: when a conversation goes live the head
turns toward the voice and then locks on the face, it holds that lock through
Nova's replies and the listening gaps, and it gives it back only when the
conversation fades.

Everything drives :class:`~reachy_nova.harness.gaze_stack.GazeStack` against a
FAKE :class:`~reachy_nova.harness.tools.IntentTools` — the same recording fake
t8 uses, extended so each TOOL NAME can be scripted independently (``ok: true``
/ a "no face known" refusal / a degraded ``ok: null``), because the whole
conversation layer is a reaction to which of those three shapes came back.

The sections map onto the task's acceptance criteria:

1. Entering a conversation submits ``look_at_sound`` then ``lock_face``, in
   that order, within one worker tick.
2. A refused lock retries on the 3/6/12/24/30/30 s backoff, logs exactly one
   "no face known" line, and summarises "lock never held" at fade. A degraded
   ``ok: null`` never counts as locked and retries the same way.
3. A held lock survives a long quiet gap and is released — once — at fade.
4. A lock the MODEL took is never taken over and never released.
5. The browsing goal stays standing under the lock and is not re-declared.
6. Start/stop hygiene: exactly once each, and a second ``stop()`` is silent.
7. ``on_lock_released`` clears the belief and the retry resumes.
8. No face-presence state lives in the module (the engine's refusal is the
   presence check).
9. Nova's OWN voice renews a live conversation but never opens one (the
   2026-09-06 live finding: "It feels rigid now. No liveness.").
10. While an AUTO-owned lock is held the antennas keep swaying underneath it —
    one bounded ``antenna-sway``, re-issued before it runs out, never for a
    model lock and never once the lock is gone.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time

import pytest

from reachy_nova.harness import gaze_stack as gaze_stack_module
from reachy_nova.harness.gaze_stack import (
    GAZE_HOLD_BEHAVIOR,
    LAYER_BROWSING,
    LAYER_CONVERSATION,
    LAYER_WANDER,
    LOCK_RETRY_BACKOFF_S,
    SWAY_BEHAVIOR,
    SWAY_DURATION_S,
    SWAY_PARAMS,
    SWAY_REISSUE_MARGIN_S,
    GazeStack,
)
from reachy_nova.harness.attention import AttentionState
from reachy_nova.harness.lock_state import LockState

TICK = 0.02
DEADLINE = 5.0

#: The tool name the liveness sway rides on (section 9).
SWAY_OP = "run_behavior"

OK = {"ok": True}
NO_FACE = {"ok": False, "error": "no face known"}
DEGRADED = {"ok": None, "submitted": "cmd-1"}


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _healthy_base_layer(monkeypatch):
    """This file describes a runtime whose ``feel-alive`` base layer is alive,
    so the stack's base-layer revive (``_tick_base_layer``) stays silent and
    every op list below counts exactly what it always counted. The revive has
    its own file: ``test_harness_gaze_stack_liveness.py``.
    """
    monkeypatch.setattr(
        gaze_stack_module, "_runtime_current_active_names", lambda: ["feel-alive"]
    )


class FakeIntents:
    """t8's recording fake, plus per-tool-name scripted results.

    ``results[name]`` may be a single result dict (answered every time) or a
    list of result dicts consumed in order, the last one repeating. Anything
    not scripted answers ``{"ok": true}``.
    """

    def __init__(self, results: dict | None = None) -> None:
        self._lock = threading.Lock()
        self.ops: list[tuple[int, str, dict]] = []
        self.results = dict(results or {})
        self._seq = 0
        self._active = 0
        self.overlaps = 0

    def script(self, tool_name: str, result) -> None:
        with self._lock:
            self.results[tool_name] = result

    def execute(self, tool_name: str, params: dict) -> str:
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._active += 1
            if self._active > 1:
                self.overlaps += 1
            self.ops.append((seq, tool_name, dict(params)))
            scripted = self.results.get(tool_name, OK)
            if isinstance(scripted, list):
                result = scripted[0] if len(scripted) == 1 else scripted.pop(0)
            else:
                result = scripted
        try:
            return json.dumps(result)
        finally:
            with self._lock:
                self._active -= 1

    # -- reads -------------------------------------------------------------- #

    def snapshot(self) -> list[tuple[int, str, dict]]:
        with self._lock:
            return list(self.ops)

    def names(self) -> list[str]:
        return [name for _seq, name, _params in self.snapshot()]

    def count(self, tool_name: str) -> int:
        return self.names().count(tool_name)

    def since(self, mark: int) -> list[str]:
        return [name for seq, name, _p in self.snapshot() if seq > mark]

    def mark(self) -> int:
        with self._lock:
            return self._seq


class FakeAttention:
    def __init__(self, live: bool = False) -> None:
        self.conversation_live = live


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self._lock = threading.Lock()
        self.now = now

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


def wait_for(predicate, deadline: float = DEADLINE, message: str = "condition") -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {message}")


def conversation_stack(intents, attention=None, clock=None, lock_state=None, **kwargs):
    """A started stack with the START HYGIENE already drained off the fake."""
    stack = GazeStack(
        intents,
        attention=attention,
        lock_state=lock_state,
        clock=clock or time.monotonic,
        tick_s=TICK,
        **kwargs,
    )
    stack.start(threading.Event())
    # The hygiene is the WORKER's first action (PR #26 review), so it lands a
    # moment after start() returns rather than on the caller's thread.
    wait_for(lambda: intents.names() == ["release_face", "declare_goal"], message="start hygiene")
    return stack


# --------------------------------------------------------------------------- #
# 1. entering: look_at_sound, then lock_face                                   #
# --------------------------------------------------------------------------- #


def test_conversation_looks_at_the_sound_then_locks_the_face():
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    lock_state = LockState()
    stack = conversation_stack(intents, attention=attention, lock_state=lock_state)
    mark = intents.mark()
    try:
        attention.conversation_live = True
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")
        wait_for(lambda: stack.lock_held is True, message="the lock")
        wait_for(lambda: intents.count(SWAY_OP) == 1, message="the liveness sway")

        # One tick, three ops, in that order and no others: the liveness sway
        # (section 9) rides directly behind the confirmed lock.
        assert intents.since(mark) == ["look_at_sound", "lock_face", SWAY_OP]
        assert lock_state.locked is True
        assert lock_state.owner == "auto"
        status = stack.status()
        assert status["lock_held"] is True
        assert status["lock_attempts"] == 1
        assert status["next_lock_retry_s"] is None
    finally:
        stack.stop()


def test_a_user_transcript_alone_drives_the_conversation_layer():
    """No AttentionState wired: the local fallback still enters the layer."""
    clock = FakeClock()
    intents = FakeIntents()
    stack = conversation_stack(intents, clock=clock)
    mark = intents.mark()
    try:
        stack.on_transcript("user", "hello nova")
        wait_for(lambda: stack.lock_held is True, message="the lock")
        wait_for(lambda: intents.count(SWAY_OP) == 1, message="the liveness sway")
        assert intents.since(mark) == ["look_at_sound", "lock_face", SWAY_OP]
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# 2. refusals: the backoff, one log line, one summary                          #
# --------------------------------------------------------------------------- #


def test_refused_lock_retries_on_the_documented_backoff(caplog):
    clock = FakeClock()
    intents = FakeIntents({"lock_face": NO_FACE})
    attention = FakeAttention(live=True)
    lock_state = LockState(clock=clock)
    caplog.set_level("INFO")
    stack = conversation_stack(
        intents, attention=attention, clock=clock, lock_state=lock_state
    )
    try:
        wait_for(lambda: intents.count("lock_face") == 1, message="the first attempt")
        # 3, 6, 12, 24, 30, then 30 again: the last value repeats forever.
        for step, delay in enumerate((3.0, 6.0, 12.0, 24.0, 30.0, 30.0), start=2):
            before = intents.count("lock_face")
            # Just short of the deadline nothing fires...
            clock.advance(delay - 0.5)
            time.sleep(TICK * 4)
            assert intents.count("lock_face") == before, f"early retry at step {step}"
            # ...and crossing it fires exactly one more.
            clock.advance(0.5)
            wait_for(
                lambda n=step: intents.count("lock_face") == n,
                message=f"retry {step}",
            )
        assert intents.count("lock_face") == 7
        assert intents.count("look_at_sound") == 1  # only on entry, never per retry
        assert lock_state.locked is None
        assert stack.lock_held is False

        # Fade: no release (nothing was ever held), and one summary line.
        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        time.sleep(TICK * 4)
        assert intents.count("release_face") == 1  # the start hygiene only
    finally:
        stack.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert sum("no face known — retrying with backoff" in m for m in messages) == 1
    assert sum("lock never held: refusals=7" in m for m in messages) == 1


def test_backoff_constant_is_the_documented_sequence():
    assert LOCK_RETRY_BACKOFF_S == (3.0, 6.0, 12.0, 24.0, 30.0)


def test_a_degraded_lock_answer_is_unknown_not_locked(caplog):
    clock = FakeClock()
    intents = FakeIntents({"lock_face": DEGRADED})
    attention = FakeAttention(live=True)
    lock_state = LockState(clock=clock)
    caplog.set_level("INFO")
    stack = conversation_stack(
        intents, attention=attention, clock=clock, lock_state=lock_state
    )
    try:
        wait_for(lambda: intents.count("lock_face") == 1, message="the first attempt")
        assert stack.lock_held is False
        assert lock_state.locked is None  # the belief is untouched, not guessed

        clock.advance(3.0)
        wait_for(lambda: intents.count("lock_face") == 2, message="the retry")
        assert stack.lock_held is False
        assert lock_state.locked is None
    finally:
        stack.stop()

    messages = [record.getMessage() for record in caplog.records]
    # An unknown is not a refusal, so the refusal line never fires for it.
    assert not any("no face known" in m for m in messages)
    assert any("ok=unknown" in m for m in messages)


def test_a_lock_that_arrives_late_stops_the_retries(caplog):
    clock = FakeClock()
    intents = FakeIntents({"lock_face": [NO_FACE, NO_FACE, OK]})
    attention = FakeAttention(live=True)
    caplog.set_level("INFO")
    stack = conversation_stack(intents, attention=attention, clock=clock)
    try:
        wait_for(lambda: intents.count("lock_face") == 1, message="attempt 1")
        clock.advance(3.0)
        wait_for(lambda: intents.count("lock_face") == 2, message="attempt 2")
        clock.advance(6.0)
        wait_for(lambda: stack.lock_held is True, message="the late lock")
        assert intents.count("lock_face") == 3

        # No further attempts once it is held, however long the gap.
        clock.advance(60.0)
        time.sleep(TICK * 5)
        assert intents.count("lock_face") == 3
    finally:
        stack.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert sum("locked after=9.0s attempts=3" in m for m in messages) == 1
    assert not any("lock never held" in m for m in messages)


# --------------------------------------------------------------------------- #
# 3. the hold, and the release at fade                                         #
# --------------------------------------------------------------------------- #


def test_the_lock_is_held_across_a_quiet_gap_and_released_at_fade():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=True)
    lock_state = LockState(clock=clock)
    stack = conversation_stack(
        intents, attention=attention, clock=clock, lock_state=lock_state
    )
    try:
        wait_for(lambda: stack.lock_held is True, message="the lock")
        wait_for(lambda: intents.count(SWAY_OP) == 1, message="the liveness sway")
        mark = intents.mark()

        # Ten seconds of nothing at all — Nova's replies and the listening
        # gaps are inside the window, so nothing is submitted (the liveness
        # sway has 50 s left on it, well short of its re-issue margin).
        clock.advance(10.0)
        time.sleep(TICK * 8)
        assert intents.since(mark) == []
        assert stack.lock_held is True

        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        wait_for(lambda: intents.since(mark) == ["release_face"], message="the release")
        assert stack.lock_held is False
        assert lock_state.locked is False
        assert lock_state.owner is None
        assert stack.status()["lock_attempts"] == 0  # reset for the next conversation
    finally:
        stack.stop()


def test_an_unconfirmed_release_still_clears_the_belief():
    clock = FakeClock()
    intents = FakeIntents({"release_face": [OK, DEGRADED]})
    attention = FakeAttention(live=True)
    stack = conversation_stack(intents, attention=attention, clock=clock)
    try:
        wait_for(lambda: stack.lock_held is True, message="the lock")
        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        wait_for(lambda: intents.count("release_face") == 2, message="the release")
        # The runtime drops the lock on its own max-hold timer anyway; a
        # belief left "held" here would go stale with nothing to correct it.
        assert stack.lock_held is False
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# 4. a lock the MODEL took is not ours to touch                                #
# --------------------------------------------------------------------------- #


def test_a_model_owned_lock_is_never_taken_over_or_released(caplog):
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    lock_state = LockState(clock=clock)
    lock_state.mark_locked(owner="model")
    caplog.set_level("INFO")
    stack = conversation_stack(
        intents, attention=attention, clock=clock, lock_state=lock_state
    )
    mark = intents.mark()
    try:
        attention.conversation_live = True
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")
        clock.advance(30.0)
        time.sleep(TICK * 6)
        assert intents.since(mark) == []
        assert stack.lock_held is False

        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        time.sleep(TICK * 6)
        assert intents.since(mark) == []
        assert lock_state.owner == "model"
    finally:
        stack.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert any("model lock standing — auto hold not taken" in m for m in messages)


# --------------------------------------------------------------------------- #
# 5. the browsing goal underneath                                              #
# --------------------------------------------------------------------------- #


def test_the_browsing_goal_stands_under_the_lock_and_survives_the_release():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    stack = conversation_stack(intents, attention=attention, clock=clock)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing layer")
        mark = intents.mark()

        attention.conversation_live = True
        wait_for(lambda: stack.lock_held is True, message="the lock")
        assert stack.status()["goal_standing"] is True

        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing again")
        time.sleep(TICK * 5)
        # Released, but the goal was never cleared and is never re-declared.
        # (task t10 added set_inhibition ops alongside the browsing
        # transitions; this test is about the goal/lock seam, so those are
        # filtered out here rather than pinned.)
        assert [n for n in intents.since(mark) if n != "set_inhibition"] == [
            "look_at_sound",
            "lock_face",
            SWAY_OP,
            "release_face",
        ]
        assert stack.status()["goal_standing"] is True

        # ...and conversation -> wander clears it, once.
        stack.on_browser_state("idle")
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        wait_for(
            lambda: "declare_goal" in intents.since(mark), message="the clear"
        )
        time.sleep(TICK * 5)
        assert [n for n in intents.since(mark) if n != "set_inhibition"] == [
            "look_at_sound",
            "lock_face",
            SWAY_OP,
            "release_face",
            "declare_goal",
        ]
        assert stack.status()["goal_standing"] is False
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# 6. start / stop hygiene                                                      #
# --------------------------------------------------------------------------- #


def test_start_submits_the_hygiene_exactly_once():
    intents = FakeIntents()
    stack = GazeStack(intents, tick_s=TICK)
    stack.start(threading.Event())
    try:
        wait_for(lambda: len(intents.snapshot()) >= 2, message="the start hygiene")
        ops = intents.snapshot()
        assert [name for _s, name, _p in ops] == ["release_face", "declare_goal"]
        assert ops[1][2]["goal"] is None
        # A second start() on a live stack is a no-op, hygiene included.
        stack.start(threading.Event())
        time.sleep(TICK * 5)
        assert len(intents.snapshot()) == 2
    finally:
        stack.stop()


def test_stop_while_locked_releases_once_and_a_second_stop_is_silent():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=True)
    lock_state = LockState(clock=clock)
    stack = conversation_stack(
        intents, attention=attention, clock=clock, lock_state=lock_state
    )
    wait_for(lambda: stack.lock_held is True, message="the lock")
    wait_for(lambda: intents.count(SWAY_OP) == 1, message="the liveness sway")
    mark = intents.mark()

    stack.stop()
    assert intents.since(mark) == ["release_face"]
    assert stack.lock_held is False
    assert lock_state.locked is False

    stack.stop()
    assert intents.since(mark) == ["release_face"]


def test_conversation_disabled_never_leaves_the_lower_layers():
    """PR #26 review (comment 3943444439): with ``NOVA_FACE_HOLD=0`` the app
    still builds the stack for the BROWSING layer, so the conversation layer
    has to be off in the stack itself — not merely unwired."""
    intents = FakeIntents()
    attention = FakeAttention(live=True)
    stack = GazeStack(
        intents, attention=attention, tick_s=TICK, conversation_enabled=False
    )
    stack.start(threading.Event())
    try:
        wait_for(lambda: len(intents.snapshot()) >= 2, message="the start hygiene")
        mark = intents.mark()
        # Transcripts flowing in as well: neither input may raise the layer.
        for _ in range(5):
            stack.on_transcript("user", "nova hello")
            stack.on_sonic_state("speaking")
            stack.on_sonic_state("idle")
            time.sleep(TICK * 2)
        assert stack.layer == LAYER_WANDER
        assert stack.conversation_live() is False
        assert stack.status()["conversation_enabled"] is False
        assert stack.status()["conversation_live"] is False

        # ...and the browsing layer still works underneath it.
        stack.on_browser_state("busy")
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing layer")
        time.sleep(TICK * 5)
        assert stack.layer == LAYER_BROWSING
        assert intents.since(mark).count("look_at_sound") == 0
        assert intents.since(mark).count("lock_face") == 0
    finally:
        stack.stop()
    assert intents.count("look_at_sound") == 0
    assert intents.count("lock_face") == 0


def test_conversation_enabled_is_on_by_default():
    stack = GazeStack(FakeIntents(), tick_s=TICK)
    assert stack.status()["conversation_enabled"] is True


def test_a_stop_event_mid_conversation_runs_the_hygiene_once():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    stack = GazeStack(
        intents, attention=attention, clock=clock, tick_s=TICK, lock_state=LockState(clock=clock)
    )
    stop = threading.Event()
    stack.start(stop)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: stack.status()["goal_standing"] is True, message="the goal")
        attention.conversation_live = True
        wait_for(lambda: stack.lock_held is True, message="the lock")
        wait_for(lambda: intents.count(SWAY_OP) == 1, message="the liveness sway")
        mark = intents.mark()

        # Nobody joins this thread: the worker's own exit path must clean up.
        stop.set()
        wait_for(lambda: not stack.is_alive(), deadline=2.0, message="worker exit")
        # The give-back of the browsing inhibits is part of the hygiene too
        # (PR #26 review), so the exit submits three ops, not two.
        wait_for(
            lambda: sorted(intents.since(mark))
            == ["declare_goal", "release_face", "set_inhibition"],
            message="the exit hygiene",
        )
    finally:
        stack.stop()
    # ...and stop() afterwards adds nothing.
    assert sorted(intents.since(mark)) == ["declare_goal", "release_face", "set_inhibition"]


def test_stop_without_a_lock_or_a_goal_submits_nothing():
    intents = FakeIntents()
    stack = GazeStack(intents, tick_s=TICK)
    stack.start(threading.Event())
    wait_for(lambda: len(intents.snapshot()) >= 2, message="the start hygiene")
    mark = intents.mark()
    stack.stop()
    assert intents.since(mark) == []


# --------------------------------------------------------------------------- #
# 7. the runtime dropping the lock under us                                    #
# --------------------------------------------------------------------------- #


def test_on_lock_released_clears_the_hold_and_the_retry_resumes():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=True)
    stack = conversation_stack(intents, attention=attention, clock=clock)
    try:
        wait_for(lambda: stack.lock_held is True, message="the lock")
        assert intents.count("lock_face") == 1

        stack.on_lock_released("max-hold")
        assert stack.lock_held is False
        # Nothing is submitted by the hook itself.
        time.sleep(TICK * 3)
        assert intents.count("release_face") == 1  # start hygiene only
        assert intents.count("lock_face") == 1

        # ...and the retry follows on schedule.
        wait_for(
            lambda: stack.status()["next_lock_retry_s"] is not None,
            message="a scheduled retry",
        )
        clock.advance(LOCK_RETRY_BACKOFF_S[0])
        wait_for(lambda: intents.count("lock_face") == 2, message="the retry")
        wait_for(lambda: stack.lock_held is True, message="the re-taken lock")
    finally:
        stack.stop()


def test_an_engine_drop_of_the_belief_resumes_the_retry():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=True)
    lock_state = LockState(clock=clock, drop_grace_s=0.0)
    stack = conversation_stack(
        intents, attention=attention, clock=clock, lock_state=lock_state
    )
    try:
        wait_for(lambda: stack.lock_held is True, message="the lock")
        lock_state.on_engine_dropped()
        assert lock_state.locked is None

        wait_for(lambda: stack.lock_held is False, message="the noticed drop")
        clock.advance(LOCK_RETRY_BACKOFF_S[0])
        wait_for(lambda: intents.count("lock_face") == 2, message="the retry")
    finally:
        stack.stop()


def test_on_lock_released_never_raises_and_leaves_a_model_lock_alone():
    clock = FakeClock()
    intents = FakeIntents()
    lock_state = LockState(clock=clock)
    lock_state.mark_locked(owner="model")
    stack = GazeStack(intents, lock_state=lock_state, clock=clock, tick_s=TICK)
    stack.lock_held = True
    stack.on_lock_released(None)
    assert stack.lock_held is True  # not ours to clear
    assert intents.snapshot() == []


# --------------------------------------------------------------------------- #
# 8. no face-presence belief lives here                                        #
# --------------------------------------------------------------------------- #


def test_the_module_keeps_no_face_presence_state():
    """The ENGINE's "no face known" refusal IS the presence check.

    A second belief kept here would be a second thing to drift, and the
    failure mode is silent: a stale "no face" would suppress a lock the
    engine would happily have granted.
    """
    source = pathlib.Path(
        "reachy_nova/harness/gaze_stack.py"
    )
    if not source.exists():  # pragma: no cover - running from another cwd
        import reachy_nova.harness.gaze_stack as module

        source = pathlib.Path(module.__file__)
    text = source.read_text(encoding="utf-8")
    for token in ("face_seen", "face_bbox", "has_face", "face_present", "last_face"):
        assert token not in text, f"gaze_stack.py should keep no face-presence state: {token}"


def test_status_reports_the_conversation_fields():
    intents = FakeIntents()
    stack = GazeStack(intents, tick_s=TICK)
    status = stack.status()
    assert status["lock_held"] is False
    assert status["lock_attempts"] == 0
    assert status["next_lock_retry_s"] is None
    assert status["sway_until"] is None
    assert set(status) >= {"layer", "browser_busy", "conversation_live", "goal_standing"}


def test_hooks_still_never_raise():
    intents = FakeIntents()
    stack = GazeStack(intents, attention=object(), lock_state=object(), tick_s=TICK)
    stack.on_lock_released("whatever")
    stack.on_browser_state(None)  # type: ignore[arg-type]
    stack.on_transcript(None, None)  # type: ignore[arg-type]
    assert intents.snapshot() == []
    assert stack.status()["conversation_live"] is False


@pytest.mark.parametrize("result", [OK, NO_FACE, DEGRADED, {"ok": "weird"}])
def test_every_lock_result_shape_is_survivable(result):
    clock = FakeClock()
    intents = FakeIntents({"lock_face": result})
    attention = FakeAttention(live=True)
    stack = conversation_stack(intents, attention=attention, clock=clock)
    try:
        wait_for(lambda: intents.count("lock_face") >= 1, message="the attempt")
        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        assert stack.is_alive()
    finally:
        stack.stop()


def test_the_browsing_goal_name_is_unchanged():
    """t8's constant is still what the browsing layer declares."""
    assert GAZE_HOLD_BEHAVIOR == "gaze-hold"


# --------------------------------------------------------------------------- #
# 9. her own voice never opens a conversation (live finding, 2026-09-06)       #
# --------------------------------------------------------------------------- #


def sway_ops(intents):
    return [op for op in intents.snapshot() if op[1] == SWAY_OP]


def test_novas_own_opening_utterance_submits_nothing():
    """Live finding: "It feels rigid now. No liveness."

    Nova's opening line at session start (and every reaction she speaks to a
    body cue) used to open ``conversation_live`` from cold, which raised the
    conversation layer, took a face lock, and inhibited ``feel-alive`` and
    ``orient-to-sound`` with nobody in the room talking. Journal:
    ``wander -> conversation`` every minute or two with nobody there.
    """
    clock = FakeClock()
    intents = FakeIntents()
    attention = AttentionState(clock=clock, window_s=45.0)
    stack = conversation_stack(intents, attention=attention, clock=clock)
    mark = intents.mark()
    try:
        # She speaks first, unprompted: the speaker notes the utterance and the
        # stack sees the speaking edge. Neither may raise the layer.
        attention.note_utterance()
        stack.on_sonic_state("speaking")
        stack.on_sonic_state("idle")
        time.sleep(TICK * 6)
        assert stack.layer == LAYER_WANDER
        assert intents.since(mark) == []

        # ...and a person actually saying something still does, immediately.
        attention.note_transcript("what time is it")
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")
        wait_for(lambda: stack.lock_held is True, message="the lock")
        assert intents.since(mark)[:2] == ["look_at_sound", "lock_face"]
    finally:
        stack.stop()


def test_her_voice_renews_a_conversation_a_person_opened():
    """The renewal half of the same rule: mid-conversation she keeps it alive."""
    clock = FakeClock()
    intents = FakeIntents()
    attention = AttentionState(clock=clock, window_s=45.0)
    stack = conversation_stack(intents, attention=attention, clock=clock)
    try:
        attention.note_transcript("what time is it")
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")

        clock.advance(40.0)
        attention.note_utterance()  # her reply, inside the window
        clock.advance(40.0)  # 80 s after the transcript
        time.sleep(TICK * 6)
        assert stack.layer == LAYER_CONVERSATION
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# 10. liveness under the hold — the antennas keep swaying                      #
# --------------------------------------------------------------------------- #


def test_a_confirmed_auto_lock_starts_one_antenna_sway():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=True)
    lock_state = LockState(clock=clock)
    stack = conversation_stack(
        intents, attention=attention, clock=clock, lock_state=lock_state
    )
    try:
        wait_for(lambda: stack.lock_held is True, message="the lock")
        wait_for(lambda: len(sway_ops(intents)) == 1, message="the liveness sway")
        time.sleep(TICK * 5)
        assert len(sway_ops(intents)) == 1  # exactly one, not one per tick

        seq, _name, params = sway_ops(intents)[0]
        assert params["name"] == SWAY_BEHAVIOR == "antenna-sway"
        assert params["params"] == {"amp": 10.0, "period": 5.0} == SWAY_PARAMS
        assert params["duration"] == SWAY_DURATION_S == 60.0

        # ...and it rides BEHIND the lock, never in front of it.
        lock_seq = [s for s, n, _p in intents.snapshot() if n == "lock_face"][0]
        assert seq > lock_seq
        assert stack.status()["sway_until"] == pytest.approx(clock.now + SWAY_DURATION_S)
    finally:
        stack.stop()


def test_the_sway_is_re_issued_before_it_runs_out_and_stops_at_the_release():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=True)
    stack = conversation_stack(intents, attention=attention, clock=clock)
    try:
        wait_for(lambda: len(sway_ops(intents)) == 1, message="the first sway")

        # 51 s in: fewer than the 10 s margin is left, so exactly one more.
        clock.advance(SWAY_DURATION_S - SWAY_REISSUE_MARGIN_S + 1.0)
        wait_for(lambda: len(sway_ops(intents)) == 2, message="the re-issue")
        time.sleep(TICK * 5)
        assert len(sway_ops(intents)) == 2

        # The conversation fades: released, and the antennas are not ours any
        # more however long we wait.
        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        assert stack.status()["sway_until"] is None
        clock.advance(SWAY_DURATION_S)
        time.sleep(TICK * 6)
        assert len(sway_ops(intents)) == 2
    finally:
        stack.stop()


def test_a_model_owned_lock_never_gets_a_liveness_sway():
    """The model asked for that hold; what the body does under it is the
    model's business too."""
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    lock_state = LockState(clock=clock)
    lock_state.mark_locked(owner="model")
    stack = conversation_stack(
        intents, attention=attention, clock=clock, lock_state=lock_state
    )
    try:
        attention.conversation_live = True
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")
        clock.advance(120.0)
        time.sleep(TICK * 8)
        assert sway_ops(intents) == []
        assert stack.status()["sway_until"] is None
    finally:
        stack.stop()


def test_lock_liveness_off_submits_no_sway_at_all():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=True)
    stack = conversation_stack(
        intents, attention=attention, clock=clock, lock_liveness=False
    )
    try:
        wait_for(lambda: stack.lock_held is True, message="the lock")
        clock.advance(120.0)
        time.sleep(TICK * 8)
        assert sway_ops(intents) == []
        assert stack.status()["sway_until"] is None
    finally:
        stack.stop()


def test_lock_liveness_is_on_by_default():
    stack = GazeStack(FakeIntents(), tick_s=TICK)
    assert stack.lock_liveness is True


def test_a_refused_lock_never_sways():
    """No hold, no stillness to make up for — and no op to waste."""
    clock = FakeClock()
    intents = FakeIntents({"lock_face": NO_FACE})
    attention = FakeAttention(live=True)
    stack = conversation_stack(intents, attention=attention, clock=clock)
    try:
        wait_for(lambda: intents.count("lock_face") == 1, message="the attempt")
        clock.advance(60.0)
        time.sleep(TICK * 6)
        assert sway_ops(intents) == []
    finally:
        stack.stop()
