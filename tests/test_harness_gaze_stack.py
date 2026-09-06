"""Tests for the gaze stack core (t8): a single-writer posture layer.

Everything here drives :class:`~reachy_nova.harness.gaze_stack.GazeStack`
against a FAKE :class:`~reachy_nova.harness.tools.IntentTools` — no spool, no
state dir, no ``reachy_mini``. The fake records every op in order with a
sequence number, returns the real JSON string shapes ``execute`` returns, and
deliberately sleeps inside each call so the "one writer" claim is actually
observable: any overlap between two ops is recorded and asserted against.

The sections map onto the task's acceptance criteria:

1. ``on_browser_state("busy")`` declares the browsing gaze-hold goal, and
   ``"idle"``/``"error"`` clear it — exactly those ops, in that order.
2. The aside yaw alternates sign across two browses.
3. ``clear_for_result()`` clears synchronously, on the CALLER's thread, and a
   later ``"idle"`` does not clear a second time.
4. Producers hammering from three threads for ~2 s still produce a strictly
   serial op list (monotonic sequence numbers, no overlapping calls).
5. A live conversation takes the top layer while the browsing goal stays
   standing; leaving it resumes browsing (no duplicate declare) or clears.
6. An op that raises does not kill the worker, and ``stop()`` returns fast.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from reachy_nova.harness.gaze_stack import (
    GAZE_HOLD_BEHAVIOR,
    LAYER_BROWSING,
    LAYER_CONVERSATION,
    LAYER_WANDER,
    GazeStack,
)

TICK = 0.02
DEADLINE = 5.0


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class FakeIntents:
    """Records ops in order, with a sequence number and overlap detection."""

    def __init__(self, call_delay: float = 0.0, result: dict | None = None) -> None:
        self._lock = threading.Lock()
        self.ops: list[tuple[int, str, dict]] = []
        self.call_delay = call_delay
        self.result = result if result is not None else {"ok": True}
        self._seq = 0
        self._active = 0
        self.overlaps = 0
        #: When set, the NEXT call raises this instead of answering.
        self.raise_once: Exception | None = None

    def execute(self, tool_name: str, params: dict) -> str:
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._active += 1
            if self._active > 1:
                self.overlaps += 1
            self.ops.append((seq, tool_name, dict(params)))
            boom, self.raise_once = self.raise_once, None
        try:
            if self.call_delay:
                time.sleep(self.call_delay)
            if boom is not None:
                raise boom
            return json.dumps(self.result)
        finally:
            with self._lock:
                self._active -= 1

    def forget(self) -> None:
        """Drop the ops recorded so far, keeping the sequence counter running.

        Called right after ``start()``, which submits its two idempotent
        start-hygiene ops (``release_face`` + ``declare_goal`` None, task t9)
        before the worker exists. Those are not what any test in THIS file is
        about — the hygiene has its own tests in
        ``test_harness_gaze_stack_conversation.py`` — so forgetting them keeps
        every assertion below counting exactly the ops it always counted.
        """
        with self._lock:
            self.ops.clear()

    # -- reads -------------------------------------------------------------- #

    def snapshot(self) -> list[tuple[int, str, dict]]:
        with self._lock:
            return list(self.ops)

    def names(self) -> list[str]:
        return [name for _seq, name, _params in self.snapshot()]

    def goals(self) -> list[object]:
        return [p.get("goal") for _s, n, p in self.snapshot() if n == "declare_goal"]

    def declare_ops(self) -> list[tuple[int, str, dict]]:
        """Only the ``declare_goal`` ops — task t10 added ``set_inhibition``
        alongside every browsing transition, so tests that pin exactly the
        goal-declare sequence filter it out rather than counting the raw
        snapshot."""
        return [op for op in self.snapshot() if op[1] == "declare_goal"]


class FakeAttention:
    """Just the one property :class:`GazeStack` reads off ``AttentionState``."""

    def __init__(self, live: bool = False) -> None:
        self.conversation_live = live


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def wait_for(predicate, deadline: float = DEADLINE, message: str = "condition") -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {message}")


def running_stack(intents, **kwargs):
    """A started GazeStack plus its stop event; caller stops it."""
    stack = GazeStack(intents, tick_s=TICK, **kwargs)
    stop = threading.Event()
    stack.start(stop)
    intents.forget()  # the t9 start hygiene; see FakeIntents.forget
    return stack, stop


# --------------------------------------------------------------------------- #
# 1. browsing: busy declares the gaze-hold goal, idle/error clear it           #
# --------------------------------------------------------------------------- #


def test_busy_declares_gaze_hold_then_idle_clears():
    intents = FakeIntents()
    stack, _stop = running_stack(intents)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing layer")
        wait_for(lambda: len(intents.declare_ops()) == 1, message="the declare op")

        _seq, name, params = intents.declare_ops()[0]
        assert name == "declare_goal"
        assert params["goal"] == GAZE_HOLD_BEHAVIOR
        assert params["params"]["pitch"] == pytest.approx(10.0)
        assert abs(params["params"]["yaw"]) == pytest.approx(15.0)
        assert stack.status()["goal_standing"] is True

        stack.on_browser_state("idle")
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        wait_for(lambda: len(intents.declare_ops()) == 2, message="the clear op")
    finally:
        stack.stop()

    ops = intents.declare_ops()
    assert [name for _s, name, _p in ops] == ["declare_goal", "declare_goal"]
    assert ops[1][2]["goal"] is None
    assert stack.status()["goal_standing"] is False


def test_browser_error_clears_the_goal_like_idle():
    intents = FakeIntents()
    stack, _stop = running_stack(intents)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: len(intents.declare_ops()) == 1, message="the declare op")
        stack.on_browser_state("error")
        wait_for(lambda: len(intents.declare_ops()) == 2, message="the clear op")
    finally:
        stack.stop()

    assert [n for _s, n, _p in intents.declare_ops()] == ["declare_goal", "declare_goal"]
    assert intents.goals() == [GAZE_HOLD_BEHAVIOR, None]


# --------------------------------------------------------------------------- #
# 2. the aside yaw alternates across browses                                   #
# --------------------------------------------------------------------------- #


def test_side_alternates_across_two_browses():
    intents = FakeIntents()
    stack, _stop = running_stack(intents)
    try:
        for expected in (2, 4):
            stack.on_browser_state("busy")
            wait_for(lambda n=expected - 1: len(intents.declare_ops()) == n, message="declare")
            stack.on_browser_state("idle")
            wait_for(lambda n=expected: len(intents.declare_ops()) == n, message="clear")
    finally:
        stack.stop()

    yaws = [
        p["params"]["yaw"]
        for _s, n, p in intents.declare_ops()
        if n == "declare_goal" and p.get("goal") == GAZE_HOLD_BEHAVIOR
    ]
    assert yaws == [pytest.approx(15.0), pytest.approx(-15.0)]


# --------------------------------------------------------------------------- #
# 3. clear_for_result — synchronous, on the caller's thread, exactly once      #
# --------------------------------------------------------------------------- #


def test_clear_for_result_clears_synchronously_and_only_once():
    intents = FakeIntents()
    stack, _stop = running_stack(intents)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: stack.status()["goal_standing"] is True, message="goal standing")
        assert len(intents.declare_ops()) == 1

        # Synchronous: the clear op is already on the fake the instant this
        # returns — no worker tick in between.
        assert stack.clear_for_result() is True
        ops = intents.declare_ops()
        assert len(ops) == 2
        assert ops[1][1] == "declare_goal"
        assert ops[1][2]["goal"] is None
        assert stack.status()["goal_standing"] is False

        # A second call with nothing standing is a no-op.
        assert stack.clear_for_result() is False
        assert len(intents.declare_ops()) == 2

        # And the browse ending later must not clear a second time (the
        # browsing inhibits ARE still given back on this transition, task
        # t10, so only the declare_goal count is pinned here).
        stack.on_browser_state("idle")
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        time.sleep(TICK * 5)
        assert len(intents.declare_ops()) == 2
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# 4. three producer threads, one writer                                        #
# --------------------------------------------------------------------------- #


def test_concurrent_producers_produce_a_strictly_serial_op_list():
    intents = FakeIntents(call_delay=0.005)
    attention = FakeAttention(live=False)
    stack, _stop = running_stack(intents, attention=attention)
    stop_producers = threading.Event()
    errors: list[Exception] = []

    def browser_flipper():
        try:
            while not stop_producers.is_set():
                stack.on_browser_state("busy")
                time.sleep(0.01)
                stack.on_browser_state("idle")
                time.sleep(0.01)
        except Exception as exc:  # pragma: no cover - a hook must never raise
            errors.append(exc)

    def transcriber():
        try:
            while not stop_producers.is_set():
                stack.on_transcript("USER", "hello nova")
                stack.on_transcript("ASSISTANT", "hello there")
                time.sleep(0.01)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def sonic_states():
        try:
            while not stop_producers.is_set():
                stack.on_sonic_state("speaking")
                stack.on_sonic_state("listening")
                stack.on_speaker_idle()
                attention.conversation_live = not attention.conversation_live
                time.sleep(0.01)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=browser_flipper, daemon=True),
        threading.Thread(target=transcriber, daemon=True),
        threading.Thread(target=sonic_states, daemon=True),
    ]
    try:
        for thread in threads:
            thread.start()
        time.sleep(2.0)
    finally:
        stop_producers.set()
        for thread in threads:
            thread.join(timeout=2.0)
        stack.stop()

    assert errors == []
    ops = intents.snapshot()
    assert ops, "the producers should have driven at least one transition"
    # Strictly serial: sequence numbers monotonic, and no call ever started
    # before the previous one returned.
    seqs = [seq for seq, _n, _p in ops]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert intents.overlaps == 0


# --------------------------------------------------------------------------- #
# 5. the conversation layer (t8 seam only — no lock ops yet)                   #
# --------------------------------------------------------------------------- #


class CountingStack(GazeStack):
    """Counts the two t9 seams without changing what they do in t8."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entered = 0
        self.left = 0

    def _enter_conversation(self) -> None:
        self.entered += 1
        super()._enter_conversation()

    def _leave_conversation(self) -> None:
        self.left += 1
        super()._leave_conversation()


def test_conversation_takes_the_top_layer_and_leaves_the_goal_standing():
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    stack = CountingStack(intents, attention=attention, tick_s=TICK)
    stack.start(threading.Event())
    intents.forget()  # the t9 start hygiene; see FakeIntents.forget
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing layer")
        assert intents.goals() == [GAZE_HOLD_BEHAVIOR]

        attention.conversation_live = True
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")
        time.sleep(TICK * 5)
        # The lock owns the head by recency; the browsing goal is NOT cleared.
        # (t9's own lock ops ride alongside it, so this counts declare_goal.)
        assert intents.goals() == [GAZE_HOLD_BEHAVIOR]
        assert stack.status()["goal_standing"] is True
        assert stack.entered == 1

        # Conversation ends with the browser still busy: back to browsing with
        # the goal already standing, so no duplicate declare.
        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing again")
        time.sleep(TICK * 5)
        assert intents.goals() == [GAZE_HOLD_BEHAVIOR]
        assert stack.left == 1
        assert stack.entered == 1
    finally:
        stack.stop()


def test_conversation_ending_with_an_idle_browser_clears_the_goal():
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    stack = CountingStack(intents, attention=attention, tick_s=TICK)
    stack.start(threading.Event())
    intents.forget()  # the t9 start hygiene; see FakeIntents.forget
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing layer")
        attention.conversation_live = True
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")

        stack.on_browser_state("idle")
        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        wait_for(lambda: len(intents.goals()) == 2, message="the clear op")
    finally:
        stack.stop()

    assert intents.goals() == [GAZE_HOLD_BEHAVIOR, None]
    assert stack.left == 1
    assert stack.status()["goal_standing"] is False


def test_local_liveness_fallback_when_no_attention_is_wired():
    clock = FakeClock()
    intents = FakeIntents()
    stack = GazeStack(intents, attention=None, clock=clock, tick_s=TICK)
    stack.start(threading.Event())
    intents.forget()  # the t9 start hygiene; see FakeIntents.forget
    try:
        assert stack.status()["conversation_live"] is False
        stack.on_transcript("user", "hello")
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")

        clock.now += 46.0
        wait_for(lambda: stack.layer == LAYER_WANDER, message="fallback expiry")
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# 6. robustness: a raising op, a fast stop                                      #
# --------------------------------------------------------------------------- #


def test_a_raising_op_does_not_kill_the_worker():
    intents = FakeIntents()
    stack, _stop = running_stack(intents)
    # Armed AFTER start(), so it lands on a real transition op rather than on
    # the start hygiene.
    intents.raise_once = RuntimeError("spool exploded")
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: len(intents.declare_ops()) == 1, message="the failed declare")

        # The worker survives and keeps serving transitions.
        stack.on_browser_state("idle")
        wait_for(lambda: len(intents.declare_ops()) == 2, message="the clear after the failure")
        stack.on_browser_state("busy")
        wait_for(lambda: len(intents.declare_ops()) == 3, message="a later declare")
        assert stack.is_alive()
    finally:
        started = time.monotonic()
        stack.stop()
        assert time.monotonic() - started < 1.0
    assert not stack.is_alive()


def test_degraded_and_refused_results_are_tolerated():
    for result in ({"ok": None, "submitted": "abc"}, {"ok": False, "error": "nope"}):
        intents = FakeIntents(result=result)
        stack, _stop = running_stack(intents)
        try:
            stack.on_browser_state("busy")
            wait_for(lambda: len(intents.declare_ops()) == 1, message="the declare op")
            assert stack.layer == LAYER_BROWSING
        finally:
            stack.stop()


def test_stop_event_stops_the_worker_within_a_tick():
    intents = FakeIntents()
    stack = GazeStack(intents, tick_s=TICK)
    stop = threading.Event()
    stack.start(stop)
    assert stack.is_alive()
    stop.set()
    wait_for(lambda: not stack.is_alive(), deadline=1.0, message="worker exit")
    stack.stop()


def test_status_reports_the_core_fields():
    intents = FakeIntents()
    stack = GazeStack(intents, tick_s=TICK)
    status = stack.status()
    # t9 added the three lock fields on top; they are asserted in
    # test_harness_gaze_stack_conversation.py.
    assert set(status) >= {"layer", "browser_busy", "conversation_live", "goal_standing"}
    assert status["layer"] == LAYER_WANDER
    assert status["browser_busy"] is False
    assert status["goal_standing"] is False


def test_hooks_never_raise_on_odd_input():
    intents = FakeIntents()
    stack = GazeStack(intents, attention=object(), tick_s=TICK)
    stack.on_browser_state(None)  # type: ignore[arg-type]
    stack.on_transcript(None, None)  # type: ignore[arg-type]
    stack.on_sonic_state(None)  # type: ignore[arg-type]
    stack.on_speaker_idle()
    # A broken attention object degrades to "not live" rather than raising.
    assert stack.status()["conversation_live"] is False
    assert intents.snapshot() == []
