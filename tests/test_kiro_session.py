"""Standing Kiro ACP session unit (task t4) — watchdog, backoff, recycle.

Everything here runs against a fake session (no real ``kiro-cli`` subprocess):
:class:`FakeSession` implements exactly the surface
``reachy_nova.harness.kiro_session.KiroSessionUnit`` relies on
(``start``/``initialize``/``new_session``/``prompt``/``close``/``alive``), and
``FakeFactory`` counts how many fake sessions get built so a test can assert
"the Nth prompt lands on a fresh session" directly.

Four properties are load-bearing, one test group each:

1. ``start()`` spawns and initializes exactly one session via the factory.
2. A dead session (``alive`` flips ``False``) is detected by the watchdog and
   replaced, with capped exponential backoff between restart attempts.
3. ``stop()`` shuts down cleanly: the watchdog thread is joined, no lingering
   non-daemon thread remains, and the session's ``close()`` is called.
4. The recycle-at-threshold path: with ``KIRO_HISTORY_MAX`` small, the
   ``(threshold + 1)``-th prompt lands on a freshly spawned session; below
   threshold no recycle happens. The threshold is honoured from both the env
   var and a constructor kwarg (kwarg wins).

All timing in the backoff tests uses sub-100ms intervals so the whole file
stays well under 10s wall time.
"""

from __future__ import annotations

import threading
import time

import pytest

from reachy_nova.harness.kiro_session import HISTORY_MAX_ENV, KiroSessionUnit
from reachy_nova.kiro_acp import KiroAcpError

# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class FakeSession:
    """A session double: records calls, lets a test flip ``alive`` at will."""

    def __init__(self, *, prompt_prefix: str = "reply") -> None:
        self.started = False
        self.initialized = False
        self.session_cwd: str | None = None
        self.closed = False
        self._alive = True
        self.prompt_prefix = prompt_prefix
        self.prompts: list[str] = []

    def start(self) -> None:
        self.started = True

    def initialize(self) -> None:
        self.initialized = True

    def new_session(self, cwd: str) -> str:
        self.session_cwd = cwd
        return "fake-session-id"

    def prompt(self, text: str, timeout: float | None = None) -> str:
        self.prompts.append(text)
        return f"{self.prompt_prefix}:{text}"

    def close(self) -> None:
        self.closed = True

    @property
    def alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        """Test helper: simulate the child process dying."""
        self._alive = False


class FakeFactory:
    """Builds :class:`FakeSession` instances, counting and recording them."""

    def __init__(self, *, always_dead: bool = False) -> None:
        self.built: list[FakeSession] = []
        self._always_dead = always_dead

    def __call__(self) -> FakeSession:
        session = FakeSession(prompt_prefix=f"reply{len(self.built)}")
        if self._always_dead:
            session._alive = False
        self.built.append(session)
        return session

    @property
    def call_count(self) -> int:
        return len(self.built)


def _poll_until(predicate, *, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# --------------------------------------------------------------------------- #
# 1. start() spawns and initializes a session via the factory.               #
# --------------------------------------------------------------------------- #


def test_start_spawns_and_initializes_session_via_factory() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0)
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        assert factory.call_count == 1
        session = factory.built[0]
        assert session.started
        assert session.initialized
        assert session.session_cwd == "/work"
        assert unit.is_alive() is True
        status = unit.status()
        assert status["alive"] is True
        assert status["restarts"] == 0
        assert status["prompts_served"] == 0
        assert status["recycles"] == 0
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_prompt_delegates_to_current_session_and_counts_it() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0)
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        result = unit.prompt("hello")
        assert result == "reply0:hello"
        assert unit.status()["prompts_served"] == 1
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_prompt_before_start_raises() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0)
    with pytest.raises(KiroAcpError):
        unit.prompt("hello")


# --------------------------------------------------------------------------- #
# 2. Dead-session detection + capped exponential backoff restart.            #
# --------------------------------------------------------------------------- #


def test_dead_session_is_detected_and_replaced() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        monitor_interval=0.01,
        backoff_initial_s=0.01,
        backoff_max_s=0.05,
    )
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        first_session = factory.built[0]
        first_session.kill()

        assert _poll_until(lambda: factory.call_count >= 2, timeout=5.0)
        # The replacement session is alive and initialized; the dead one was closed.
        assert first_session.closed is True
        second_session = factory.built[1]
        assert second_session.started and second_session.initialized
        assert unit.status()["restarts"] >= 1
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_restart_backoff_grows_and_is_capped() -> None:
    # A factory that always hands back a dead session forces repeated restarts,
    # so the backoff sequence (0.01, 0.02, 0.04, capped at 0.05, ...) is
    # observable within a couple hundred milliseconds of wall time.
    factory = FakeFactory(always_dead=True)
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        monitor_interval=0.005,
        backoff_initial_s=0.01,
        backoff_max_s=0.05,
        backoff_reset_after_s=999.0,  # never resets mid-test
    )
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        # Poll on the spawn count, not the restart counter: the counter ticks
        # up BEFORE the backoff wait + respawn complete, so waiting on it alone
        # can observe restarts==4 while the 4th respawn is still in flight.
        assert _poll_until(lambda: factory.call_count >= 5, timeout=5.0)
        assert unit.status()["restarts"] >= 4
        # Backoff must have grown past its initial value and never exceeded the cap.
        assert unit._backoff <= 0.05 + 1e-9
        assert unit._backoff >= 0.01
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_backoff_resets_after_a_healthy_period() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        monitor_interval=0.01,
        backoff_initial_s=0.01,
        backoff_max_s=0.05,
        backoff_reset_after_s=0.03,
    )
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        first_session = factory.built[0]
        first_session.kill()
        assert _poll_until(lambda: factory.call_count >= 2, timeout=5.0)
        # Backoff grew past its initial value from the one restart.
        assert unit._backoff > 0.01

        # Stay healthy (do not kill the replacement) long enough for the
        # reset-after window to elapse, then let the watchdog observe it.
        assert _poll_until(lambda: unit._backoff == pytest.approx(0.01), timeout=5.0)
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_stuck_prompt_triggers_a_restart_independent_of_the_call_lock() -> None:
    """A prompt() call wedged inside _call_lock must not block the watchdog."""
    factory = FakeFactory()
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        monitor_interval=10.0,  # driven manually below, not by the thread
        backoff_initial_s=0.01,
        backoff_max_s=0.05,
        prompt_stuck_deadline_s=0.05,
    )
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        # Simulate a prompt that started a while ago and never finished.
        with unit._status_lock:
            unit._prompt_started_at = time.monotonic() - 1.0
        assert unit._is_prompt_stuck() is True

        # Drive one watchdog tick directly (deterministic, no thread timing).
        unit._watchdog_tick()
        assert factory.call_count == 2
        assert unit.status()["restarts"] == 1
    finally:
        # Clear the simulated stuck marker so stop() does not itself trip on it.
        with unit._status_lock:
            unit._prompt_started_at = None
        unit.stop(timeout=2.0)
        stop_event.set()


# --------------------------------------------------------------------------- #
# 3. stop() shuts down cleanly.                                               #
# --------------------------------------------------------------------------- #


def test_stop_joins_thread_and_closes_session_with_no_lingering_threads() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=0.01)
    stop_event = threading.Event()
    unit.start(stop_event)
    session = factory.built[0]

    before_names = {t.name for t in threading.enumerate()}
    assert "kiro-session-monitor" in before_names

    unit.stop(timeout=2.0)

    assert session.closed is True
    after_names = {t.name for t in threading.enumerate()}
    assert "kiro-session-monitor" not in after_names
    assert unit.status()["alive"] is False


def test_stop_is_idempotent_and_safe_before_start() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work")
    # Never started — must not raise.
    unit.stop(timeout=1.0)
    unit.stop(timeout=1.0)


# --------------------------------------------------------------------------- #
# 4. Recycle at the threshold boundary.                                      #
# --------------------------------------------------------------------------- #


def test_recycle_happens_on_the_prompt_that_crosses_the_threshold() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0, history_max=3)
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        first_session = factory.built[0]

        unit.prompt("one")
        unit.prompt("two")
        assert factory.call_count == 1  # still below threshold
        assert first_session.closed is False

        unit.prompt("three")  # crosses the threshold -> recycle happens now
        assert factory.call_count == 2
        assert first_session.closed is True
        assert unit.status()["recycles"] == 1
        assert unit.status()["prompts_served"] == 0

        result = unit.prompt("four")  # the 4th prompt: fresh (second) session
        second_session = factory.built[1]
        assert result == f"{second_session.prompt_prefix}:four"
        assert second_session.prompts == ["four"]
        assert factory.call_count == 2  # no further recycle yet
        assert unit.status()["prompts_served"] == 1
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_below_threshold_prompts_never_recycle() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0, history_max=3)
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        unit.prompt("one")
        unit.prompt("two")
        assert factory.call_count == 1
        assert unit.status()["recycles"] == 0
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_history_max_env_var_is_honoured() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(
        factory, cwd="/work", env={HISTORY_MAX_ENV: "2"}
    )
    assert unit._history_max == 2


def test_history_max_kwarg_overrides_env_var() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        history_max=7,
        env={HISTORY_MAX_ENV: "2"},
    )
    assert unit._history_max == 7


def test_history_max_defaults_to_fifty_with_no_env_or_kwarg() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work", env={})
    assert unit._history_max == 50
