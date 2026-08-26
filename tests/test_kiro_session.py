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


class FailingSession(FakeSession):
    """A session double whose handshake fails at a chosen step.

    ``close_calls`` counts invocations so a test can assert "closed exactly
    once", and closing can itself be made to raise via ``close_error`` to
    prove that a broken ``close()`` never masks the original failure.
    """

    def __init__(
        self,
        *,
        fail_at: str,
        error: Exception,
        close_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._fail_at = fail_at
        self._error = error
        self._close_error = close_error
        self.close_calls = 0

    def start(self) -> None:
        super().start()
        if self._fail_at == "start":
            raise self._error

    def initialize(self) -> None:
        super().initialize()
        if self._fail_at == "initialize":
            raise self._error

    def new_session(self, cwd: str) -> str:
        super().new_session(cwd)
        if self._fail_at == "new_session":
            raise self._error

    def close(self) -> None:
        self.close_calls += 1
        super().close()
        if self._close_error is not None:
            raise self._close_error


class FlakyRestartFactory:
    """Simulates a ``kiro-cli`` executable that disappears and comes back.

    The first call (used by ``start()``) always succeeds. The next
    ``num_failures`` calls raise ``FileNotFoundError`` — the real-world
    precedent from PR review comment 3812045193 (systemd, no ``kiro-cli`` on
    PATH). Calls after that succeed again, so a test can assert the watchdog
    survives the failures and recovers once the binary is back.
    """

    def __init__(self, num_failures: int) -> None:
        self._num_failures = num_failures
        self.built: list[FakeSession] = []
        self.call_count = 0

    def __call__(self) -> FakeSession:
        self.call_count += 1
        if 1 < self.call_count <= 1 + self._num_failures:
            raise FileNotFoundError("kiro-cli: No such file or directory")
        session = FakeSession(prompt_prefix=f"reply{len(self.built)}")
        self.built.append(session)
        return session


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


# --------------------------------------------------------------------------- #
# 5. _spawn_session closes a partially-started session on handshake failure  #
#    (qodo review comment 3812045184).                                      #
# --------------------------------------------------------------------------- #


def test_spawn_session_closes_session_when_initialize_fails() -> None:
    session = FailingSession(fail_at="initialize", error=KiroAcpError("init boom"))
    unit = KiroSessionUnit(lambda: session, cwd="/work")

    with pytest.raises(KiroAcpError) as exc_info:
        unit._spawn_session()

    assert str(exc_info.value) == "init boom"
    assert session.started is True
    assert session.close_calls == 1


def test_spawn_session_closes_session_when_new_session_fails() -> None:
    session = FailingSession(fail_at="new_session", error=KiroAcpError("new_session boom"))
    unit = KiroSessionUnit(lambda: session, cwd="/work")

    with pytest.raises(KiroAcpError) as exc_info:
        unit._spawn_session()

    assert str(exc_info.value) == "new_session boom"
    assert session.initialized is True
    assert session.close_calls == 1


def test_spawn_session_preserves_original_exception_type_and_message() -> None:
    original = KiroAcpError("original message, unchanged")
    session = FailingSession(fail_at="initialize", error=original)
    unit = KiroSessionUnit(lambda: session, cwd="/work")

    with pytest.raises(KiroAcpError) as exc_info:
        unit._spawn_session()

    assert exc_info.value is original
    assert type(exc_info.value) is KiroAcpError
    assert str(exc_info.value) == "original message, unchanged"


def test_spawn_session_close_failure_does_not_mask_original_error() -> None:
    session = FailingSession(
        fail_at="new_session",
        error=KiroAcpError("real failure"),
        close_error=RuntimeError("close is also broken"),
    )
    unit = KiroSessionUnit(lambda: session, cwd="/work")

    with pytest.raises(KiroAcpError, match="real failure"):
        unit._spawn_session()

    assert session.close_calls == 1


def test_watchdog_restart_calls_spawn_session_which_closes_partial_session() -> None:
    """End-to-end: a recycle-triggering failure still closes the half-open session."""
    good_session = FakeSession()
    bad_session = FailingSession(fail_at="initialize", error=KiroAcpError("recycle boom"))
    sessions = iter([good_session, bad_session])
    factory = lambda: next(sessions)  # noqa: E731
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0, history_max=1)
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        # This prompt crosses the threshold and triggers a recycle; the
        # replacement session's handshake fails, so the recycle keeps the
        # existing (good) session but must still close the half-open one.
        result = unit.prompt("hello")
        assert result == "reply:hello"
        assert bad_session.close_calls == 1
        assert unit.status()["recycles"] == 0
        assert unit.is_alive() is True
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


# --------------------------------------------------------------------------- #
# 6. Non-KiroAcpError spawn failures (OSError/FileNotFoundError) do not kill #
#    the watchdog thread (qodo review comment 3812045193).                  #
# --------------------------------------------------------------------------- #


def test_spawn_session_normalizes_non_kiro_errors_to_kiro_acp_error() -> None:
    session = FailingSession(fail_at="start", error=FileNotFoundError("no kiro-cli on PATH"))
    unit = KiroSessionUnit(lambda: session, cwd="/work")

    with pytest.raises(KiroAcpError) as exc_info:
        unit._spawn_session()

    assert "no kiro-cli on PATH" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_watchdog_survives_file_not_found_and_recovers() -> None:
    """The exact class of failure that fired live under systemd: kiro-cli
    disappearing from PATH must not kill the monitor thread — it keeps
    retrying with backoff and recovers once the factory succeeds again."""
    factory = FlakyRestartFactory(num_failures=2)
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        monitor_interval=0.01,
        backoff_initial_s=0.01,
        backoff_max_s=0.02,
    )
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        first_session = factory.built[0]
        first_session.kill()

        # Two restart attempts hit FileNotFoundError before the third
        # succeeds and produces a second built (and alive) FakeSession.
        assert _poll_until(lambda: len(factory.built) >= 2, timeout=5.0)
        assert unit.is_alive() is True
        assert unit.status()["restarts"] >= 3
        # The monitor thread must still be running, never having died on the
        # unhandled FileNotFoundError.
        assert unit._thread is not None and unit._thread.is_alive()
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_spawn_session_normalizes_oserror_from_factory_itself() -> None:
    def factory() -> FakeSession:
        raise OSError("process creation failed")

    unit = KiroSessionUnit(factory, cwd="/work")

    with pytest.raises(KiroAcpError) as exc_info:
        unit._spawn_session()

    assert "process creation failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, OSError)


# --------------------------------------------------------------------------- #
# 7. Degraded start: an INITIAL spawn failure is retried, never propagated    #
#    (task t5 — the 2026-08-26 cold-boot bug).                                #
# --------------------------------------------------------------------------- #


class DeferredFactory:
    """Fails the first ``num_failures`` calls, then builds real fake sessions.

    Unlike :class:`FlakyRestartFactory` the FIRST call fails too — this is the
    cold-boot shape: the harness starts before Wi-Fi associates, kiro-cli
    exits immediately, and the initial spawn is the one that fails.
    """

    def __init__(self, num_failures: int, *, error: Exception | None = None) -> None:
        self._num_failures = num_failures
        self._error = error or KiroAcpError("kiro-cli process exited")
        self.built: list[FakeSession] = []
        self.call_count = 0

    def __call__(self) -> FakeSession:
        self.call_count += 1
        if self.call_count <= self._num_failures:
            raise self._error
        session = FakeSession(prompt_prefix=f"reply{len(self.built)}")
        self.built.append(session)
        return session


def test_initial_spawn_failure_does_not_raise_and_arms_the_watchdog() -> None:
    """start() returns on a failed first spawn; the unit is up-but-degraded."""
    factory = DeferredFactory(num_failures=1000)
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        monitor_interval=10.0,
        backoff_initial_s=30.0,  # the retry is armed but will not land during the test
    )
    stop_event = threading.Event()
    try:
        unit.start(stop_event)  # must NOT raise

        assert unit._thread is not None and unit._thread.is_alive()
        assert unit.is_alive() is False
        status = unit.status()
        assert status["alive"] is False
        assert status["degraded"] is True
        assert isinstance(status["restarts"], int)
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_degraded_start_logs_the_named_line(caplog) -> None:
    factory = DeferredFactory(num_failures=1)
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0)
    stop_event = threading.Event()
    with caplog.at_level("INFO"):
        try:
            unit.start(stop_event)
        finally:
            unit.stop(timeout=2.0)
            stop_event.set()
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "started degraded" in messages
    assert "kiro-cli process exited" in messages
    assert "retrying under watchdog" in messages


def test_watchdog_tick_respawns_when_there_is_no_session() -> None:
    """A session-less unit is a RESTARTABLE state, not a resting one.

    Driven WITHOUT the monitor thread (no ``start()``) so the restart count is
    exactly the manual tick's, with no race against the thread's own first tick.
    """
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0, backoff_initial_s=0.01)
    assert unit.is_alive() is False

    unit._watchdog_tick()

    assert unit.is_alive() is True
    status = unit.status()
    assert status["alive"] is True
    assert status["degraded"] is False
    assert status["restarts"] == 1


def test_degraded_unit_recovers_under_its_own_monitor_thread(caplog) -> None:
    factory = DeferredFactory(num_failures=1)
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        monitor_interval=0.01,
        backoff_initial_s=0.01,
        backoff_max_s=0.02,
    )
    stop_event = threading.Event()
    with caplog.at_level("INFO"):
        try:
            unit.start(stop_event)
            assert _poll_until(lambda: unit.is_alive(), timeout=5.0)
            # Exactly one restart: the degraded start's very first watchdog
            # tick respawned, and the session stayed alive afterwards.
            assert unit.status()["restarts"] == 1
        finally:
            unit.stop(timeout=2.0)
            stop_event.set()
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "recovered" in messages


def test_repeated_initial_failures_keep_backing_off_without_raising() -> None:
    """Nothing leaks out, the thread survives, and the backoff grows to its cap."""
    factory = DeferredFactory(num_failures=1000)
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        monitor_interval=0.001,
        backoff_initial_s=0.001,
        backoff_max_s=0.004,
    )
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        assert _poll_until(lambda: unit.status()["restarts"] >= 4, timeout=5.0)
        assert unit._thread is not None and unit._thread.is_alive()
        assert unit.is_alive() is False
        assert unit.status()["degraded"] is True
        assert unit._backoff <= 0.004  # capped, never unbounded
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_supervisor_lists_a_degraded_kiro_unit_as_started(caplog) -> None:
    """The supervisor must log 'started', never 'start failed' (h12)."""
    from reachy_nova.harness import supervisor

    factory = DeferredFactory(num_failures=1)
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0)
    stop_event = threading.Event()
    with caplog.at_level("INFO"):
        try:
            started = supervisor._start_components([unit], stop_event)
        finally:
            unit.stop(timeout=2.0)
            stop_event.set()
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert started == [unit]
    assert "started name=kiro_session" in messages
    assert "start failed name=kiro_session" not in messages


# --------------------------------------------------------------------------- #
# 8. request_restart(): the network-change trigger (task t5).                 #
# --------------------------------------------------------------------------- #


def test_request_restart_closes_the_session_and_respawns_immediately() -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(
        factory,
        cwd="/work",
        monitor_interval=0.01,
        backoff_initial_s=5.0,  # would be a 5s wait if the reset did not happen
        backoff_max_s=10.0,
    )
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        first = factory.built[0]

        unit.request_restart("joined ssid=iPhone (5)")

        assert _poll_until(lambda: factory.call_count >= 2, timeout=3.0)
        assert first.closed is True
        assert unit.is_alive() is True
        assert unit.status()["restarts"] == 1
        # Backoff was reset by the request, not doubled from a 5s ladder.
        assert unit._backoff == 5.0
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()


def test_request_restart_logs_the_reason(caplog) -> None:
    factory = FakeFactory()
    unit = KiroSessionUnit(factory, cwd="/work", monitor_interval=10.0)
    stop_event = threading.Event()
    with caplog.at_level("INFO"):
        try:
            unit.start(stop_event)
            unit.request_restart("moved ip=172.20.10.2")
        finally:
            unit.stop(timeout=2.0)
            stop_event.set()
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "restart requested reason=moved ip=172.20.10.2" in messages


def test_request_restart_before_start_is_safe() -> None:
    unit = KiroSessionUnit(FakeFactory(), cwd="/work", monitor_interval=10.0)
    unit.request_restart("joined")  # must not raise
    assert unit.status()["degraded"] is True


def test_request_restart_survives_a_close_that_raises() -> None:
    session = FailingSession(fail_at="never", error=RuntimeError("unused"))
    session._close_error = RuntimeError("close is broken")
    unit = KiroSessionUnit(lambda: session, cwd="/work", monitor_interval=10.0)
    stop_event = threading.Event()
    try:
        unit.start(stop_event)
        unit.request_restart("joined")  # must not raise
        assert unit.status()["degraded"] is True
    finally:
        unit.stop(timeout=2.0)
        stop_event.set()
