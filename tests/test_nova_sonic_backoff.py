"""Exponential restart backoff for Sonic stream death (t7).

With no network, opening a Bedrock bidirectional stream fails fast
(``AWS_IO_DNS_QUERY_FAILED``). The old fixed 3s retry reopened a stream every
~6s for as long as the robot stayed offline — noisy, and pointless (the
network is not coming back in the next 3 seconds).

This module covers:

1. ``NovaSonic._compute_restart_delay`` — exponential backoff (3s → 6 → 12 →
   24 → 48, capped at 60s) with small jitter, attempt numbering.
2. ``NovaSonic._maybe_reset_backoff_for_healthy_session`` — a session that
   proved itself (armed, heard at least one response, stayed up ≥60s)
   resets the backoff to the base delay on its next death.
3. ``NovaSonic.request_immediate_restart`` — resets the backoff to 0,
   idempotent, and (via ``_run_loop`` integration) forces a restart even of
   a currently healthy session.
4. ``NOVA_SONIC_RESTART_BASE_S`` / ``NOVA_SONIC_RESTART_MAX_S`` env parsing,
   defensive like ``_liveness_window``.

Style follows ``tests/test_sonic_resilience.py``: fake client/stream, no
real network connections, a fake clock + fast asyncio.sleep for the
integration tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import types

import pytest

from reachy_nova import nova_sonic
from reachy_nova.nova_sonic import NovaSonic

# --------------------------------------------------------------------------
# fakes (mirrors tests/test_sonic_resilience.py)
# --------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, wall: float = 1_700_000_000.0, mono: float = 1_000.0):
        self.wall = wall
        self.mono = mono

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds

    def as_module(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(time=self.time, monotonic=self.monotonic)


class _FastAsyncio:
    def __init__(self, clock: _FakeClock):
        self._clock = clock

    def __getattr__(self, name):  # pragma: no cover - trivial delegation
        return getattr(asyncio, name)

    async def sleep(self, delay: float = 0, result=None):
        self._clock.advance(delay or 0)
        await asyncio.sleep(0)
        return result


class _FakeInputStream:
    def __init__(self, sent: list[dict]):
        self._sent = sent
        self.closed = False

    async def send(self, chunk) -> None:
        self._sent.append(json.loads(chunk.value.bytes_.decode("utf-8"))["event"])

    async def close(self) -> None:
        self.closed = True


class _FakeStream:
    """Answers ``reply_after`` response events, then the stream dies.

    "Dies" means ``receive()`` raises — the same shape a real dropped
    connection takes, and what actually makes ``_process_responses``'s task
    complete (see ``NovaSonic._process_responses``: a non-"Invalid event
    bytes" exception breaks its loop). A ``receive()`` that merely hangs
    forever never completes the response task and is indistinguishable from
    a healthy, quiet connection — that shape belongs to the liveness-watchdog
    tests in ``test_sonic_resilience.py``, not here.
    """

    def __init__(self, reply_after: int = 0, on_first_receive=None, hang_after_death=False):
        self.sent: list[dict] = []
        self.input_stream = _FakeInputStream(self.sent)
        self._on_first_receive = on_first_receive
        self._received_once = False
        self._replies_left = reply_after
        self._hang = asyncio.Event()
        self._hang_after_death = hang_after_death

    async def await_output(self):
        return (None, self)

    async def receive(self):
        if not self._received_once:
            self._received_once = True
            if self._on_first_receive is not None:
                self._on_first_receive()
        if self._replies_left > 0:
            self._replies_left -= 1
            return _fake_response_chunk()
        if self._hang_after_death:
            await self._hang.wait()  # never set: used to stall out a running loop
        raise ConnectionError("simulated dropped Bedrock stream")

    def event_types(self) -> list[str]:
        return [next(iter(e.keys()), "unknown") for e in self.sent]


def _fake_response_chunk():
    """A minimal chunk shaped enough for _process_responses to call it a response."""

    class _Value:
        bytes_ = json.dumps({"event": {"completionStart": {}}}).encode("utf-8")

    class _Chunk:
        value = _Value()

    return _Chunk()


class _FailThenDieClient:
    """First N ``invoke_model_with_bidirectional_stream`` calls raise; then streams die immediately."""

    def __init__(self, fail_times: int = 0):
        self.streams: list[_FakeStream] = []
        self._fail_times = fail_times
        self._calls = 0

    async def invoke_model_with_bidirectional_stream(self, _operation_input):
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("AWS_IO_DNS_QUERY_FAILED")
        stream = _FakeStream()
        self.streams.append(stream)
        return stream


class _DyingStreamClient:
    """Every stream it hands out dies (receive() raises) right after opening."""

    def __init__(self):
        self.streams: list[_FakeStream] = []

    async def invoke_model_with_bidirectional_stream(self, _operation_input):
        stream = _FakeStream()
        self.streams.append(stream)
        return stream


class _HangingStreamClient:
    """Every stream it hands out looks perfectly healthy and never dies on its own."""

    def __init__(self):
        self.streams: list[_FakeStream] = []

    async def invoke_model_with_bidirectional_stream(self, _operation_input):
        stream = _FakeStream(hang_after_death=True)
        self.streams.append(stream)
        return stream


def _make_sonic(client, restart_rng: random.Random | None = None) -> NovaSonic:
    sonic = NovaSonic(
        system_prompt="You are Nova, a small curious robot.",
        restart_rng=restart_rng,
    )
    sonic._init_client = lambda: setattr(sonic, "_client", client)  # type: ignore[method-assign]
    return sonic


@pytest.fixture
def faketime(monkeypatch):
    monkeypatch.delenv("NOVA_SONIC_RESTART_BASE_S", raising=False)
    monkeypatch.delenv("NOVA_SONIC_RESTART_MAX_S", raising=False)
    clock = _FakeClock()
    monkeypatch.setattr(nova_sonic, "time", clock.as_module())
    monkeypatch.setattr(nova_sonic, "asyncio", _FastAsyncio(clock))
    return clock


def _run_loop_for(sonic: NovaSonic, fake_seconds: float, clock: _FakeClock) -> None:
    stop = threading.Event()

    async def _stopper() -> None:
        deadline = clock.mono + fake_seconds
        while clock.mono < deadline:
            await asyncio.sleep(0)
        stop.set()

    async def _drive() -> None:
        sonic._loop = asyncio.get_running_loop()
        stopper = asyncio.create_task(_stopper())
        try:
            await asyncio.wait_for(sonic._run_loop(stop), timeout=60)
        finally:
            stopper.cancel()

    asyncio.run(_drive())
    return stop


# --------------------------------------------------------------------------
# 1. _compute_restart_delay — unit level
# --------------------------------------------------------------------------


class TestComputeRestartDelay:
    def test_delay_sequence_for_five_consecutive_deaths(self, monkeypatch):
        monkeypatch.delenv("NOVA_SONIC_RESTART_BASE_S", raising=False)
        monkeypatch.delenv("NOVA_SONIC_RESTART_MAX_S", raising=False)
        sonic = NovaSonic(restart_rng=random.Random(0))

        expected_floor = [3.0, 6.0, 12.0, 24.0, 48.0]
        got = []
        for floor in expected_floor:
            delay, attempt = sonic._compute_restart_delay()
            got.append((delay, attempt))
            assert delay >= floor, f"expected >= {floor}, got {delay}"
            # jitter is at most 10% on top
            assert delay <= floor * 1.10 + 1e-9, f"expected <= {floor * 1.10}, got {delay}"

        assert [a for _, a in got] == [1, 2, 3, 4, 5]

    def test_capped_at_60s(self, monkeypatch):
        monkeypatch.delenv("NOVA_SONIC_RESTART_BASE_S", raising=False)
        monkeypatch.delenv("NOVA_SONIC_RESTART_MAX_S", raising=False)
        sonic = NovaSonic(restart_rng=random.Random(1))
        for _ in range(10):
            delay, _ = sonic._compute_restart_delay()
        assert delay == pytest.approx(60.0)

    def test_attempt_counter_increments_and_persists(self):
        sonic = NovaSonic(restart_rng=random.Random(2))
        _, a1 = sonic._compute_restart_delay()
        _, a2 = sonic._compute_restart_delay()
        _, a3 = sonic._compute_restart_delay()
        assert (a1, a2, a3) == (1, 2, 3)

    def test_jitter_is_never_negative_and_respects_base(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_RESTART_BASE_S", "3")
        sonic = NovaSonic(restart_rng=random.Random(3))
        delay, attempt = sonic._compute_restart_delay()
        assert attempt == 1
        assert 3.0 <= delay <= 3.3 + 1e-9


class TestBackoffHealthyReset:
    def test_resets_after_a_healthy_period(self):
        sonic = NovaSonic(restart_rng=random.Random(4))
        # Escalate a few times first.
        sonic._compute_restart_delay()
        sonic._compute_restart_delay()
        sonic._compute_restart_delay()
        assert sonic._restart_attempt == 3

        # Session armed, heard a response, ran for >= 60s before dying.
        sonic._arm_watchdogs(wall=2_000.0, mono=1_000.0)
        sonic._note_response_event(mono=1_010.0)
        sonic._maybe_reset_backoff_for_healthy_session(death_mono=1_061.0)

        assert sonic._restart_attempt == 0
        delay, attempt = sonic._compute_restart_delay()
        assert attempt == 1
        assert delay >= 3.0

    def test_does_not_reset_before_the_healthy_window(self):
        sonic = NovaSonic(restart_rng=random.Random(5))
        sonic._compute_restart_delay()
        sonic._compute_restart_delay()
        assert sonic._restart_attempt == 2

        sonic._arm_watchdogs(wall=2_000.0, mono=1_000.0)
        sonic._note_response_event(mono=1_010.0)
        # Only 59s alive — not enough.
        sonic._maybe_reset_backoff_for_healthy_session(death_mono=1_059.0)

        assert sonic._restart_attempt == 2

    def test_does_not_reset_without_any_sign_of_life(self):
        """A session that ran for a long time but never actually heard back
        is not "healthy" — it looks exactly like a zombie."""
        sonic = NovaSonic(restart_rng=random.Random(6))
        sonic._compute_restart_delay()
        sonic._compute_restart_delay()
        assert sonic._restart_attempt == 2

        sonic._arm_watchdogs(wall=2_000.0, mono=1_000.0)
        # No _note_response_event call.
        sonic._maybe_reset_backoff_for_healthy_session(death_mono=1_200.0)

        assert sonic._restart_attempt == 2

    def test_unarmed_session_is_a_no_op(self):
        sonic = NovaSonic(restart_rng=random.Random(7))
        sonic._compute_restart_delay()
        assert sonic._restart_attempt == 1
        sonic._maybe_reset_backoff_for_healthy_session(death_mono=999_999.0)
        assert sonic._restart_attempt == 1


# --------------------------------------------------------------------------
# 2. request_immediate_restart — unit level
# --------------------------------------------------------------------------


class TestRequestImmediateRestart:
    def test_resets_backoff_to_zero(self):
        sonic = NovaSonic(restart_rng=random.Random(8))
        sonic._compute_restart_delay()
        sonic._compute_restart_delay()
        sonic._compute_restart_delay()
        assert sonic._restart_attempt == 3

        sonic.request_immediate_restart("network changed")

        assert sonic._restart_attempt == 0
        delay, attempt = sonic._compute_restart_delay()
        assert attempt == 1
        assert delay >= 3.0

    def test_sets_the_event_and_reason(self):
        sonic = NovaSonic()
        assert not sonic._restart_now_event.is_set()
        sonic.request_immediate_restart("network changed")
        assert sonic._restart_now_event.is_set()
        assert sonic._immediate_restart_reason == "network changed"

    def test_idempotent_when_called_twice_quickly(self):
        sonic = NovaSonic(restart_rng=random.Random(9))
        sonic._compute_restart_delay()
        sonic.request_immediate_restart("first reason")
        sonic.request_immediate_restart("second reason")

        assert sonic._restart_attempt == 0
        assert sonic._restart_now_event.is_set()
        assert sonic._immediate_restart_reason == "second reason"

    def test_callable_from_another_thread(self):
        sonic = NovaSonic()
        errors: list[Exception] = []

        def _call():
            try:
                sonic.request_immediate_restart("from another thread")
            except Exception as e:  # pragma: no cover - failure path only
                errors.append(e)

        t = threading.Thread(target=_call)
        t.start()
        t.join(timeout=5)

        assert not errors
        assert sonic._restart_now_event.is_set()


# --------------------------------------------------------------------------
# 3. env parsing — defensive like _liveness_window
# --------------------------------------------------------------------------


class TestEnvParsing:
    def test_base_defaults_to_3s(self, monkeypatch):
        monkeypatch.delenv("NOVA_SONIC_RESTART_BASE_S", raising=False)
        assert nova_sonic._restart_base_s() == 3.0

    def test_max_defaults_to_60s(self, monkeypatch):
        monkeypatch.delenv("NOVA_SONIC_RESTART_MAX_S", raising=False)
        assert nova_sonic._restart_max_s() == 60.0

    def test_base_is_overridable(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_RESTART_BASE_S", "1")
        assert nova_sonic._restart_base_s() == 1.0

    def test_max_is_overridable(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_RESTART_MAX_S", "10")
        assert nova_sonic._restart_max_s() == 10.0

    def test_garbage_base_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_RESTART_BASE_S", "not-a-number")
        assert nova_sonic._restart_base_s() == 3.0
        monkeypatch.setenv("NOVA_SONIC_RESTART_BASE_S", "-1")
        assert nova_sonic._restart_base_s() == 3.0
        monkeypatch.setenv("NOVA_SONIC_RESTART_BASE_S", "0")
        assert nova_sonic._restart_base_s() == 3.0

    def test_garbage_max_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_RESTART_MAX_S", "not-a-number")
        assert nova_sonic._restart_max_s() == 60.0
        monkeypatch.setenv("NOVA_SONIC_RESTART_MAX_S", "0")
        assert nova_sonic._restart_max_s() == 60.0

    def test_compute_restart_delay_honours_env_override(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_RESTART_BASE_S", "1")
        monkeypatch.setenv("NOVA_SONIC_RESTART_MAX_S", "5")
        sonic = NovaSonic(restart_rng=random.Random(10))
        d1, _ = sonic._compute_restart_delay()  # 1
        d2, _ = sonic._compute_restart_delay()  # 2
        d3, _ = sonic._compute_restart_delay()  # 4
        d4, _ = sonic._compute_restart_delay()  # capped at 5
        assert 1.0 <= d1 <= 1.1 + 1e-9
        assert 2.0 <= d2 <= 2.2 + 1e-9
        assert 4.0 <= d3 <= 4.4 + 1e-9
        assert d4 == pytest.approx(5.0)


# --------------------------------------------------------------------------
# 4. integration — the backoff actually drives the existing restart path
# --------------------------------------------------------------------------


class TestBackoffIntegration:
    def test_repeated_stream_death_backs_off(self, faketime, caplog):
        """Repeated deaths (no healthy period in between) escalate the delay."""
        client = _DyingStreamClient()
        sonic = _make_sonic(client, restart_rng=random.Random(11))

        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, fake_seconds=200.0, clock=faketime)

        # 3 + 6 + 12 ~= 21s of backoff sleeps fit inside 200 fake seconds,
        # each restart opens one more stream than the last.
        assert len(client.streams) >= 4
        warnings = [
            r.getMessage()
            for r in caplog.records
            if "restarting session" in r.getMessage()
        ]
        assert len(warnings) >= 3
        assert "attempt 1" in warnings[0]
        assert "attempt 2" in warnings[1]
        assert "attempt 3" in warnings[2]

    def test_session_start_failure_also_backs_off(self, faketime, caplog):
        client = _FailThenDieClient(fail_times=3)
        sonic = _make_sonic(client, restart_rng=random.Random(12))

        with caplog.at_level(logging.ERROR, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, fake_seconds=100.0, clock=faketime)

        errors = [
            r.getMessage()
            for r in caplog.records
            if "Session start failed" in r.getMessage()
        ]
        assert len(errors) == 3
        assert "attempt 1" in errors[0]
        assert "attempt 2" in errors[1]
        assert "attempt 3" in errors[2]
        # eventually the client stops raising and a real stream opens
        assert len(client.streams) >= 1

    def test_immediate_restart_fires_even_when_healthy(self, faketime):
        """A session with an open, silent-but-not-yet-stalled stream must
        still restart the moment request_immediate_restart() is called."""
        client = _HangingStreamClient()
        sonic = _make_sonic(client, restart_rng=random.Random(13))

        stop = threading.Event()

        async def _driver():
            sonic._loop = asyncio.get_running_loop()

            async def _requester():
                # Give the loop a couple of ticks to open the first stream
                # and settle into the wait loop, then fire the request.
                for _ in range(5):
                    await asyncio.sleep(0)
                sonic.request_immediate_restart("network changed")
                for _ in range(50):
                    if len(client.streams) >= 2:
                        break
                    await asyncio.sleep(0)
                stop.set()

            req_task = asyncio.create_task(_requester())
            try:
                await asyncio.wait_for(sonic._run_loop(stop), timeout=10)
            finally:
                req_task.cancel()

        asyncio.run(_driver())

        assert len(client.streams) >= 2, "expected an immediate restart to open a new stream"

    def test_immediate_restart_uses_zero_delay(self, faketime, caplog):
        client = _HangingStreamClient()
        sonic = _make_sonic(client, restart_rng=random.Random(14))
        stop = threading.Event()

        async def _driver():
            sonic._loop = asyncio.get_running_loop()

            async def _requester():
                for _ in range(5):
                    await asyncio.sleep(0)
                sonic.request_immediate_restart("network changed")
                for _ in range(50):
                    if len(client.streams) >= 2:
                        break
                    await asyncio.sleep(0)
                stop.set()

            req_task = asyncio.create_task(_requester())
            try:
                await asyncio.wait_for(sonic._run_loop(stop), timeout=10)
            finally:
                req_task.cancel()

        start_mono = faketime.mono
        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            asyncio.run(_driver())

        # No 3s+ backoff sleep should have elapsed on the fake clock for the
        # immediate restart itself — the loop's 0.1s poll ticks dominate.
        assert faketime.mono - start_mono < 3.0
        messages = [r.getMessage() for r in caplog.records]
        assert any("Immediate restart requested" in m for m in messages)


# --------------------------------------------------------------------------- #
# start() never propagates a failure to the supervisor (task t5)              #
# --------------------------------------------------------------------------- #


class TestStartNeverRaises:
    """A network-less start must degrade, never fail the component.

    The supervisor treats a raising ``start()`` as ``start failed name=...``
    and NEVER retries it — the exact shape that left the Kiro writer absent
    for hours on 2026-08-26. Sonic's Bedrock connection is made inside
    ``_run_loop`` on its own retrying thread, so the only way ``start()``
    could raise at all is the thread spawn itself; that is guarded too.
    """

    def test_start_returns_and_names_the_degradation_when_the_thread_cannot_spawn(
        self, monkeypatch, caplog
    ):
        import reachy_nova.nova_sonic as nova_sonic

        sonic = NovaSonic(system_prompt="hi")

        class _RefusingThread:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("can't start new thread")

        monkeypatch.setattr(nova_sonic.threading, "Thread", _RefusingThread)

        with caplog.at_level("INFO"):
            sonic.start(threading.Event())  # must NOT raise

        text = " | ".join(r.getMessage() for r in caplog.records)
        assert "component degraded name=sonic" in text
        assert sonic._thread is None

    def test_start_spawns_the_thread_when_it_can(self):
        sonic = NovaSonic(system_prompt="hi")
        stop_event = threading.Event()
        stop_event.set()  # the loop exits immediately; we only assert the spawn
        sonic.start(stop_event)
        assert sonic._thread is not None
        sonic._thread.join(timeout=5.0)
