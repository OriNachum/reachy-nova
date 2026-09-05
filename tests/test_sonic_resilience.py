"""Resilience watchdogs for the Nova Sonic bidirectional stream (t1).

The robot is a Raspberry Pi with no RTC. It can boot with a stale clock and
have NTP step time forward by hours a minute later. When that happened, the
already-open Bedrock bidirectional stream turned into a *zombie*: injects and
audio kept flowing out, zero response events ever came back, and no error was
ever raised — so the existing "stream died" restart path never triggered and
the robot stayed mute until somebody restarted the service by hand.

Two watchdogs close that hole, both riding the *existing* restart path in
``NovaSonic._run_loop`` (fresh client, fresh UUIDs, system prompt only — no
conversation recap):

1. **Clock-step detector** — the wall/monotonic offset captured at session
   start is compared every tick; a drift over 60s in either direction means
   the wall clock was stepped underneath a live stream, so restart.
2. **Response-liveness watchdog** — if input was sent since the last response
   event and no response event has arrived for the liveness window (default
   180s, ``NOVA_SONIC_LIVENESS_S``), the stream is a zombie, so restart.

Both must fire *at most once per cause* and both must re-arm on the new
session. Neither may fire in a quiet room where nothing was sent.

"Input" is the load-bearing word in watchdog 2, and the first version got it
wrong: the harness feeds the microphone to Sonic 10x/s whether or not anybody
is talking, so "input flowing, no response for 180s" meant "nobody spoke for
three minutes" and the robot's 2026-09-05 journal carried six forced restarts
exactly 180s apart in a quiet room — each one a clean session that wiped the
conversation. A mic chunk therefore counts as input only when its RMS clears
``NOVA_SONIC_SPEECH_FLOOR`` (default 0.02, between the measured ~0.002 room
floor and the ~0.09 RMS of human speech on this mic — see
``harness/hearing.py``); an inject or a tool result still counts
unconditionally, because those really are us talking to Bedrock.

Time is faked throughout (a ``_FakeClock`` swapped into the module's ``time``
slot plus an ``asyncio`` proxy whose ``sleep`` advances that clock) so the
integration tests cover minutes of simulated time in milliseconds of real
time, without sleeping and without touching the real event-loop clock.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import types

import numpy as np
import pytest

from reachy_nova import nova_sonic
from reachy_nova.nova_sonic import NovaSonic

# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class _FakeClock:
    """A wall clock and a monotonic clock that can be moved independently."""

    def __init__(self, wall: float = 1_700_000_000.0, mono: float = 1_000.0):
        self.wall = wall
        self.mono = mono

    # -- the two functions nova_sonic reads off its ``time`` module --------
    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        """Ordinary passage of time — both clocks move together."""
        self.wall += seconds
        self.mono += seconds

    def step_wall(self, seconds: float) -> None:
        """An NTP step: the wall clock jumps, monotonic does not."""
        self.wall += seconds

    def as_module(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(time=self.time, monotonic=self.monotonic)


class _FastAsyncio:
    """``asyncio`` proxy whose ``sleep`` advances the fake clock instead of waiting.

    Every other attribute is delegated to the real module, so ``Lock``,
    ``create_task``, ``timeout`` and friends keep working untouched.
    """

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
    """A bidirectional stream that accepts everything and answers nothing.

    ``receive`` blocks forever after invoking ``on_first_receive`` — the exact
    shape of the zombie: the send side stays healthy, the response side is
    silent, and nothing ever raises.
    """

    def __init__(self, on_first_receive=None):
        self.sent: list[dict] = []
        self.input_stream = _FakeInputStream(self.sent)
        self._on_first_receive = on_first_receive
        self._received_once = False
        self._silent = asyncio.Event()

    async def await_output(self):
        return (None, self)

    async def receive(self):
        if not self._received_once:
            self._received_once = True
            if self._on_first_receive is not None:
                self._on_first_receive()
        await self._silent.wait()  # never set: no response event, ever

    def event_types(self) -> list[str]:
        return [next(iter(e.keys()), "unknown") for e in self.sent]


class _FakeClient:
    """Hands out a fresh ``_FakeStream`` per session and counts sessions."""

    def __init__(self, first_stream_hook=None):
        self.streams: list[_FakeStream] = []
        self._first_stream_hook = first_stream_hook

    async def invoke_model_with_bidirectional_stream(self, _operation_input):
        hook = self._first_stream_hook if not self.streams else None
        stream = _FakeStream(on_first_receive=hook)
        self.streams.append(stream)
        return stream


def _make_sonic(client: _FakeClient) -> NovaSonic:
    sonic = NovaSonic(system_prompt="You are Nova, a small curious robot.")
    sonic._init_client = lambda: setattr(sonic, "_client", client)  # type: ignore[method-assign]
    return sonic


def _run_loop_for(
    sonic: NovaSonic,
    clock: _FakeClock,
    fake_seconds: float,
    injector: bool = False,
) -> None:
    """Drive ``_run_loop`` until ``fake_seconds`` of simulated time have passed.

    The loop itself moves the clock (its 0.1s tick sleep and its 3s restart
    sleep both go through ``_FastAsyncio``), so a stopper task only has to
    watch the clock and set the stop event. When ``injector`` is set, a task
    keeps calling ``inject_text`` — input flowing into a stream that never
    answers.
    """
    stop = threading.Event()

    async def _stopper() -> None:
        deadline = clock.mono + fake_seconds
        while clock.mono < deadline:
            await asyncio.sleep(0)
        stop.set()

    async def _injector() -> None:
        # Every 5 fake seconds: one text inject (so the stream really is being
        # fed) AND half a second of speech-level mic audio — since the
        # speech-burst gate (2026-09-06) only sustained speech counts as
        # liveness input; a text inject alone no longer does.
        next_at = clock.wall
        while not stop.is_set():
            if clock.wall >= next_at:
                sonic.inject_text("I see someone moving")
                if sonic._active and sonic._loop is not None:
                    for _ in range(nova_sonic.SPEECH_BURST_CHUNKS):
                        sonic.feed_audio(_speech())
                next_at = clock.wall + 5.0
            await asyncio.sleep(0)

    async def _drive() -> None:
        sonic._loop = asyncio.get_running_loop()
        helpers = [asyncio.create_task(_stopper())]
        if injector:
            helpers.append(asyncio.create_task(_injector()))
        try:
            await asyncio.wait_for(sonic._run_loop(stop), timeout=60)
        finally:
            for task in helpers:
                task.cancel()

    asyncio.run(_drive())


def _restart_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "reachy_nova.nova_sonic"
        and r.levelno >= logging.WARNING
        and "restarting session" in r.getMessage()
    ]


@pytest.fixture
def faketime(monkeypatch):
    """Swap the module's ``time`` and ``asyncio`` for fake-clock-driven ones."""
    # The integration tests below rely on the 180s default window, so an
    # ambient override in the developer's environment must not leak in.
    monkeypatch.delenv("NOVA_SONIC_LIVENESS_S", raising=False)
    # These cases run up to 600 fake seconds, past the 420s default rotation
    # interval (t12) — a rotation there would be correct behaviour and a
    # second restart these assertions do not expect. Rotation has its own
    # file (tests/test_sonic_rotation.py); here it is switched off so each
    # restart counted below is a *watchdog's* doing and nothing else.
    monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "0")
    clock = _FakeClock()
    monkeypatch.setattr(nova_sonic, "time", clock.as_module())
    monkeypatch.setattr(nova_sonic, "asyncio", _FastAsyncio(clock))
    return clock


# --------------------------------------------------------------------------
# 1. clock-step detector — unit level
# --------------------------------------------------------------------------


class TestClockStepDetector:
    def _armed(self, wall: float = 1_000.0, mono: float = 500.0) -> NovaSonic:
        sonic = NovaSonic()
        sonic._arm_watchdogs(wall, mono)
        return sonic

    def test_no_step_means_no_restart(self):
        sonic = self._armed()
        # 10 minutes of ordinary time: both clocks move together.
        assert sonic._check_clock_step(1_600.0, 1_100.0) is False

    def test_drift_under_the_threshold_is_tolerated(self):
        sonic = self._armed()
        # 59s of wall-only drift — under the 60s threshold, so no restart.
        assert sonic._check_clock_step(1_000.0 + 59.0, 500.0) is False

    def test_forward_step_requests_a_restart(self, caplog):
        sonic = self._armed()
        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            stepped = sonic._check_clock_step(1_000.0 + 14 * 3600, 500.0)
        assert stepped is True
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1
        assert "clock step" in messages[0].lower()
        assert "50400" in messages[0]  # the delta, in seconds

    def test_backward_step_requests_a_restart(self, caplog):
        sonic = self._armed()
        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            stepped = sonic._check_clock_step(1_000.0 - 3600.0, 500.0)
        assert stepped is True
        assert "clock step" in caplog.records[0].getMessage().lower()

    def test_fires_exactly_once_per_step(self, caplog):
        sonic = self._armed()
        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            first = sonic._check_clock_step(1_000.0 + 50_000.0, 500.0)
            repeats = [
                sonic._check_clock_step(1_000.0 + 50_000.0 + i, 500.0 + i)
                for i in range(1, 20)
            ]
        assert first is True
        assert not any(repeats)
        assert len(caplog.records) == 1

    def test_rearms_on_the_next_session(self):
        sonic = self._armed()
        assert sonic._check_clock_step(1_000.0 + 50_000.0, 500.0) is True
        # A new session re-captures the offset against the *stepped* clock.
        sonic._arm_watchdogs(1_000.0 + 50_000.0, 500.0)
        assert sonic._check_clock_step(1_000.0 + 50_010.0, 510.0) is False
        assert sonic._check_clock_step(1_000.0 + 100_000.0, 510.0) is True


# --------------------------------------------------------------------------
# 2. response-liveness watchdog — unit level
# --------------------------------------------------------------------------


class TestResponseLivenessWatchdog:
    @pytest.fixture(autouse=True)
    def _default_window(self, monkeypatch):
        """Most cases here assert against the 180s default, so drop any override."""
        monkeypatch.delenv("NOVA_SONIC_LIVENESS_S", raising=False)

    def _armed(self, mono: float = 500.0) -> NovaSonic:
        sonic = NovaSonic()
        sonic._arm_watchdogs(1_000.0, mono)
        return sonic

    def test_quiet_room_never_trips(self):
        """Nothing sent → silence from Bedrock is expected, not a stall."""
        sonic = self._armed()
        assert sonic._check_response_liveness(500.0 + 10_000.0) is False

    def test_input_but_still_inside_the_window_does_not_trip(self):
        sonic = self._armed()
        sonic._note_input_sent()
        assert sonic._check_response_liveness(500.0 + 179.0) is False

    def test_input_with_no_response_past_the_window_trips(self, caplog):
        sonic = self._armed()
        sonic._note_input_sent()
        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            stalled = sonic._check_response_liveness(500.0 + 181.0)
        assert stalled is True
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1
        assert "liveness" in messages[0].lower()
        assert "181" in messages[0]

    def test_a_response_event_resets_the_timer(self):
        sonic = self._armed()
        sonic._note_input_sent()
        sonic._note_response_event(500.0 + 100.0)
        # 181s after arming, but only 81s after the last response event.
        assert sonic._check_response_liveness(500.0 + 181.0) is False
        # ...and the response cleared "input since last response" too.
        assert sonic._check_response_liveness(500.0 + 100.0 + 181.0) is False

    def test_fires_exactly_once_per_stall(self, caplog):
        sonic = self._armed()
        sonic._note_input_sent()
        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            first = sonic._check_response_liveness(500.0 + 181.0)
            repeats = [
                sonic._check_response_liveness(500.0 + 181.0 + i) for i in range(1, 20)
            ]
        assert first is True
        assert not any(repeats)
        assert len(caplog.records) == 1

    def test_window_is_overridable_by_env(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_LIVENESS_S", "20")
        assert nova_sonic._liveness_window() == 20.0
        sonic = self._armed()
        sonic._note_input_sent()
        assert sonic._check_response_liveness(500.0 + 19.0) is False
        assert sonic._check_response_liveness(500.0 + 21.0) is True

    def test_window_defaults_to_180s(self, monkeypatch):
        monkeypatch.delenv("NOVA_SONIC_LIVENESS_S", raising=False)
        assert nova_sonic._liveness_window() == 180.0

    def test_garbage_env_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_LIVENESS_S", "not-a-number")
        assert nova_sonic._liveness_window() == 180.0
        monkeypatch.setenv("NOVA_SONIC_LIVENESS_S", "0")
        assert nova_sonic._liveness_window() == 180.0

    def test_feed_audio_without_a_loop_claims_nothing(self):
        """No loop → nothing was sent, so nothing may be claimed as input."""
        sonic = self._armed()
        sonic._active = True
        sonic._loop = None  # feed_audio bails out before scheduling anything
        sonic.feed_audio(_speech())
        assert sonic._input_since_response is False


# --------------------------------------------------------------------------
# 2b. speech-gated input — a quiet room is not a zombie stream
# --------------------------------------------------------------------------


def _tone(rms: float, seconds: float = 0.1) -> np.ndarray:
    """One mic chunk at a given RMS, shaped like the hearing leg's output.

    100 ms of float32 mono at 16 kHz — exactly what ``TeeHearing`` hands to
    ``feed_audio`` ten times a second.
    """
    n = int(nova_sonic.INPUT_SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / nova_sonic.INPUT_SAMPLE_RATE
    return (rms * np.sqrt(2.0) * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _room_tone() -> np.ndarray:
    """The measured quiet-room floor on this mic (~0.002 RMS)."""
    return _tone(0.002)


def _speech() -> np.ndarray:
    """The measured RMS of human speech on this mic (~0.09)."""
    return _tone(0.09)


class _Feeder:
    """A live-enough NovaSonic whose sends are captured instead of awaited."""

    def __init__(self, monkeypatch, mono: float = 500.0):
        self.sonic = NovaSonic()
        self.sonic._arm_watchdogs(1_000.0, mono)
        self.sonic._active = True
        self.sonic._loop = types.SimpleNamespace()
        self.sent: list = []
        monkeypatch.setattr(nova_sonic.asyncio, "run_coroutine_threadsafe", self._capture)

    def _capture(self, coro, loop):
        coro.close()  # never awaited — we only care that it was scheduled
        self.sent.append(coro)
        return None


class TestSpeechGatedInput:
    @pytest.fixture(autouse=True)
    def _default_floor(self, monkeypatch):
        monkeypatch.delenv("NOVA_SONIC_SPEECH_FLOOR", raising=False)
        monkeypatch.delenv("NOVA_SONIC_LIVENESS_S", raising=False)

    # -- the floor itself --------------------------------------------------

    def test_floor_defaults_to_0_02(self, monkeypatch):
        assert nova_sonic._speech_floor() == 0.02

    def test_floor_is_overridable_by_env(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_SPEECH_FLOOR", "0.05")
        assert nova_sonic._speech_floor() == 0.05

    def test_garbage_floor_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_SPEECH_FLOOR", "not-a-number")
        assert nova_sonic._speech_floor() == 0.02
        monkeypatch.setenv("NOVA_SONIC_SPEECH_FLOOR", "-1")
        assert nova_sonic._speech_floor() == 0.02

    # -- what feed_audio does with it --------------------------------------

    def test_silence_is_still_sent_but_is_not_input(self, monkeypatch):
        feeder = _Feeder(monkeypatch)
        feeder.sonic.feed_audio(np.zeros(1600, dtype=np.float32))
        assert feeder.sent, "the audio itself is sent either way"
        assert feeder.sonic._input_since_response is False

    def test_room_tone_is_not_input(self, monkeypatch):
        feeder = _Feeder(monkeypatch)
        feeder.sonic.feed_audio(_room_tone())
        assert feeder.sent
        assert feeder.sonic._input_since_response is False

    def test_a_single_loud_chunk_is_not_input(self, monkeypatch):
        """A cough, a door: one 100 ms chunk above the floor is not speech."""
        feeder = _Feeder(monkeypatch)
        feeder.sonic.feed_audio(_speech())
        assert feeder.sent
        assert feeder.sonic._input_since_response is False

    def test_a_burst_of_speech_is_input(self, monkeypatch):
        """Half a second of speech-level audio within two seconds is a person."""
        feeder = _Feeder(monkeypatch)
        for _ in range(nova_sonic.SPEECH_BURST_CHUNKS):
            feeder.sonic.feed_audio(_speech())
        assert feeder.sonic._input_since_response is True

    def test_a_burst_spread_thin_is_not_input(self, monkeypatch):
        """Loud chunks further apart than the window never add up."""
        feeder = _Feeder(monkeypatch)
        for _ in range(nova_sonic.SPEECH_BURST_CHUNKS):
            feeder.sonic.feed_audio(_speech())
            for _ in range(nova_sonic.SPEECH_BURST_WINDOW):
                feeder.sonic.feed_audio(_room_tone())
        assert feeder.sonic._input_since_response is False

    def test_the_floor_moves_with_the_env(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_SPEECH_FLOOR", "0.5")
        feeder = _Feeder(monkeypatch)
        feeder.sonic.feed_audio(_speech())  # 0.09 RMS, now below the floor
        assert feeder.sonic._input_since_response is False

    def test_a_stereo_chunk_is_measured_after_the_downmix(self, monkeypatch):
        """Stereo speech is still speech (and an empty chunk is not)."""
        feeder = _Feeder(monkeypatch)
        mono = _speech()
        for _ in range(nova_sonic.SPEECH_BURST_CHUNKS):
            feeder.sonic.feed_audio(np.stack([mono, mono], axis=1))
        assert feeder.sonic._input_since_response is True

    def test_an_empty_chunk_is_harmless(self, monkeypatch):
        feeder = _Feeder(monkeypatch)
        feeder.sonic.feed_audio(np.zeros(0, dtype=np.float32))
        assert feeder.sonic._input_since_response is False

    # -- and what the watchdog therefore concludes -------------------------

    def test_a_quiet_room_fed_past_the_window_never_trips(self, monkeypatch):
        """The regression: 200s of mic chunks with nobody talking is not a stall."""
        feeder = _Feeder(monkeypatch)
        for _ in range(2000):  # 200s at 10 chunks/s
            feeder.sonic.feed_audio(_room_tone())
        assert len(feeder.sent) == 2000, "every chunk still reached the stream"
        assert feeder.sonic._check_response_liveness(500.0 + 200.0) is False

    def test_one_word_then_silence_still_trips(self, monkeypatch, caplog):
        """Somebody spoke and Bedrock never answered — that IS the zombie."""
        feeder = _Feeder(monkeypatch)
        for _ in range(1000):
            feeder.sonic.feed_audio(_room_tone())
        assert feeder.sonic._input_since_response is False, "the mic alone claimed nothing"
        for _ in range(nova_sonic.SPEECH_BURST_CHUNKS):
            feeder.sonic.feed_audio(_speech())  # one word, half a second of it
        for _ in range(999 - nova_sonic.SPEECH_BURST_CHUNKS):
            feeder.sonic.feed_audio(_room_tone())
        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            stalled = feeder.sonic._check_response_liveness(500.0 + 200.0)
        assert stalled is True
        assert "liveness" in caplog.records[0].getMessage().lower()

    def test_an_inject_alone_is_not_input(self, monkeypatch):
        """A quiet body cue legitimately gets no answer — it must not look like a zombie."""
        feeder = _Feeder(monkeypatch)
        for _ in range(2000):
            feeder.sonic.feed_audio(_room_tone())
        assert feeder.sonic._input_since_response is False, "the mic alone claimed nothing"
        feeder.sonic.inject_text("(someone is petting you)")
        assert feeder.sonic._input_since_response is False
        assert feeder.sonic._check_response_liveness(500.0 + 200.0) is False

    def test_a_tool_result_alone_is_not_input(self, monkeypatch):
        feeder = _Feeder(monkeypatch)
        for _ in range(2000):
            feeder.sonic.feed_audio(_room_tone())
        assert feeder.sonic._input_since_response is False, "the mic alone claimed nothing"
        feeder.sonic.send_tool_result("tu-1", "done")
        assert feeder.sonic._input_since_response is False
        assert feeder.sonic._check_response_liveness(500.0 + 200.0) is False


# --------------------------------------------------------------------------
# 3. integration: the watchdogs actually drive the existing restart path
# --------------------------------------------------------------------------


class TestForcedRestartIntegration:
    def test_clock_step_forces_exactly_one_restart(self, faketime, caplog):
        """NTP steps time +14h under a live stream → one clean restart."""
        client = _FakeClient(first_stream_hook=lambda: faketime.step_wall(14 * 3600))
        sonic = _make_sonic(client)

        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, faketime, fake_seconds=600.0)

        assert len(client.streams) == 2, "expected exactly one restart"
        warnings = _restart_warnings(caplog)
        assert len(warnings) == 1
        assert "clock step" in warnings[0].lower()

    def test_clock_step_restart_is_clean_system_prompt_only(self, faketime):
        """The forced restart replays the system prompt and nothing else."""
        client = _FakeClient(first_stream_hook=lambda: faketime.step_wall(14 * 3600))
        sonic = _make_sonic(client)
        sonic.last_user_text = "RECAP-USER-SENTINEL"
        sonic.last_assistant_text = "RECAP-ASSISTANT-SENTINEL"

        _run_loop_for(sonic, faketime, fake_seconds=600.0)

        assert len(client.streams) == 2
        fresh = client.streams[1]
        dumped = json.dumps(fresh.sent)
        assert "RECAP-USER-SENTINEL" not in dumped
        assert "RECAP-ASSISTANT-SENTINEL" not in dumped

        text_inputs = [e["textInput"] for e in fresh.sent if "textInput" in e]
        assert len(text_inputs) == 1, "a clean restart sends exactly one text input"
        assert text_inputs[0]["content"] == sonic.system_prompt

        # ...and it is the SYSTEM role content that carries it.
        roles = [e["contentStart"].get("role") for e in fresh.sent if "contentStart" in e]
        assert roles == ["SYSTEM", "USER"]  # system prompt, then the audio channel

    def test_zombie_stream_forces_exactly_one_restart(self, faketime, caplog):
        """Injects keep flowing, response events are muted → one forced restart."""
        client = _FakeClient()
        sonic = _make_sonic(client)

        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            # 300 fake seconds: long enough for one 180s stall + the 3s restart,
            # too short for the fresh session to stall a second time.
            _run_loop_for(sonic, faketime, fake_seconds=300.0, injector=True)

        assert len(client.streams) == 2, "expected exactly one restart"
        warnings = _restart_warnings(caplog)
        assert len(warnings) == 1
        assert "liveness" in warnings[0].lower()

        # The zombie really was being fed while it stayed silent.
        assert client.streams[0].event_types().count("textInput") > 1

    def test_zombie_restart_is_clean_system_prompt_only(self, faketime):
        client = _FakeClient()
        sonic = _make_sonic(client)
        sonic.last_assistant_text = "RECAP-ASSISTANT-SENTINEL"

        _run_loop_for(sonic, faketime, fake_seconds=190.0, injector=True)

        assert len(client.streams) == 2
        fresh = client.streams[1]
        assert "RECAP-ASSISTANT-SENTINEL" not in json.dumps(fresh.sent)
        text_inputs = [e["textInput"] for e in fresh.sent if "textInput" in e]
        assert text_inputs and text_inputs[0]["content"] == sonic.system_prompt

    def test_quiet_room_is_never_restarted(self, faketime, caplog):
        """No input sent → 10 fake minutes of silence is not a stall."""
        client = _FakeClient()
        sonic = _make_sonic(client)

        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, faketime, fake_seconds=600.0)

        assert len(client.streams) == 1, "a quiet room must not be restarted"
        assert _restart_warnings(caplog) == []
