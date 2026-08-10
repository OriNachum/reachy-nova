"""Speaker path + mouth-loss grace (task t8) — ``reachy_nova/harness/speaking.py``.

``SonicSpeaker`` buffers Nova Sonic's 24 kHz float32 output chunks per
utterance, and on the transition out of ``"speaking"`` posts one complete WAV
to the daemon HTTP media route (upload + play_sound) via an injectable
``poster``. A single worker thread enforces one-speaker-at-a-time discipline
through the shared :class:`EchoGate`, and ANY playback failure routes to the
interruption path (gate cleared, pending queue emptied,
``on_playback_failure`` fired) so there is never a stuck speaking state.

All tests here use a fake poster — no network, no daemon.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import wave

import numpy as np
import pytest

from reachy_nova.harness.gate import EchoGate
from reachy_nova.harness.speaking import SonicSpeaker

SAMPLE_RATE = 24000


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


class RecordingPoster:
    """Fake HTTP transport: records every (wav_bytes, filename) upload+play."""

    def __init__(self, fail_times: int = 0):
        self.calls: list[dict] = []
        self.fail_times = fail_times
        self._lock = threading.Lock()

    def __call__(self, base_url: str, wav_bytes: bytes, filename: str) -> None:
        with self._lock:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise OSError("daemon unreachable (simulated)")
            self.calls.append(
                {
                    "base_url": base_url,
                    "wav": wav_bytes,
                    "filename": filename,
                    "t": time.monotonic(),
                }
            )

    def wait_for_calls(self, n: int, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.calls) >= n:
                    return True
            time.sleep(0.005)
        with self._lock:
            return len(self.calls) >= n


def parse_wav(wav_bytes: bytes) -> tuple[int, int, int, int]:
    """Return (channels, sampwidth, framerate, nframes) of a WAV byte blob."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()


def make_chunk(n_samples: int, value: float = 0.25) -> np.ndarray:
    return np.full(n_samples, value, dtype=np.float32)


def speak_utterance(speaker: SonicSpeaker, chunks: list[np.ndarray]) -> None:
    """Drive the sonic-callback sequence for one complete utterance."""
    speaker.on_state_change("speaking")
    for chunk in chunks:
        speaker.on_audio_chunk(chunk)
    speaker.on_state_change("listening")


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def stop_event():
    ev = threading.Event()
    yield ev
    ev.set()


# --------------------------------------------------------------------------- #
# 1. Happy path: one utterance -> exactly one well-formed WAV posted.         #
# --------------------------------------------------------------------------- #


def test_one_utterance_posts_exactly_one_wav(stop_event):
    gate = EchoGate(margin_s=0.05)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster)
    speaker.start(stop_event)
    try:
        # 0.2 s of audio fed as two chunks during "speaking".
        speak_utterance(speaker, [make_chunk(2400), make_chunk(2400)])
        assert poster.wait_for_calls(1)
        # Give a straggler post a chance to appear — there must be exactly one.
        time.sleep(0.05)
        assert len(poster.calls) == 1
        channels, sampwidth, framerate, nframes = parse_wav(poster.calls[0]["wav"])
        assert channels == 1
        assert sampwidth == 2  # int16
        assert framerate == SAMPLE_RATE
        assert nframes == 4800  # every fed sample, no more
        # Gate armed for ~duration (0.2 s) + margin (0.05 s).
        remaining = gate.remaining()
        assert 0.0 < remaining <= 0.2 + 0.05 + 0.02
        assert speaker.utterances_played == 1
        assert speaker.playback_failures == 0
    finally:
        stop_event.set()
        speaker.stop()


def test_wav_payload_is_the_clipped_int16_conversion(stop_event):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster)
    speaker.start(stop_event)
    try:
        # Include an out-of-range sample: must be clipped, not wrapped.
        samples = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(samples)
        speaker.on_state_change("listening")
        assert poster.wait_for_calls(1)
        with wave.open(io.BytesIO(poster.calls[0]["wav"]), "rb") as wf:
            frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        expected = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        assert np.array_equal(frames, expected)
    finally:
        stop_event.set()
        speaker.stop()


def test_no_post_without_a_speaking_transition(stop_event):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster)
    speaker.start(stop_event)
    try:
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(make_chunk(2400))
        # Still "speaking": nothing may be posted yet.
        time.sleep(0.1)
        assert poster.calls == []
        # Transition to "thinking" (any state that is not "speaking") flushes.
        speaker.on_state_change("thinking")
        assert poster.wait_for_calls(1)
    finally:
        stop_event.set()
        speaker.stop()


# --------------------------------------------------------------------------- #
# 2. Mouth-loss grace: poster failure -> interruption path, worker survives.  #
# --------------------------------------------------------------------------- #


def test_playback_failure_routes_to_interruption_path(stop_event, caplog):
    gate = EchoGate(margin_s=0.05)
    poster = RecordingPoster(fail_times=1)
    failures: list[float] = []
    speaker = SonicSpeaker(
        gate,
        sample_rate=SAMPLE_RATE,
        poster=poster,
        on_playback_failure=lambda: failures.append(time.monotonic()),
    )
    speaker.start(stop_event)
    try:
        # Pre-arm the gate so utterances A and B are BOTH pending when the
        # worker wakes up: A fails, B must be discarded (pending queue emptied).
        gate.arm_for(0.2)
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            speak_utterance(speaker, [make_chunk(2400)])  # A: 0.1 s -> fails
            speak_utterance(speaker, [make_chunk(1200)])  # B: 0.05 s -> dropped
            assert wait_until(lambda: speaker.playback_failures == 1)
            # Interruption semantics: gate cleared, callback fired, queue empty.
            assert wait_until(lambda: gate.remaining() == 0.0)
            assert len(failures) == 1
            assert wait_until(lambda: speaker.idle)
        assert any(
            "stage=speak" in rec.getMessage()
            and "reason=playback-http-failed" in rec.getMessage()
            for rec in caplog.records
        )
        # B was cleared, never posted.
        assert poster.calls == []
        # The worker survives: a following utterance plays normally.
        speak_utterance(speaker, [make_chunk(720)])  # C: 0.03 s
        assert poster.wait_for_calls(1)
        _, _, _, nframes = parse_wav(poster.calls[0]["wav"])
        assert nframes == 720  # it is C, not A or B
        assert speaker.utterances_played == 1
        assert len(failures) == 1  # no spurious extra interruptions
    finally:
        stop_event.set()
        speaker.stop()


def test_on_playback_failure_exception_does_not_kill_worker(stop_event):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster(fail_times=1)

    def bad_callback():
        raise RuntimeError("integrator bug")

    speaker = SonicSpeaker(
        gate, sample_rate=SAMPLE_RATE, poster=poster, on_playback_failure=bad_callback
    )
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(480)])
        assert wait_until(lambda: speaker.playback_failures == 1)
        speak_utterance(speaker, [make_chunk(480)])
        assert poster.wait_for_calls(1)
    finally:
        stop_event.set()
        speaker.stop()


# --------------------------------------------------------------------------- #
# 3. One-speaker discipline: second utterance waits for the first's window.   #
# --------------------------------------------------------------------------- #


def test_second_utterance_waits_for_first_gate_window(stop_event):
    gate = EchoGate(margin_s=0.05)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster)
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(3600)])  # A: 0.15 s
        speak_utterance(speaker, [make_chunk(1200)])  # B: 0.05 s
        assert poster.wait_for_calls(2)
        t_a = poster.calls[0]["t"]
        t_b = poster.calls[1]["t"]
        # B may not post until A's gate window (0.15 s + 0.05 s margin) elapses.
        assert t_b - t_a >= 0.15
        # And B's own window is armed after it plays.
        assert gate.remaining() > 0.0
    finally:
        stop_event.set()
        speaker.stop()


# --------------------------------------------------------------------------- #
# 4. Long-monologue safety: >cap buffer flushes mid-"speaking".               #
# --------------------------------------------------------------------------- #


def test_long_monologue_flushes_mid_speaking(stop_event):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    # A short cap keeps the test fast; the production default is ~15 s.
    speaker = SonicSpeaker(
        gate, sample_rate=SAMPLE_RATE, poster=poster, max_buffer_s=0.5
    )
    speaker.start(stop_event)
    try:
        speaker.on_state_change("speaking")
        # 0.6 s of audio fed while STILL "speaking" — exceeds the 0.5 s cap.
        for _ in range(6):
            speaker.on_audio_chunk(make_chunk(2400))  # 0.1 s each
        # No transition out of "speaking", yet a segment must flush and play.
        assert poster.wait_for_calls(1)
        _, _, _, nframes = parse_wav(poster.calls[0]["wav"])
        assert nframes >= int(0.5 * SAMPLE_RATE)
        # The remainder still flushes on the eventual transition.
        speaker.on_state_change("listening")
        assert poster.wait_for_calls(2, timeout=5.0)
        total = sum(parse_wav(c["wav"])[3] for c in poster.calls)
        assert total == 6 * 2400  # no samples lost across the split
    finally:
        stop_event.set()
        speaker.stop()


def test_default_cap_is_about_15_seconds():
    speaker = SonicSpeaker(EchoGate())
    assert 10.0 <= speaker.max_buffer_s <= 20.0


# --------------------------------------------------------------------------- #
# 5. Bounded queue: overflow is a named drop.                                 #
# --------------------------------------------------------------------------- #


def test_queue_overflow_drop_is_named(caplog):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    # Never started: no worker drains the queue, so it fills deterministically.
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, queue_size=2)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        for _ in range(3):
            speak_utterance(speaker, [make_chunk(240)])
    rendered = [rec.getMessage() for rec in caplog.records]
    drops = [line for line in rendered if "reason=queue-full" in line]
    assert len(drops) == 1
    assert all("stage=speak" in line and "source=nova" in line for line in drops)
    assert poster.calls == []


def test_default_queue_is_bounded():
    speaker = SonicSpeaker(EchoGate())
    assert speaker.queue_size == 8


# --------------------------------------------------------------------------- #
# 6. Lifecycle + observability.                                               #
# --------------------------------------------------------------------------- #


def test_idle_reflects_pending_and_in_flight_work(stop_event):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster)
    assert speaker.idle  # nothing queued, nothing playing, worker not started
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(480)])
        assert wait_until(lambda: speaker.idle)
        assert speaker.utterances_played == 1
    finally:
        stop_event.set()
        speaker.stop()


def test_stop_terminates_the_worker(stop_event):
    gate = EchoGate(margin_s=0.01)
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=RecordingPoster())
    speaker.start(stop_event)
    stop_event.set()
    speaker.stop()
    assert wait_until(lambda: not speaker.worker_alive)


def test_stop_is_idempotent(stop_event):
    speaker = SonicSpeaker(EchoGate(), poster=RecordingPoster())
    speaker.start(stop_event)
    speaker.stop()
    speaker.stop()  # must not raise


def test_queued_and_played_sense_lines_carry_duration(stop_event, caplog):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster)
    speaker.start(stop_event)
    try:
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            speak_utterance(speaker, [make_chunk(2400)])  # 0.1 s
            assert poster.wait_for_calls(1)
            assert wait_until(lambda: speaker.idle)
        rendered = [rec.getMessage() for rec in caplog.records]
        queued = [line for line in rendered if "stage=speak" in line and "queued" in line]
        played = [line for line in rendered if "stage=speak" in line and "played" in line]
        assert queued and "0.10" in queued[0]
        assert played
    finally:
        stop_event.set()
        speaker.stop()
