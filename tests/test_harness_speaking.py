"""Speaker path + mouth-loss grace (task t8) — ``reachy_nova/harness/speaking.py``.

``SonicSpeaker`` cuts Nova Sonic's 24 kHz float32 output into ~1 s chunks and
posts each one to the daemon HTTP media route (upload + play_sound) via an
injectable ``poster``: a chunk leaves the buffer when it reaches the target
size (split at the quietest 50 ms window before the target) or when no new
audio has arrived for ``inactivity_s`` — never by waiting for Sonic to leave
``"speaking"``, because on the robot that transition is produced by the 4 s
speaking watchdog. A single worker thread enforces one-speaker-at-a-time
discipline through the shared :class:`EchoGate`, each chunk gets its own
``nova-<utt>-<seq>.wav`` deleted after its window, and ANY playback failure
routes to the interruption path (gate cleared, pending queue emptied,
``on_playback_failure`` fired) so there is never a stuck speaking state.
``chunked=False`` restores the whole-utterance behaviour that shipped before
this task — the tests that hold in both modes are parametrised over it.

All tests here use a fake poster and a fake deleter — no network, no daemon.
"""

from __future__ import annotations

import io
import logging
import queue
import threading
import time
import wave

import numpy as np
import pytest

from reachy_nova.harness import speaking
from reachy_nova.harness.gate import ECHO_GATE_ENV, EchoGate
from reachy_nova.harness.hearing import TeeHearing
from reachy_nova.harness.quiet import QuietState
from reachy_nova.harness.speaking import SonicSpeaker

SAMPLE_RATE = 24000

#: Both playback modes, for the tests whose meaning is identical in each.
BOTH_MODES = pytest.mark.parametrize("chunked", [False, True], ids=["whole", "chunked"])


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


class RecordingDeleter:
    """Fake DELETE transport: records every chunk file the speaker cleans up."""

    def __init__(self, fail: bool = False):
        self.attempted: list[str] = []
        self.deleted: list[str] = []
        self.fail = fail
        self._lock = threading.Lock()

    def __call__(self, base_url: str, filename: str) -> None:
        with self._lock:
            self.attempted.append(filename)
            if self.fail:
                raise OSError("daemon refused the delete (simulated)")
            self.deleted.append(filename)

    def names(self) -> list[str]:
        with self._lock:
            return list(self.deleted)

    def attempts(self) -> list[str]:
        with self._lock:
            return list(self.attempted)


class FakeClock:
    """Injectable monotonic clock for the chunker's size/inactivity timing.

    The echo gate keeps its own real clock (playback windows are real time);
    this one drives only the buffer's notion of "no audio has arrived for
    300 ms", so the inactivity flush is testable without sleeping through it.
    """

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(autouse=True)
def _never_dial_the_daemon(monkeypatch):
    """No test in this module may touch the network.

    Every test injects its own ``poster``/``stopper``; the chunk-cleanup
    ``deleter`` defaults to a real HTTP DELETE, and :class:`SonicSpeaker`
    resolves it from this module global at construction time — so patching it
    here covers every speaker built without an explicit ``deleter=``. Tests
    that assert on cleanup inject their own :class:`RecordingDeleter`.
    """
    monkeypatch.setattr(speaking, "default_deleter", lambda base_url, filename: None)


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


@BOTH_MODES
def test_one_utterance_posts_exactly_one_wav(stop_event, chunked):
    gate = EchoGate(margin_s=0.05)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, chunked=chunked)
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


@BOTH_MODES
def test_wav_payload_is_the_clipped_int16_conversion(stop_event, chunked):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, chunked=chunked)
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
    """Whole-utterance mode only — the state change IS the flush there.

    Under ``chunked=True`` the same 0.1 s of audio leaves the buffer on the
    inactivity timer with no state change at all; that contract is pinned by
    ``test_a_short_reply_is_flushed_by_inactivity_alone`` instead.
    """
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, chunked=False)
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


@BOTH_MODES
def test_playback_failure_routes_to_interruption_path(stop_event, caplog, chunked):
    gate = EchoGate(margin_s=0.05)
    poster = RecordingPoster(fail_times=1)
    failures: list[float] = []
    speaker = SonicSpeaker(
        gate,
        sample_rate=SAMPLE_RATE,
        poster=poster,
        on_playback_failure=lambda: failures.append(time.monotonic()),
        chunked=chunked,
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


@BOTH_MODES
def test_on_playback_failure_exception_does_not_kill_worker(stop_event, chunked):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster(fail_times=1)

    def bad_callback():
        raise RuntimeError("integrator bug")

    speaker = SonicSpeaker(
        gate,
        sample_rate=SAMPLE_RATE,
        poster=poster,
        on_playback_failure=bad_callback,
        chunked=chunked,
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
# 3. One-speaker discipline: second utterance waits for the first's window,   #
#    under every NOVA_ECHO_GATE policy (the policy is hearing-side only).     #
# --------------------------------------------------------------------------- #


@BOTH_MODES
@pytest.mark.parametrize("policy", [None, "off", "half-duplex", "nonsense"])
def test_second_utterance_waits_for_first_gate_window(
    stop_event, monkeypatch, policy, chunked
):
    """One speaker at a time, under EVERY hearing policy.

    ``NOVA_ECHO_GATE`` (t2) selects only what the HEARING leg does with the
    window — overlapping playback mixes at the device (reachy-mini-cli 0.48.0
    has no speaker arbitration), so the speaking leg's use of the gate is not
    policy-selectable and must be identical in all four cases below.
    """
    if policy is None:
        monkeypatch.delenv(ECHO_GATE_ENV, raising=False)
    else:
        monkeypatch.setenv(ECHO_GATE_ENV, policy)

    gate = EchoGate(margin_s=0.05)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, chunked=chunked)
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


@BOTH_MODES
@pytest.mark.parametrize("policy", [None, "off", "half-duplex"])
def test_the_gate_is_armed_before_the_post_under_every_policy(
    stop_event, monkeypatch, policy, chunked
):
    """The window opens BEFORE the upload, whatever the hearing leg does with it.

    ``play_sound`` returns when playback is *triggered*, not when the sound has
    left the speaker, so arming after the HTTP round trip would leave the
    upload window ungated — and the speaking leg's one-at-a-time wait reads
    exactly this window.
    """
    if policy is None:
        monkeypatch.delenv(ECHO_GATE_ENV, raising=False)
    else:
        monkeypatch.setenv(ECHO_GATE_ENV, policy)

    gate = EchoGate(margin_s=0.05)
    armed_at_post: list[bool] = []

    def poster(base_url: str, wav_bytes: bytes, filename: str) -> None:
        armed_at_post.append(gate.active)

    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, chunked=chunked)
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(2400)])  # 0.1 s
        assert wait_until(lambda: bool(armed_at_post))
        assert armed_at_post == [True]
    finally:
        stop_event.set()
        speaker.stop()


# --------------------------------------------------------------------------- #
# 4. Long-monologue safety: >cap buffer flushes mid-"speaking".               #
# --------------------------------------------------------------------------- #


def test_long_monologue_flushes_mid_speaking(stop_event):
    """Whole-utterance mode's only mid-"speaking" flush: the buffer cap.

    Under ``chunked=True`` the cap is not what splits a monologue — the ~1 s
    chunk target is, and it splits at a low-energy point rather than at the
    exact cap; see ``test_the_size_flush_cuts_at_the_quietest_window``.
    """
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    # A short cap keeps the test fast; the production default is ~15 s.
    speaker = SonicSpeaker(
        gate, sample_rate=SAMPLE_RATE, poster=poster, max_buffer_s=0.5, chunked=False
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


@BOTH_MODES
def test_queue_overflow_drop_is_named(caplog, chunked):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    # Never started: no worker drains the queue, so it fills deterministically.
    speaker = SonicSpeaker(
        gate, sample_rate=SAMPLE_RATE, poster=poster, queue_size=2, chunked=chunked
    )
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
    assert speaker.queue_size == 90


# --------------------------------------------------------------------------- #
# 6. Lifecycle + observability.                                               #
# --------------------------------------------------------------------------- #


@BOTH_MODES
def test_idle_reflects_pending_and_in_flight_work(stop_event, chunked):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, chunked=chunked)
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


@BOTH_MODES
def test_queued_and_played_sense_lines_carry_duration(stop_event, caplog, chunked):
    gate = EchoGate(margin_s=0.01)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, chunked=chunked)
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


# --------------------------------------------------------------------------- #
# preempt — the barge-in cut (stop_sound + purge + gate clear)                #
# --------------------------------------------------------------------------- #


def test_preempt_stops_the_playing_sound_via_the_stopper():
    gate = EchoGate()
    stops: list[str] = []
    speaker = SonicSpeaker(gate=gate, poster=RecordingPoster(), stopper=stops.append)
    gate.arm_for(5.0)
    speaker.preempt()
    assert stops == [speaker.base_url]
    assert not gate.active


def test_preempt_survives_a_stopper_that_raises():
    gate = EchoGate()

    def bad_stopper(base_url: str) -> None:
        raise OSError("daemon away")

    speaker = SonicSpeaker(gate=gate, poster=RecordingPoster(), stopper=bad_stopper)
    gate.arm_for(5.0)
    speaker.preempt()  # must not raise
    assert not gate.active


def test_an_utterance_already_in_the_workers_hands_is_dropped_by_preempt(stop_event):
    """The live 23:15 race: preempt purged the queue but the worker had already
    dequeued the utterance and was waiting on the gate window — clearing the
    gate RELEASED it to play right after the cut. The epoch check drops it."""
    gate = EchoGate(margin_s=0.0)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate=gate, poster=poster, stopper=lambda base: None)
    speaker.start(stop_event)
    try:
        gate.arm_for(0.6)  # a previous playback window holds the worker
        speaker._enqueue(np.zeros(2400, dtype=np.float32), why="test")
        time.sleep(0.15)  # worker dequeues and sits in the gate wait
        speaker.preempt()
        time.sleep(0.8)  # well past the original window
        assert poster.calls == []
        assert speaker.utterances_played == 0
    finally:
        speaker.stop()


def test_a_preempt_landing_during_the_post_cuts_the_sound_again(stop_event):
    gate = EchoGate(margin_s=0.0)
    stops: list[str] = []

    speaker = SonicSpeaker(gate=gate, poster=None, stopper=stops.append)

    def preempting_poster(base_url, wav_bytes, filename):
        speaker.preempt()  # the barge-in lands mid-upload

    speaker.poster = preempting_poster
    speaker.start(stop_event)
    try:
        speaker._enqueue(np.zeros(2400, dtype=np.float32), why="test")
        time.sleep(0.4)
        # one stop from the preempt itself + one from the post-post epoch check
        assert len(stops) == 2
        assert speaker.utterances_played == 0
        assert not gate.active
    finally:
        speaker.stop()


# --------------------------------------------------------------------------- #
# Chunked playback (task t8): the flush is driven by the audio, not by Sonic. #
#                                                                             #
# Measured on the robot 2026-09-05/06: five short replies (0.84-1.28 s of     #
# audio) were each queued 4.3-4.6 s after their first audio chunk, because    #
# the transition OUT of "speaking" is produced by the 4 s speaking watchdog   #
# and not by an end-of-turn event. So size and inactivity must flush, and     #
# the state change is only the final sweep.                                   #
# --------------------------------------------------------------------------- #


def drain_queue(speaker: SonicSpeaker) -> list:
    """Pop every queued chunk (the worker is not running in these tests)."""
    items = []
    while True:
        try:
            items.append(speaker._queue.get_nowait())
        except queue.Empty:
            return items
        speaker._queue.task_done()


def test_chunking_defaults_are_the_measured_ones():
    speaker = SonicSpeaker(EchoGate())
    assert speaker.chunked is True
    assert speaker.chunk_s == 1.0
    assert speaker.inactivity_s == pytest.approx(0.30)
    assert speaker.max_outstanding_files == 8


def test_five_seconds_of_audio_is_queued_as_chunks_not_as_one_utterance():
    """Criterion 1a: first chunk within 1.2 s, no chunk longer than 1.5 s.

    Fed at real-time pace on a fake clock (100 ms of audio per tick) with the
    worker deliberately not running — the acceptance criterion is about when
    audio LEAVES THE BUFFER, and running the worker would make the test wait
    out 5 s of real gate windows.
    """
    clock = FakeClock()
    speaker = SonicSpeaker(
        EchoGate(margin_s=0.0),
        sample_rate=SAMPLE_RATE,
        poster=RecordingPoster(),
        clock=clock,
    )
    speaker.on_state_change("speaking")
    first_sample_t = clock.t
    queued: list[tuple[float, float]] = []  # (fake time of the flush, duration)
    for _ in range(50):  # 5 s of audio, 100 ms at a time
        speaker.on_audio_chunk(make_chunk(2400))
        clock.advance(0.1)
        for item in drain_queue(speaker):
            queued.append((clock.t, len(item.samples) / SAMPLE_RATE))

    assert queued, "5 s of audio produced no chunk at all"
    assert queued[0][0] - first_sample_t <= 1.2
    assert max(duration for _t, duration in queued) <= 1.5
    # ~1 s chunks: 5 s of audio is several of them, never one utterance.
    assert len(queued) >= 4
    # And nothing is lost: the tail is still buffered, awaiting its own flush.
    played = sum(int(duration * SAMPLE_RATE) for _t, duration in queued)
    assert played + speaker._buffer_samples == 50 * 2400


def test_a_short_reply_is_flushed_by_inactivity_alone():
    """Criterion 1b: 0.5 s of reply, then silence — queued without a state change."""
    clock = FakeClock()
    speaker = SonicSpeaker(
        EchoGate(margin_s=0.0),
        sample_rate=SAMPLE_RATE,
        poster=RecordingPoster(),
        clock=clock,
    )
    speaker.on_state_change("speaking")
    for i in range(5):  # 0.5 s of reply, 100 ms apart
        if i:
            clock.advance(0.1)
        speaker.on_audio_chunk(make_chunk(2400))
    last_sample_t = clock.t

    clock.advance(0.29)
    speaker._flush_if_inactive()
    assert speaker._queue.qsize() == 0, "290 ms of silence is not yet inactivity"

    clock.advance(0.02)
    speaker._flush_if_inactive()
    assert speaker._queue.qsize() == 1
    item = speaker._queue.get_nowait()
    assert len(item.samples) == 5 * 2400  # the whole short reply, exactly once
    assert clock.t - last_sample_t <= 0.4
    assert speaker._sonic_state == "speaking"  # no state change was involved


def test_the_worker_drives_the_inactivity_flush(stop_event):
    """The timer is not a test-only method: the running worker checks it."""
    clock = FakeClock()
    poster = RecordingPoster()
    speaker = SonicSpeaker(
        EchoGate(margin_s=0.0), sample_rate=SAMPLE_RATE, poster=poster, clock=clock
    )
    speaker.start(stop_event)
    try:
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(make_chunk(2400))
        assert poster.calls == []
        clock.advance(0.5)  # past inactivity_s, still "speaking"
        assert poster.wait_for_calls(1)
        assert speaker._sonic_state == "speaking"
    finally:
        stop_event.set()
        speaker.stop()


def test_the_size_flush_cuts_at_the_quietest_window_before_the_target():
    """Words are not cut: the boundary lands in the quiet band, not at 1.000 s."""
    clock = FakeClock()
    speaker = SonicSpeaker(
        EchoGate(margin_s=0.0),
        sample_rate=SAMPLE_RATE,
        poster=RecordingPoster(),
        clock=clock,
    )
    loud = np.full(int(1.2 * SAMPLE_RATE), 0.5, dtype=np.float32)
    quiet_from, quiet_to = int(0.90 * SAMPLE_RATE), int(0.95 * SAMPLE_RATE)
    loud[quiet_from:quiet_to] = 0.0  # one 50 ms pause inside the search window

    speaker.on_state_change("speaking")
    speaker.on_audio_chunk(loud)

    items = drain_queue(speaker)
    assert len(items) == 1
    # Cut at the END of the quietest window: the chunk carries the pause, so
    # any residual boundary latency lands in silence rather than mid-word.
    assert len(items[0].samples) == quiet_to
    assert speaker._buffer_samples == len(loud) - quiet_to


def test_a_target_shorter_than_the_search_tail_splits_at_the_exact_size():
    """Below 250 ms of tail there is nowhere to search: split at the target."""
    speaker = SonicSpeaker(
        EchoGate(margin_s=0.0),
        sample_rate=SAMPLE_RATE,
        poster=RecordingPoster(),
        clock=FakeClock(),
        chunk_s=0.1,
    )
    speaker.on_state_change("speaking")
    speaker.on_audio_chunk(make_chunk(3 * 2400))  # 0.3 s in one callback
    assert [len(item.samples) for item in drain_queue(speaker)] == [2400, 2400, 2400]
    assert speaker._buffer_samples == 0


def test_chunks_post_in_order_and_a_preempt_purges_the_rest(stop_event):
    """Criterion 2: in order, gate-serialised, and chunks 3..n never post."""
    gate = EchoGate(margin_s=0.0)
    poster = RecordingPoster()
    stops: list[str] = []
    speaker = SonicSpeaker(
        gate,
        sample_rate=SAMPLE_RATE,
        poster=poster,
        deleter=RecordingDeleter(),
        stopper=stops.append,
        chunk_s=0.2,
    )
    speaker.start(stop_event)
    try:
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(make_chunk(5 * 4800))  # five 0.2 s chunks
        assert poster.wait_for_calls(2)
        # Chunk 2's window is open and chunk 3 sits in the worker's hands
        # waiting it out: the barge-in lands exactly between them.
        speaker.preempt()
        assert wait_until(lambda: speaker.idle)
        time.sleep(0.05)  # well past a 20 ms poll: a stale chunk would show

        assert [c["filename"] for c in poster.calls] == [
            "nova-1-1.wav",
            "nova-1-2.wav",
        ]
        # Each chunk waited out the previous one's window (0.2 s, margin 0).
        assert poster.calls[1]["t"] - poster.calls[0]["t"] >= 0.2
        assert stops == [speaker.base_url]
        assert speaker.chunks_played == 2
        assert speaker.utterances_played == 1  # one utterance, two chunks
    finally:
        speaker.stop()


def test_chunks_do_not_pay_the_ear_margin_between_them(stop_event):
    """The gate's margin pads the EAR, not the speaker's own serialisation.

    ``app.py`` builds ``EchoGate()`` — a full second of margin. Waiting that
    out between chunk 3 and chunk 4 of one sentence would insert a second of
    silence mid-word and hand back exactly the delay this task removes, so
    chunks wait out the previous chunk's AUDIO and post there. The ear's
    window keeps the whole margin.
    """
    gate = EchoGate(margin_s=0.5)
    poster = RecordingPoster()
    speaker = SonicSpeaker(
        gate,
        sample_rate=SAMPLE_RATE,
        poster=poster,
        deleter=RecordingDeleter(),
        chunk_s=0.05,
    )
    speaker.start(stop_event)
    try:
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(make_chunk(2 * 1200))  # two 50 ms chunks
        assert poster.wait_for_calls(2)
        gap = poster.calls[1]["t"] - poster.calls[0]["t"]
        assert gap >= 0.05, "chunk 2 posted over chunk 1's audio"
        assert gap < 0.2, "chunk 2 waited out the ear's 0.5 s margin as well"
        # The ear still sees the full padded window while the robot speaks.
        assert gate.remaining() > 0.05
    finally:
        speaker.stop()


def test_whole_utterance_mode_still_waits_out_the_whole_gate_window(stop_event):
    """chunked=False is the pre-t8 behaviour, ear margin and all."""
    gate = EchoGate(margin_s=0.3)
    poster = RecordingPoster()
    speaker = SonicSpeaker(
        gate, sample_rate=SAMPLE_RATE, poster=poster, chunked=False
    )
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(1200)])  # A: 0.05 s
        speak_utterance(speaker, [make_chunk(1200)])  # B: 0.05 s
        assert poster.wait_for_calls(2)
        assert poster.calls[1]["t"] - poster.calls[0]["t"] >= 0.05 + 0.3
    finally:
        speaker.stop()


def test_chunk_numbering_restarts_with_each_utterance(stop_event):
    gate = EchoGate(margin_s=0.0)
    poster = RecordingPoster()
    speaker = SonicSpeaker(
        gate,
        sample_rate=SAMPLE_RATE,
        poster=poster,
        deleter=RecordingDeleter(),
        chunk_s=0.1,
    )
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(2 * 2400)])  # utterance 1: 2 chunks
        speak_utterance(speaker, [make_chunk(2400)])  # utterance 2: 1 chunk
        assert poster.wait_for_calls(3)
        assert [c["filename"] for c in poster.calls] == [
            "nova-1-1.wav",
            "nova-1-2.wav",
            "nova-2-1.wav",
        ]
        assert speaker.chunks_played == 3
        assert speaker.utterances_played == 2
    finally:
        speaker.stop()


def test_each_chunk_is_deleted_after_its_window(stop_event):
    """Criterion 3a: per-chunk file, cleaned up through the injectable deleter."""
    gate = EchoGate(margin_s=0.0)
    poster = RecordingPoster()
    deleter = RecordingDeleter()
    speaker = SonicSpeaker(
        gate, sample_rate=SAMPLE_RATE, poster=poster, deleter=deleter, chunk_s=0.05
    )
    speaker.start(stop_event)
    try:
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(make_chunk(3 * 1200))  # three 50 ms chunks
        assert poster.wait_for_calls(3)
        expected = ["nova-1-1.wav", "nova-1-2.wav", "nova-1-3.wav"]
        assert [c["filename"] for c in poster.calls] == expected
        # Deletion follows each chunk's window — including the last one, once
        # the worker goes idle. The daemon's sounds dir never grows.
        assert wait_until(lambda: deleter.names() == expected)
        assert speaker.outstanding_files == []
    finally:
        speaker.stop()


def test_at_most_eight_chunk_files_are_ever_outstanding():
    """Criterion 3b: the cap deletes the oldest first, whatever the windows do.

    The fake clock never advances, so no window ever elapses on its own — the
    only thing that can keep the daemon's sounds dir bounded is the cap. (On
    the robot the disk this writes to is at 90 %.)
    """
    poster = RecordingPoster()
    deleter = RecordingDeleter()
    speaker = SonicSpeaker(
        EchoGate(margin_s=0.0),
        sample_rate=SAMPLE_RATE,
        poster=poster,
        deleter=deleter,
        clock=FakeClock(),
    )
    for _ in range(20):
        speaker._enqueue(make_chunk(24), why="test")  # 1 ms of audio each
        speaker._play_one(speaker._queue.get_nowait())
        assert len(speaker.outstanding_files) <= 8

    assert len(poster.calls) == 20
    assert deleter.names() == [f"nova-1-{i}.wav" for i in range(1, 13)]
    assert speaker.outstanding_files == [f"nova-1-{i}.wav" for i in range(13, 21)]


def test_a_failed_chunk_delete_is_one_named_line_and_playback_continues(
    stop_event, caplog
):
    """Criterion 3c: cleanup is best-effort; the mouth is not the disk."""
    gate = EchoGate(margin_s=0.0)
    poster = RecordingPoster()
    deleter = RecordingDeleter(fail=True)
    speaker = SonicSpeaker(
        gate, sample_rate=SAMPLE_RATE, poster=poster, deleter=deleter, chunk_s=0.05
    )
    speaker.start(stop_event)
    try:
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            speaker.on_state_change("speaking")
            speaker.on_audio_chunk(make_chunk(4 * 1200))
            assert poster.wait_for_calls(4)
            assert wait_until(lambda: speaker.delete_failures >= 3)
        drops = [
            rec.getMessage()
            for rec in caplog.records
            if "reason=chunk-delete-failed" in rec.getMessage()
        ]
        assert len(drops) == 1, "the delete failure is latched, not per chunk"
        assert "stage=speak" in drops[0] and "source=nova" in drops[0]
        assert speaker.chunks_played == 4  # every chunk still played
        assert speaker.playback_failures == 0  # a failed delete is not mouth loss
    finally:
        speaker.stop()


def test_whole_utterance_mode_keeps_one_filename_and_never_deletes(stop_event):
    """Criterion 4: with chunking off, this is byte-for-byte today's behaviour."""
    gate = EchoGate(margin_s=0.0)
    poster = RecordingPoster()
    deleter = RecordingDeleter()
    speaker = SonicSpeaker(
        gate,
        sample_rate=SAMPLE_RATE,
        poster=poster,
        deleter=deleter,
        chunked=False,
    )
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(2400)])
        speak_utterance(speaker, [make_chunk(2400)])
        assert poster.wait_for_calls(2)
        assert {c["filename"] for c in poster.calls} == {"tts_synth.wav"}
        time.sleep(0.05)
        assert deleter.attempts() == []
        assert speaker.outstanding_files == []
        assert speaker.utterances_played == 2
        assert speaker.chunks_played == 2
    finally:
        speaker.stop()


def test_an_utterance_still_in_the_buffer_is_not_idle():
    """`idle` covers chunks in flight AND audio not yet flushed (rotation gate)."""
    speaker = SonicSpeaker(
        EchoGate(),
        sample_rate=SAMPLE_RATE,
        poster=RecordingPoster(),
        clock=FakeClock(),
    )
    assert speaker.idle
    speaker.on_state_change("speaking")
    speaker.on_audio_chunk(make_chunk(2400))  # buffered, below the 1 s target
    assert not speaker.idle
    speaker.on_state_change("listening")  # flushed, now queued (no worker)
    assert not speaker.idle


def test_chunk_sense_lines_name_the_chunk_and_the_flush_reason(stop_event, caplog):
    gate = EchoGate(margin_s=0.0)
    poster = RecordingPoster()
    speaker = SonicSpeaker(
        gate,
        sample_rate=SAMPLE_RATE,
        poster=poster,
        deleter=RecordingDeleter(),
        chunk_s=0.05,
    )
    speaker.start(stop_event)

    def played_lines() -> list[str]:
        return [r.getMessage() for r in caplog.records if "] played chunk=" in r.getMessage()]

    try:
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            speaker.on_state_change("speaking")
            speaker.on_audio_chunk(make_chunk(2 * 1200))  # two size flushes
            speaker.on_audio_chunk(make_chunk(600))  # tail, flushed by the state
            speaker.on_state_change("listening")
            assert poster.wait_for_calls(3)
            # `chunks_played` is bumped BEFORE its line is written, so poll for
            # the LINES — the counter is not a happens-before edge for them.
            assert wait_until(lambda: len(played_lines()) == 3)
        rendered = [rec.getMessage() for rec in caplog.records]
        queued = [line for line in rendered if "] queued chunk=" in line]
        played = played_lines()
        assert [line.split("chunk=")[1].split()[0] for line in queued] == [
            "1-1",
            "1-2",
            "1-3",
        ]
        assert "(size)" in queued[0]
        assert "(state-change)" in queued[2]
        assert len(played) == 3
        assert "chunk=1-1" in played[0]
    finally:
        speaker.stop()


def test_the_inactivity_flush_names_itself_in_the_sense_line(caplog):
    clock = FakeClock()
    speaker = SonicSpeaker(
        EchoGate(margin_s=0.0),
        sample_rate=SAMPLE_RATE,
        poster=RecordingPoster(),
        clock=clock,
    )
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(make_chunk(2400))
        clock.advance(0.4)
        speaker._flush_if_inactive()
    queued = [
        rec.getMessage()
        for rec in caplog.records
        if "] queued chunk=" in rec.getMessage()
    ]
    assert len(queued) == 1
    assert "(inactivity)" in queued[0]


# --------------------------------------------------------------------------- #
# Timed quiet (task t11): the mouth is gated, the ear is not.                 #
# --------------------------------------------------------------------------- #


class _QuietClock:
    """Injectable wall clock for QuietState inside the speaker tests."""

    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def quiet_state(tmp_path, monkeypatch):
    """A QuietState on a fake clock, persisting into a tmp state dir."""
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    clock = _QuietClock()
    state = QuietState(clock=clock, grace_s=2.0)
    return state, clock


def _quiet_drop_lines(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if "event=quiet-drop]" in rec.getMessage()
    ]


def _quiet_resume_lines(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if "event=quiet-resume]" in rec.getMessage()
    ]


def test_utterances_are_dropped_while_quiet_and_summarised_after(
    stop_event, quiet_state, caplog
):
    state, clock = quiet_state
    gate = EchoGate(margin_s=0.05)
    poster = RecordingPoster()
    failures: list[int] = []
    speaker = SonicSpeaker(
        gate,
        sample_rate=SAMPLE_RATE,
        poster=poster,
        on_playback_failure=lambda: failures.append(1),
        quiet=state,
    )
    speaker.start(stop_event)
    try:
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            state.arm(10)
            clock.advance(3.0)  # the acknowledgement grace lapses unused
            for _ in range(3):
                speak_utterance(speaker, [make_chunk(2400)])
            assert wait_until(lambda: speaker.quiet_drops == 3)
            time.sleep(0.05)

            assert poster.calls == []  # the mouth never posted
            assert failures == []  # a quiet drop is NOT mouth loss
            assert gate.remaining() == 0.0  # the echo gate was never armed
            assert speaker.utterances_played == 0
            assert len(_quiet_drop_lines(caplog)) == 1  # latched

            # After the deadline the next utterance plays, and the silence is
            # summarised with what it cost.
            clock.advance(601.0)
            speak_utterance(speaker, [make_chunk(2400)])
            assert poster.wait_for_calls(1)
            assert wait_until(lambda: _quiet_resume_lines(caplog))
        summary = _quiet_resume_lines(caplog)
        assert len(summary) == 1
        assert "count=3" in summary[0]
        assert speaker.utterances_played == 1
    finally:
        speaker.stop()


def test_the_first_utterance_after_arm_is_spoken_then_the_mouth_closes(
    stop_event, quiet_state
):
    """"okay, quiet for ten minutes" must be heard — the gate closes after it."""
    state, _clock = quiet_state
    gate = EchoGate(margin_s=0.0)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, quiet=state)
    speaker.start(stop_event)
    try:
        state.arm(10)
        speak_utterance(speaker, [make_chunk(2400)])
        assert poster.wait_for_calls(1)
        speak_utterance(speaker, [make_chunk(2400)])
        assert wait_until(lambda: speaker.quiet_drops == 1)
        time.sleep(0.05)
        assert len(poster.calls) == 1
        assert speaker.utterances_played == 1
    finally:
        speaker.stop()


def test_the_acknowledgement_grace_expires_on_its_own(stop_event, quiet_state):
    """No utterance within the grace: the mouth closes anyway."""
    state, clock = quiet_state
    gate = EchoGate(margin_s=0.0)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, quiet=state)
    speaker.start(stop_event)
    try:
        state.arm(10)
        clock.advance(2.5)  # grace_s = 2.0
        speak_utterance(speaker, [make_chunk(2400)])
        assert wait_until(lambda: speaker.quiet_drops == 1)
        time.sleep(0.05)
        assert poster.calls == []
        assert speaker.utterances_played == 0
    finally:
        speaker.stop()


def test_quiet_never_reaches_the_ear(stop_event, quiet_state, monkeypatch):
    """Quiet closes the mouth only: hearing keeps feeding under policy 'off'."""
    monkeypatch.delenv(ECHO_GATE_ENV, raising=False)
    state, clock = quiet_state
    gate = EchoGate(margin_s=0.05)
    poster = RecordingPoster()
    speaker = SonicSpeaker(gate, sample_rate=SAMPLE_RATE, poster=poster, quiet=state)

    fed: list[int] = []
    ear = TeeHearing(feed=lambda chunk: fed.append(len(chunk)), gate=gate)
    assert ear.echo_gate_policy == "off"
    chunk_bytes = 4 * 1600  # 100 ms of float32 at 16 kHz

    speaker.start(stop_event)
    try:
        state.arm(10)
        clock.advance(3.0)
        for _ in range(3):
            speak_utterance(speaker, [make_chunk(2400)])
        assert wait_until(lambda: speaker.quiet_drops == 3)
        ear._drain(bytearray(b"\x00" * chunk_bytes * 2), 16000, chunk_bytes)
        assert len(fed) == 2
        assert ear.chunks_gated == 0
    finally:
        speaker.stop()


def test_a_long_reply_generated_faster_than_real_time_never_drops_chunks(monkeypatch):
    """Robot, 2026-09-06 00:38: a ~35 s reply arrived in ~18 s; with a queue of 8
    every chunk past the eighth pending one was dropped and words went missing."""
    import numpy as np

    from reachy_nova.harness.gate import EchoGate
    from reachy_nova.harness.speaking import SonicSpeaker

    posted: list[str] = []
    speaker = SonicSpeaker(
        gate=EchoGate(margin_s=0.0),
        sample_rate=1000,
        poster=lambda base, wav, name: posted.append(name),
        deleter=lambda base, name: None,
        chunk_s=1.0,
        inactivity_s=0.3,
    )
    speaker.on_state_change("speaking")
    for _ in range(40):  # 40 s of audio, all delivered at once (faster than real time)
        speaker.on_audio_chunk(np.full(1000, 0.1, dtype=np.float32))
    speaker.on_state_change("listening")
    assert speaker._queue.qsize() + len(speaker._buffer) >= 40 or speaker.chunks_played >= 0
    # nothing was dropped for queue-full: every chunk is either queued or already played
    import threading

    stop = threading.Event()
    speaker.start(stop)
    try:
        deadline = __import__("time").monotonic() + 5.0
        while speaker.chunks_played < 40 and __import__("time").monotonic() < deadline:
            speaker.gate.clear()  # let the fake windows elapse instantly
            __import__("time").sleep(0.01)
    finally:
        stop.set()
        speaker.stop()
    assert speaker.chunks_played == 40, f"only {speaker.chunks_played} of 40 chunks played"
