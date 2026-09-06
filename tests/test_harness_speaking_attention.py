"""Voice gating on the attention window (task t6) — the speaker half.

``SonicSpeaker`` learns one new question, asked once per utterance at the
moment Sonic starts speaking: *was anybody talking to us?* The answer comes
from :class:`~reachy_nova.harness.attention.AttentionState` alone
(:meth:`SonicSpeaker.attention_verdict`), and a ``not-addressed`` verdict
drops every chunk of that utterance exactly the way a timed-quiet drop does —
no post, no gate arm, no queue purge, no ``on_playback_failure``, one named
senselog line for the whole utterance.

The mic feed is untouched: this module gates PLAYBACK only, which is why the
tests below assert on the poster and the gate and never on hearing.

All tests use a fake poster/stopper/deleter — no network, no daemon.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pytest

from reachy_nova.harness.attention import AttentionState
from reachy_nova.harness.gate import EchoGate
from reachy_nova.harness.speaking import SonicSpeaker

SAMPLE_RATE = 24000


# --------------------------------------------------------------------------- #
# Helpers (same shapes as tests/test_harness_speaking.py)                     #
# --------------------------------------------------------------------------- #


class RecordingPoster:
    """Fake HTTP transport: records every (wav_bytes, filename) upload+play."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, base_url: str, wav_bytes: bytes, filename: str) -> None:
        with self._lock:
            self.calls.append({"filename": filename, "wav": wav_bytes})

    def count(self) -> int:
        with self._lock:
            return len(self.calls)

    def wait_for_calls(self, n: int, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.count() >= n:
                return True
            time.sleep(0.005)
        return self.count() >= n


class FakeClock:
    """One monotonic clock shared by the speaker and the attention window."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_chunk(n_samples: int, value: float = 0.25) -> np.ndarray:
    return np.full(n_samples, value, dtype=np.float32)


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def speak_utterance(speaker: SonicSpeaker, chunks: list[np.ndarray]) -> None:
    speaker.on_state_change("speaking")
    for chunk in chunks:
        speaker.on_audio_chunk(chunk)
    speaker.on_state_change("listening")


def attention_drop_lines(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if "event=attention]" in rec.getMessage()
    ]


def attention_summary_lines(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if "event=attention-resume]" in rec.getMessage()
    ]


@pytest.fixture
def stop_event():
    ev = threading.Event()
    yield ev
    ev.set()


@pytest.fixture
def clock():
    return FakeClock()


def build_speaker(
    attention: AttentionState | None,
    poster: RecordingPoster,
    clock: FakeClock,
    *,
    gate: EchoGate | None = None,
    stops: list[int] | None = None,
) -> SonicSpeaker:
    return SonicSpeaker(
        gate if gate is not None else EchoGate(margin_s=0.0),
        sample_rate=SAMPLE_RATE,
        poster=poster,
        stopper=lambda base_url: (stops.append(1) if stops is not None else None),
        deleter=lambda base_url, filename: None,
        attention=attention,
        clock=clock,
    )


# --------------------------------------------------------------------------- #
# 1. Cold + nameless: the utterance is dropped like a quiet drop.             #
# --------------------------------------------------------------------------- #


def test_cold_nameless_reply_is_dropped_at_the_speaker(stop_event, clock, caplog):
    """Nobody said "nova": Nova's reply never reaches the daemon at all."""
    attention = AttentionState(clock=clock, window_s=45.0)
    assert attention.note_transcript("is the kettle boiled yet") == "ignored"
    assert not attention.warm
    before_utterance_at = attention.last_utterance_at

    poster = RecordingPoster()
    gate = EchoGate(margin_s=0.0)
    failures: list[int] = []
    speaker = build_speaker(attention, poster, clock, gate=gate)
    speaker.on_playback_failure = lambda: failures.append(1)
    speaker.start(stop_event)
    try:
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            speak_utterance(speaker, [make_chunk(2400)])
            assert wait_until(lambda: speaker.attention_drops == 1)
            time.sleep(0.05)

            assert poster.calls == []  # never posted
            assert gate.remaining() == 0.0  # never armed the ear's gate
            assert failures == []  # a gate drop is NOT mouth loss
            assert speaker.utterances_played == 0
            assert speaker.playback_failures == 0
            drops = attention_drop_lines(caplog)
            assert len(drops) == 1
            assert "dropped reason=not-addressed" in drops[0]
            summary = attention_summary_lines(caplog)
            assert len(summary) == 1
            assert "count=1" in summary[0]

        # The window is untouched: a suppressed utterance never renews it.
        assert attention.last_utterance_at == before_utterance_at
        assert not attention.warm
    finally:
        speaker.stop()


def test_every_chunk_of_a_suppressed_utterance_is_dropped_with_one_line(
    stop_event, clock, caplog
):
    """A long cold reply costs one log line, not one per chunk."""
    attention = AttentionState(clock=clock, window_s=45.0)
    attention.note_transcript("just talking to myself over here")

    poster = RecordingPoster()
    speaker = build_speaker(attention, poster, clock)
    speaker.start(stop_event)
    try:
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            # 3.5 s of audio: three whole 1 s chunks plus the final sweep.
            long_reply = [make_chunk(SAMPLE_RATE) for _ in range(3)]
            long_reply.append(make_chunk(SAMPLE_RATE // 2))
            speak_utterance(speaker, long_reply)
            assert wait_until(lambda: speaker.attention_drops >= 4)
            time.sleep(0.05)
            assert poster.calls == []
            assert len(attention_drop_lines(caplog)) == 1
            assert len(attention_summary_lines(caplog)) == 1
    finally:
        speaker.stop()


# --------------------------------------------------------------------------- #
# 2. The three ways an utterance is still allowed.                            #
# --------------------------------------------------------------------------- #


def test_a_warm_window_plays_exactly_as_before(stop_event, clock):
    attention = AttentionState(clock=clock, window_s=45.0)
    assert attention.note_transcript("nova, what time is it") == "opened"
    assert attention.warm

    poster = RecordingPoster()
    speaker = build_speaker(attention, poster, clock)
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(2400)])
        assert poster.wait_for_calls(1)
        assert speaker.attention_drops == 0
        # Nova speaking renews the window it was allowed to speak in.
        assert attention.last_utterance_at == pytest.approx(clock.t)
    finally:
        speaker.stop()


def test_a_reply_to_a_body_cue_plays_while_cold(stop_event, clock):
    """A pat/vision inject one second before the edge: this is a reaction."""
    attention = AttentionState(clock=clock, window_s=45.0)
    attention.note_transcript("nothing to do with the robot")
    clock.advance(1.0)
    attention.note_inject()
    assert not attention.warm

    poster = RecordingPoster()
    speaker = build_speaker(attention, poster, clock)
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(2400)])
        assert poster.wait_for_calls(1)
        assert speaker.attention_drops == 0
    finally:
        speaker.stop()


def test_an_inject_older_than_the_grace_does_not_excuse_a_cold_reply(
    stop_event, clock
):
    attention = AttentionState(clock=clock, window_s=45.0)
    attention.note_inject()
    clock.advance(1.0)
    attention.note_transcript("still not talking to the robot")
    clock.advance(5.0)  # grace is 3 s

    poster = RecordingPoster()
    speaker = build_speaker(attention, poster, clock)
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(2400)])
        assert wait_until(lambda: speaker.attention_drops == 1)
        time.sleep(0.05)
        assert poster.calls == []
    finally:
        speaker.stop()


def test_no_transcript_at_all_plays(stop_event, clock):
    """Startup context, a greeting on boot: nothing was ever misheard."""
    attention = AttentionState(clock=clock, window_s=45.0)
    assert attention.last_transcript_at is None

    poster = RecordingPoster()
    speaker = build_speaker(attention, poster, clock)
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(2400)])
        assert poster.wait_for_calls(1)
        assert speaker.attention_drops == 0
    finally:
        speaker.stop()


def test_no_attention_object_means_the_gate_does_not_exist(stop_event, clock):
    poster = RecordingPoster()
    speaker = build_speaker(None, poster, clock)
    assert speaker.attention_verdict() == "allowed"
    speaker.start(stop_event)
    try:
        speak_utterance(speaker, [make_chunk(2400)])
        assert poster.wait_for_calls(1)
        assert speaker.attention_drops == 0
    finally:
        speaker.stop()


# --------------------------------------------------------------------------- #
# 3. Ordering: the verdict is taken on the speaking edge.                     #
# --------------------------------------------------------------------------- #


def test_transcript_before_the_edge_stops_the_first_chunk(stop_event, clock):
    """Scripted order: nameless transcript, THEN speaking — nothing posts."""
    attention = AttentionState(clock=clock, window_s=45.0)
    attention.note_transcript("pass me the spanner")

    poster = RecordingPoster()
    speaker = build_speaker(attention, poster, clock)
    speaker.start(stop_event)
    try:
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(make_chunk(SAMPLE_RATE))  # a whole size-flush chunk
        assert wait_until(lambda: speaker.attention_drops >= 1)
        speaker.on_state_change("listening")
        time.sleep(0.05)
        assert poster.calls == []  # not even the first chunk
    finally:
        speaker.stop()


def test_a_late_transcript_preempts_the_utterance_exactly_once(stop_event, clock):
    """Sonic emitted audio before the transcript landed: the fallback path.

    Costs at most one clipped chunk — the one already posted — and every
    later chunk of the same utterance is dropped.
    """
    attention = AttentionState(clock=clock, window_s=45.0)
    poster = RecordingPoster()
    stops: list[int] = []
    speaker = build_speaker(attention, poster, clock, stops=stops)

    preempts: list[int] = []
    real_preempt = speaker.preempt

    def counting_preempt() -> None:
        preempts.append(1)
        real_preempt()

    speaker.preempt = counting_preempt  # type: ignore[method-assign]
    speaker.start(stop_event)
    try:
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(make_chunk(SAMPLE_RATE))
        assert poster.wait_for_calls(1)  # one chunk got out: the clipped one

        attention.note_transcript("anyway as I was saying to you")
        speaker.recheck_attention()
        assert preempts == [1]

        # A second recheck for the same utterance must not preempt again.
        speaker.recheck_attention()
        assert preempts == [1]

        speaker.on_audio_chunk(make_chunk(SAMPLE_RATE))
        speaker.on_state_change("listening")
        assert wait_until(lambda: speaker.attention_drops >= 1)
        time.sleep(0.05)
        assert poster.count() == 1  # nothing after the clipped chunk
    finally:
        speaker.stop()


def test_recheck_does_nothing_when_no_utterance_is_in_flight(stop_event, clock):
    attention = AttentionState(clock=clock, window_s=45.0)
    poster = RecordingPoster()
    stops: list[int] = []
    speaker = build_speaker(attention, poster, clock, stops=stops)
    speaker.start(stop_event)
    try:
        attention.note_transcript("nobody is talking to the robot")
        speaker.recheck_attention()
        assert stops == []  # no preempt, so no stop_sound
        assert speaker.attention_drops == 0
    finally:
        speaker.stop()


def test_recheck_leaves_an_allowed_utterance_alone(stop_event, clock):
    """Allowed because WARM, not because the transcript was missing."""
    attention = AttentionState(clock=clock, window_s=45.0)
    attention.note_transcript("nova, tell me a story")
    poster = RecordingPoster()
    stops: list[int] = []
    speaker = build_speaker(attention, poster, clock, stops=stops)
    speaker.start(stop_event)
    try:
        speaker.on_state_change("speaking")
        speaker.on_audio_chunk(make_chunk(SAMPLE_RATE))
        assert poster.wait_for_calls(1)
        attention.note_transcript("mm-hmm")  # renews, stays warm
        speaker.recheck_attention()
        assert stops == []
        assert speaker.attention_drops == 0
        speaker.on_state_change("listening")
    finally:
        speaker.stop()


# --------------------------------------------------------------------------- #
# 4. The verdict is a pure function of the attention state.                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (lambda a: None, "allowed"),
        (lambda a: a.note_transcript("nova, hello"), "allowed"),
        (lambda a: a.note_transcript("mind the step"), "not-addressed"),
        (lambda a: a.note_inject(), "allowed"),
    ],
)
def test_attention_verdict_is_pure(clock, script, expected):
    attention = AttentionState(clock=clock, window_s=45.0)
    script(attention)
    speaker = SonicSpeaker(
        EchoGate(margin_s=0.0),
        sample_rate=SAMPLE_RATE,
        poster=lambda *a: None,
        deleter=lambda *a: None,
        attention=attention,
        clock=clock,
    )
    snapshot = (
        attention.last_transcript_at,
        attention.last_transcript_named,
        attention.last_utterance_at,
        attention.last_inject_at,
        speaker.attention_drops,
    )
    first = speaker.attention_verdict()
    second = speaker.attention_verdict()
    assert first == second == expected
    assert (
        attention.last_transcript_at,
        attention.last_transcript_named,
        attention.last_utterance_at,
        attention.last_inject_at,
        speaker.attention_drops,
    ) == snapshot


def test_attention_verdict_accepts_an_explicit_now(clock):
    attention = AttentionState(clock=clock, window_s=45.0)
    attention.note_inject()
    clock.advance(1.0)
    attention.note_transcript("not addressing anyone in particular")
    speaker = SonicSpeaker(
        EchoGate(margin_s=0.0),
        sample_rate=SAMPLE_RATE,
        poster=lambda *a: None,
        deleter=lambda *a: None,
        attention=attention,
        attention_grace_s=3.0,
        clock=clock,
    )
    # The inject is 1 s old: this utterance is still a reaction to it.
    assert speaker.attention_verdict(now=clock.t) == "allowed"
    # Ten seconds later the same inject explains nothing, and the nameless
    # transcript that followed it is the most recent thing that happened.
    assert speaker.attention_verdict(now=clock.t + 10.0) == "not-addressed"
