"""Tests for the speech-capture lane (reachy_nova/speech_events.py).

Covers spec targets c2/h2 (task t2 of the event-based-senses plan):

1. A synthetic silence-then-speech stream proves the emitted clip's audio
   begins at or before the known onset t0 — backtrack MEASURED (via an
   amplitude scan over the buffered audio), not merely assumed.
2. The detector reuses an injected ASR handle (no second parakeet load);
   with no handle it falls back to an XMOS ``speech_flag_provider`` as the
   onset trigger.
3. The emitted payload carries clip_path, transcript, duration_seconds and
   onset_ts, and clips are written under a local, never-uploaded directory.

No real parakeet/nemo model is loaded anywhere in this file — the ASR path
is exercised with a FakeASR test double whose ``transcribe()`` only returns
non-empty text when the fed audio contains an amplitude marker, mirroring
how a real ASR would only "hear" speech once it's actually present.
"""

import inspect
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from reachy_nova.speech_events import SpeechEventDetector

SAMPLE_RATE = 16000


class FakeASR:
    """Stands in for an already-loaded parakeet handle (see wake_word.py).

    transcribe() only "hears" speech when the fed snapshot contains a
    sample whose amplitude clears ``marker_threshold`` — a stand-in for a
    real model only transcribing non-empty text once speech is present.
    """

    def __init__(self, marker_threshold: float = 0.1, transcript: str = "hello there"):
        self.marker_threshold = marker_threshold
        self.transcript = transcript
        self.call_count = 0

    def transcribe(self, audio_batch):
        self.call_count += 1
        audio = audio_batch[0]
        if audio.size and float(np.max(np.abs(audio))) >= self.marker_threshold:
            return [self.transcript]
        return [""]


def _read_wav_float(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _drain_pending(detector: SpeechEventDetector, events: list, timeout: float = 3.0) -> None:
    """Keep feeding tiny silence chunks until the background ASR catches up."""
    deadline = time.time() + timeout
    silence = np.zeros(160, dtype=np.float32)
    while not events and time.time() < deadline:
        evt = detector.feed(silence)
        if evt is not None:
            events.append(evt)
        time.sleep(0.005)


def test_backtrack_is_measured_not_assumed(tmp_path):
    """Criterion 1: the emitted clip begins at or before the known onset t0."""
    silence_seconds = 3.0
    speech_seconds = 1.0
    n_silence = int(silence_seconds * SAMPLE_RATE)
    n_speech = int(speech_seconds * SAMPLE_RATE)
    t0 = n_silence  # onset sample index — speech starts exactly here

    stream = np.zeros(n_silence + n_speech, dtype=np.float32)
    stream[t0:] = 0.5  # marker amplitude, well above FakeASR's threshold

    fake_asr = FakeASR(marker_threshold=0.1)
    pre_roll_seconds = 2.0
    detector = SpeechEventDetector(
        asr_handle=fake_asr,
        pre_roll_seconds=pre_roll_seconds,
        sample_rate=SAMPLE_RATE,
        buffer_seconds=10.0,
        transcribe_interval=0.0,
        clip_dir=tmp_path,
    )

    events: list[dict] = []
    total_fed = 0
    chunk_size = 1600  # 100ms chunks, like a real mic pipeline
    for i in range(0, len(stream), chunk_size):
        chunk = stream[i : i + chunk_size]
        total_fed += len(chunk)
        evt = detector.feed(chunk)
        if evt is not None:
            events.append(evt)
        time.sleep(0.001)  # let the single-worker executor make progress

    _drain_pending(detector, events)

    assert events, "expected a speech event to fire"
    event = events[0]

    # Recompute where the clip must have started, purely from the test's own
    # cumulative sample count and the reported duration — no internal state.
    clip_samples = round(event["duration_seconds"] * SAMPLE_RATE)
    clip_start_absolute = total_fed - clip_samples

    assert 0 <= clip_start_absolute <= t0, (
        f"clip must begin at or before t0={t0}, got clip_start={clip_start_absolute}"
    )

    # And it must be a genuine ~pre_roll backtrack, not "the whole buffer" —
    # this is what distinguishes MEASURED from ASSUMED.
    pre_roll_samples = int(pre_roll_seconds * SAMPLE_RATE)
    slack_samples = int(0.05 * SAMPLE_RATE)  # onset-scan quantization + chunking slack
    assert t0 - clip_start_absolute <= pre_roll_samples + slack_samples, (
        "backtrack pulled in far more audio than pre_roll_seconds — looks assumed, not measured"
    )

    # Verify the clip file itself: content matches the computed backtrack.
    clip_path = Path(event["clip_path"])
    assert clip_path.exists()
    assert clip_path.parent == tmp_path
    clip_audio = _read_wav_float(clip_path)
    assert len(clip_audio) == clip_samples

    marker_offset_in_clip = t0 - clip_start_absolute
    if marker_offset_in_clip > 0:
        # Pre-roll prefix should be (near-)silent.
        assert np.max(np.abs(clip_audio[:marker_offset_in_clip])) < 0.05
    # From the marker onward, the clip should carry the speech amplitude.
    assert np.max(np.abs(clip_audio[marker_offset_in_clip:])) > 0.1


def test_reuses_injected_asr_handle_no_second_load(tmp_path):
    """Criterion 2 (first half): the SAME handle is reused, never reloaded."""
    fake_asr = FakeASR(marker_threshold=0.1, transcript="hey there")
    detector = SpeechEventDetector(
        asr_handle=fake_asr,
        transcribe_interval=0.0,
        clip_dir=tmp_path,
        buffer_seconds=10.0,
    )

    events: list[dict] = []
    silence = np.zeros(1600, dtype=np.float32)
    speech = np.full(1600, 0.5, dtype=np.float32)

    for chunk in (silence, silence, speech, speech):
        evt = detector.feed(chunk)
        if evt is not None:
            events.append(evt)
        time.sleep(0.005)
    _drain_pending(detector, events)

    assert events
    assert fake_asr.call_count > 0, "the injected handle's transcribe() must actually be used"
    assert events[0]["transcript"] == "hey there"


def test_speech_events_module_never_loads_its_own_parakeet():
    """Criterion 2 (first half, static): speech_events.py has no loading capability.

    wake_word.py owns model loading (nemo_asr.models.ASRModel.from_pretrained);
    this module only ever *consumes* an already-loaded handle passed in.
    """
    from reachy_nova import speech_events

    source = inspect.getsource(speech_events)
    assert "from_pretrained" not in source
    assert "nemo" not in source.lower()


def test_xmos_flag_fallback_when_no_asr_handle(tmp_path):
    """Criterion 2 (second half): no handle -> XMOS speech_flag_provider drives onset."""
    flag = {"speaking": False}

    detector = SpeechEventDetector(
        asr_handle=None,
        speech_flag_provider=lambda: flag["speaking"],
        pre_roll_seconds=1.0,
        sample_rate=SAMPLE_RATE,
        buffer_seconds=10.0,
        clip_dir=tmp_path,
    )

    silence = np.zeros(1600, dtype=np.float32)
    marker = np.full(1600, 0.7, dtype=np.float32)

    events: list[dict] = []
    total_fed = 0

    # Feed 2s of silence with the flag down — must not fire.
    for _ in range(20):
        total_fed += len(silence)
        evt = detector.feed(silence)
        assert evt is None
        assert not events

    # XMOS flips the flag as speech starts — feed the marker now.
    flag["speaking"] = True
    t0 = total_fed
    total_fed += len(marker)
    evt = detector.feed(marker)

    assert evt is not None, "flag rising edge must trigger a speech event"
    events.append(evt)

    clip_samples = round(evt["duration_seconds"] * SAMPLE_RATE)
    clip_start_absolute = total_fed - clip_samples
    assert 0 <= clip_start_absolute <= t0

    # Flag drop must reset the edge trigger so a later rise fires again.
    flag["speaking"] = False
    for _ in range(5):
        detector.feed(silence)
    flag["speaking"] = True
    evt2 = detector.feed(marker)
    assert evt2 is not None, "a later rising edge should fire a new event"


def test_no_trigger_configured_never_fires(tmp_path):
    """Degenerate case: neither an ASR handle nor a flag provider -> always None."""
    detector = SpeechEventDetector(clip_dir=tmp_path)
    audio = np.full(1600, 0.9, dtype=np.float32)
    for _ in range(5):
        assert detector.feed(audio) is None


def test_event_payload_shape_and_local_clip_policy(tmp_path):
    """Criterion 3: payload carries clip_path/transcript/duration_seconds/onset_ts,
    and clips are written locally (never uploaded — no network code in this module)."""
    fake_asr = FakeASR(marker_threshold=0.1, transcript="a short phrase")
    captured: list[dict] = []
    detector = SpeechEventDetector(
        asr_handle=fake_asr,
        on_speech=captured.append,
        transcribe_interval=0.0,
        clip_dir=tmp_path,
        buffer_seconds=10.0,
    )

    silence = np.zeros(1600, dtype=np.float32)
    speech = np.full(1600, 0.5, dtype=np.float32)
    events: list[dict] = []
    for chunk in (silence, silence, speech, speech):
        evt = detector.feed(chunk)
        if evt is not None:
            events.append(evt)
        time.sleep(0.005)
    _drain_pending(detector, events)

    assert events
    event = events[0]
    assert set(event) >= {"clip_path", "transcript", "duration_seconds", "onset_ts"}
    assert isinstance(event["clip_path"], str)
    assert isinstance(event["transcript"], str) and event["transcript"]
    assert isinstance(event["duration_seconds"], float) and event["duration_seconds"] > 0
    assert isinstance(event["onset_ts"], float)

    # on_speech callback must have received the identical payload.
    assert captured and captured[0] == event

    # Local-only policy: clip lives under the configured (local) clip_dir.
    clip_path = Path(event["clip_path"])
    assert clip_path.is_file()
    assert tmp_path in clip_path.parents or clip_path.parent == tmp_path

    source = inspect.getsource(__import__("reachy_nova.speech_events", fromlist=["speech_events"]))
    for forbidden in ("requests", "urllib", "boto3", "socket"):
        assert forbidden not in source


def test_default_clip_dir_is_under_reachy_nova_home():
    """Criterion 3: default clip directory policy — under ~/.reachy_nova/, unwritten
    until an event actually fires (mirrors face_manager.py / session_state.py)."""
    detector = SpeechEventDetector()
    expected = Path.home() / ".reachy_nova" / "speech_clips"
    assert detector.clip_dir == expected


@pytest.mark.parametrize("marker_threshold", [0.1])
def test_transcript_empty_never_fires(tmp_path, marker_threshold):
    """Pure silence (transcript always empty) never emits a speech event."""
    fake_asr = FakeASR(marker_threshold=marker_threshold, transcript="should not appear")
    detector = SpeechEventDetector(
        asr_handle=fake_asr,
        transcribe_interval=0.0,
        clip_dir=tmp_path,
        buffer_seconds=10.0,
    )
    silence = np.zeros(1600, dtype=np.float32)
    events: list[dict] = []
    for _ in range(10):
        evt = detector.feed(silence)
        if evt is not None:
            events.append(evt)
        time.sleep(0.005)
    time.sleep(0.1)
    assert not events
