"""Speech-capture lane: rolling ring buffer + ASR-driven (or XMOS-driven) VAD.

Feeds the same preprocessed 16 kHz mono audio the main loop hands to Sonic
into a rolling ``SpeechEventDetector``. Onset is detected either by an
injected, already-loaded ASR handle (mirroring wake_word.py's single-worker
background-transcription pattern — the SAME loaded parakeet instance, never
a second load) becoming non-empty, or, when no handle is given, by polling an
XMOS ``speech_flag_provider`` callable.

Once onset is detected, the clip emitted with ``on_speech`` is backtracked a
measured ``pre_roll_seconds`` before the onset — measured by scanning the
buffered audio for where its energy actually rises above a silence
threshold, not merely assumed to be some fixed offset from "now".

This module never touches main.py — wiring it into the main loop is a later
integration task. It never uploads anything: clips are written under a
local-only directory (default ``~/.reachy_nova/speech_clips``).
"""

import concurrent.futures
import logging
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Hz — must match the existing mic pipeline

_ONSET_WINDOW_SECONDS = 0.01  # 10ms analysis window for the amplitude scan
_DEFAULT_SILENCE_THRESHOLD = 0.02  # RMS


class SpeechEventDetector:
    """Detects utterance onset and emits a backtracked local clip + transcript.

    Two mutually-exclusive onset triggers:

    - ``asr_handle`` given: periodic background transcription (single-worker
      thread pool, mirroring ``wake_word.WakeWordDetector``) of the rolling
      buffer; onset = transcript becomes non-empty.
    - no ``asr_handle``: a ``speech_flag_provider`` callable is polled on
      every ``feed()`` call; onset = the flag's rising edge (the XMOS
      hardware speech_detected flag).

    Either way, once onset fires the emitted clip is the buffered audio from
    a *measured* onset sample (an amplitude scan over the buffer, not an
    assumed fixed offset) backtracked by ``pre_roll_seconds``.
    """

    def __init__(
        self,
        asr_handle=None,
        speech_flag_provider: Callable[[], bool] | None = None,
        on_speech: Callable[[dict], None] | None = None,
        pre_roll_seconds: float = 2.0,
        sample_rate: int = SAMPLE_RATE,
        buffer_seconds: float | None = None,
        transcribe_interval: float = 2.0,
        silence_threshold: float = _DEFAULT_SILENCE_THRESHOLD,
        clip_dir: str | Path | None = None,
    ):
        """
        Args:
            asr_handle: An ALREADY-LOADED ASR model/handle exposing
                ``transcribe([audio]) -> [text_or_result]`` (e.g. the same
                parakeet instance wake_word.py loaded — this lane never
                loads its own). None disables the ASR path.
            speech_flag_provider: Zero-arg callable returning the current
                XMOS hardware speech_detected flag. Only polled when
                ``asr_handle`` is None.
            on_speech: Callback invoked with the event payload dict whenever
                a speech event is emitted.
            pre_roll_seconds: How far before the measured onset the emitted
                clip should start.
            sample_rate: Audio sample rate in Hz (must match what's fed).
            buffer_seconds: Size of the rolling ring buffer. Must be large
                enough to still hold the true onset by the time it's
                measured; defaults to ``pre_roll_seconds + 8.0``.
            transcribe_interval: Seconds between background transcription
                runs (ASR path only), mirroring wake_word.py.
            silence_threshold: RMS level (over a 10ms window) above which
                audio counts as "speech" for onset measurement.
            clip_dir: Local directory clips are written under. Defaults to
                ``~/.reachy_nova/speech_clips``. Never uploaded anywhere —
                this module has no network code.
        """
        self._asr = asr_handle
        self._speech_flag_provider = speech_flag_provider
        self.on_speech = on_speech

        self._sample_rate = sample_rate
        self._pre_roll_seconds = pre_roll_seconds
        self._pre_roll_samples = int(pre_roll_seconds * sample_rate)
        if buffer_seconds is None:
            buffer_seconds = pre_roll_seconds + 8.0
        self._buffer_max = int(buffer_seconds * sample_rate)
        self._transcribe_interval = transcribe_interval
        self._silence_threshold = silence_threshold
        self._onset_window = max(1, int(_ONSET_WINDOW_SECONDS * sample_rate))

        self.clip_dir = Path(clip_dir) if clip_dir is not None else Path.home() / ".reachy_nova" / "speech_clips"

        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._total_samples_seen = 0

        self._speaking = False  # edge-trigger state (transcript/flag non-empty)
        self._event_count = 0

        # ASR (background transcription) bookkeeping — mirrors wake_word.py.
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        if self._asr is not None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="speech-events"
            )
        self._pending: concurrent.futures.Future | None = None
        self._pending_snapshot: np.ndarray | None = None
        self._pending_snapshot_start = 0
        self._last_transcribe = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, audio: np.ndarray) -> dict | None:
        """Feed a chunk of float32 mono audio at ``sample_rate``.

        Returns the event payload dict when a speech event fires on this
        call, otherwise None. ``on_speech`` (if set) is also invoked.
        """
        self._push(audio)

        if self._asr is not None:
            return self._feed_asr()
        return self._feed_flag()

    # ------------------------------------------------------------------
    # Ring buffer
    # ------------------------------------------------------------------

    def _push(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.size == 0:
            return
        self._buffer.append(chunk)
        self._buffer_samples += len(chunk)
        self._total_samples_seen += len(chunk)
        while len(self._buffer) > 1 and self._buffer_samples - len(self._buffer[0]) >= self._buffer_max:
            popped = self._buffer.pop(0)
            self._buffer_samples -= len(popped)

    def _concat_buffer(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros(0, dtype=np.float32)
        if len(self._buffer) == 1:
            return self._buffer[0]
        return np.concatenate(self._buffer)

    @property
    def _buffer_start_absolute(self) -> int:
        return self._total_samples_seen - self._buffer_samples

    # ------------------------------------------------------------------
    # ASR-driven onset (mirrors wake_word.WakeWordDetector.detect)
    # ------------------------------------------------------------------

    def _feed_asr(self) -> dict | None:
        result = None

        if self._pending is not None and self._pending.done():
            fut, self._pending = self._pending, None
            snapshot, snapshot_start = self._pending_snapshot, self._pending_snapshot_start
            self._pending_snapshot = None
            try:
                transcript = fut.result()
            except Exception as e:
                logger.warning(f"[SpeechEvents] Transcription error: {e}")
                transcript = ""

            if transcript:
                if not self._speaking:
                    self._speaking = True
                    result = self._emit_event(transcript=transcript, snapshot=snapshot, snapshot_start=snapshot_start)
            else:
                self._speaking = False

        now = time.time()
        if (
            self._pending is None
            and self._buffer_samples > 0
            and now - self._last_transcribe >= self._transcribe_interval
        ):
            self._last_transcribe = now
            snapshot = self._concat_buffer().copy()
            self._pending_snapshot = snapshot
            self._pending_snapshot_start = self._buffer_start_absolute
            self._pending = self._executor.submit(self._transcribe, snapshot)

        return result

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run inference on a float32 audio snapshot (called from thread pool)."""
        results = self._asr.transcribe([audio])
        if not results:
            return ""
        r = results[0]
        text = r.text if hasattr(r, "text") else str(r)
        return text.strip()

    # ------------------------------------------------------------------
    # XMOS flag-driven onset (fallback when no ASR handle is injected)
    # ------------------------------------------------------------------

    def _feed_flag(self) -> dict | None:
        if self._speech_flag_provider is None:
            return None

        flag = bool(self._speech_flag_provider())
        if flag:
            if not self._speaking:
                self._speaking = True
                snapshot = self._concat_buffer()
                snapshot_start = self._buffer_start_absolute
                return self._emit_event(transcript="", snapshot=snapshot, snapshot_start=snapshot_start)
        else:
            self._speaking = False
        return None

    # ------------------------------------------------------------------
    # Onset measurement + clip emission
    # ------------------------------------------------------------------

    def _measure_onset(self, snapshot: np.ndarray) -> int:
        """Scan for the first window whose RMS clears the silence threshold.

        Returns an offset (samples, relative to snapshot start) — this is a
        MEASUREMENT over the actual buffered audio, not an assumed fixed
        duration. Falls back to 0 (start of the snapshot) if nothing clears
        the threshold, so backtracking still applies conservatively.
        """
        win = self._onset_window
        n = len(snapshot)
        for start in range(0, n, win):
            window = snapshot[start : start + win]
            if window.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(window))))
            if rms >= self._silence_threshold:
                return start
        return 0

    def _emit_event(self, transcript: str, snapshot: np.ndarray, snapshot_start: int) -> dict:
        onset_offset = self._measure_onset(snapshot)
        onset_absolute = snapshot_start + onset_offset

        buffer_start_absolute = self._buffer_start_absolute
        clip_start_absolute = max(buffer_start_absolute, onset_absolute - self._pre_roll_samples)

        full_buffer = self._concat_buffer()
        clip_offset = clip_start_absolute - buffer_start_absolute
        clip_audio = full_buffer[clip_offset:]

        now = time.time()
        elapsed_since_onset = max(0.0, (self._total_samples_seen - onset_absolute) / self._sample_rate)
        onset_ts = now - elapsed_since_onset

        clip_path = self._write_clip(clip_audio)
        duration_seconds = len(clip_audio) / self._sample_rate

        payload = {
            "clip_path": str(clip_path),
            "transcript": transcript,
            "duration_seconds": float(duration_seconds),
            "onset_ts": float(onset_ts),
        }

        logger.info(
            f"[SpeechEvents] speech_detected — duration={duration_seconds:.2f}s, "
            f"transcript={transcript!r}, clip={clip_path}"
        )

        if self.on_speech is not None:
            try:
                self.on_speech(payload)
            except Exception as e:
                logger.warning(f"[SpeechEvents] on_speech callback error: {e}")

        return payload

    def _write_clip(self, samples: np.ndarray) -> Path:
        """Write a float32 [-1, 1] clip to a local-only WAV file (stdlib wave)."""
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self._event_count += 1
        filename = f"speech_{int(time.time() * 1000)}_{self._event_count:04d}.wav"
        path = self.clip_dir / filename

        clipped = np.clip(samples, -1.0, 1.0)
        int16_samples = (clipped * 32767.0).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(int16_samples.tobytes())

        return path
