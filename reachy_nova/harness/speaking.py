"""Speaker leg — Nova Sonic's voice out through the daemon HTTP media route.

The deployed reachy-mini-cli 0.48.0 has no streaming speaker feed (that seam
is reachy-mini-cli#162, unshipped), so the harness speaks the way the proven
``agent embody`` path does (``reachy/speech/playback.py`` http transport):

1. ``POST {base}/api/media/sounds/upload`` — multipart/form-data, one file
   field, body = a complete WAV (mono int16 LE PCM; the WAV header declares
   the true sample rate and the daemon resamples). Response is JSON
   ``{"path": "<saved name>"}`` (fall back to the uploaded filename).
2. ``POST {base}/api/media/play_sound`` with ``{"file": "<path>"}``.

The runtime owns the media session — this module NEVER touches
``/api/media/acquire`` / ``/api/media/release``. And because 0.48.0 has no
speaker arbitration (concurrent plays mix at the device), :class:`SonicSpeaker`
enforces the harness's own one-speaker-at-a-time discipline through the shared
:class:`~reachy_nova.harness.gate.EchoGate`: the worker waits for the previous
playback window to elapse before posting the next utterance, and arms the gate
for each utterance's duration after triggering it.

Utterance assembly mirrors Sonic's callback contract (``nova_sonic.py``):
``on_audio_chunk`` buffers the float32 24 kHz output chunks, and
``on_state_change`` flushes the buffer to the playback queue on the transition
OUT of ``"speaking"``. A long monologue that exceeds ``max_buffer_s`` (~15 s)
flushes mid-"speaking" so playback starts in chunks rather than after an
unbounded wait.

Mouth-loss grace: ANY failure in upload/play resolves to a named senselog drop
(``reason=playback-http-failed``), an immediate ``gate.clear()``, a purge of
all pending queued utterances, and one ``on_playback_failure()`` call — the
integrator wires that to Sonic's interruption handling, so a dead/preempted
speaker can never leave a stuck speaking state. The worker never retries and
survives for the next utterance.

stdlib + numpy only; never imports ``reachy_mini``
(``tests/test_harness_boundary.py``).
"""

from __future__ import annotations

import io
import json
import os
import queue
import threading
import urllib.request
import wave
from collections.abc import Callable

import numpy as np

from reachy_nova import sensory_log
from reachy_nova.harness.gate import EchoGate

STAGE_SPEAK = "speak"
SOURCE = "nova"

DEFAULT_BASE_URL = "http://localhost:8000"
BASE_URL_ENV = "NOVA_DAEMON_URL"
_UPLOAD_PATH = "/api/media/sounds/upload"
_PLAY_PATH = "/api/media/play_sound"
_UPLOAD_FILENAME = "tts_synth.wav"
_HTTP_TIMEOUT_S = 10.0

#: Poll interval while the worker waits for the gate window / queue items.
_POLL_S = 0.02

_INT16_BYTES = 2
_INT16_MAX = 32767.0


# --------------------------------------------------------------------------- #
# Default poster — stdlib urllib multipart transport (no third-party deps).   #
# --------------------------------------------------------------------------- #


def _make_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """float32 [-1, 1] mono samples -> complete WAV bytes (int16 LE PCM)."""
    pcm = (np.clip(samples, -1.0, 1.0) * _INT16_MAX).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(_INT16_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _multipart_encode(filename: str, wav_bytes: bytes) -> tuple[bytes, str]:
    """Encode a single-file multipart/form-data body; return (body, content-type)."""
    boundary = "----ReachyNovaSpeakBoundary"
    ctype = f"multipart/form-data; boundary={boundary}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n"
        f"\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + wav_bytes + tail, ctype


def _post(url: str, data: bytes, content_type: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read())


def default_poster(base_url: str, wav_bytes: bytes, filename: str) -> None:
    """Upload *wav_bytes* to the daemon and trigger playback (two POSTs).

    Raises on any HTTP/network failure — the worker's mouth-loss grace path
    catches everything.
    """
    base = base_url.rstrip("/")
    body, ctype = _multipart_encode(filename, wav_bytes)
    upload_resp = _post(f"{base}{_UPLOAD_PATH}", body, ctype, _HTTP_TIMEOUT_S)
    saved_path = upload_resp.get("path", filename)
    play_body = json.dumps({"file": saved_path}).encode("utf-8")
    _post(f"{base}{_PLAY_PATH}", play_body, "application/json", _HTTP_TIMEOUT_S)


# --------------------------------------------------------------------------- #
# SonicSpeaker                                                                #
# --------------------------------------------------------------------------- #


class SonicSpeaker:
    """Buffers Sonic's audio output and plays it utterance-by-utterance.

    Parameters
    ----------
    gate:
        The shared half-duplex :class:`EchoGate`. The worker waits for
        ``gate.remaining() == 0`` before each playback (one-speaker
        discipline) and arms it for the utterance duration after triggering.
    sample_rate:
        Rate of the incoming float32 chunks (Sonic outputs 24 kHz).
    base_url:
        Daemon base URL; defaults to ``$NOVA_DAEMON_URL`` or
        ``http://localhost:8000``.
    poster:
        Injectable HTTP transport ``poster(base_url, wav_bytes, filename)``.
        Defaults to :func:`default_poster` (stdlib urllib). Tests inject a
        fake so no network is touched.
    on_playback_failure:
        Called once per playback failure, AFTER the gate is cleared and the
        pending queue is purged — the integrator wires this to Sonic's
        interruption handling.
    max_buffer_s:
        Long-monologue cap: a buffer exceeding this many seconds flushes to
        the queue even while still ``"speaking"``.
    queue_size:
        Bound of the playback queue; overflow is a named senselog drop
        (``reason=queue-full``), never a block.
    """

    def __init__(
        self,
        gate: EchoGate,
        sample_rate: int = 24000,
        base_url: str | None = None,
        poster: Callable[[str, bytes, str], None] | None = None,
        on_playback_failure: Callable[[], None] | None = None,
        max_buffer_s: float = 15.0,
        queue_size: int = 8,
    ) -> None:
        self.gate = gate
        self.sample_rate = sample_rate
        self.base_url = base_url or os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
        self.poster = poster or default_poster
        self.on_playback_failure = on_playback_failure
        self.max_buffer_s = max_buffer_s
        self.queue_size = queue_size

        self.utterances_played = 0
        self.playback_failures = 0

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_lock = threading.Lock()
        self._sonic_state = "idle"
        self._utterance_seq = 0

        self._stop_event: threading.Event | None = None
        self._local_stop = threading.Event()
        self._worker: threading.Thread | None = None

    # -- read-only status ---------------------------------------------------

    @property
    def idle(self) -> bool:
        """True when nothing is queued and nothing is playing.

        Uses the queue's unfinished-task accounting (``task_done`` is called
        only after a playback attempt fully resolves), so an in-flight
        utterance keeps ``idle`` False with no separate flag race.
        """
        return self._queue.unfinished_tasks == 0  # type: ignore[attr-defined]

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # -- Sonic callback wire targets ----------------------------------------

    def on_audio_chunk(self, samples: np.ndarray) -> None:
        """Wire target for ``sonic.on_audio_output`` (float32 mono chunks).

        Appends to the current utterance buffer; flushes mid-"speaking" when
        the long-monologue cap is exceeded (chunked playback).
        """
        if samples is None or len(samples) == 0:
            return
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        with self._buffer_lock:
            self._buffer.append(chunk)
            self._buffer_samples += len(chunk)
            flush = self._buffer_samples >= self.max_buffer_s * self.sample_rate
            segment = self._take_buffer_locked() if flush else None
        if segment is not None:
            self._enqueue(segment, why="monologue-cap")

    def on_state_change(self, state: str) -> None:
        """Wire target for ``sonic.on_state_change``.

        On the transition OUT of ``"speaking"`` the buffered utterance is
        complete: enqueue it for playback and reset the buffer.
        """
        previous, self._sonic_state = self._sonic_state, state
        if previous == "speaking" and state != "speaking":
            with self._buffer_lock:
                segment = self._take_buffer_locked()
            if segment is not None:
                self._enqueue(segment, why="utterance-complete")

    # -- lifecycle ----------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        """Start the single playback worker thread."""
        self._stop_event = stop_event
        self._local_stop.clear()
        self._worker = threading.Thread(
            target=self._run, name="nova-speaker", daemon=True
        )
        self._worker.start()
        sensory_log.stage(
            STAGE_SPEAK, SOURCE, "start", f"speaker worker up base_url={self.base_url}"
        )

    def stop(self) -> None:
        """Stop the worker. Idempotent; never blocks longer than a poll tick."""
        self._local_stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)

    # -- internals ----------------------------------------------------------

    def _take_buffer_locked(self) -> np.ndarray | None:
        """Concatenate + reset the buffer. Caller holds ``_buffer_lock``."""
        if not self._buffer:
            return None
        segment = np.concatenate(self._buffer)
        self._buffer = []
        self._buffer_samples = 0
        return segment

    def _enqueue(self, samples: np.ndarray, why: str) -> None:
        self._utterance_seq += 1
        event = f"utt-{self._utterance_seq}"
        duration_s = len(samples) / self.sample_rate
        try:
            self._queue.put_nowait(samples)
        except queue.Full:
            sensory_log.stage(
                STAGE_SPEAK,
                SOURCE,
                event,
                f"dropped reason=queue-full duration={duration_s:.2f}s",
            )
            return
        sensory_log.stage(
            STAGE_SPEAK, SOURCE, event, f"queued duration={duration_s:.2f}s ({why})"
        )

    def _stopping(self) -> bool:
        if self._local_stop.is_set():
            return True
        return self._stop_event is not None and self._stop_event.is_set()

    def _run(self) -> None:
        """Worker loop: gate-wait -> WAV -> upload+play -> arm gate."""
        while not self._stopping():
            try:
                samples = self._queue.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            try:
                self._play_one(samples)
            finally:
                self._queue.task_done()

    def _play_one(self, samples: np.ndarray) -> None:
        # One-speaker discipline: wait out any previous playback window.
        while not self._stopping():
            remaining = self.gate.remaining()
            if remaining <= 0.0:
                break
            self._local_stop.wait(min(remaining, _POLL_S))
        if self._stopping():
            return

        duration_s = len(samples) / self.sample_rate
        wav_bytes = _make_wav_bytes(samples, self.sample_rate)
        # Arm BEFORE the post: play_sound returns when playback is triggered,
        # not when sound leaves the speaker — arming after the HTTP round trip
        # leaves the upload window plus the sink's start latency ungated, and
        # the mic transcribes the robot's own tail (observed live: Nova held a
        # conversation with its own echo).
        self.gate.arm_for(duration_s)
        try:
            self.poster(self.base_url, wav_bytes, _UPLOAD_FILENAME)
        except Exception as err:
            self._mouth_loss(duration_s, err)
            return
        self.utterances_played += 1
        sensory_log.stage(
            STAGE_SPEAK, SOURCE, "play", f"played duration={duration_s:.2f}s"
        )

    def preempt(self) -> None:
        """Barge-in: drop the building buffer and every queued utterance, free the gate.

        Wired to Sonic's ``on_interruption`` — the user spoke over the robot, so
        anything not yet on the speaker must never play.
        """
        with self._buffer_lock:
            self._buffer = []
            self._buffer_samples = 0
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            cleared += 1
        self.gate.clear()
        sensory_log.stage(
            STAGE_SPEAK,
            SOURCE,
            "preempt",
            f"dropped reason=barge-in pending={cleared}",
        )

    def _mouth_loss(self, duration_s: float, err: Exception) -> None:
        """Playback failed: route to the interruption path, never hang/retry."""
        self.playback_failures += 1
        sensory_log.stage(
            STAGE_SPEAK,
            SOURCE,
            "play",
            f"dropped reason=playback-http-failed duration={duration_s:.2f}s ({err})",
        )
        self.gate.clear()
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            cleared += 1
        if cleared:
            sensory_log.stage(
                STAGE_SPEAK,
                SOURCE,
                "play",
                f"dropped reason=preempted-after-failure pending={cleared}",
            )
        if self.on_playback_failure is not None:
            try:
                self.on_playback_failure()
            except Exception as cb_err:  # integrator bug must not kill the worker
                sensory_log.stage(
                    STAGE_SPEAK, SOURCE, "play", f"failure callback raised: {cb_err}"
                )
