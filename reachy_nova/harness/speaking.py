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
:class:`~reachy_nova.harness.gate.EchoGate`: the worker waits out the previous
playback before posting the next chunk, and arms the gate for each chunk's
duration after triggering it (see :meth:`SonicSpeaker._wait_for_the_speaker`
for why chunked mode waits out the audio rather than the gate's ear-side
margin).

Chunk assembly mirrors Sonic's callback contract (``nova_sonic.py``):
``on_audio_chunk`` buffers the float32 24 kHz output chunks, and the buffer is
flushed to the playback queue by the AUDIO ITSELF — never by waiting for Sonic
to leave ``"speaking"``:

* **size** — the buffer reaches ``chunk_s`` (~1 s), split at the lowest-RMS
  50 ms window inside the last 200 ms before the target so the cut lands in a
  pause instead of mid-word (below 250 ms of tail there is nowhere to search,
  and the split is the exact target);
* **inactivity** — no new audio for ``inactivity_s`` (~300 ms), which is what
  ends a short reply;
* **state change** — the transition OUT of ``"speaking"`` stays as the final
  sweep of whatever is left, no longer as the thing that starts playback.

Why: measured on the robot 2026-09-05/06, that state transition is produced by
the 4 s speaking watchdog and not by an end-of-turn event — five short replies
(0.84-1.28 s of audio) were each queued 4.3-4.6 s after their first audio
chunk, and a 12.5 s reply 9.9 s after. Buffering a whole utterance therefore
bought a 4 s silence on every single reply. ``max_buffer_s`` survives only as
an upper bound on the chunk target; ``chunked=False`` restores the whole-
utterance behaviour exactly (one ``tts_synth.wav``, no per-chunk cleanup).

Each chunk uploads under its own ``nova-<utt>-<seq>.wav`` (a single reused
filename was safe only while the next post waited out the previous window) and
is deleted through ``DELETE /api/media/sounds/{filename}`` once its window has
elapsed, with at most ``max_outstanding_files`` undeleted at a time — the
robot's root disk is at 90 %. Deletes run in the slack AFTER a post, never
before one, so cleanup can never sit on the latency path; a failed delete is
one latched senselog line and playback continues.

Chunks post with NO pre-roll: a playback probe on the robot (2026-09-06) put
two 1 s tones back to back, posting the second exactly when the first's window
ended, and the join was seamless to the ear — no gap, no click — with
``play_sound`` returning in 29-73 ms.

Mouth-loss grace: ANY failure in upload/play resolves to a named senselog drop
(``reason=playback-http-failed``), an immediate ``gate.clear()``, a purge of
all pending queued utterances, and one ``on_playback_failure()`` call — the
integrator wires that to Sonic's interruption handling, so a dead/preempted
speaker can never leave a stuck speaking state. The worker never retries and
survives for the next utterance.

Timed quiet (``quiet.QuietState``, optional) gates this leg in ONE place —
the top of ``_play_one``. A dropped-for-quiet utterance is a no-op everywhere
else: no post, no gate arm, no queue purge, no ``on_playback_failure``. Quiet
is not mouth loss, and the ear is untouched by it.

stdlib + numpy only; never imports ``reachy_mini``
(``tests/test_harness_boundary.py``).
"""

from __future__ import annotations

import io
import os
import queue
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from reachy_nova import sensory_log
from reachy_nova.harness.daemon_client import (
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    DaemonClient,
)
from reachy_nova.harness.gate import EchoGate
from reachy_nova.harness.quiet import QuietState

STAGE_SPEAK = "speak"
SOURCE = "nova"

#: The single reused filename of whole-utterance mode (``chunked=False``).
_UPLOAD_FILENAME = "tts_synth.wav"
#: Per-chunk upload name, ``<utterance counter>-<chunk index within it>``.
_CHUNK_FILENAME = "nova-{utt}-{seq}.wav"

#: Poll interval while the worker waits for the gate window / queue items.
_POLL_S = 0.02

#: How far back from the chunk target a low-energy split point is looked for.
_SPLIT_SEARCH_S = 0.20
#: Width of the RMS window whose end becomes the split point.
_SPLIT_WINDOW_S = 0.05
#: Below this much tail (search + window) there is nowhere to search at all.
_MIN_SPLIT_TARGET_S = _SPLIT_SEARCH_S + _SPLIT_WINDOW_S

#: How long past a chunk's own audio its file is left on the daemon before it
#: is deleted — cover for ``play_sound``'s trigger latency (29-73 ms measured
#: on the robot 2026-09-06), not for the echo gate's ear-side margin.
_DELETE_GRACE_S = 0.25

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


def default_poster(base_url: str, wav_bytes: bytes, filename: str) -> None:
    """Upload *wav_bytes* to the daemon and trigger playback (two POSTs).

    Delegates to the shared :class:`~reachy_nova.harness.daemon_client.DaemonClient`
    (task t10) — same two calls (upload, then play) this module used to make
    directly. Raises on any HTTP/network failure — the worker's mouth-loss
    grace path catches everything.
    """
    client = DaemonClient(base_url=base_url)
    saved_path = client.upload_sound(wav_bytes, filename)
    client.play_sound(saved_path)


def default_stopper(base_url: str) -> None:
    """Stop whatever the daemon speaker is playing right now (barge-in cut).

    ``POST /api/media/stop_sound`` exists on the deployed daemon (verified in
    its openapi 2026-08-12). Raises on HTTP/network failure — ``preempt()``
    treats a failed stop as best-effort and keeps going.
    """
    DaemonClient(base_url=base_url).stop_sound()


def default_deleter(base_url: str, filename: str) -> None:
    """Remove one played-out chunk file from the daemon's sounds directory.

    ``DELETE /api/media/sounds/{filename}`` takes the BARE filename the chunk
    was uploaded under — the daemon owns the directory it saved it into.
    Raises on HTTP/network failure; the caller names the drop itself and keeps
    speaking (see :meth:`SonicSpeaker._delete_file`).
    """
    DaemonClient(base_url=base_url).delete_sound(filename)


# --------------------------------------------------------------------------- #
# Chunking                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Chunk:
    """One queued unit of playback: the samples plus the name they upload as.

    In whole-utterance mode there is exactly one chunk per utterance and
    ``filename`` is the single reused ``tts_synth.wav``; in chunked mode
    ``utt``/``seq`` number the chunk within its utterance and give it its own
    file to be deleted after its window.
    """

    samples: np.ndarray
    utt: int
    seq: int
    filename: str

    @property
    def label(self) -> str:
        """``<utt>-<seq>`` — how a chunk names itself in the senselog."""
        return f"{self.utt}-{self.seq}"

    def duration_s(self, sample_rate: int) -> float:
        return len(self.samples) / sample_rate


def _split_index(samples: np.ndarray, target: int, sample_rate: int) -> int:
    """Where to cut a buffer that has reached *target* samples.

    The cut lands at the END of the lowest-RMS :data:`_SPLIT_WINDOW_S` window
    inside the last :data:`_SPLIT_SEARCH_S` before *target*, so the outgoing
    chunk carries the pause and the next one begins on sound: any residual
    latency at the join is then spent in silence rather than mid-word. Ties
    resolve to the earliest window (shortest chunk, earliest audio).

    A target with less than :data:`_MIN_SPLIT_TARGET_S` of tail has nowhere to
    search and is cut at exactly *target*.
    """
    window = int(_SPLIT_WINDOW_S * sample_rate)
    search = int(_SPLIT_SEARCH_S * sample_rate)
    if window < 1 or target < _MIN_SPLIT_TARGET_S * sample_rate:
        return target
    start = target - search
    region = np.square(samples[start:target].astype(np.float64))
    # Sliding-window energy in one pass: cumulative sums, differenced by width.
    # RMS and energy rank identically for a fixed window, so the sqrt/mean are
    # never computed.
    cumulative = np.concatenate(([0.0], np.cumsum(region)))
    energy = cumulative[window:] - cumulative[:-window]
    return start + int(np.argmin(energy)) + window


# --------------------------------------------------------------------------- #
# SonicSpeaker                                                                #
# --------------------------------------------------------------------------- #


class SonicSpeaker:
    """Cuts Sonic's audio output into chunks and plays them one at a time.

    Parameters
    ----------
    gate:
        The shared half-duplex :class:`EchoGate`. The worker waits out the
        previous playback on it before each post (one-speaker discipline) and
        arms it for the chunk's duration after triggering — see
        :meth:`_wait_for_the_speaker` for what "waits out" means in each mode.
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
        Upper bound on the chunk target. In whole-utterance mode this is the
        long-monologue cap it always was; in chunked mode ``chunk_s`` is far
        below it and it never bites (kept for call-site compatibility).
    queue_size:
        Bound of the playback queue; overflow is a named senselog drop
        (``reason=queue-full``), never a block.
    quiet:
        Optional :class:`~reachy_nova.harness.quiet.QuietState`. While it is
        active every chunk is DROPPED before anything else happens — no post,
        no gate arm, no queue purge, no ``on_playback_failure``: a quiet robot
        is quiet, not broken (see :meth:`_quiet_blocks`).
    chunked:
        Chunked playback (the default). ``False`` restores the whole-utterance
        behaviour that shipped before task t8 — one buffer per utterance, the
        single reused ``tts_synth.wav``, no per-chunk cleanup — and is what
        ``NOVA_CHUNKED_PLAYBACK=0`` selects.
    chunk_s:
        Target chunk size. 1 s is the measured compromise: short enough that
        the first audio lands about a second after Sonic's first sample, long
        enough that the ~30-70 ms ``play_sound`` round trip is a small
        fraction of each window.
    inactivity_s:
        Flush the buffer when no new audio has arrived for this long. This is
        what ends a SHORT reply, whose buffer never reaches ``chunk_s``.
    deleter:
        Injectable ``deleter(base_url, filename)`` removing a played-out chunk
        file. Defaults to :func:`default_deleter`.
    max_outstanding_files:
        How many uploaded-but-undeleted chunk files may exist at once; over
        the cap the oldest is deleted immediately, whatever its window says.
    clock:
        Injectable monotonic clock for the size/inactivity timing (tests drive
        it directly). The echo gate keeps its own real clock — playback
        windows are real time.
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
        stopper: Callable[[str], None] | None = None,
        quiet: QuietState | None = None,
        chunked: bool = True,
        chunk_s: float = 1.0,
        inactivity_s: float = 0.30,
        deleter: Callable[[str, str], None] | None = None,
        max_outstanding_files: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gate = gate
        self.quiet = quiet
        self.sample_rate = sample_rate
        self.base_url = base_url or os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
        self.poster = poster or default_poster
        self.stopper = stopper or default_stopper
        self.deleter = deleter or default_deleter
        self.on_playback_failure = on_playback_failure
        self.max_buffer_s = max_buffer_s
        self.queue_size = queue_size

        self.chunked = chunked
        self.chunk_s = chunk_s
        self.inactivity_s = inactivity_s
        self.max_outstanding_files = max_outstanding_files
        self._clock = clock
        #: The size flush target — ``max_buffer_s`` is only ever an upper bound.
        self._target_s = min(chunk_s, max_buffer_s)

        self.utterances_played = 0
        self.chunks_played = 0
        self.playback_failures = 0
        #: Chunk files successfully removed from the daemon's sounds dir.
        self.files_deleted = 0
        #: Failed deletes (logged once per run, counted every time).
        self.delete_failures = 0
        self._delete_drop_logged = False
        #: Chunks dropped by the quiet deadline since it was armed.
        self.quiet_drops = 0
        self._quiet_drop_logged = False

        #: Bumped by every preempt(). The worker snapshots it when it dequeues
        #: an utterance and re-checks before (and after) posting: an utterance
        #: that was in the worker's hands when the barge-in landed must never
        #: play — purging the queue alone misses it, and clearing the gate
        #: actually RELEASES it (observed live 2026-08-11 23:15: 'preempt ...
        #: stopped=True' followed by 'played duration=5.76s' the same second).
        self._preempt_epoch = 0

        self._queue: queue.Queue[_Chunk] = queue.Queue(maxsize=queue_size)
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_lock = threading.Lock()
        self._last_audio_t: float | None = None
        self._sonic_state = "idle"
        self._utterance_seq = 0
        #: Chunk index within the current utterance; 0 means "none flushed yet".
        self._chunk_seq = 0

        #: (filename, time its playback window ends), oldest first. Mutated
        #: only by whichever thread runs :meth:`_play_one` — the single
        #: playback worker — but read by anyone asking for
        #: :attr:`outstanding_files`, hence the lock. The DELETE itself always
        #: happens outside it; housekeeping must not hold a lock over HTTP.
        self._outstanding: deque[tuple[str, float]] = deque()
        self._files_lock = threading.Lock()

        self._stop_event: threading.Event | None = None
        self._local_stop = threading.Event()
        self._worker: threading.Thread | None = None

    # -- read-only status ---------------------------------------------------

    @property
    def idle(self) -> bool:
        """True when nothing is buffered, nothing is queued and nothing plays.

        Uses the queue's unfinished-task accounting (``task_done`` is called
        only after a playback attempt fully resolves), so a chunk in flight
        keeps ``idle`` False with no separate flag race — and the buffer is
        checked too, because under chunking an utterance is routinely between
        chunks with an empty queue and audio still waiting to be cut.
        """
        if self._queue.unfinished_tasks:  # type: ignore[attr-defined]
            return False
        with self._buffer_lock:
            return self._buffer_samples == 0

    @property
    def outstanding_files(self) -> list[str]:
        """Chunk files uploaded to the daemon and not yet deleted, oldest first."""
        with self._files_lock:
            return [filename for filename, _due in self._outstanding]

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # -- Sonic callback wire targets ----------------------------------------

    def on_audio_chunk(self, samples: np.ndarray) -> None:
        """Wire target for ``sonic.on_audio_output`` (float32 mono chunks).

        Appends to the buffer and, in chunked mode, hands every whole chunk
        the buffer now contains to the playback queue immediately — this is
        the size flush, and it is what makes the first audio audible about a
        second after Sonic's first sample instead of after the speaking
        watchdog. In whole-utterance mode only the long-monologue cap flushes
        here, exactly as before.
        """
        if samples is None or len(samples) == 0:
            return
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        segments: list[np.ndarray] = []
        with self._buffer_lock:
            self._buffer.append(chunk)
            self._buffer_samples += len(chunk)
            self._last_audio_t = self._clock()
            if self.chunked:
                segments = self._take_chunks_locked()
            elif self._buffer_samples >= self.max_buffer_s * self.sample_rate:
                segment = self._take_buffer_locked()
                if segment is not None:
                    segments = [segment]
        why = "size" if self.chunked else "monologue-cap"
        for segment in segments:
            self._enqueue(segment, why=why)

    def on_state_change(self, state: str) -> None:
        """Wire target for ``sonic.on_state_change``.

        The transition OUT of ``"speaking"`` is the FINAL sweep, not the start
        of playback: whatever is left in the buffer (a sub-chunk tail the size
        and inactivity flushes have not claimed) goes to the queue, and the
        chunk numbering restarts for the next utterance. Entering ``"speaking"``
        restarts it too, so an utterance's chunks always begin at ``seq=1``.
        """
        previous, self._sonic_state = self._sonic_state, state
        if state == "speaking" and previous != "speaking":
            self._chunk_seq = 0
            return
        if previous == "speaking" and state != "speaking":
            with self._buffer_lock:
                segment = self._take_buffer_locked()
            if segment is not None:
                self._enqueue(
                    segment,
                    why="state-change" if self.chunked else "utterance-complete",
                )
            self._chunk_seq = 0

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

    def _take_chunks_locked(self) -> list[np.ndarray]:
        """Cut every whole chunk the buffer now holds. Caller holds the lock.

        A loop rather than a single cut, because one Sonic callback may carry
        several chunks' worth of audio; the remainder stays buffered for the
        next size flush, the inactivity timer, or the final state change.
        """
        target = int(self._target_s * self.sample_rate)
        if target <= 0:
            return []
        segments: list[np.ndarray] = []
        while self._buffer_samples >= target:
            if len(self._buffer) == 1:
                merged = self._buffer[0]
            else:
                merged = np.concatenate(self._buffer)
            split = _split_index(merged, target, self.sample_rate)
            segments.append(merged[:split].copy())
            rest = merged[split:]
            self._buffer = [rest] if len(rest) else []
            self._buffer_samples = len(rest)
        return segments

    def _flush_if_inactive(self) -> None:
        """Queue the buffer when no new audio has arrived for ``inactivity_s``.

        The other half of the chunker, and the half that ends a SHORT reply:
        a 0.8 s answer never reaches the size target, and waiting for Sonic to
        leave ``"speaking"`` would cost the 4 s the watchdog takes to say so.
        Driven by the playback worker's own poll loop (``_POLL_S`` is 20 ms),
        so there is no second thread to own and stop.
        """
        if not self.chunked:
            return
        with self._buffer_lock:
            if not self._buffer or self._last_audio_t is None:
                return
            if self._clock() - self._last_audio_t < self.inactivity_s:
                return
            segment = self._take_buffer_locked()
        if segment is not None:
            self._enqueue(segment, why="inactivity")

    def _next_chunk_identity(self) -> tuple[int, int, str]:
        """Number the chunk about to be queued and name the file it uploads as.

        Chunked mode numbers chunks within an utterance (``nova-<utt>-<seq>``,
        each its own file to delete after its window); whole-utterance mode
        keeps the single reused ``tts_synth.wav`` it always had — safe there
        precisely because the next post waits out the previous window.
        """
        if not self.chunked:
            self._utterance_seq += 1
            return self._utterance_seq, 1, _UPLOAD_FILENAME
        if self._chunk_seq == 0:
            self._utterance_seq += 1
        self._chunk_seq += 1
        filename = _CHUNK_FILENAME.format(utt=self._utterance_seq, seq=self._chunk_seq)
        return self._utterance_seq, self._chunk_seq, filename

    def _enqueue(self, samples: np.ndarray, why: str) -> None:
        utt, seq, filename = self._next_chunk_identity()
        chunk = _Chunk(samples=samples, utt=utt, seq=seq, filename=filename)
        event = f"utt-{utt}"
        duration_s = chunk.duration_s(self.sample_rate)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            sensory_log.stage(
                STAGE_SPEAK,
                SOURCE,
                event,
                f"dropped reason=queue-full chunk={chunk.label} "
                f"duration={duration_s:.2f}s",
            )
            return
        sensory_log.stage(
            STAGE_SPEAK,
            SOURCE,
            event,
            f"queued chunk={chunk.label} duration={duration_s:.2f}s ({why})",
        )

    def _stopping(self) -> bool:
        if self._local_stop.is_set():
            return True
        return self._stop_event is not None and self._stop_event.is_set()

    def _run(self) -> None:
        """Worker loop: inactivity check -> gate-wait -> WAV -> upload+play.

        The idle tick does the chunker's timekeeping and the cleanup nobody
        else will get to: the inactivity flush (which is what ends a short
        reply) and the reaping of the LAST chunk's file, which has no
        following post to sweep it up.
        """
        while not self._stopping():
            self._flush_if_inactive()
            try:
                chunk = self._queue.get(timeout=_POLL_S)
            except queue.Empty:
                self._reap_due()
                continue
            try:
                self._play_one(chunk)
            finally:
                self._queue.task_done()

    def _quiet_blocks(self, duration_s: float) -> bool:
        """Timed quiet: is the mouth closed for this utterance? Latched logging.

        Called FIRST in :meth:`_play_one`, before the gate wait and before any
        HTTP work, because a quiet drop must leave no trace on the speaker
        state machine: the poster is never called, the echo gate is never
        armed (so the ear is untouched), the queue is never purged and
        ``on_playback_failure`` never fires — quiet is not mouth loss and the
        mind must not think the mouth is gone.

        Logging is latched the way ``hearing.py`` latches its gate window: ONE
        line for the first drop, then one summary carrying the count when the
        first chunk gets through again. A ten-minute quiet during a chatty
        session must not cost ten minutes of log lines.
        """
        if self.quiet is None:
            return False
        if not self.quiet.allow_utterance():
            self.quiet_drops += 1
            if not self._quiet_drop_logged:
                self._quiet_drop_logged = True
                sensory_log.stage(
                    STAGE_SPEAK,
                    SOURCE,
                    "quiet-drop",
                    f"dropped reason=quiet duration={duration_s:.2f}s "
                    "(further drops summarised on release)",
                )
            return True
        if self.quiet_drops:
            sensory_log.stage(
                STAGE_SPEAK,
                SOURCE,
                "quiet-resume",
                f"speaking again count={self.quiet_drops} (utterances dropped while quiet)",
            )
            self.quiet_drops = 0
            self._quiet_drop_logged = False
        return False

    def _wait_for_the_speaker(self) -> bool:
        """Hold until the previous playback has left the speaker. False = stop.

        One-speaker discipline: reachy-mini-cli 0.48.0 has no speaker
        arbitration, so two overlapping posts literally mix at the device.
        The wait therefore reads the SAME echo-gate window the previous
        playback armed — but in chunked mode it stops at the gate's tail
        margin instead of at the end of it.

        Why: ``EchoGate.arm_for`` pads every window with ``margin_s`` (1 s by
        default, and that is what ``app.py`` builds), because the EAR wants a
        pad after the robot's audio ends. Playback arbitration does not — the
        audio is over at ``duration_s``. Paying the ear's pad between chunk 3
        and chunk 4 of one sentence would insert a second of silence mid-word
        and hand back the delay this whole task removes. So the speaker waits
        out the audio and posts there, with no pre-roll: a robot probe on
        2026-09-06 put two 1 s tones together exactly that way and heard no
        gap and no click (``play_sound`` returns in 29-73 ms, which is the
        entire join). The ear's window is untouched — ``gate.active`` still
        carries the full margin for ``hearing.py``.

        Whole-utterance mode keeps the original ``remaining() <= 0`` wait,
        margin included, exactly as it shipped.
        """
        floor = self.gate.margin_s if self.chunked else 0.0
        while not self._stopping():
            remaining = self.gate.remaining() - floor
            if remaining <= 0.0:
                return True
            self._local_stop.wait(min(remaining, _POLL_S))
        return False

    def _play_one(self, chunk: _Chunk) -> None:
        epoch = self._preempt_epoch
        duration_s = chunk.duration_s(self.sample_rate)
        if self._quiet_blocks(duration_s):
            return
        if not self._wait_for_the_speaker():
            return
        if self._preempt_epoch != epoch:
            # A barge-in landed while this chunk sat in the worker's hands.
            sensory_log.stage(
                STAGE_SPEAK,
                SOURCE,
                "play",
                f"dropped reason=preempted chunk={chunk.label} (stale chunk)",
            )
            return

        wav_bytes = _make_wav_bytes(chunk.samples, self.sample_rate)
        # Arm BEFORE the post: play_sound returns when playback is triggered,
        # not when sound leaves the speaker — arming after the HTTP round trip
        # leaves the upload window plus the sink's start latency ungated, and
        # the mic transcribes the robot's own tail (observed live: Nova held a
        # conversation with its own echo).
        self.gate.arm_for(duration_s)
        self._track_file(chunk.filename, duration_s)
        try:
            self.poster(self.base_url, wav_bytes, chunk.filename)
        except Exception as err:
            self._mouth_loss(duration_s, err, chunk)
            return
        if self._preempt_epoch != epoch:
            # The barge-in landed during the upload/play round trip: the sound
            # just started against the user's interruption. Cut it again.
            try:
                self.stopper(self.base_url)
            except Exception:  # noqa: BLE001 - best-effort
                pass
            self.gate.clear()
            sensory_log.stage(
                STAGE_SPEAK,
                SOURCE,
                "play",
                f"dropped reason=preempted chunk={chunk.label} (cut after post)",
            )
            return
        self.chunks_played += 1
        if chunk.seq == 1:
            self.utterances_played += 1
        sensory_log.stage(
            STAGE_SPEAK,
            SOURCE,
            "play",
            f"played chunk={chunk.label} duration={duration_s:.2f}s",
        )
        # Housekeeping in the slack: this chunk's whole window is ahead of us,
        # so the previous chunk's file goes now and the next post never waits
        # behind a DELETE round trip.
        self._reap_due()

    # -- chunk file housekeeping --------------------------------------------

    def _track_file(self, filename: str, duration_s: float) -> None:
        """Remember a chunk file to delete once its playback window ends.

        Recorded BEFORE the post, so a half-completed upload/play still leaves
        the file accounted for — the robot's root disk is at 90 % and an
        orphaned WAV per failed post is exactly the outage this avoids.

        The window is the chunk's AUDIO plus :data:`_DELETE_GRACE_S`, not the
        echo gate's padded one: the daemon is done with the file when the
        sound has played, and the grace only covers ``play_sound``'s trigger
        latency (29-73 ms measured on the robot 2026-09-06).
        """
        if not self.chunked:
            return
        over_cap: list[str] = []
        with self._files_lock:
            due = self._clock() + duration_s + _DELETE_GRACE_S
            self._outstanding.append((filename, due))
            while len(self._outstanding) > self.max_outstanding_files:
                over_cap.append(self._outstanding.popleft()[0])
        for stale in over_cap:
            self._delete_file(stale)

    def _reap_due(self) -> None:
        """Delete every chunk file whose playback window has elapsed."""
        if not self._outstanding:
            return
        now = self._clock()
        expired: list[str] = []
        with self._files_lock:
            while self._outstanding and self._outstanding[0][1] <= now:
                expired.append(self._outstanding.popleft()[0])
        for filename in expired:
            self._delete_file(filename)

    def _delete_file(self, filename: str) -> None:
        """One best-effort DELETE. A failure is latched, never a lost voice.

        The disk filling up is a real outage path, but so is a mouth that
        stops because housekeeping failed — so this counts every failure,
        names the FIRST one, and returns.
        """
        try:
            self.deleter(self.base_url, filename)
        except Exception as err:  # noqa: BLE001 - cleanup is never fatal
            self.delete_failures += 1
            if not self._delete_drop_logged:
                self._delete_drop_logged = True
                sensory_log.stage(
                    STAGE_SPEAK,
                    SOURCE,
                    "cleanup",
                    f"dropped reason=chunk-delete-failed file={filename} ({err}) "
                    "(further failures counted, not logged)",
                )
            return
        self.files_deleted += 1

    def preempt(self) -> None:
        """Barge-in: cut the playing sound, drop everything queued, free the gate.

        Wired to Sonic's ``on_interruption`` AND to user-speech-during-playback
        (``app.py``) — the user spoke over the robot, so the sound on the
        speaker stops now (``stopper`` → ``POST /api/media/stop_sound``) and
        anything not yet playing must never play. Under chunking that is the
        whole rest of the utterance: chunks 3..n are purged from the queue,
        the one already in the worker's hands dies on the epoch check, and the
        buffered tail is dropped before it can ever become a chunk. The stop
        is best-effort: a daemon that cannot stop still gets its queue purged
        and gate cleared. The chunk files already uploaded stay tracked and
        are reaped on their own due times.
        """
        self._preempt_epoch += 1
        with self._buffer_lock:
            self._buffer = []
            self._buffer_samples = 0
        self._chunk_seq = 0  # what survives the cut is a NEW utterance
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            cleared += 1
        stopped = True
        try:
            self.stopper(self.base_url)
        except Exception:  # noqa: BLE001 - best-effort; the purge already happened
            stopped = False
        self.gate.clear()
        sensory_log.stage(
            STAGE_SPEAK,
            SOURCE,
            "preempt",
            f"dropped reason=barge-in pending={cleared} stopped={stopped}",
        )

    def _mouth_loss(self, duration_s: float, err: Exception, chunk: _Chunk) -> None:
        """Playback failed: route to the interruption path, never hang/retry.

        ``playback_failures`` is bumped AFTER the purge, not before it: the
        counter is the only signal an outside observer has that the grace path
        has finished, and anything enqueued between "a failure happened" and
        "the queue was purged" would be swallowed by that purge. Publishing
        the count first made the counter a lie about the state it names (a
        flaky ordering in tests/chaos/test_chaos_aws_loss.py, ~2 runs in 10).
        """
        sensory_log.stage(
            STAGE_SPEAK,
            SOURCE,
            "play",
            f"dropped reason=playback-http-failed chunk={chunk.label} "
            f"duration={duration_s:.2f}s ({err})",
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
        self.playback_failures += 1
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
