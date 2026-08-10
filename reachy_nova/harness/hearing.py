"""Audio-tee client — the harness's ear, with an echo-safe half-duplex gate.

The wireless Reachy Mini's microphone belongs to the reachy-mini-cli runtime,
not to us. The runtime publishes what it hears on an ``AF_UNIX``
``SOCK_STREAM`` **audio tee**, and this module is that tee's reader: it
connects, reads the wire, and hands 16 kHz float32 mono chunks to
``sonic.feed_audio`` — one thread, no robot SDK, no motion code.

Wire contract (reachy-mini-cli 0.48.0, ``reachy/behavior/audio_tee.py``;
reference reader: ``reachy/embody/media.py::_RobotTeeSourceBackend``)::

    {"stream":"reachy-audio-tee","version":1,"format":"f32le",
     "channels":1,"samplerate":16000}\\n
    <little-endian float32 mono samples in [-1, 1], contiguous, forever>

Four consequences shape everything below:

* **The header is read first, in full, and validated.** It is the one thing a
  hearer cannot guess, so an absent, truncated, unparseable or foreign header
  is a NAMED refusal (:data:`TEE_HEADER_INVALID` / :data:`TEE_HEADER_FOREIGN`)
  plus a disconnect and a backoff — never "start reading samples anyway",
  which is exactly how ASCII header bytes would reach Sonic as audio.
* **The rate comes off the wire, and may be ``null``.** A cold media holder
  cannot report the mic's real rate yet, and the runtime says so rather than
  inventing one. We do the same: :data:`TEE_RATE_UNKNOWN`, disconnect, retry.
  Resampling against a guessed rate would pitch-shift everything Sonic hears,
  silently — a wrong answer is worse than a named gap.
* **A recv may land MID-SAMPLE.** A sample is 4 bytes, so the trailing partial
  sample is buffered and prefixed onto the next read. An off-by-one here
  shifts every sample after it, for the life of the connection.
* **The socket is usually absent.** The runtime being down is the ORDINARY
  resting state of a peripheral, not an error: reconnect with backoff, report
  it once (latched), never crash the thread.

Half-duplex gate
----------------
The wireless capture path has no verified hardware AEC, so while the robot is
speaking the mic feed to Sonic is suppressed rather than echoed back at it
(``gate.EchoGate``, armed by the speaking leg). A gate window costs exactly
two log lines — one when it opens, one summarising the suppressed chunks when
it clears — never one line per 100 ms chunk.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from reachy_nova import sensory_log
from reachy_nova.audio_pipeline import preprocess_mic_audio
from reachy_nova.harness import statedir
from reachy_nova.harness.gate import EchoGate

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Senselog identity                                                            #
# --------------------------------------------------------------------------- #

#: ``[SENSE stage=hear source=nova event=...]`` — every line this module emits.
STAGE = "hear"
SOURCE = "nova"

# --------------------------------------------------------------------------- #
# Wire constants — CITED from reachy-mini-cli 0.48.0, never negotiated         #
# --------------------------------------------------------------------------- #

TEE_WIRE_NAME = "reachy-audio-tee"
TEE_WIRE_VERSION = 1
TEE_WIRE_FORMAT = "f32le"
#: Explicit little-endian float32: the wire's byte order, not the machine's.
TEE_SAMPLE_DTYPE = "<f4"
BYTES_PER_SAMPLE = 4
TEE_HEADER_TERMINATOR = b"\n"
#: Wire versions this reader understands. A tuple, not a set, so an unhashable
#: ``version`` (e.g. ``[1]``) is REFUSED rather than raising inside the test.
UNDERSTOOD_WIRE_VERSIONS = (TEE_WIRE_VERSION,)
#: Cap on the header line — a peer that never sends a newline is refused
#: rather than buffered without bound. The real header is well under 200 bytes.
MAX_HEADER_BYTES = 4096

# --------------------------------------------------------------------------- #
# NAMED drop reasons — every silence this module produces has a name           #
# --------------------------------------------------------------------------- #

#: The socket is not there (runtime down / tee not started). Ordinary.
TEE_UNAVAILABLE = "tee-unavailable"
#: A zero-length read: the writer detached.
TEE_CLOSED = "tee-closed"
#: ``recv`` itself failed.
TEE_READ_FAILED = "tee-read-failed"
#: Whatever is on that socket is not this protocol at all.
TEE_HEADER_INVALID = "tee-header-invalid"
#: Well-formed, but announces a stream/version/format/channel count we cannot
#: consume.
TEE_HEADER_FOREIGN = "tee-header-foreign"
#: A legitimate header whose ``samplerate`` is ``null`` — the mic's real rate
#: is unknown, so nothing is read against a guess.
TEE_RATE_UNKNOWN = "tee-rate-unknown"
#: The downstream consumer (``sonic.feed_audio``) raised. Latched per run.
FEED_FAILED = "feed-failed"

# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #

DEFAULT_CHUNK_MS = 100
DEFAULT_TARGET_SR = 16000
DEFAULT_RECV_BYTES = 4096
DEFAULT_CONNECT_TIMEOUT_S = 0.5
#: Short enough that ``stop_event`` is honoured promptly, long enough that an
#: idle tee does not spin the CPU.
DEFAULT_READ_TIMEOUT_S = 0.1
DEFAULT_BACKOFF_MIN_S = 1.0
DEFAULT_BACKOFF_MAX_S = 5.0
#: A cold media holder warms up on its own clock; 2 s is the runtime's own
#: order of magnitude for "ask again shortly".
DEFAULT_RATE_UNKNOWN_BACKOFF_S = 2.0
#: Granularity of the interruptible sleep — the shutdown latency ceiling.
_WAIT_SLICE_S = 0.05


class TeeHearing:
    """Reads the runtime's audio tee into ``feed`` — gated, resampled, named.

    Parameters
    ----------
    feed:
        Called with each chunk of float32 mono samples at ``target_sr``. This
        is ``sonic.feed_audio`` in the harness. An exception from it is a
        NAMED drop, never a dead reader thread.
    gate:
        The half-duplex :class:`~reachy_nova.harness.gate.EchoGate`. While it
        is active every chunk is dropped (counted, summarised once).
    chunk_ms:
        Chunk size in milliseconds of AUDIO, computed at the header's
        announced rate — so a 100 ms chunk stays 100 ms whatever the mic runs
        at.
    socket_path:
        The tee socket. Defaults to
        :func:`reachy_nova.harness.statedir.audio_tee_socket` (which honours
        ``$REACHY_AUDIO_TEE_SOCKET``).
    target_sr:
        The rate ``feed`` wants — 16 kHz, Nova Sonic's input rate.
    """

    def __init__(
        self,
        feed: Callable[[np.ndarray], None],
        gate: EchoGate,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        socket_path: Path | str | None = None,
        target_sr: int = DEFAULT_TARGET_SR,
        *,
        recv_bytes: int = DEFAULT_RECV_BYTES,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
        backoff_min_s: float = DEFAULT_BACKOFF_MIN_S,
        backoff_max_s: float = DEFAULT_BACKOFF_MAX_S,
        rate_unknown_backoff_s: float = DEFAULT_RATE_UNKNOWN_BACKOFF_S,
    ) -> None:
        self.feed = feed
        self.gate = gate
        self.chunk_ms = int(chunk_ms)
        self.target_sr = int(target_sr)
        self.socket_path = (
            Path(socket_path) if socket_path is not None else statedir.audio_tee_socket()
        )
        self._recv_bytes = int(recv_bytes)
        self._connect_timeout_s = float(connect_timeout_s)
        self._read_timeout_s = float(read_timeout_s)
        self._backoff_min_s = float(backoff_min_s)
        self._backoff_max_s = float(backoff_max_s)
        self._rate_unknown_backoff_s = float(rate_unknown_backoff_s)

        # Counters — the status surface (``/api`` + tests).
        self.chunks_fed = 0
        self.chunks_gated = 0
        self.reconnects = 0

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._external_stop: threading.Event | None = None

        # One preallocated recv buffer for the life of the reader: the tee is
        # a hot loop and a fresh bytes object per recv is pure garbage.
        self._recv_buf = bytearray(self._recv_bytes)
        self._recv_view = memoryview(self._recv_buf)

        # Latches, so a persistent condition costs ONE line, not one per loop.
        self._down_reported = False
        self._feed_fault_reported = False
        self._connected_once = False
        # Gate-window bookkeeping (see _drain / _flush_gate_summary).
        self._gate_window_open = False
        self._gate_suppressed = 0
        self._gate_suppressed_ms = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        """Start the reader daemon thread, shutting down on *stop_event*."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._external_stop = stop_event
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tee-hearing", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the reader to finish and join it (best effort, never raises)."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- the reader loop ---------------------------------------------------

    def _run(self) -> None:
        backoff = self._backoff_min_s
        while not self._should_stop():
            sock = self._connect()
            if sock is None:
                self._wait(backoff)
                backoff = min(backoff * 2.0, self._backoff_max_s)
                continue
            backoff = self._backoff_min_s
            try:
                hold = self._session(sock)
            finally:
                _close(sock)
            self._flush_gate_summary()
            if hold > 0.0 and not self._should_stop():
                self._sense("backoff", f"retrying the tee in {hold:.2f}s")
                self._wait(hold)
        self._flush_gate_summary()

    def _connect(self) -> socket.socket | None:
        """One connect attempt. ``None`` (named, latched) when the tee is absent."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout_s)
        try:
            sock.connect(str(self.socket_path))
        except OSError as err:
            _close(sock)
            if not self._down_reported:
                self._down_reported = True
                self._sense("connect", f"{TEE_UNAVAILABLE} ({err}) at {self.socket_path}")
            return None
        sock.settimeout(self._read_timeout_s)
        self._down_reported = False
        if self._connected_once:
            self.reconnects += 1
            self._sense("connect", f"reconnected to the tee (reconnects={self.reconnects})")
        else:
            self._connected_once = True
            self._sense("connect", f"connected to the tee at {self.socket_path}")
        return sock

    def _session(self, sock: socket.socket) -> float:
        """Read one connection to its end. Returns the seconds to back off."""
        header = self._read_header(sock)
        if header is None:
            return self._backoff_min_s
        rate, leftover = header
        if rate is None:
            self._sense(
                "header",
                f"{TEE_RATE_UNKNOWN} (the tee announced samplerate=null; "
                "nothing is read against a guessed rate)",
            )
            return self._rate_unknown_backoff_s

        chunk_samples = max(1, int(round(rate * self.chunk_ms / 1000.0)))
        chunk_bytes = chunk_samples * BYTES_PER_SAMPLE
        self._sense(
            "header",
            f"tee header accepted (format={TEE_WIRE_FORMAT} rate={rate} Hz "
            f"chunk={self.chunk_ms}ms/{chunk_samples} samples -> {self.target_sr} Hz)",
        )

        pending = bytearray(leftover)
        self._drain(pending, rate, chunk_bytes)
        while not self._should_stop():
            received = self._recv(sock)
            if received is None:
                continue  # a read timeout is silence, not a disconnect
            if received == 0:
                self._sense("disconnect", f"{TEE_CLOSED} (the writer detached)")
                return self._backoff_min_s
            if received < 0:
                return self._backoff_min_s
            pending.extend(self._recv_view[:received])
            self._drain(pending, rate, chunk_bytes)
        return 0.0

    def _recv(self, sock: socket.socket) -> int | None:
        """Bytes read into the shared buffer; ``None`` on timeout, ``-1`` on error."""
        try:
            return sock.recv_into(self._recv_buf, self._recv_bytes)
        except (TimeoutError, socket.timeout):
            return None
        except OSError as err:
            self._sense("disconnect", f"{TEE_READ_FAILED} ({err})")
            return -1

    # -- the header --------------------------------------------------------

    def _read_header(self, sock: socket.socket) -> tuple[int | None, bytes] | None:
        """Read + validate the header line.

        Returns ``(rate_or_None, leftover_sample_bytes)``; ``None`` means the
        header was refused (already named) and the connection is finished.
        A ``rate`` of ``None`` is the announced-``null`` case — a legitimate
        header whose samples we still refuse to interpret.
        """
        buffered = bytearray()
        while not self._should_stop():
            index = buffered.find(TEE_HEADER_TERMINATOR)
            if index >= 0:
                line = bytes(buffered[:index])
                leftover = bytes(buffered[index + 1 :])
                return self._accept_header(line, leftover)
            if len(buffered) > MAX_HEADER_BYTES:
                self._sense(
                    "header",
                    f"{TEE_HEADER_INVALID} (no header line in the first "
                    f"{len(buffered)} bytes)",
                )
                return None
            received = self._recv(sock)
            if received is None:
                continue
            if received == 0:
                self._sense("disconnect", f"{TEE_CLOSED} (before the header line)")
                return None
            if received < 0:
                return None
            buffered.extend(self._recv_view[:received])
        return None

    def _accept_header(self, line: bytes, leftover: bytes) -> tuple[int | None, bytes] | None:
        """Fail CLOSED on every declared field — each one changes how bytes read."""
        try:
            header = json.loads(line.decode("utf-8"))
        except ValueError:  # UnicodeDecodeError is a ValueError: same verdict
            self._sense("header", f"{TEE_HEADER_INVALID} (the first line is not JSON)")
            return None
        if not isinstance(header, dict):
            self._sense(
                "header",
                f"{TEE_HEADER_INVALID} (the header is {type(header).__name__}, not an object)",
            )
            return None
        stream = header.get("stream")
        if stream != TEE_WIRE_NAME:
            self._sense(
                "header", f"{TEE_HEADER_FOREIGN} (stream={stream!r}, expected {TEE_WIRE_NAME!r})"
            )
            return None
        version = header.get("version")
        if version not in UNDERSTOOD_WIRE_VERSIONS:
            self._sense(
                "header",
                f"{TEE_HEADER_FOREIGN} (version={version!r}, understood "
                f"{list(UNDERSTOOD_WIRE_VERSIONS)})",
            )
            return None
        wire_format = header.get("format")
        if wire_format != TEE_WIRE_FORMAT:
            self._sense(
                "header",
                f"{TEE_HEADER_FOREIGN} (format={wire_format!r}, expected {TEE_WIRE_FORMAT!r})",
            )
            return None
        channels = header.get("channels", 1)
        if channels != 1:
            self._sense(
                "header", f"{TEE_HEADER_FOREIGN} (channels={channels!r}, this reader is mono)"
            )
            return None
        return (_positive_rate(header.get("samplerate")), leftover)

    # -- the samples -------------------------------------------------------

    def _drain(self, pending: bytearray, rate: int, chunk_bytes: int) -> None:
        """Emit every WHOLE chunk in *pending*; the part-chunk tail waits.

        The wire is already mono float32, so the only work per chunk is the
        little-endian decode and (when the mic is not already at ``target_sr``)
        the ``np.interp`` resample this project uses everywhere else.
        """
        while len(pending) >= chunk_bytes:
            raw = pending[:chunk_bytes]
            del pending[:chunk_bytes]
            if self.gate.active:
                self._suppress(chunk_bytes, rate)
                continue
            self._flush_gate_summary()
            samples = np.frombuffer(raw, dtype=TEE_SAMPLE_DTYPE).astype(np.float32)
            chunk = preprocess_mic_audio(samples, rate, self.target_sr)
            try:
                self.feed(chunk)
            except Exception as err:  # a sense must never kill its own thread
                if not self._feed_fault_reported:
                    self._feed_fault_reported = True
                    self._sense("feed", f"{FEED_FAILED} ({err})")
                continue
            self._feed_fault_reported = False
            self.chunks_fed += 1

    def _suppress(self, chunk_bytes: int, rate: int) -> None:
        """Drop one chunk into the echo gate — ONE line per window, not per chunk."""
        self.chunks_gated += 1
        self._gate_suppressed += 1
        self._gate_suppressed_ms += (chunk_bytes / BYTES_PER_SAMPLE) / rate * 1000.0
        if not self._gate_window_open:
            self._gate_window_open = True
            self._sense(
                "gate",
                "echo gate armed — suppressing mic chunks while the robot speaks "
                f"(half-duplex, ~{self.gate.remaining():.2f}s left)",
            )

    def _flush_gate_summary(self) -> None:
        """Close the gate window with the count it cost, once."""
        if self._gate_suppressed <= 0:
            self._gate_window_open = False
            return
        self._sense(
            "gate",
            f"echo gate cleared — suppressed {self._gate_suppressed} chunks "
            f"({self._gate_suppressed_ms:.0f} ms of mic audio)",
        )
        self._gate_suppressed = 0
        self._gate_suppressed_ms = 0.0
        self._gate_window_open = False

    # -- plumbing ----------------------------------------------------------

    def _sense(self, event: str, detail: str) -> None:
        sensory_log.stage(STAGE, SOURCE, event, detail)

    def _should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        return self._external_stop is not None and self._external_stop.is_set()

    def _wait(self, seconds: float) -> None:
        """Sleep, interruptibly: either stop signal cuts it short."""
        deadline = time.monotonic() + seconds
        while not self._should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            self._stop.wait(min(remaining, _WAIT_SLICE_S))


def _positive_rate(announced: object) -> int | None:
    """The header's ``samplerate`` as a usable int, or ``None`` — never a guess."""
    try:
        rate = int(announced) if announced is not None else None
    except (TypeError, ValueError):
        return None
    if rate is None or rate <= 0:
        return None
    return rate


def _close(sock: socket.socket) -> None:
    try:
        sock.close()
    except OSError:
        pass
