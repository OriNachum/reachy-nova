"""Audio-tee client + the echo-gate POLICY (task t7; policy split in t2).

Everything here runs against a **fake tee server**: a real ``AF_UNIX``
``SOCK_STREAM`` listener in ``tmp_path`` that speaks reachy-mini-cli 0.48.0's
audio-tee wire (one JSON header line, then contiguous little-endian float32
mono samples). No robot, no ``reachy_mini`` import, no daemon.

The wire under test (reachy-mini-cli ``reachy/behavior/audio_tee.py``, read by
``reachy/embody/media.py::_RobotTeeSourceBackend``)::

    {"stream":"reachy-audio-tee","version":1,"format":"f32le",
     "channels":1,"samplerate":16000}\\n
    <little-endian float32 samples, contiguous, forever>

Four properties are load-bearing and each has a test below:

1. the announced rate governs resampling (the header wins; nothing is guessed);
2. a ``recv`` may land MID-SAMPLE — the remainder is buffered, never spliced,
   because an off-by-one there shifts the whole stream silently;
3. ``samplerate: null`` (a cold media holder) is a NAMED drop plus a
   reconnect, never a guessed rate;
4. the echo gate is read under a POLICY (``NOVA_ECHO_GATE``): under the
   default ``off`` the mic keeps feeding Sonic while the robot speaks (that is
   what barge-in needs, and the XVF3800's hardware AEC was verified active on
   this capture path live on 2026-08-10), while under ``half-duplex`` the
   chunks are dropped, counted, and summarised in ONE line when the gate
   clears — not one line per chunk.
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from reachy_nova.audio_pipeline import preprocess_mic_audio
from reachy_nova.harness import gate as gate_mod
from reachy_nova.harness import hearing
from reachy_nova.harness.gate import EchoGate

TEE_HEADER = {
    "stream": "reachy-audio-tee",
    "version": 1,
    "format": "f32le",
    "channels": 1,
    "samplerate": 16000,
}


# --------------------------------------------------------------------------- #
# The fake tee server                                                          #
# --------------------------------------------------------------------------- #


class FakeTee:
    """A unix-socket listener that speaks the audio-tee wire.

    ``header`` is the dict sent as the first line (``None`` sends a raw
    non-JSON line instead). ``payload`` is streamed right after the header, in
    ``send_bytes``-sized writes so a test can force splits at non-4-byte
    boundaries. :meth:`push` streams more onto the LIVE connection later (that
    is how the gate test resumes audio after clearing the gate).
    ``close_after`` closes the connection once the initial payload is out,
    which drives the disconnect/reconnect path.
    """

    def __init__(
        self,
        path: Path,
        header: dict | None = TEE_HEADER,
        payload: np.ndarray | bytes | None = None,
        send_bytes: int = 4096,
        close_after: bool = False,
    ) -> None:
        self.path = path
        self.header = header
        self.payload = payload
        self.send_bytes = send_bytes
        self.close_after = close_after
        self.connections = 0
        self._outbox: queue.Queue[bytes] = queue.Queue()
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(0.1)
        self._sock.bind(str(path))
        self._sock.listen(4)
        self._conns: list[socket.socket] = []
        self._handlers: list[threading.Thread] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    # -- test-side controls -------------------------------------------------

    def push(self, payload: np.ndarray | bytes) -> None:
        """Queue more samples for the currently-open connection."""
        self._outbox.put(_as_wire_bytes(payload))

    # -- server internals ---------------------------------------------------

    def _serve(self) -> None:
        """Accept forever, one handler thread per connection.

        Per-connection threads matter: a reader that refuses a header and
        reconnects must find the listener still accepting, not blocked inside
        the previous connection's send loop.
        """
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            self._conns.append(conn)
            self.connections += 1
            handler = threading.Thread(target=self._handle, args=(conn,), daemon=True)
            self._handlers.append(handler)
            handler.start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            self._speak(conn)
        except OSError:
            pass
        if self.close_after:
            conn.close()

    def _speak(self, conn: socket.socket) -> None:
        if self.header is not None:
            conn.sendall(json.dumps(self.header).encode() + b"\n")
        else:
            conn.sendall(b"not-json-at-all\n")
        if self.payload is not None:
            self._send_chunked(conn, _as_wire_bytes(self.payload))
        if self.close_after:
            return
        # Hold the connection open: a live tee with nothing to say is silence,
        # not a disconnect. Drain anything a test pushes in the meantime.
        while not self._stop.is_set():
            try:
                blob = self._outbox.get(timeout=0.05)
            except queue.Empty:
                continue
            self._send_chunked(conn, blob)

    def _send_chunked(self, conn: socket.socket, raw: bytes) -> None:
        for start in range(0, len(raw), self.send_bytes):
            conn.sendall(raw[start : start + self.send_bytes])

    def close(self) -> None:
        self._stop.set()
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)
        for handler in self._handlers:
            handler.join(timeout=2.0)


def _as_wire_bytes(payload: np.ndarray | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return payload.astype("<f4").tobytes()


@pytest.fixture
def tee_path(tmp_path: Path) -> Path:
    return tmp_path / "audio_tee.sock"


class Recorder:
    """Collects the chunks handed to ``feed`` (this is ``sonic.feed_audio``)."""

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    def __call__(self, chunk: np.ndarray) -> None:
        with self._lock:
            self.chunks.append(chunk)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.chunks)

    def concat(self) -> np.ndarray:
        with self._lock:
            if not self.chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self.chunks)


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def sense_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "nova.sensory"]


def hear_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [m for m in sense_lines(caplog) if "stage=hear" in m and "source=nova" in m]


def start_hearing(**kwargs) -> tuple[hearing.TeeHearing, threading.Event]:
    stop = threading.Event()
    hear = hearing.TeeHearing(**kwargs)
    hear.start(stop)
    return hear, stop


# --------------------------------------------------------------------------- #
# 0. Surface                                                                   #
# --------------------------------------------------------------------------- #


def test_socket_path_defaults_to_the_statedir_resolver(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("REACHY_AUDIO_TEE_SOCKET", raising=False)
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    hear = hearing.TeeHearing(feed=lambda chunk: None, gate=EchoGate())
    assert hear.socket_path == tmp_path / "audio_tee.sock"


def test_socket_path_honours_the_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REACHY_AUDIO_TEE_SOCKET", str(tmp_path / "elsewhere.sock"))
    hear = hearing.TeeHearing(feed=lambda chunk: None, gate=EchoGate())
    assert hear.socket_path == tmp_path / "elsewhere.sock"


def test_counters_start_at_zero() -> None:
    hear = hearing.TeeHearing(feed=lambda chunk: None, gate=EchoGate())
    assert (hear.chunks_fed, hear.chunks_gated, hear.reconnects) == (0, 0, 0)


# --------------------------------------------------------------------------- #
# 1. The header's rate governs the resample                                    #
# --------------------------------------------------------------------------- #


def test_a_32khz_sine_is_resampled_to_the_16khz_target(tee_path: Path) -> None:
    src_rate = 32000
    t = np.arange(src_rate, dtype=np.float32) / src_rate
    sine = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    server = FakeTee(tee_path, header={**TEE_HEADER, "samplerate": src_rate}, payload=sine)
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=EchoGate(),
        chunk_ms=100,
        socket_path=tee_path,
        target_sr=16000,
    )
    try:
        assert wait_for(lambda: rec.count >= 10), f"only {rec.count} chunks arrived"
    finally:
        stop.set()
        hear.stop()
        server.close()

    first = rec.chunks[0]
    assert first.dtype == np.float32
    # 100 ms at 32 kHz is 3200 source samples -> 1600 at the 16 kHz target.
    assert len(first) == 1600
    assert hear.chunks_fed >= 10
    assert hear.chunks_gated == 0

    # Exactly the project's own np.interp path, over exactly the right slice.
    assert np.array_equal(first, preprocess_mic_audio(sine[:3200], src_rate, 16000))
    # Content sanity independent of that helper: it is still the sine, halved.
    assert np.allclose(first, sine[:3200][::2], atol=0.05)
    assert 0.4 < float(np.max(np.abs(first))) <= 0.51


def test_a_matching_rate_is_passed_through_unresampled(tee_path: Path) -> None:
    ramp = (np.arange(1600, dtype=np.float32) / 1600.0).astype(np.float32)
    server = FakeTee(tee_path, payload=ramp)
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec, gate=EchoGate(), chunk_ms=100, socket_path=tee_path, target_sr=16000
    )
    try:
        assert wait_for(lambda: rec.count >= 1)
    finally:
        stop.set()
        hear.stop()
        server.close()
    assert np.array_equal(rec.chunks[0], ramp)


def test_the_announced_rate_is_logged_on_connect(tee_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    server = FakeTee(tee_path, header={**TEE_HEADER, "samplerate": 48000})
    hear, stop = start_hearing(feed=lambda chunk: None, gate=EchoGate(), socket_path=tee_path)
    try:
        assert wait_for(lambda: any("event=header" in m for m in hear_lines(caplog)))
    finally:
        stop.set()
        hear.stop()
        server.close()
    header_line = next(m for m in hear_lines(caplog) if "event=header" in m)
    assert "48000" in header_line
    assert any("event=connect" in m for m in hear_lines(caplog))


# --------------------------------------------------------------------------- #
# 2. Mid-sample framing                                                        #
# --------------------------------------------------------------------------- #


def test_a_recv_split_off_the_sample_boundary_does_not_corrupt_the_stream(
    tee_path: Path,
) -> None:
    """7-byte writes are coprime with the 4-byte sample: every recv lands mid-sample."""
    samples = (np.arange(480, dtype=np.float32) / 1000.0).astype(np.float32)
    server = FakeTee(tee_path, payload=samples, send_bytes=7)
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=EchoGate(),
        chunk_ms=10,  # 160 samples per chunk at 16 kHz -> exactly 3 chunks
        socket_path=tee_path,
        target_sr=16000,
    )
    try:
        assert wait_for(lambda: rec.count >= 3), f"only {rec.count} chunks arrived"
    finally:
        stop.set()
        hear.stop()
        server.close()
    got = rec.concat()[: len(samples)]
    assert np.array_equal(got, samples), "samples were spliced across a recv boundary"


def test_a_header_split_across_recvs_is_still_accepted(tee_path: Path) -> None:
    samples = (np.arange(320, dtype=np.float32) / 1000.0).astype(np.float32)
    # 3-byte writes split the header line itself, not just the samples.
    server = FakeTee(tee_path, payload=samples, send_bytes=3)
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec, gate=EchoGate(), chunk_ms=10, socket_path=tee_path, target_sr=16000
    )
    try:
        assert wait_for(lambda: rec.count >= 2)
    finally:
        stop.set()
        hear.stop()
        server.close()
    assert np.array_equal(rec.concat()[: len(samples)], samples)


def test_a_partial_trailing_sample_is_never_emitted(tee_path: Path) -> None:
    """A stream cut two bytes short yields whole chunks only — the tail waits."""
    samples = ((np.arange(320, dtype=np.float32) + 1.0) / 1000.0).astype(np.float32)
    raw = samples.astype("<f4").tobytes()[:-2]  # two bytes short of the last sample
    assert len(raw) % 4 == 2

    server = FakeTee(tee_path, payload=raw, send_bytes=13)
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec, gate=EchoGate(), chunk_ms=10, socket_path=tee_path, target_sr=16000
    )
    try:
        assert wait_for(lambda: rec.count >= 1)
        time.sleep(0.2)
        # 319 whole samples arrived: chunk 1 (160) is emitted, chunk 2 is one
        # sample short and must NOT be emitted early or padded.
        assert rec.count == 1
    finally:
        stop.set()
        hear.stop()
        server.close()
    assert np.array_equal(rec.chunks[0], samples[:160])


# --------------------------------------------------------------------------- #
# 3. samplerate: null -> named drop, no guessing, reconnect                    #
# --------------------------------------------------------------------------- #


def test_a_null_samplerate_header_feeds_nothing_and_reconnects(tee_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    samples = np.full(1600, 0.25, dtype=np.float32)
    server = FakeTee(tee_path, header={**TEE_HEADER, "samplerate": None}, payload=samples)
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=EchoGate(),
        chunk_ms=100,
        socket_path=tee_path,
        target_sr=16000,
        rate_unknown_backoff_s=0.05,
    )
    try:
        assert wait_for(lambda: server.connections >= 2), "no reconnect after the null rate"
        time.sleep(0.1)
    finally:
        stop.set()
        hear.stop()
        server.close()

    assert rec.count == 0, "samples were consumed against a guessed rate"
    assert hear.chunks_fed == 0
    assert any(hearing.TEE_RATE_UNKNOWN in m for m in hear_lines(caplog))
    assert hear.reconnects >= 1


def test_a_foreign_header_is_refused_without_reading_samples(tee_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    server = FakeTee(
        tee_path,
        header={**TEE_HEADER, "version": 99},
        payload=np.full(1600, 0.25, dtype=np.float32),
    )
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=EchoGate(),
        socket_path=tee_path,
        backoff_min_s=0.05,
        backoff_max_s=0.05,
    )
    try:
        assert wait_for(lambda: any(hearing.TEE_HEADER_FOREIGN in m for m in hear_lines(caplog)))
        time.sleep(0.1)
    finally:
        stop.set()
        hear.stop()
        server.close()
    assert rec.count == 0


def test_a_non_json_first_line_is_refused(tee_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    server = FakeTee(tee_path, header=None, payload=np.full(160, 0.25, dtype=np.float32))
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=EchoGate(),
        socket_path=tee_path,
        backoff_min_s=0.05,
        backoff_max_s=0.05,
    )
    try:
        assert wait_for(lambda: any(hearing.TEE_HEADER_INVALID in m for m in hear_lines(caplog)))
    finally:
        stop.set()
        hear.stop()
        server.close()
    assert rec.count == 0


# --------------------------------------------------------------------------- #
# 4. The echo gate, under its POLICY (NOVA_ECHO_GATE — see gate.py)            #
# --------------------------------------------------------------------------- #


def test_the_default_hearing_policy_is_off(monkeypatch) -> None:
    monkeypatch.delenv(gate_mod.ECHO_GATE_ENV, raising=False)
    hear = hearing.TeeHearing(feed=lambda chunk: None, gate=EchoGate())
    assert hear.echo_gate_policy == "off"
    assert hear.suppresses_while_speaking is False


def test_the_env_selects_the_half_duplex_policy(monkeypatch) -> None:
    monkeypatch.setenv(gate_mod.ECHO_GATE_ENV, "half-duplex")
    hear = hearing.TeeHearing(feed=lambda chunk: None, gate=EchoGate())
    assert hear.echo_gate_policy == "half-duplex"
    assert hear.suppresses_while_speaking is True


def test_an_unrecognised_env_policy_falls_back_to_off(monkeypatch) -> None:
    monkeypatch.setenv(gate_mod.ECHO_GATE_ENV, "sort-of-duplex")
    hear = hearing.TeeHearing(feed=lambda chunk: None, gate=EchoGate())
    assert hear.echo_gate_policy == "off"


def test_the_constructor_argument_beats_the_env(monkeypatch) -> None:
    monkeypatch.setenv(gate_mod.ECHO_GATE_ENV, "half-duplex")
    hear = hearing.TeeHearing(
        feed=lambda chunk: None, gate=EchoGate(), echo_gate_policy="off"
    )
    assert hear.echo_gate_policy == "off"


def test_the_off_policy_keeps_feeding_while_the_gate_is_armed(
    tee_path: Path, caplog
) -> None:
    """The default: the robot hears itself being spoken over — that IS barge-in.

    The XVF3800's hardware AEC was verified active on this capture path live on
    2026-08-10, so suppressing the mic for the whole playback window only made
    the robot deaf while speaking.
    """
    caplog.set_level(logging.INFO, logger="nova.sensory")
    gate = EchoGate(margin_s=0.0)
    gate.arm_for(60.0)  # armed across the WHOLE burst

    server = FakeTee(tee_path, payload=np.full(1600 * 5, 0.4, dtype=np.float32))
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=gate,
        chunk_ms=100,
        socket_path=tee_path,
        target_sr=16000,
        echo_gate_policy="off",
    )
    try:
        assert wait_for(lambda: rec.count >= 5), f"only {rec.count} chunks reached Sonic"
    finally:
        stop.set()
        hear.stop()
        server.close()

    assert gate.active, "the speaking window must still be open (speaking leg's job)"
    assert hear.chunks_gated == 0
    assert hear.chunks_fed >= 5
    assert np.allclose(rec.chunks[0], 0.4)
    # Nothing is being gated, so the hearing leg says nothing about the gate.
    assert [m for m in hear_lines(caplog) if "event=gate" in m] == []


def test_the_off_policy_is_what_an_unset_env_gives_the_reader(
    tee_path: Path, monkeypatch
) -> None:
    """Same proof, but through the env default rather than an explicit argument."""
    monkeypatch.delenv(gate_mod.ECHO_GATE_ENV, raising=False)
    gate = EchoGate(margin_s=0.0)
    gate.arm_for(60.0)
    server = FakeTee(tee_path, payload=np.full(1600 * 3, 0.4, dtype=np.float32))
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec, gate=gate, chunk_ms=100, socket_path=tee_path, target_sr=16000
    )
    try:
        assert wait_for(lambda: rec.count >= 3), f"only {rec.count} chunks reached Sonic"
    finally:
        stop.set()
        hear.stop()
        server.close()
    assert hear.chunks_gated == 0


def test_half_duplex_from_the_env_suppresses_the_feed(tee_path: Path, monkeypatch) -> None:
    """The opt-in policy is reachable by env alone — one flip, no code change."""
    monkeypatch.setenv(gate_mod.ECHO_GATE_ENV, "half-duplex")
    gate = EchoGate(margin_s=0.0)
    gate.arm_for(60.0)
    server = FakeTee(tee_path, payload=np.full(1600 * 3, 0.4, dtype=np.float32))
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec, gate=gate, chunk_ms=100, socket_path=tee_path, target_sr=16000
    )
    try:
        assert wait_for(lambda: hear.chunks_gated >= 3), f"gated {hear.chunks_gated}"
        assert rec.count == 0, "half-duplex did not suppress the feed"
    finally:
        stop.set()
        hear.stop()
        server.close()


def test_chunks_are_dropped_while_the_gate_is_armed_and_summarised_once(
    tee_path: Path, caplog
) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    gate = EchoGate(margin_s=0.0)
    gate.arm_for(60.0)  # armed across the whole first burst

    server = FakeTee(tee_path, payload=np.full(1600 * 5, 0.4, dtype=np.float32))
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=gate,
        chunk_ms=100,
        socket_path=tee_path,
        target_sr=16000,
        echo_gate_policy="half-duplex",
    )
    try:
        assert wait_for(lambda: hear.chunks_gated >= 5), f"gated {hear.chunks_gated}"
        assert rec.count == 0, "the gate did not suppress the feed"
        gated_before = hear.chunks_gated

        # ONE throttled line per gate window, not one per chunk.
        gate_lines = [m for m in hear_lines(caplog) if "event=gate" in m]
        assert len(gate_lines) == 1, gate_lines

        # Clear the gate, push more audio: the summary lands, the feed resumes.
        gate.clear()
        server.push(np.full(1600, -0.4, dtype=np.float32))
        assert wait_for(lambda: rec.count >= 1), "the feed never resumed"
        assert wait_for(lambda: any("suppressed" in m for m in hear_lines(caplog)))
    finally:
        stop.set()
        hear.stop()
        server.close()

    summary = next(m for m in hear_lines(caplog) if "suppressed" in m)
    assert str(gated_before) in summary
    assert hear.chunks_gated == gated_before
    assert np.allclose(rec.chunks[0], -0.4)


def test_an_unarmed_gate_feeds_every_chunk(tee_path: Path) -> None:
    server = FakeTee(tee_path, payload=np.full(1600 * 3, 0.2, dtype=np.float32))
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec, gate=EchoGate(), chunk_ms=100, socket_path=tee_path, target_sr=16000
    )
    try:
        assert wait_for(lambda: rec.count >= 3)
    finally:
        stop.set()
        hear.stop()
        server.close()
    assert hear.chunks_gated == 0
    assert hear.chunks_fed >= 3


# --------------------------------------------------------------------------- #
# 5. Disconnect / reconnect                                                    #
# --------------------------------------------------------------------------- #


def test_a_closed_server_is_a_named_drop_then_a_reconnect(tee_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    server = FakeTee(
        tee_path, payload=np.full(1600, 0.3, dtype=np.float32), close_after=True
    )
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=EchoGate(),
        chunk_ms=100,
        socket_path=tee_path,
        target_sr=16000,
        backoff_min_s=0.05,
        backoff_max_s=0.1,
    )
    try:
        assert wait_for(lambda: rec.count >= 2), f"only {rec.count} chunks across reconnects"
        assert wait_for(lambda: hear.reconnects >= 1)
    finally:
        stop.set()
        hear.stop()
        server.close()
    assert any(hearing.TEE_CLOSED in m for m in hear_lines(caplog))
    assert server.connections >= 2


def test_an_absent_socket_is_the_ordinary_resting_state(tee_path: Path, caplog) -> None:
    """No runtime yet: back off, say so once, never raise."""
    caplog.set_level(logging.INFO, logger="nova.sensory")
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=EchoGate(),
        socket_path=tee_path,
        backoff_min_s=0.02,
        backoff_max_s=0.05,
    )
    try:
        assert wait_for(lambda: any(hearing.TEE_UNAVAILABLE in m for m in hear_lines(caplog)))
        time.sleep(0.2)
        assert hear.is_alive(), "the reader thread died on a missing socket"
    finally:
        stop.set()
        hear.stop()

    # Latched: an absent socket costs ONE line, not one per backoff period.
    unavailable = [m for m in hear_lines(caplog) if hearing.TEE_UNAVAILABLE in m]
    assert len(unavailable) == 1, unavailable
    assert rec.count == 0


def test_a_socket_appearing_later_is_picked_up(tee_path: Path) -> None:
    rec = Recorder()
    hear, stop = start_hearing(
        feed=rec,
        gate=EchoGate(),
        chunk_ms=100,
        socket_path=tee_path,
        backoff_min_s=0.02,
        backoff_max_s=0.05,
    )
    server = None
    try:
        time.sleep(0.1)  # the reader is already looping against nothing
        server = FakeTee(tee_path, payload=np.full(1600, 0.1, dtype=np.float32))
        assert wait_for(lambda: rec.count >= 1), "the late tee was never picked up"
    finally:
        stop.set()
        hear.stop()
        if server is not None:
            server.close()


def test_stop_event_ends_the_thread(tee_path: Path) -> None:
    server = FakeTee(tee_path, payload=np.full(1600, 0.1, dtype=np.float32))
    rec = Recorder()
    hear, stop = start_hearing(feed=rec, gate=EchoGate(), socket_path=tee_path)
    try:
        assert wait_for(lambda: rec.count >= 1)
        stop.set()
        assert wait_for(lambda: not hear.is_alive(), timeout=3.0)
    finally:
        stop.set()
        hear.stop()
        server.close()


def test_a_raising_feed_callback_never_kills_the_reader(tee_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    calls: list[int] = []

    def boom(chunk: np.ndarray) -> None:
        calls.append(len(chunk))
        raise RuntimeError("sonic is unhappy")

    server = FakeTee(tee_path, payload=np.full(1600 * 3, 0.1, dtype=np.float32))
    hear, stop = start_hearing(feed=boom, gate=EchoGate(), chunk_ms=100, socket_path=tee_path)
    try:
        assert wait_for(lambda: len(calls) >= 3)
        assert hear.is_alive()
    finally:
        stop.set()
        hear.stop()
        server.close()
    assert any("feed-failed" in m for m in hear_lines(caplog))
