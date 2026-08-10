"""Chaos: the body dies under the mind — a reachy-runtime restart mid-stream.

On the robot this is ``systemctl --user restart reachy-runtime`` while the
harness keeps running: the audio-tee socket vanishes mid-stream, the intent
spool loses its consumer, and a little later the runtime comes back on the
SAME paths. The harness is a peripheral — it must go quiet with a NAMED drop,
then re-attach by itself, without being restarted or rebuilt.

Two properties are pinned here, each in one test:

1. **The tee reader re-attaches as the same instance.** The fake tee server
   is killed abruptly (connections closed, listener closed, path unlinked)
   while audio is flowing; the reader names the disconnect, stops feeding,
   and — when a new listener appears on the same path — reconnects and feeds
   again, all on the ONE ``TeeHearing`` object built at startup. With the
   test-tuned backoff (0.05–0.2 s vs the production 1–5 s) the re-attach
   lands well inside the scaled equivalent of the 10 s bound.
2. **Commands spooled while the engine is down survive for the late engine.**
   A tool call with no engine returns the degraded submitted-only payload
   (named ``[SENSE stage=act ...] degraded ... reason=...`` line) AND leaves
   the command file on disk, so the restarted engine still applies it — and
   its late answer is still consumable through the same ``IntentTools``.
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

from reachy_nova.harness import hearing, statedir
from reachy_nova.harness.gate import EchoGate
from reachy_nova.harness.tools import IntentTools

TEE_HEADER = {
    "stream": "reachy-audio-tee",
    "version": 1,
    "format": "f32le",
    "channels": 1,
    "samplerate": 16000,
}

#: 100 ms of 16 kHz audio — exactly one chunk for the reader below.
ONE_CHUNK = np.full(1600, 0.25, dtype=np.float32)


class ChaosTee:
    """A fake runtime audio tee that can be killed abruptly and reborn.

    Speaks the reachy-mini-cli 0.48.0 wire (JSON header line, then contiguous
    little-endian float32 mono samples). :meth:`push` queues samples for
    whichever connection is (or becomes) live; :meth:`kill` is the chaotic
    part — connections closed mid-stream, listener closed, path unlinked —
    which is what a ``systemctl --user restart reachy-runtime`` does to the
    socket from the harness's point of view.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connections = 0
        self._outbox: queue.Queue[bytes] = queue.Queue()
        self._stop = threading.Event()
        self._conns: list[socket.socket] = []
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(0.05)
        self._sock.bind(str(path))
        self._sock.listen(4)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def push(self, samples: np.ndarray) -> None:
        self._outbox.put(samples.astype("<f4").tobytes())

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            self.connections += 1
            self._conns.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.sendall(json.dumps(TEE_HEADER).encode() + b"\n")
            while not self._stop.is_set():
                try:
                    blob = self._outbox.get(timeout=0.02)
                except queue.Empty:
                    continue
                conn.sendall(blob)
        except OSError:
            pass

    def kill(self) -> None:
        """The runtime dies: sockets torn down mid-stream, path removed."""
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
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._thread.join(timeout=2.0)


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def hear_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "nova.sensory" and "stage=hear" in r.getMessage()
    ]


class Recorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0

    def __call__(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._count += 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


def test_runtime_restart_reattaches_the_same_hearing_instance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    tee_path = tmp_path / "audio_tee.sock"
    server = ChaosTee(tee_path)
    rec = Recorder()
    stop = threading.Event()
    hear = hearing.TeeHearing(
        feed=rec,
        gate=EchoGate(),
        chunk_ms=100,
        socket_path=tee_path,
        target_sr=16000,
        backoff_min_s=0.05,
        backoff_max_s=0.2,
    )
    hear.start(stop)
    reborn = None
    try:
        # Phase 1 — alive: audio flows.
        server.push(ONE_CHUNK)
        server.push(ONE_CHUNK)
        assert wait_for(lambda: rec.count >= 2), "audio never flowed before the kill"
        fed_before_kill = hear.chunks_fed

        # Phase 2 — the runtime dies mid-stream.
        server.kill()
        assert wait_for(
            lambda: any(
                "event=disconnect" in m
                and (hearing.TEE_CLOSED in m or hearing.TEE_READ_FAILED in m)
                for m in hear_lines(caplog)
            )
        ), "the mid-stream kill was not a NAMED disconnect drop"
        # The now-absent socket is also named (latched, once).
        assert wait_for(
            lambda: any(hearing.TEE_UNAVAILABLE in m for m in hear_lines(caplog))
        ), "the missing socket was never named"

        # Audio feeding has STOPPED — the count does not creep while dead.
        count_while_dead = rec.count
        time.sleep(0.3)
        assert rec.count == count_while_dead, "chunks were fed while the runtime was dead"
        assert hear.is_alive(), "the reader thread died with the runtime"

        # Phase 3 — the runtime restarts on the SAME path.
        rebirth_t0 = time.monotonic()
        reborn = ChaosTee(tee_path)
        reborn.push(ONE_CHUNK)
        reborn.push(ONE_CHUNK)
        assert wait_for(lambda: rec.count >= count_while_dead + 2), (
            "the harness never re-attached to the reborn runtime"
        )
        reattach_s = time.monotonic() - rebirth_t0
    finally:
        stop.set()
        hear.stop()
        server.kill()
        if reborn is not None:
            reborn.kill()

    # Bounded re-attach: with 0.05–0.2 s backoff (production 1–5 s) this is
    # the scaled equivalent of well under 10 s on the robot.
    assert reattach_s < 10.0, f"re-attach took {reattach_s:.2f}s"
    # Same instance, counted as a reconnect — never a rebuilt TeeHearing.
    assert hear.reconnects >= 1
    assert hear.chunks_fed >= fed_before_kill + 2
    assert any(
        "event=connect" in m and "reconnected" in m for m in hear_lines(caplog)
    ), "the re-attach was not a named reconnect"


def test_commands_spooled_while_the_engine_is_down_survive_for_the_late_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    caplog.set_level(logging.INFO, logger="nova.sensory")
    tools = IntentTools(await_timeout=0.15)

    # The engine is down (no results ever appear): the call degrades, named.
    payload = json.loads(tools.execute("run_behavior", {"name": "nod"}))
    assert payload["ok"] is None
    cmd_id = payload["submitted"]
    assert payload["note"]
    degraded_lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "nova.sensory"
        and "stage=act" in r.getMessage()
        and "degraded" in r.getMessage()
        and "reason=" in r.getMessage()
    ]
    assert degraded_lines, "the engine-down degradation was not a named [SENSE] line"
    assert cmd_id in degraded_lines[0]

    # The command file SURVIVES on disk for the later engine.
    spooled = list(statedir.intents_commands_dir().glob(f"*-{cmd_id}.json"))
    assert len(spooled) == 1, "the degraded command did not stay spooled"
    on_disk = json.loads(spooled[0].read_text())
    assert on_disk["cmd_id"] == cmd_id
    assert on_disk["op"] == "run_behavior"
    assert on_disk["name"] == "nod"

    # Recovery: the engine restarts later, drains the spool, and answers —
    # and that late answer is consumable through the SAME IntentTools.
    spooled[0].unlink()  # the reborn engine consumes the command
    (statedir.intents_results_dir() / f"{cmd_id}.json").write_text(
        json.dumps({"ok": True, "applied": "nod", "cmd_id": cmd_id})
    )
    late = tools.await_result(cmd_id, timeout=1.0)
    assert late is not None and late["ok"] is True and late["applied"] == "nod"
    # And a fresh call against the now-live engine confirms end to end.
    engine = threading.Thread(target=_answer_next_command, daemon=True)
    engine.start()
    confirmed = json.loads(tools.execute("run_behavior", {"name": "shake"}))
    engine.join(timeout=2.0)
    assert confirmed["ok"] is True, "the recovered engine path did not confirm"


def _answer_next_command() -> None:
    """A minimal live engine: wait for one spooled command, confirm it."""
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        pending = sorted(statedir.intents_commands_dir().glob("*.json"))
        if pending:
            cmd = json.loads(pending[0].read_text())
            pending[0].unlink()
            (statedir.intents_results_dir() / f"{cmd['cmd_id']}.json").write_text(
                json.dumps({"ok": True, "applied": cmd.get("name")})
            )
            return
        time.sleep(0.01)
