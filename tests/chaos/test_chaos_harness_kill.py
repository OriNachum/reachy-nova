"""Chaos: the harness dies — the robot is unaffected, the rebirth is clean.

On the robot this is ``kill -9`` on the harness PID (or an OOM kill): the
symbolic runtime keeps the body alive because the harness is only a
peripheral, and the next harness start must reclaim the corpse's PID file and
come up cleanly. The unit tests pin each PID-file rule in isolation; the
chaos sequence here runs the WHOLE death-and-rebirth arc in one test:

* a genuinely dead PID (a real, reaped child process — no monkeypatching of
  liveness) left behind by the "killed" harness is reclaimed, named;
* the reborn harness runs real (fake) components, survives an engine
  heartbeat loss AND recovery mid-run (``dropped
  reason=engine-heartbeat-lost`` then ``engine live`` again — the named
  drop + recovery pair), and a SIGTERM-equivalent stop lands as an orderly
  shutdown: every component's ``stop()`` runs and ``run()`` returns;
* a component whose ``stop()`` RAISES does not prevent its siblings from
  stopping (named ``stop failed`` line) — one bad actor never turns a clean
  shutdown into a hung process for systemd to escalate on;
* the released PID file removes only our own — a sibling's claim is never
  clobbered by a dying harness.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time

import pytest

from reachy_nova.harness import statedir, supervisor


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    (tmp_path / "behavior").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_heartbeat(age_s: float) -> None:
    statedir.state_json_path().write_text(json.dumps({"updated": time.time() - age_s}))


def _really_dead_pid() -> int:
    """The PID of a child that ran and was reaped — dead for real, not mocked."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])  # nosec B603
    proc.wait()
    return proc.pid


class RecordingComponent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.started_with = None
        self.stopped = False

    def start(self, stop_event: threading.Event) -> None:
        self.started_with = stop_event

    def stop(self) -> None:
        self.stopped = True


class RaisingStopComponent(RecordingComponent):
    def stop(self) -> None:
        raise RuntimeError("stop exploded (simulated)")


def sense_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(
        r.getMessage() for r in caplog.records if r.name == "nova.sensory"
    )


def test_harness_death_and_rebirth_full_arc(state_dir, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")

    # --- The previous harness was kill -9'd: its PID file names a real corpse.
    corpse = _really_dead_pid()
    statedir.harness_pid_path().write_text(str(corpse))

    # --- Rebirth: the stale claim is reclaimed, named, and rewritten to us.
    assert supervisor.acquire_pid_file() is True
    assert statedir.harness_pid_path().read_text().strip() == str(os.getpid())
    assert f"reclaimed stale pid={corpse}" in sense_text(caplog)

    # --- The reborn harness runs: components up, engine dies and returns
    #     under it, then a SIGTERM-equivalent stop (stop_event set — exactly
    #     what install_signal_handlers wires SIGTERM to).
    first, bad, last = (
        RecordingComponent("hearing"),
        RaisingStopComponent("bus"),
        RecordingComponent("speaking"),
    )
    stop = threading.Event()
    _write_heartbeat(age_s=0.0)  # the engine is live at start

    def tick_hook(count: int) -> None:
        if count == 1:
            _write_heartbeat(age_s=60.0)  # the engine dies under us
        elif count == 2:
            _write_heartbeat(age_s=0.0)  # ...and restarts
        elif count >= 3:
            stop.set()  # SIGTERM arrives

    supervisor.run([first, bad, last], stop, poll_interval=0.0, tick_hook=tick_hook)
    # run() RETURNED — the SIGTERM-equivalent was an orderly stop, not a hang.

    text = sense_text(caplog)
    # Named drop + recovery for the engine dying UNDER the running harness.
    assert "dropped reason=engine-heartbeat-lost" in text
    assert text.count("engine live") == 2, "the engine's recovery was not re-named"
    # Every component stopped, even though one stop() raised (named).
    assert first.stopped is True
    assert last.stopped is True
    assert "stop failed name=bus" in text
    assert "harness down" in text

    # --- Clean release: our own file goes, a successor's claim never would.
    supervisor.release_pid_file()
    assert not statedir.harness_pid_path().exists()
    statedir.harness_pid_path().write_text("424242")  # a successor's claim
    supervisor.release_pid_file()
    assert statedir.harness_pid_path().read_text().strip() == "424242"


def test_sigterm_mid_run_stops_promptly_with_live_components(state_dir) -> None:
    """The stop must land within a poll tick, not after a full poll interval."""
    _write_heartbeat(age_s=0.0)
    component = RecordingComponent("only")
    stop = threading.Event()

    def tick_hook(count: int) -> None:
        stop.set()  # SIGTERM on the very first observation

    t0 = time.monotonic()
    supervisor.run([component], stop, poll_interval=5.0, tick_hook=tick_hook)
    elapsed = time.monotonic() - t0

    assert component.started_with is stop
    assert component.stopped is True
    assert elapsed < 2.0, f"shutdown took {elapsed:.2f}s against a 5s poll interval"
