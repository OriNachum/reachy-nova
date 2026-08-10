"""The harness supervisor: exclusivity, PID identity, and named drops.

The three behaviours pinned here are the ones a silent regression makes
expensive on the robot rather than in CI:

* **exclusivity** — the harness and the ``agent embody`` layer would fight
  over the audio tee, the intent spool and the speaker, so a live embody
  layer must REFUSE the start (exit 3) with a named line, never a warning;
* **PID identity** — a stale PID file is reclaimed, but a live one whose
  argv is really another harness refuses (exit 2). Identity is exact argv
  tokens, never a substring of the joined command line;
* **observability** — losing the engine heartbeat is a NAMED transition in
  the log, not a silence.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import types

import pytest

from reachy_nova.harness import statedir, supervisor


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    """A throwaway REACHY_STATE_DIR so no test touches the real one."""
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    (tmp_path / "behavior").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_heartbeat(state_dir, age_s: float) -> None:
    statedir.state_json_path().write_text(
        json.dumps({"updated": time.monotonic() - age_s})
    )


# --------------------------------------------------------------------------- #
# Exclusivity with the embody layer.
# --------------------------------------------------------------------------- #


def test_run_refuses_while_embody_is_live(state_dir, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    statedir.embody_pid_path().write_text(str(os.getpid()))
    # The identity check reads real argv; force the peer's shape.
    monkeypatch.setattr(statedir, "_pid_argv", lambda pid: ["python", "-m", "reachy", "embody"])

    code = supervisor.main(["run"])

    assert code == 3
    assert (
        "[SENSE stage=supervise source=nova event=start] refused reason=embody-live" in caplog.text
    )
    # Refusing must not leave our own PID file behind.
    assert not statedir.harness_pid_path().exists()


def test_run_proceeds_when_the_embody_pid_is_a_stranger(state_dir, monkeypatch):
    statedir.embody_pid_path().write_text(str(os.getpid()))
    monkeypatch.setattr(statedir, "_pid_argv", lambda pid: ["python", "-m", "http.server"])

    assert statedir.embody_is_live() is False


# --------------------------------------------------------------------------- #
# Own PID file: stale is reclaimed, live is refused.
# --------------------------------------------------------------------------- #


def test_stale_pid_file_is_reclaimed(state_dir, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    statedir.harness_pid_path().write_text("999999")
    monkeypatch.setattr(supervisor, "_is_alive", lambda pid: False)

    assert supervisor.acquire_pid_file() is True

    assert statedir.harness_pid_path().read_text().strip() == str(os.getpid())
    assert "reclaimed" in caplog.text


def test_live_sibling_harness_refuses_the_start(state_dir, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    statedir.harness_pid_path().write_text("4242")
    monkeypatch.setattr(supervisor, "_is_alive", lambda pid: True)
    monkeypatch.setattr(
        supervisor, "_pid_argv", lambda pid: ["python", "-m", "reachy_nova.harness", "run"]
    )

    code = supervisor.main(["run"])

    assert code == 2
    assert "refused reason=already-running pid=4242" in caplog.text
    # The sibling's PID file is left exactly as it was.
    assert statedir.harness_pid_path().read_text().strip() == "4242"


def test_live_pid_belonging_to_a_stranger_is_reclaimed(state_dir, monkeypatch):
    """PID reuse: the recorded PID is alive but is NOT a harness."""
    statedir.harness_pid_path().write_text("4242")
    monkeypatch.setattr(supervisor, "_is_alive", lambda pid: True)
    monkeypatch.setattr(supervisor, "_pid_argv", lambda pid: ["python", "-m", "http.server"])

    assert supervisor.acquire_pid_file() is True
    assert statedir.harness_pid_path().read_text().strip() == str(os.getpid())


def test_identity_is_exact_tokens_not_a_substring(state_dir, monkeypatch):
    """A path merely CONTAINING the module name is not a harness."""
    statedir.harness_pid_path().write_text("4242")
    monkeypatch.setattr(supervisor, "_is_alive", lambda pid: True)
    monkeypatch.setattr(
        supervisor,
        "_pid_argv",
        lambda pid: ["/home/pi/git/reachy_nova.harness/bin/python", "-c", "pass"],
    )

    assert supervisor.acquire_pid_file() is True


def test_release_pid_file_only_removes_our_own(state_dir):
    statedir.harness_pid_path().write_text("4242")

    supervisor.release_pid_file()

    assert statedir.harness_pid_path().read_text().strip() == "4242"

    statedir.harness_pid_path().write_text(str(os.getpid()))
    supervisor.release_pid_file()
    assert not statedir.harness_pid_path().exists()


# --------------------------------------------------------------------------- #
# Observability: the engine heartbeat transition is named.
# --------------------------------------------------------------------------- #


def test_engine_heartbeat_loss_is_a_named_drop(state_dir, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    _write_heartbeat(state_dir, age_s=0.0)
    stop = threading.Event()

    def tick_hook(count: int) -> None:
        if count == 1:
            _write_heartbeat(state_dir, age_s=60.0)  # heartbeat goes stale
        if count >= 2:
            stop.set()

    supervisor.run([], stop, poll_interval=0.0, tick_hook=tick_hook)

    assert "[SENSE stage=supervise source=nova event=engine] engine live" in caplog.text
    assert (
        "[SENSE stage=supervise source=nova event=engine] dropped reason=engine-heartbeat-lost"
        in caplog.text
    )


def test_engine_recovery_is_named_too(state_dir, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    _write_heartbeat(state_dir, age_s=60.0)
    stop = threading.Event()

    def tick_hook(count: int) -> None:
        if count == 1:
            _write_heartbeat(state_dir, age_s=0.0)
        if count >= 2:
            stop.set()

    supervisor.run([], stop, poll_interval=0.0, tick_hook=tick_hook)

    assert "engine absent" in caplog.text
    assert "engine live" in caplog.text


# --------------------------------------------------------------------------- #
# Component lifecycle.
# --------------------------------------------------------------------------- #


class _FakeComponent:
    name = "fake"

    def __init__(self):
        self.started_with = None
        self.stopped = False

    def start(self, stop_event):
        self.started_with = stop_event

    def stop(self):
        self.stopped = True


def test_components_are_started_and_stopped(state_dir):
    component = _FakeComponent()
    stop = threading.Event()
    stop.set()

    supervisor.run([component], stop, poll_interval=0.0)

    assert component.started_with is stop
    assert component.stopped is True


def test_a_component_that_fails_to_start_does_not_take_down_the_loop(state_dir, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")

    class Exploding(_FakeComponent):
        name = "exploding"

        def start(self, stop_event):
            raise RuntimeError("no device")

    ok = _FakeComponent()
    stop = threading.Event()
    stop.set()

    supervisor.run([Exploding(), ok], stop, poll_interval=0.0)

    assert ok.started_with is stop
    assert "start failed" in caplog.text


def test_build_components_names_absent_modules_and_continues(caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")

    components = supervisor.build_components(names=("definitely_not_here",))

    assert components == []
    assert "component absent name=definitely_not_here" in caplog.text


def test_build_components_collects_modules_that_exist(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    component = _FakeComponent()
    module = types.ModuleType("reachy_nova.harness.fakecomp")
    module.build_component = lambda: component  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "reachy_nova.harness.fakecomp", module)

    assert supervisor.build_components(names=("fakecomp",)) == [component]


def test_build_components_skips_a_module_without_a_factory(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    module = types.ModuleType("reachy_nova.harness.nofactory")
    monkeypatch.setitem(sys.modules, "reachy_nova.harness.nofactory", module)

    assert supervisor.build_components(names=("nofactory",)) == []
    assert "reason=no-factory" in caplog.text


# --------------------------------------------------------------------------- #
# CLI surface.
# --------------------------------------------------------------------------- #


def test_status_reports_liveness_as_json(state_dir, capsys, monkeypatch):
    _write_heartbeat(state_dir, age_s=0.0)
    statedir.harness_pid_path().write_text(str(os.getpid()))
    monkeypatch.setattr(supervisor, "_is_alive", lambda pid: True)
    monkeypatch.setattr(
        supervisor, "_pid_argv", lambda pid: ["python", "-m", "reachy_nova.harness"]
    )

    code = supervisor.main(["status"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["engine_live"] is True
    assert payload["embody_live"] is False
    assert payload["harness_pid"] == os.getpid()
    assert payload["harness_running"] is True


def test_install_unit_command_uses_the_injected_runner(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    calls: list[list[str]] = []

    code = supervisor.main(["install-unit"], runner=lambda cmd, **kw: calls.append(list(cmd)))

    assert code == 0
    assert (tmp_path / "systemd" / "user" / "reachy-nova-harness.service").exists()
    assert calls == [["systemctl", "--user", "daemon-reload"]]


def test_run_loads_the_env_file_before_anything_else(state_dir, tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    env_file = tmp_path / "harness.env"
    env_file.write_text("REACHY_NOVA_T11_PROBE=loaded\n")
    monkeypatch.delenv("REACHY_NOVA_T11_PROBE", raising=False)
    # Refuse right after loading so the loop never starts.
    statedir.embody_pid_path().write_text(str(os.getpid()))
    monkeypatch.setattr(statedir, "_pid_argv", lambda pid: ["reachy", "embody"])

    code = supervisor.main(["run", "--env-file", str(env_file)])

    assert code == 3
    assert os.environ.get("REACHY_NOVA_T11_PROBE") == "loaded"


def test_run_starts_the_loop_when_nothing_else_is_live(state_dir, monkeypatch):
    started: dict[str, object] = {}

    def fake_run(components, stop_event, **kwargs):
        started["components"] = components
        started["stop_event"] = stop_event

    monkeypatch.setattr(supervisor, "run", fake_run)
    monkeypatch.setattr(supervisor, "build_components", lambda: [])

    code = supervisor.main(["run"])

    assert code == 0
    assert isinstance(started["stop_event"], threading.Event)
    # The PID file is released on a clean exit.
    assert not statedir.harness_pid_path().exists()
