"""The harness's systemd ``--user`` unit: pure text + the install seam.

Two things are pinned here because getting either wrong is a silent
field failure rather than a test failure:

* the unit is a **peripheral**, not a presence unit — it orders ``After=``
  the runtime but must NEVER ``Requires=`` it, or a runtime restart takes
  the harness down with it instead of letting the harness re-attach;
* installing writes the file and reloads the daemon, and does **not**
  enable or start anything — that stays an explicit operator act.
"""

from __future__ import annotations

from pathlib import Path

from reachy_nova.harness import unit


def test_unit_orders_after_runtime_without_requiring_it():
    text = unit.harness_unit_text(python="/usr/bin/python3")

    assert "After=reachy-runtime.service network-online.target" in text
    # A peripheral never Requires= the runtime: the runtime restarting must
    # not stop the harness (it re-attaches through the filesystem seams).
    assert "Requires=" not in text


def test_unit_carries_the_shared_service_grammar():
    text = unit.harness_unit_text(python="/usr/bin/python3")

    assert "Type=simple" in text
    assert "Restart=on-failure" in text
    assert "RestartSec=5" in text
    assert "WantedBy=default.target" in text


def test_exec_start_runs_the_harness_module_entry():
    text = unit.harness_unit_text(python="/usr/bin/python3")

    assert 'ExecStart="/usr/bin/python3" -m reachy_nova.harness run' in text
    assert "--env-file" not in text


def test_exec_start_forwards_an_env_file_when_given():
    text = unit.harness_unit_text(python="/usr/bin/python3", env_file="/home/pi/.env")

    assert (
        'ExecStart="/usr/bin/python3" -m reachy_nova.harness run --env-file "/home/pi/.env"'
        in text
    )


def test_workdir_renders_only_when_given():
    assert "WorkingDirectory=" not in unit.harness_unit_text(python="/usr/bin/python3")

    text = unit.harness_unit_text(python="/usr/bin/python3", workdir="/home/pi/reachy_nova")
    assert 'WorkingDirectory="/home/pi/reachy_nova"' in text


def test_unit_arguments_are_escaped_for_the_systemd_grammar():
    text = unit.harness_unit_text(python="/opt/py 3.12/bin/python", env_file="/tmp/100%/.env")

    # Spaces survive inside quotes; '%' is a systemd specifier and must double.
    assert '"/opt/py 3.12/bin/python"' in text
    assert '"/tmp/100%%/.env"' in text


def test_unit_text_is_pure(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    unit.harness_unit_text(python="/usr/bin/python3")

    assert not (tmp_path / "systemd").exists()


def test_unit_path_honors_xdg_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert unit.unit_path() == tmp_path / "systemd" / "user" / "reachy-nova-harness.service"


def test_unit_path_falls_back_to_dot_config(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    expected = tmp_path / ".config" / "systemd" / "user" / "reachy-nova-harness.service"
    assert unit.unit_path() == expected


def test_install_unit_writes_the_file_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):
        calls.append(list(cmd))
        return None

    path = unit.install_unit(python="/usr/bin/python3", runner=runner)

    assert path == tmp_path / "systemd" / "user" / "reachy-nova-harness.service"
    assert "ExecStart=" in path.read_text()
    assert calls == [["systemctl", "--user", "daemon-reload"]]


def test_install_unit_never_enables_or_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    calls: list[list[str]] = []

    unit.install_unit(python="/usr/bin/python3", runner=lambda cmd, **kw: calls.append(list(cmd)))

    flattened = [token for cmd in calls for token in cmd]
    assert "enable" not in flattened
    assert "start" not in flattened
