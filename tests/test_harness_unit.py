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


# --- runtime + demo-mode templates: boot exclusivity hardening -------------
#
# reachy-runtime.service and reachy-demo-mode.service both drive the body, so
# each must Conflicts= the other (systemd itself refuses to run both at once,
# rather than relying on operator discipline), and the demo unit must be
# reachable only by an explicit `systemctl --user start` — never `enable`.


def test_runtime_unit_conflicts_with_demo_mode():
    text = unit.runtime_unit_text(python="/opt/cli/.venv/bin/python")

    assert "Conflicts=reachy-demo-mode.service" in text


def test_demo_mode_unit_conflicts_with_runtime():
    text = unit.demo_mode_unit_text(python="/opt/cli/.venv/bin/python")

    assert "Conflicts=reachy-runtime.service" in text


def test_demo_mode_unit_has_no_install_section():
    text = unit.demo_mode_unit_text(python="/opt/cli/.venv/bin/python")

    assert "[Install]" not in text
    assert "WantedBy=" not in text


def test_runtime_unit_is_enabled_and_ordered_after_network():
    text = unit.runtime_unit_text(python="/opt/cli/.venv/bin/python")

    assert "[Install]" in text
    assert "WantedBy=default.target" in text
    assert "After=network-online.target" in text


def test_runtime_unit_exec_start_runs_the_behavior_engine():
    text = unit.runtime_unit_text(python="/opt/cli/.venv/bin/python")

    assert 'ExecStart="/opt/cli/.venv/bin/python" -m reachy behavior engine run' in text


def test_demo_mode_unit_exec_start_runs_demo_mode_with_config():
    text = unit.demo_mode_unit_text(python="/opt/cli/.venv/bin/python")

    assert (
        'ExecStart="/opt/cli/.venv/bin/python" -m reachy demo-mode run '
        "--config %h/.config/reachy/demo-mode.json" in text
    )
    # %h is a systemd specifier (per-user home) and must reach the unit file
    # literally, not doubled the way a literal '%' in a path would be.
    assert "%%h" not in text


def test_runtime_and_demo_mode_units_are_pure():
    """Neither renderer touches the filesystem or spawns anything."""
    unit.runtime_unit_text(python="/opt/cli/.venv/bin/python")
    unit.demo_mode_unit_text(python="/opt/cli/.venv/bin/python")
    # No assertion needed beyond "did not raise / did not require a runner" —
    # both are plain str-returning functions with no side-effecting call.


def test_harness_unit_is_unchanged_by_the_new_templates():
    """t3 adds runtime/demo templates; the harness's own unit stays as-is."""
    text = unit.harness_unit_text(python="/usr/bin/python3")

    assert "Conflicts=" not in text


# --------------------------------------------------------------------------- #
# Network ordering is ordering only — never a dependency (task t5)            #
# --------------------------------------------------------------------------- #


def test_unit_never_depends_on_network_online_target():
    """The harness must start with Wi-Fi down (spec h14).

    ``After=`` is kept (free ordering), but a ``Wants=``/``Requires=`` on
    ``network-online.target`` would make systemd hold the harness back — and on
    this device that target is reached BEFORE wlan0 associates anyway, so it
    would buy a delay and still not guarantee a route.
    """
    text = unit.harness_unit_text(python="/usr/bin/python3")

    assert "After=reachy-runtime.service network-online.target" in text
    assert "Wants=" not in text
    assert "Requires=" not in text


def test_unit_text_explains_that_the_network_ordering_is_not_a_dependency():
    text = unit.harness_unit_text(python="/usr/bin/python3")

    assert "# Ordering only" in text
    assert "does NOT depend on the network" in text
    assert "Never turn this ordering into a Wants or Requires dependency." in text
    # The comment sits in [Unit], immediately above the After= line it explains.
    assert text.index("# Ordering only") > text.index("[Unit]")
    assert text.index("# Ordering only") < text.index("After=reachy-runtime.service")
    assert text.index("After=reachy-runtime.service") < text.index("[Service]")


def test_network_ordering_comment_lines_are_all_systemd_comments():
    for line in unit.NETWORK_ORDERING_COMMENT.splitlines():
        assert line.startswith("#")
