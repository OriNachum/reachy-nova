"""Static checks for the failover dispatcher hook and its installer step (task t3).

The hook itself (``config/network/90-reachy-failover``) runs as ROOT, on the
robot's only radio, on every NetworkManager event — it is the one script in
this repo that can take the whole machine off the network. It cannot be
exercised end-to-end in CI (no NetworkManager, no root, no wlan0), so what is
pinned here are the properties that make it *safe to install*: it parses, it
ignores every interface but wlan0, it ignores actions it does not own, it
never exits non-zero back into NM, it logs to journald, and it drives the
tested Python driver rather than open-coding nmcli logic.

Also pins the installer's contract for the step: `bash -n` parses,
``--no-failover`` exists and is documented, the self-test runs, and the
revert-on-failure branch is really there.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / "config" / "network" / "90-reachy-failover"
INSTALLER_PATH = REPO_ROOT / "scripts" / "install-device-units.sh"


def _hook() -> str:
    return HOOK_PATH.read_text()


def _installer() -> str:
    return INSTALLER_PATH.read_text()


def _failover_step() -> str:
    """The body of install_failover_hook(), from its definition line.

    Split on the definition (``install_failover_hook() {``), never on the bare
    name: the usage header mentions the function too, and everything the
    installer grew above it would otherwise be read as part of the step.
    """
    return _installer().split("install_failover_hook() {", 1)[1]


# --------------------------------------------------------------------------
# the dispatcher hook
# --------------------------------------------------------------------------


def test_hook_exists_with_a_bash_shebang():
    assert HOOK_PATH.is_file()
    assert _hook().splitlines()[0].startswith("#!")
    assert "bash" in _hook().splitlines()[0]


def test_hook_parses_as_bash():
    proc = subprocess.run(
        ["bash", "-n", str(HOOK_PATH)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_hook_uses_set_u_and_not_set_e():
    """`set -u` catches typos; `set -e` would turn any hiccup into a failed
    dispatcher run on the robot's only radio."""
    text = _hook()
    assert re.search(r"^set -u\s*$", text, re.M)
    assert not re.search(r"^set -e", text, re.M)
    assert not re.search(r"^set -[a-z]*e[a-z]*u", text, re.M)


def test_hook_only_acts_on_the_wifi_interface():
    text = _hook()
    assert 'WIFI_IFACE="${REACHY_WIFI_IFACE:-wlan0}"' in text
    assert '[ "$IFACE" != "$WIFI_IFACE" ]' in text


def test_hook_handles_exactly_the_four_owned_actions():
    text = _hook()
    assert "up|down|pre-down|connectivity-change" in text


def test_hook_never_exits_non_zero():
    """NM logs a non-zero dispatcher script as a failure; every path here ends
    in `exit 0`."""
    text = _hook()
    exits = re.findall(r"^\s*exit\s+(\d+)", text, re.M)
    assert exits, "the hook should exit explicitly"
    assert set(exits) == {"0"}
    assert "exit 1" not in text


def test_hook_invokes_the_tested_python_driver_not_hand_rolled_nmcli():
    text = _hook()
    assert "reachy_nova.netfailover --once" in text
    assert "reachy_nova.netfailover --loop" in text
    # the policy/nmcli logic lives in the tested module, never inline here
    # (comment lines are stripped: the header explains WHY nmcli is not here)
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "nmcli" not in code


def test_hook_logs_to_journald_with_a_stable_tag():
    text = _hook()
    assert 'logger -t "$TAG"' in text
    assert 'TAG="reachy-failover"' in text


def test_hook_starts_the_loop_unit_idempotently():
    text = _hook()
    assert 'LOOP_UNIT="reachy-netfailover-loop"' in text
    assert "systemctl is-active" in text
    assert "systemd-run" in text
    assert '--unit="$LOOP_UNIT"' in text


def test_hook_passes_the_state_dir_and_rescan_env():
    text = _hook()
    assert "REACHY_STATE_DIR" in text
    assert "REACHY_NET_RESCAN=1" in text
    assert "/home/pollen/.local/state/reachy" in text


def test_hook_chowns_the_network_change_file_to_the_harness_user():
    """The harness reads <state>/network-change as pollen; the hook writes it
    as root."""
    text = _hook()
    assert "chown $NOVA_USER" in text
    assert "network-change" in text
    assert 'NOVA_USER="${REACHY_NOVA_USER:-pollen}"' in text


def test_hook_guards_on_the_interpreter_existing():
    text = _hook()
    assert "/home/pollen/git/reachy-nova/.venv/bin/python" in text
    assert '[ ! -x "$NOVA_PYTHON" ]' in text


def test_hook_sources_the_installer_rendered_defaults_file():
    """The hook's compiled-in defaults are for the reference device only. A
    custom install renders /etc/default/reachy-failover with the interpreter,
    state dir and user it actually resolved, and the hook must source it
    BEFORE its own `${VAR:-default}` fallbacks — otherwise a custom install
    passes the installer's self-test and then silently skips every event on
    the `[ ! -x "$NOVA_PYTHON" ]` guard.
    """
    text = _hook()
    assert 'DEFAULTS_FILE="${REACHY_FAILOVER_DEFAULTS:-/etc/default/reachy-failover}"' in text
    assert '[ -r "$DEFAULTS_FILE" ]' in text
    assert '. "$DEFAULTS_FILE"' in text
    # ordering: sourcing must precede the defaults it is meant to override
    assert text.index('. "$DEFAULTS_FILE"') < text.index('NOVA_PYTHON="${REACHY_NOVA_PYTHON:-')


def test_hook_still_reads_all_three_values_from_the_environment():
    text = _hook()
    for name in ("REACHY_NOVA_PYTHON", "REACHY_STATE_DIR", "REACHY_NOVA_USER"):
        assert name in text


def test_hook_never_blocks_networkmanager_for_the_activation():
    """The activation can take 45 s; the dispatcher queue must not wait for it."""
    text = _hook()
    assert "systemd-run" in text
    assert "timeout 60" in text  # the no-systemd-run fallback is still bounded


# --------------------------------------------------------------------------
# the installer step
# --------------------------------------------------------------------------


def test_installer_parses_as_bash():
    proc = subprocess.run(
        ["bash", "-n", str(INSTALLER_PATH)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_installer_has_the_guarded_failover_step():
    text = _installer()
    assert "install_failover_hook()" in text
    assert "/etc/NetworkManager/dispatcher.d" in text
    assert "90-reachy-failover" in text


def test_installer_copies_the_hook_root_owned_0755_via_sudo_n():
    text = _installer()
    assert "sudo -n install -o root -g root -m 0755" in text


def test_installer_runs_the_self_test():
    text = _installer()
    assert "reachy_nova.netfailover --self-test" in text


def test_installer_reverts_the_hook_when_the_self_test_fails():
    """The whole point of the self-test: a hook that cannot run is removed."""
    step = _failover_step()
    assert "self-test FAILED" in step
    assert "sudo -n rm -f" in step


def test_installer_failover_step_is_never_fatal():
    """Every guard in the step returns 0 — a missing dispatcher dir, no sudo,
    or a failed self-test must not fail the whole install."""
    step = _failover_step().split("\n}", 1)[0]
    # no `exit` at all inside the function; only `return 0`
    assert not re.search(r"^\s*exit\s", step, re.M)
    assert re.search(r"^\s*return 0", step, re.M)


def test_installer_documents_and_implements_no_failover_flag():
    text = _installer()
    assert "--no-failover" in text
    # documented in the usage header...
    header = text.split("set -euo pipefail", 1)[0]
    assert "--no-failover" in header
    # ...and actually parsed
    assert "INSTALL_FAILOVER=0" in text
    assert 'if [ "$INSTALL_FAILOVER" -eq 1 ]' in text


def test_installer_still_accepts_the_two_positional_pythons():
    text = _installer()
    assert 'CLI_PYTHON="${POSITIONAL[0]:-' in text
    assert 'NOVA_PYTHON="${POSITIONAL[1]:-' in text


# --------------------------------------------------------------------------
# the installer renders the hook's config instead of relying on its defaults
# --------------------------------------------------------------------------


def _dry_run(tmp_path, nova_python: str, env_extra=None):
    """Run the installer in its INSTALL_DRY_RUN=1 seam — prints what the
    failover step would render, touches nothing, needs no sudo."""
    import os

    env = dict(os.environ)
    env.update(
        {
            "INSTALL_DRY_RUN": "1",
            "HOME": str(tmp_path),
            "PATH": env.get("PATH", "/usr/bin:/bin"),
        }
    )
    env.pop("REACHY_STATE_DIR", None)
    env.pop("XDG_STATE_HOME", None)
    env.pop("REACHY_NOVA_USER", None)
    env.update(env_extra or {})
    proc = subprocess.run(
        ["bash", str(INSTALLER_PATH), str(tmp_path / "cli-python"), nova_python],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout + proc.stderr


def _fake_python(tmp_path) -> str:
    path = tmp_path / "custom-venv" / "bin" / "python"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


def test_installer_has_a_dry_run_seam():
    text = _installer()
    assert "INSTALL_DRY_RUN" in text


def test_installer_renders_the_defaults_file_with_the_resolved_interpreter(tmp_path):
    """Finding 6: a custom NOVA_PYTHON must reach the installed hook."""
    nova_python = _fake_python(tmp_path)
    out = _dry_run(tmp_path, nova_python)
    assert "/etc/default/reachy-failover" in out
    assert f"REACHY_NOVA_PYTHON={nova_python}" in out
    assert "/home/pollen/git/reachy-nova/.venv/bin/python" not in out


def test_installer_renders_the_resolved_state_dir_and_user(tmp_path):
    nova_python = _fake_python(tmp_path)
    out = _dry_run(
        tmp_path,
        nova_python,
        env_extra={"REACHY_STATE_DIR": str(tmp_path / "st"), "REACHY_NOVA_USER": "someone"},
    )
    assert f"REACHY_STATE_DIR={tmp_path / 'st'}" in out
    assert "REACHY_NOVA_USER=someone" in out


def test_installer_state_dir_defaults_like_the_python_resolver(tmp_path):
    """No REACHY_STATE_DIR / XDG_STATE_HOME -> $HOME/.local/state/reachy, the
    same cascade as reachy_nova.netfailover.default_statedir()."""
    out = _dry_run(tmp_path, _fake_python(tmp_path))
    assert f"REACHY_STATE_DIR={tmp_path}/.local/state/reachy" in out


def test_installer_resolves_a_bare_interpreter_name_to_an_absolute_path(tmp_path):
    """The hook guards on `[ -x "$NOVA_PYTHON" ]`, which a bare `python3`
    never satisfies."""
    out = _dry_run(tmp_path, "python3")
    match = re.search(r"^REACHY_NOVA_PYTHON=(.+)$", out, re.M)
    assert match, out
    assert match.group(1).startswith("/")


def test_installer_self_tests_through_the_same_interpreter_the_hook_will_use():
    step = _failover_step()
    assert '"$NOVA_PYTHON_RESOLVED" -m reachy_nova.netfailover --self-test' in step
    # ...and the defaults file is written before the hook is trusted
    assert "FAILOVER_DEFAULTS_FILE" in step
