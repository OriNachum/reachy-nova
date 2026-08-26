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
    text = _installer()
    step = text.split("install_failover_hook()", 1)[1]
    assert "self-test FAILED" in step
    assert "sudo -n rm -f" in step


def test_installer_failover_step_is_never_fatal():
    """Every guard in the step returns 0 — a missing dispatcher dir, no sudo,
    or a failed self-test must not fail the whole install."""
    text = _installer()
    step = text.split("install_failover_hook()", 1)[1].split("\n}", 1)[0]
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
