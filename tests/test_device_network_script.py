"""Tests for scripts/device-network.sh (task t1).

Covers:
  - the script exists, is executable (on disk and as git records it), uses
    `set -euo pipefail`, and is shellcheck-clean when shellcheck is available
  - --check prints both profiles' autoconnect / autoconnect-priority /
    timestamp and exits 0 iff the preferred profile outranks the fallback
    and both are autoconnect=yes, else exits non-zero
  - --dry-run prints the exact nmcli commands --apply would run and issues
    no `connection modify` call
  - --apply issues the expected `sudo -n nmcli connection modify` calls and
    schedules a revert (forcing the non-systemd-run fallback path via
    REACHY_NET_NO_SYSTEMD_RUN=1, verified via the pidfile under
    XDG_STATE_HOME/reachy/)
  - --commit cancels a pending scheduled revert (kills the background job,
    clears the pidfile/state)
  - --revert restores the previously recorded priorities

All system-touching commands (nmcli, sudo, systemctl, systemd-run) are
stubbed on PATH / via REACHY_NMCLI; nothing here touches the real network.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "device-network.sh"

PREFERRED = "iPhone (5)"
FALLBACK = "bar-nachum"

# A fake nmcli: records every invocation to a log file, and answers
# `-t -f <field> connection show <name>` from a small canned table keyed by
# connection name. The table is injected via env vars so each test can pick
# its own scenario (right order / wrong order / post-apply state).
NMCLI_STUB = """#!/usr/bin/env bash
printf '%s\\n' "nmcli $*" >> "{log_file}"

if [[ "$1" == "-t" && "$2" == "-f" ]]; then
    field="$3"
    name="$6"
    if [[ "$name" == "{preferred}" ]]; then
        ac="$NMCLI_STUB_PREFERRED_AC"
        prio="$NMCLI_STUB_PREFERRED_PRIO"
        ts="$NMCLI_STUB_PREFERRED_TS"
    elif [[ "$name" == "{fallback}" ]]; then
        ac="$NMCLI_STUB_FALLBACK_AC"
        prio="$NMCLI_STUB_FALLBACK_PRIO"
        ts="$NMCLI_STUB_FALLBACK_TS"
    else
        exit 1
    fi
    case "$field" in
        connection.autoconnect) echo "connection.autoconnect:$ac" ;;
        connection.autoconnect-priority) echo "connection.autoconnect-priority:$prio" ;;
        connection.timestamp) echo "connection.timestamp:$ts" ;;
    esac
fi
exit 0
"""

SUDO_STUB = """#!/usr/bin/env bash
printf '%s\\n' "sudo $*" >> "{log_file}"
shift  # drop -n
"$@"
"""

SYSTEMCTL_STUB = """#!/usr/bin/env bash
printf '%s\\n' "systemctl $*" >> "{log_file}"
exit 0
"""

SYSTEMD_RUN_STUB = """#!/usr/bin/env bash
printf '%s\\n' "systemd-run $*" >> "{log_file}"
exit 0
"""


def _write_stub(path: Path, template: str, **kwargs) -> None:
    path.write_text(template.format(**kwargs))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# nmcli stub that fails the *apply-time* modify call to FALLBACK (identified
# by its target priority, distinct from the recorded/original priority used
# on revert) but succeeds on every other invocation, including the revert.
FAILING_SECOND_MODIFY_NMCLI_STUB = """#!/usr/bin/env bash
printf '%s\\n' "nmcli $*" >> "{log_file}"

if [[ "$1" == "-t" && "$2" == "-f" ]]; then
    field="$3"
    name="$6"
    if [[ "$name" == "{preferred}" ]]; then
        ac="$NMCLI_STUB_PREFERRED_AC"
        prio="$NMCLI_STUB_PREFERRED_PRIO"
        ts="$NMCLI_STUB_PREFERRED_TS"
    elif [[ "$name" == "{fallback}" ]]; then
        ac="$NMCLI_STUB_FALLBACK_AC"
        prio="$NMCLI_STUB_FALLBACK_PRIO"
        ts="$NMCLI_STUB_FALLBACK_TS"
    else
        exit 1
    fi
    case "$field" in
        connection.autoconnect) echo "connection.autoconnect:$ac" ;;
        connection.autoconnect-priority) echo "connection.autoconnect-priority:$prio" ;;
        connection.timestamp) echo "connection.timestamp:$ts" ;;
    esac
    exit 0
fi

if [[ "$1" == "connection" && "$2" == "modify" && "$3" == "{fallback}" ]]; then
    if [[ "$*" == *"autoconnect-priority {fallback_target_prio}"* ]]; then
        exit 1
    fi
fi
exit 0
"""

# systemd-run stub that always fails, to exercise the nohup fallback.
FAILING_SYSTEMD_RUN_STUB = """#!/usr/bin/env bash
printf '%s\\n' "systemd-run $*" >> "{log_file}"
exit 1
"""


@pytest.fixture()
def env(tmp_path: Path):
    """Builds an isolated env dict + stub paths for one test run."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state_dir = tmp_path / "state"
    log_file = tmp_path / "calls.log"

    nmcli_path = bin_dir / "nmcli"
    _write_stub(nmcli_path, NMCLI_STUB, log_file=log_file, preferred=PREFERRED, fallback=FALLBACK)
    _write_stub(bin_dir / "sudo", SUDO_STUB, log_file=log_file)
    _write_stub(bin_dir / "systemctl", SYSTEMCTL_STUB, log_file=log_file)
    _write_stub(bin_dir / "systemd-run", SYSTEMD_RUN_STUB, log_file=log_file)

    base_env = dict(os.environ)
    base_env["PATH"] = f"{bin_dir}:{base_env['PATH']}"
    base_env["REACHY_NMCLI"] = str(nmcli_path)
    base_env["XDG_STATE_HOME"] = str(state_dir)
    base_env["HOME"] = str(tmp_path / "home")
    Path(base_env["HOME"]).mkdir(exist_ok=True)
    # canned connection-show scenario: right order by default
    base_env["NMCLI_STUB_PREFERRED_AC"] = "yes"
    base_env["NMCLI_STUB_PREFERRED_PRIO"] = "20"
    base_env["NMCLI_STUB_PREFERRED_TS"] = "1700000100"
    base_env["NMCLI_STUB_FALLBACK_AC"] = "yes"
    base_env["NMCLI_STUB_FALLBACK_PRIO"] = "10"
    base_env["NMCLI_STUB_FALLBACK_TS"] = "1700000200"

    return {
        "env": base_env,
        "log_file": log_file,
        "state_dir": state_dir,
    }


def run_script(env_fixture, *args, timeout=15) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        env=env_fixture["env"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# --- static shape ------------------------------------------------------


def test_script_exists_and_is_executable_on_disk() -> None:
    assert SCRIPT_PATH.is_file(), f"missing {SCRIPT_PATH}"
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, f"{SCRIPT_PATH} is not executable on disk (mode {oct(mode)})"


def test_script_is_executable_in_git() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", "scripts/device-network.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()
    assert output, (
        "scripts/device-network.sh is not staged/tracked in git yet - "
        "run `git add scripts/device-network.sh` (with the executable bit)"
    )
    git_mode = output.split()[0]
    assert git_mode == "100755", f"git records mode {git_mode}, expected 100755"


def test_script_uses_set_euo_pipefail() -> None:
    assert "set -euo pipefail" in SCRIPT_PATH.read_text()


def test_script_never_hardcodes_the_real_nmcli_path_only() -> None:
    """REACHY_NMCLI must be honoured so tests can stub nmcli out."""
    text = SCRIPT_PATH.read_text()
    assert "REACHY_NMCLI" in text


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_script_is_shellcheck_clean() -> None:
    result = subprocess.run(
        ["shellcheck", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"shellcheck findings:\n{result.stdout}\n{result.stderr}"


# --- --check -------------------------------------------------------------


def test_check_exits_zero_and_prints_both_profiles_when_order_is_right(env) -> None:
    result = run_script(env, "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert PREFERRED in result.stdout
    assert FALLBACK in result.stdout
    assert "autoconnect=yes" in result.stdout
    assert "autoconnect-priority=20" in result.stdout
    assert "autoconnect-priority=10" in result.stdout
    assert "1700000100" in result.stdout
    assert "1700000200" in result.stdout
    # --check must never touch the system
    calls = env["log_file"].read_text() if env["log_file"].exists() else ""
    assert "sudo" not in calls


def test_check_exits_nonzero_when_priority_order_is_wrong(env) -> None:
    env["env"]["NMCLI_STUB_PREFERRED_PRIO"] = "5"
    env["env"]["NMCLI_STUB_FALLBACK_PRIO"] = "10"
    result = run_script(env, "--check")
    assert result.returncode != 0


def test_check_exits_nonzero_when_autoconnect_is_not_yes(env) -> None:
    env["env"]["NMCLI_STUB_FALLBACK_AC"] = "no"
    result = run_script(env, "--check")
    assert result.returncode != 0


# --- --dry-run -------------------------------------------------------------


def test_dry_run_prints_commands_and_changes_nothing(env) -> None:
    result = run_script(env, "--dry-run")
    assert result.returncode == 0
    assert "connection modify" in result.stdout
    assert PREFERRED in result.stdout
    assert FALLBACK in result.stdout
    assert "20" in result.stdout
    assert "10" in result.stdout

    calls = env["log_file"].read_text() if env["log_file"].exists() else ""
    assert "sudo" not in calls
    assert "modify" not in calls  # nmcli itself was never invoked to mutate


def test_reachy_net_dry_forces_apply_into_dry_run(env) -> None:
    env["env"]["REACHY_NET_DRY"] = "1"
    result = run_script(env, "--apply")
    assert result.returncode == 0
    assert "connection modify" in result.stdout

    calls = env["log_file"].read_text() if env["log_file"].exists() else ""
    assert "sudo" not in calls
    state_file = env["state_dir"] / "reachy" / "network-revert-state.env"
    assert not state_file.exists()


# --- --apply (non-systemd-run fallback path) --------------------------------


def test_apply_issues_expected_modify_commands_and_schedules_revert(env) -> None:
    env["env"]["REACHY_NET_NO_SYSTEMD_RUN"] = "1"
    result = run_script(env, "--apply", "--revert-after", "120")
    assert result.returncode == 0, result.stdout + result.stderr

    calls = env["log_file"].read_text()
    assert f"sudo -n {env['env']['REACHY_NMCLI']} connection modify {PREFERRED} connection.autoconnect yes connection.autoconnect-priority 20" in calls
    assert f"sudo -n {env['env']['REACHY_NMCLI']} connection modify {FALLBACK} connection.autoconnect yes connection.autoconnect-priority 10" in calls
    # systemd-run must NOT have been used in the forced-fallback path
    assert "systemd-run" not in calls

    pid_file = env["state_dir"] / "reachy" / "network-revert.pid"
    assert pid_file.exists(), "no pending revert pidfile written in the fallback path"
    pid = int(pid_file.read_text().strip())
    # give the backgrounded job a beat to actually start
    time.sleep(0.3)
    # process should still be alive (waiting out the 120s sleep)
    alive = True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        alive = False
    assert alive, "scheduled revert job is not running"

    state_file = env["state_dir"] / "reachy" / "network-revert-state.env"
    assert state_file.exists()
    state_text = state_file.read_text()
    assert "PREFERRED_PRIORITY=20" in state_text
    assert "FALLBACK_PRIORITY=10" in state_text

    # cleanup: don't leave a live background process after the test
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass


def test_apply_uses_systemd_run_when_available_and_not_forced_off(env) -> None:
    result = run_script(env, "--apply", "--revert-after", "60")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = env["log_file"].read_text()
    assert "systemd-run" in calls
    assert "--on-active=60s" in calls
    # the fallback pidfile path must not be used when systemd-run is taken
    pid_file = env["state_dir"] / "reachy" / "network-revert.pid"
    assert not pid_file.exists()


def test_apply_records_current_priorities_before_mutating(env) -> None:
    env["env"]["REACHY_NET_NO_SYSTEMD_RUN"] = "1"
    env["env"]["NMCLI_STUB_PREFERRED_PRIO"] = "5"
    env["env"]["NMCLI_STUB_FALLBACK_PRIO"] = "10"
    result = run_script(env, "--apply", "--revert-after", "120")
    assert result.returncode == 0

    state_file = env["state_dir"] / "reachy" / "network-revert-state.env"
    state_text = state_file.read_text()
    assert "PREFERRED_PRIORITY=5" in state_text
    assert "FALLBACK_PRIORITY=10" in state_text

    pid_file = env["state_dir"] / "reachy" / "network-revert.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 9)
        except ProcessLookupError:
            pass


# --- --commit --------------------------------------------------------------


def test_commit_cancels_pending_revert_and_clears_state(env) -> None:
    env["env"]["REACHY_NET_NO_SYSTEMD_RUN"] = "1"
    apply_result = run_script(env, "--apply", "--revert-after", "120")
    assert apply_result.returncode == 0

    pid_file = env["state_dir"] / "reachy" / "network-revert.pid"
    assert pid_file.exists()
    pid = int(pid_file.read_text().strip())
    time.sleep(0.3)

    commit_result = run_script(env, "--commit")
    assert commit_result.returncode == 0, commit_result.stdout + commit_result.stderr

    assert not pid_file.exists(), "pidfile should be removed after --commit"
    state_file = env["state_dir"] / "reachy" / "network-revert-state.env"
    assert not state_file.exists(), "revert state should be cleared after --commit"

    # the background job should have been killed, not left running
    time.sleep(0.3)
    alive = True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        alive = False
    assert not alive, "background revert job should be dead after --commit"


def test_commit_with_no_pending_revert_is_a_safe_noop(env) -> None:
    result = run_script(env, "--commit")
    assert result.returncode == 0, result.stdout + result.stderr


# --- --revert ----------------------------------------------------------


def test_revert_restores_recorded_values(env) -> None:
    env["env"]["REACHY_NET_NO_SYSTEMD_RUN"] = "1"
    env["env"]["NMCLI_STUB_PREFERRED_PRIO"] = "5"
    env["env"]["NMCLI_STUB_FALLBACK_PRIO"] = "10"

    apply_result = run_script(env, "--apply", "--revert-after", "120")
    assert apply_result.returncode == 0

    pid_file = env["state_dir"] / "reachy" / "network-revert.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 9)
        except ProcessLookupError:
            pass

    revert_result = run_script(env, "--revert")
    assert revert_result.returncode == 0, revert_result.stdout + revert_result.stderr

    calls = env["log_file"].read_text()
    assert f"sudo -n {env['env']['REACHY_NMCLI']} connection modify {PREFERRED} connection.autoconnect yes connection.autoconnect-priority 5" in calls
    assert f"sudo -n {env['env']['REACHY_NMCLI']} connection modify {FALLBACK} connection.autoconnect yes connection.autoconnect-priority 10" in calls

    state_file = env["state_dir"] / "reachy" / "network-revert-state.env"
    assert not state_file.exists(), "revert state should be cleared after --revert"


def test_revert_with_no_pending_state_is_a_safe_noop(env) -> None:
    result = run_script(env, "--revert")
    assert result.returncode == 0, result.stdout + result.stderr


# --- env-configurable profile names ----------------------------------------


def test_profile_names_are_configurable_via_env(env) -> None:
    env["env"]["REACHY_NET_PREFERRED"] = "custom-preferred"
    env["env"]["REACHY_NET_FALLBACK"] = "custom-fallback"
    result = run_script(env, "--dry-run")
    assert result.returncode == 0
    assert "custom-preferred" in result.stdout
    assert "custom-fallback" in result.stdout


# --- never requires root for --check / --dry-run ----------------------------


def test_check_and_dry_run_never_invoke_sudo(env) -> None:
    for args in (("--check",), ("--dry-run",)):
        log_file = env["log_file"]
        if log_file.exists():
            log_file.unlink()
        run_script(env, *args)
        calls = log_file.read_text() if log_file.exists() else ""
        assert "sudo" not in calls, f"{args} unexpectedly invoked sudo"


# --- finding 4: revert must restore original autoconnect too ---------------


def test_apply_records_original_autoconnect_alongside_priority(env) -> None:
    env["env"]["REACHY_NET_NO_SYSTEMD_RUN"] = "1"
    env["env"]["NMCLI_STUB_FALLBACK_AC"] = "no"

    result = run_script(env, "--apply", "--revert-after", "120")
    assert result.returncode == 0, result.stdout + result.stderr

    state_file = env["state_dir"] / "reachy" / "network-revert-state.env"
    state_text = state_file.read_text()
    assert "PREFERRED_AUTOCONNECT=yes" in state_text
    assert "FALLBACK_AUTOCONNECT=no" in state_text
    # priorities are still recorded as before
    assert "PREFERRED_PRIORITY=20" in state_text
    assert "FALLBACK_PRIORITY=10" in state_text

    pid_file = env["state_dir"] / "reachy" / "network-revert.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 9)
        except ProcessLookupError:
            pass


def test_revert_restores_original_autoconnect_no(env) -> None:
    env["env"]["REACHY_NET_NO_SYSTEMD_RUN"] = "1"
    env["env"]["NMCLI_STUB_FALLBACK_AC"] = "no"

    apply_result = run_script(env, "--apply", "--revert-after", "120")
    assert apply_result.returncode == 0, apply_result.stdout + apply_result.stderr

    pid_file = env["state_dir"] / "reachy" / "network-revert.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 9)
        except ProcessLookupError:
            pass

    revert_result = run_script(env, "--revert")
    assert revert_result.returncode == 0, revert_result.stdout + revert_result.stderr

    calls = env["log_file"].read_text()
    nmcli_bin = env["env"]["REACHY_NMCLI"]
    assert (
        f"sudo -n {nmcli_bin} connection modify {FALLBACK} "
        f"connection.autoconnect no connection.autoconnect-priority 10" in calls
    ), calls
    assert (
        f"sudo -n {nmcli_bin} connection modify {PREFERRED} "
        f"connection.autoconnect yes connection.autoconnect-priority 20" in calls
    ), calls

    state_file = env["state_dir"] / "reachy" / "network-revert-state.env"
    assert not state_file.exists()


# --- finding 5: rollback must be armed before mutating NetworkManager ------


def test_systemd_run_failure_falls_back_to_nohup_and_apply_proceeds(env) -> None:
    _write_stub(
        Path(env["env"]["PATH"].split(":", 1)[0]) / "systemd-run",
        FAILING_SYSTEMD_RUN_STUB,
        log_file=env["log_file"],
    )

    result = run_script(env, "--apply", "--revert-after", "60")
    assert result.returncode == 0, result.stdout + result.stderr

    calls = env["log_file"].read_text()
    assert "systemd-run" in calls  # attempted
    assert "connection modify" in calls  # apply proceeded anyway

    pid_file = env["state_dir"] / "reachy" / "network-revert.pid"
    assert pid_file.exists(), "nohup fallback should have armed a pidfile after systemd-run failed"
    pid = int(pid_file.read_text().strip())
    time.sleep(0.3)
    alive = True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        alive = False
    assert alive, "fallback revert job is not running"
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass


def test_second_modify_failure_triggers_immediate_revert_and_nonzero_exit(env) -> None:
    # Original fallback priority (3) is deliberately different from the
    # apply target (10) so the two modify calls are distinguishable in the
    # stub's log/matching, and so the revert-restoring-the-original
    # assertion below can't be satisfied by the (failing) apply call itself.
    env["env"]["NMCLI_STUB_FALLBACK_PRIO"] = "3"

    nmcli_path = Path(env["env"]["PATH"].split(":", 1)[0]) / "nmcli"
    _write_stub(
        nmcli_path,
        FAILING_SECOND_MODIFY_NMCLI_STUB,
        log_file=env["log_file"],
        preferred=PREFERRED,
        fallback=FALLBACK,
        fallback_target_prio="10",
    )

    result = run_script(env, "--apply", "--revert-after", "60")
    assert result.returncode != 0, "apply must fail non-zero when a modify call fails"

    calls = env["log_file"].read_text()
    log_lines = [line for line in calls.splitlines() if line.strip()]

    # rollback must be armed (revert state recorded + revert scheduled) before
    # the first mutating nmcli call
    arm_idx = next(i for i, line in enumerate(log_lines) if line.startswith("systemd-run"))
    first_modify_idx = next(i for i, line in enumerate(log_lines) if "connection modify" in line)
    assert arm_idx < first_modify_idx, calls

    # the original fallback priority (3, not the failed apply target of 10)
    # must have been restored via an explicit revert modify call after the
    # failure
    nmcli_bin = env["env"]["REACHY_NMCLI"]
    assert (
        f"sudo -n {nmcli_bin} connection modify {FALLBACK} "
        f"connection.autoconnect yes connection.autoconnect-priority 3" in calls
    ), calls

    # no leftover pending revert state after the immediate rollback
    state_file = env["state_dir"] / "reachy" / "network-revert-state.env"
    assert not state_file.exists()
