"""Tests for scripts/install-device-units.sh (task t3).

Covers:
  - the script exists, is executable (on disk and as git records it), and
    uses `set -euo pipefail`
  - static assertions that it renders reachy-runtime.service and
    reachy-demo-mode.service from reachy_nova.harness.unit, installs the
    harness unit via the existing `install-unit` CLI, enables runtime+harness
    but never demo-mode, masks the legacy reachy-nova-autostart.service, and
    writes the journald persistence drop-in (Storage=persistent,
    SystemMaxUse=64M) before restarting systemd-journald
  - an end-to-end run (stubbed systemctl/sudo on PATH, no real root/systemd
    touched) that the script actually writes correct unit files and is
    idempotent: running it twice produces identical unit files and journald
    drop-in content, and exits 0 both times
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "install-device-units.sh"

STUB_TEMPLATE = """#!/usr/bin/env bash
# Test stub — records its invocation, never touches the real system.
printf '%s' "$0 $*" >> "{log_file}"
printf '\\n' >> "{log_file}"
if [[ "$1" == "tee" ]] || [[ "$(basename "$0")" == "tee" ]]; then
    cat > "{capture_dir}/journald.conf"
fi
exit 0
"""


def _write_stub(bin_dir: Path, name: str, log_file: Path, capture_dir: Path) -> None:
    path = bin_dir / name
    path.write_text(STUB_TEMPLATE.format(log_file=log_file, capture_dir=capture_dir))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_script(tmp_path: Path, log_file: Path, capture_dir: Path) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir(exist_ok=True)
    # sudo and tee are stubbed so the script never touches real /etc or
    # systemd; systemctl is stubbed so no real user manager is required.
    _write_stub(bin_dir, "sudo", log_file, capture_dir)
    _write_stub(bin_dir, "systemctl", log_file, capture_dir)
    _write_stub(bin_dir, "tee", log_file, capture_dir)

    home = tmp_path / "home"
    xdg_config = home / ".config"
    (home).mkdir(exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(home)
    env["XDG_CONFIG_HOME"] = str(xdg_config)
    env["PYTHONPATH"] = str(REPO_ROOT)

    return subprocess.run(
        [str(SCRIPT_PATH), "/opt/cli/.venv/bin/python", sys.executable],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


# --- static shape ----------------------------------------------------------


def test_install_script_exists_and_is_executable_on_disk() -> None:
    assert SCRIPT_PATH.is_file(), f"missing {SCRIPT_PATH}"
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, f"{SCRIPT_PATH} is not executable on disk (mode {oct(mode)})"


def test_install_script_is_executable_in_git() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", "scripts/install-device-units.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()
    assert output, (
        "scripts/install-device-units.sh is not staged/tracked in git yet - "
        "run `git add scripts/install-device-units.sh` (with the executable bit)"
    )
    git_mode = output.split()[0]
    assert git_mode == "100755", (
        f"git records scripts/install-device-units.sh as mode {git_mode}, expected 100755"
    )


def test_install_script_uses_set_euo_pipefail() -> None:
    assert "set -euo pipefail" in SCRIPT_PATH.read_text()


def test_install_script_renders_runtime_and_demo_units_from_unit_py() -> None:
    text = SCRIPT_PATH.read_text()
    assert "runtime_unit_text" in text
    assert "demo_mode_unit_text" in text
    assert "reachy-runtime.service" in text
    assert "reachy-demo-mode.service" in text


def test_install_script_installs_harness_unit_via_existing_cli() -> None:
    text = SCRIPT_PATH.read_text()
    assert "reachy_nova.harness install-unit" in text


def test_install_script_enables_runtime_and_harness_never_demo() -> None:
    text = SCRIPT_PATH.read_text()
    enable_lines = [line for line in text.splitlines() if "systemctl --user enable" in line]
    assert enable_lines, "no `systemctl --user enable` invocation found"
    for line in enable_lines:
        assert "reachy-runtime.service" in line
        assert "reachy-nova-harness.service" in line
        assert "reachy-demo-mode.service" not in line


def test_install_script_masks_legacy_autostart_unit() -> None:
    text = SCRIPT_PATH.read_text()
    assert "systemctl mask reachy-nova-autostart.service" in text
    assert "sudo systemctl mask reachy-nova-autostart.service" in text


def test_install_script_writes_journald_persistence_drop_in() -> None:
    text = SCRIPT_PATH.read_text()
    assert "/etc/systemd/journald.conf.d/reachy-nova.conf" in text
    assert "Storage=persistent" in text
    assert "SystemMaxUse=64M" in text
    assert "systemctl restart systemd-journald" in text


def test_install_script_documents_idempotency() -> None:
    assert "idempotent" in SCRIPT_PATH.read_text().lower()


def test_install_script_guards_optional_kiro_config_copy() -> None:
    """The nova-writer Kiro config `cp` must warn-and-continue on failure,
    not abort the whole install under the script-wide `set -e` (qodo review
    comment 3812045214)."""
    text = SCRIPT_PATH.read_text()
    assert 'if ! cp "$repo_config" "$kiro_agents_dir/nova-writer.json"' in text
    # the guard must actually recover (warn + return 0), matching the
    # function's other guarded steps, not just suppress the error code
    guard_start = text.index('if ! cp "$repo_config" "$kiro_agents_dir/nova-writer.json"')
    guard_block = text[guard_start : guard_start + 400]
    assert "warn " in guard_block
    assert "return 0" in guard_block


# --- end-to-end, stubbed systemctl/sudo -------------------------------------


def test_install_script_runs_and_renders_correct_units(tmp_path) -> None:
    log_file = tmp_path / "calls.log"
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()

    result = _run_script(tmp_path, log_file, capture_dir)

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    home = tmp_path / "home"
    unit_dir = home / ".config" / "systemd" / "user"

    runtime_text = (unit_dir / "reachy-runtime.service").read_text()
    assert "Conflicts=reachy-demo-mode.service" in runtime_text
    assert '"/opt/cli/.venv/bin/python" -m reachy behavior engine run' in runtime_text
    assert "[Install]" in runtime_text

    demo_text = (unit_dir / "reachy-demo-mode.service").read_text()
    assert "Conflicts=reachy-runtime.service" in demo_text
    assert "[Install]" not in demo_text
    assert "reachy demo-mode run --config %h/.config/reachy/demo-mode.json" in demo_text

    harness_text = (unit_dir / "reachy-nova-harness.service").read_text()
    assert "ExecStart=" in harness_text

    journald_conf = (capture_dir / "journald.conf").read_text()
    assert "Storage=persistent" in journald_conf
    assert "SystemMaxUse=64M" in journald_conf

    calls = log_file.read_text()
    assert "mask reachy-nova-autostart.service" in calls
    assert "enable reachy-runtime.service reachy-nova-harness.service" in calls
    enable_lines = [line for line in calls.splitlines() if "enable" in line]
    assert enable_lines, "no `enable` invocation was logged"
    assert not any("reachy-demo-mode.service" in line for line in enable_lines)


def test_install_script_is_idempotent_on_rerun(tmp_path) -> None:
    log_file = tmp_path / "calls.log"
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()

    first = _run_script(tmp_path, log_file, capture_dir)
    assert first.returncode == 0, f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}"

    home = tmp_path / "home"
    unit_dir = home / ".config" / "systemd" / "user"
    runtime_after_first = (unit_dir / "reachy-runtime.service").read_text()
    demo_after_first = (unit_dir / "reachy-demo-mode.service").read_text()
    journald_after_first = (capture_dir / "journald.conf").read_text()

    second = _run_script(tmp_path, log_file, capture_dir)
    assert second.returncode == 0, f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}"

    runtime_after_second = (unit_dir / "reachy-runtime.service").read_text()
    demo_after_second = (unit_dir / "reachy-demo-mode.service").read_text()
    journald_after_second = (capture_dir / "journald.conf").read_text()

    assert runtime_after_first == runtime_after_second
    assert demo_after_first == demo_after_second
    assert journald_after_first == journald_after_second
