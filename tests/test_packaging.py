"""Packaging + harness-entrypoint contract (t3).

Verifies pyproject.toml's distribution metadata (name, console scripts, the
reachy_mini_apps entry point) and that the harness CLI stub fails cleanly —
not with a traceback — while ``reachy_nova.harness.supervisor`` doesn't exist
yet (a sibling task adds it).
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def test_distribution_name_is_reachy_nova():
    data = _load_pyproject()
    assert data["project"]["name"] == "reachy-nova"


def test_harness_console_script_entry():
    data = _load_pyproject()
    scripts = data["project"]["scripts"]
    assert scripts["reachy-nova-harness"] == "reachy_nova.harness.__main__:main"


def test_sonic_demo_console_script_survives():
    data = _load_pyproject()
    scripts = data["project"]["scripts"]
    assert scripts["sonic-demo"] == "tools.sonic_demo:main"


def test_reachy_mini_apps_entry_point_survives():
    data = _load_pyproject()
    entry_points = data["project"]["entry-points"]
    assert entry_points["reachy_mini_apps"]["reachy_nova"] == "reachy_nova.main:ReachyNova"


def test_dependencies_are_unchanged():
    data = _load_pyproject()
    deps = data["project"]["dependencies"]
    # Spot-check a handful of deps that must survive the rename untouched.
    for expected in ("reachy-mini", "boto3>=1.35.0", "nova-act>=3.0.0", "ultralytics"):
        assert expected in deps


def test_harness_entry_point_help_exits_cleanly():
    """``python -m reachy_nova.harness --help`` exits 0 with no traceback.

    (The original stub-era test ran ``run`` expecting failure while the
    supervisor was absent; the supervisor exists now, so ``run`` would start a
    real harness — ``--help`` proves the entry point wiring instead.)
    """
    import os

    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "reachy_nova.harness", "--help"],
        cwd=REPO_ROOT,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Traceback" not in result.stderr


def test_harness_stub_help_lists_subcommands():
    import os

    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "reachy_nova.harness", "--help"],
        cwd=REPO_ROOT,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "run" in result.stdout
    assert "install-unit" in result.stdout
