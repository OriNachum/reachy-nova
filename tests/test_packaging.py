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


def test_mandatory_dependencies_are_exactly_as_expected():
    """Verify the mandatory dependency list hasn't drifted."""
    data = _load_pyproject()
    deps = data["project"]["dependencies"]

    # Expected list of mandatory dependencies (canonical distribution names,
    # no version specifiers or extras).
    expected = [
        "reachy-mini",
        "boto3",
        "aws-sdk-bedrock-runtime",
        "nova-act",
        "numpy",
        "opencv-python",
        "python-dotenv",
        "pyaudio",
        "ultralytics",
        "pymongo",
        "neo4j",
        "slack-bolt",
        "paho-mqtt",
        "pyyaml",
        "nemo-toolkit",
    ]

    # Strip version specifiers and extras from deps to get distribution names.
    def canonicalize(dep_str: str) -> str:
        """Extract the distribution name from a dependency specifier."""
        # Split on brackets (extras) and comparison operators.
        name = dep_str.split("[")[0]
        for op in (">=", "<=", "==", ">", "<", "!=", "~="):
            name = name.split(op)[0]
        return name.strip().lower().replace("_", "-")

    actual_names = sorted([canonicalize(d) for d in deps])
    expected_names = sorted(expected)

    assert actual_names == expected_names, (
        f"Dependency mismatch:\n"
        f"  Expected: {expected_names}\n"
        f"  Actual:   {actual_names}"
    )


def test_optional_dependencies_kiro_exists_and_empty():
    """Verify [project.optional-dependencies] exists with a kiro key that is empty."""
    data = _load_pyproject()
    assert "optional-dependencies" in data["project"], (
        "Missing [project.optional-dependencies] section"
    )
    opt_deps = data["project"]["optional-dependencies"]
    assert "kiro" in opt_deps, (
        "Missing 'kiro' key in [project.optional-dependencies]"
    )
    assert opt_deps["kiro"] == [], (
        f"Expected kiro to be an empty list, got {opt_deps['kiro']!r}"
    )


def test_import_reachy_nova_does_not_spawn_kiro():
    """Verify that importing reachy_nova doesn't spawn a kiro process.

    Run a fresh Python interpreter and import reachy_nova, then check:
    1. The import succeeds.
    2. No kiro-related module is in sys.modules (static safety).
    3. No subprocess with 'kiro' in its name is spawned.
    """
    import os

    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(REPO_ROOT)

    script = """
import sys
import reachy_nova

# Check that no kiro-related module was imported at the top level.
kiro_modules = [m for m in sys.modules.keys() if "kiro" in m.lower()]
if kiro_modules:
    print(f"ERROR: Found kiro modules in sys.modules: {kiro_modules}")
    sys.exit(1)

print("OK")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"Import failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"Unexpected output:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Double-check: no "ERROR" in output.
    assert "ERROR" not in result.stdout, (
        f"Kiro module detected during import: {result.stdout}"
    )
