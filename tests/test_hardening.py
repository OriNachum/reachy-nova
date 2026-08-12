"""Tests for the P0 security hardening deliverables (task t13).

Covers:
  - scripts/harden-robot.sh exists, is executable (both on disk and as the
    mode git will record for it), and contains the three required checks
    (chmod 600 on the .env, the ss/:1883 mosquitto-bind check, and the
    residual-exposure summary).
  - docs/security.md's fenced IAM policy JSON parses and references only the
    Bedrock model IDs the harness actually invokes (reachy_nova/config.py):
    amazon.nova-2-sonic-v1:0, us.amazon.nova-2-lite-v1:0 (an inference
    profile — its underlying, unprefixed foundation-model ID is the same
    "amazon.nova-2-lite-v1:0" string and is expected to appear too), and
    amazon.nova-2-multimodal-embeddings-v1:0.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "harden-robot.sh"
DOC_PATH = REPO_ROOT / "docs" / "security.md"

# The exact model IDs the harness invokes, per reachy_nova/config.py.
# Cross-region inference profile IDs ("us."-prefixed) fan out to an
# underlying foundation model of the same unprefixed name, so that
# unprefixed form is an expected/allowed appearance too, not an extraneous
# model.
ALLOWED_MODEL_BASENAMES = {
    "amazon.nova-2-sonic-v1:0",
    "amazon.nova-2-lite-v1:0",
    "amazon.nova-2-multimodal-embeddings-v1:0",
}

# Matches Bedrock-style Nova model IDs, with or without a leading
# cross-region-inference-profile prefix like "us.".
MODEL_ID_RE = re.compile(r"(?:[a-z]{2}\.)?amazon\.nova-[a-z0-9.-]*-v\d+:\d+")


def _strip_profile_prefix(model_id: str) -> str:
    """Drop a leading two-letter cross-region-inference-profile prefix."""
    parts = model_id.split(".", 1)
    if len(parts) == 2 and len(parts[0]) == 2 and parts[0].isalpha():
        return parts[1]
    return model_id


def test_harden_script_exists_and_is_executable_on_disk() -> None:
    assert SCRIPT_PATH.is_file(), f"missing {SCRIPT_PATH}"
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, f"{SCRIPT_PATH} is not executable on disk (mode {oct(mode)})"


def test_harden_script_is_executable_in_git() -> None:
    """git must record the executable bit, not just the filesystem."""
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", "scripts/harden-robot.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()
    assert output, (
        "scripts/harden-robot.sh is not staged/tracked in git yet - "
        "run `git add scripts/harden-robot.sh` (with the executable bit) before testing"
    )
    git_mode = output.split()[0]
    assert git_mode == "100755", (
        f"git records scripts/harden-robot.sh as mode {git_mode}, expected 100755 (executable)"
    )


def test_harden_script_uses_set_euo_pipefail() -> None:
    text = SCRIPT_PATH.read_text()
    assert "set -euo pipefail" in text


def test_harden_script_chmods_env_to_600() -> None:
    text = SCRIPT_PATH.read_text()
    assert "chmod 600" in text


def test_harden_script_checks_mosquitto_1883_binding() -> None:
    text = SCRIPT_PATH.read_text()
    assert "ss -tlnp" in text
    assert ":1883" in text


def test_harden_script_prints_residual_exposure_summary() -> None:
    text = SCRIPT_PATH.read_text().lower()
    assert "residual" in text
    assert ":8000" in SCRIPT_PATH.read_text()
    assert "7447" in SCRIPT_PATH.read_text()


def test_harden_script_requires_no_sudo() -> None:
    """No line actually invokes sudo (mentioning "no sudo" in a comment is fine)."""
    lines = SCRIPT_PATH.read_text().splitlines()
    sudo_invocations = [
        line for line in lines if re.match(r"^\s*sudo\b", line) or " sudo " in line
    ]
    assert not sudo_invocations, f"script invokes sudo: {sudo_invocations}"


def _extract_fenced_json_blocks(markdown_text: str) -> list[str]:
    return re.findall(r"```json\n(.*?)```", markdown_text, flags=re.DOTALL)


def test_security_doc_exists() -> None:
    assert DOC_PATH.is_file(), f"missing {DOC_PATH}"


def test_security_doc_policy_json_parses() -> None:
    text = DOC_PATH.read_text()
    blocks = _extract_fenced_json_blocks(text)
    assert blocks, "docs/security.md has no fenced ```json block for the IAM policy"

    policy = json.loads(blocks[0])
    assert policy.get("Version") == "2012-10-17"
    assert policy.get("Statement"), "IAM policy has no Statement entries"


def test_security_doc_policy_references_only_invoked_model_ids() -> None:
    text = DOC_PATH.read_text()
    blocks = _extract_fenced_json_blocks(text)
    assert blocks, "docs/security.md has no fenced ```json block for the IAM policy"
    policy_text = blocks[0]

    found_ids = set(MODEL_ID_RE.findall(policy_text))
    assert found_ids, "IAM policy JSON references no Bedrock model IDs at all"

    for model_id in found_ids:
        base = _strip_profile_prefix(model_id)
        assert base in ALLOWED_MODEL_BASENAMES, (
            f"IAM policy references unexpected model id {model_id!r} "
            f"(not one of {sorted(ALLOWED_MODEL_BASENAMES)})"
        )

    # And every allowed model must actually be present - "scoped to exactly
    # the invoked models" means neither over- nor under-inclusive.
    found_basenames = {_strip_profile_prefix(m) for m in found_ids}
    missing = ALLOWED_MODEL_BASENAMES - found_basenames
    assert not missing, f"IAM policy is missing required model id(s): {sorted(missing)}"


def test_security_doc_covers_env_and_mosquitto_and_residual_sections() -> None:
    text = DOC_PATH.read_text().lower()
    assert "chmod 600" in text
    assert "ss -tlnp" in text
    assert "127.0.0.1" in text
    assert "residual" in text
    assert ":8000" in DOC_PATH.read_text()
    assert "7447" in DOC_PATH.read_text()
