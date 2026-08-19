"""Shape tests for config/kiro/nova-writer.json (task t5).

The nova-writer agent config is a Kiro agent-config JSON file, provisioned
onto the device at ~/.kiro/agents/nova-writer.json (see
scripts/install-device-units.sh), that grants the on-device Kiro writer
engine the full read/write/shell tool surface as user pollen. This test
proves the shipped config is well-formed and matches kiro's agent_config
schema closely enough to load: valid JSON, required keys with correct
types, the full tool set repeated verbatim in allowedTools (so nothing
prompts for permission), an empty mcpServers, and no hardcoded model.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "kiro" / "nova-writer.json"

EXPECTED_TOOLS = {
    "read",
    "write",
    "shell",
    "aws",
    "report",
    "introspect",
    "knowledge",
    "thinking",
    "todo",
    "delegate",
    "grep",
    "glob",
}


def _load() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_config_file_exists() -> None:
    assert CONFIG_PATH.is_file(), f"missing {CONFIG_PATH}"


def test_config_is_valid_json() -> None:
    # _load() itself raises on malformed JSON; getting a dict back is proof.
    data = _load()
    assert isinstance(data, dict)


def test_required_keys_present_with_correct_types() -> None:
    data = _load()
    expected_types = {
        "name": str,
        "description": str,
        "prompt": (str, type(None)),
        "mcpServers": dict,
        "tools": list,
        "toolAliases": dict,
        "allowedTools": list,
        "resources": list,
        "toolsSettings": dict,
        "includeMcpJson": bool,
        "model": (str, type(None)),
    }
    for key, expected_type in expected_types.items():
        assert key in data, f"missing required key {key!r}"
        assert isinstance(data[key], expected_type), (
            f"{key!r} has type {type(data[key]).__name__}, expected {expected_type}"
        )


def test_name_is_nova_writer() -> None:
    data = _load()
    assert data["name"] == "nova-writer"


def test_prompt_is_a_nonempty_writer_role_system_prompt() -> None:
    data = _load()
    assert data["prompt"], "prompt must not be empty/null"
    assert isinstance(data["prompt"], str)


def test_full_tool_set_granted() -> None:
    data = _load()
    assert set(data["tools"]) == EXPECTED_TOOLS


def test_every_allowed_tool_appears_in_tools() -> None:
    data = _load()
    tools = set(data["tools"])
    allowed = set(data["allowedTools"])
    assert allowed, "allowedTools must not be empty"
    assert allowed <= tools, (
        f"allowedTools contains entries not present in tools: {allowed - tools}"
    )


def test_allowed_tools_matches_full_tool_set() -> None:
    # The whole point of the config is that nothing prompts for permission —
    # allowedTools should carry the same full set as tools, not a subset.
    data = _load()
    assert set(data["allowedTools"]) == EXPECTED_TOOLS


def test_model_is_null() -> None:
    # Model selection happens per-session via --model (default minimax-m2.5),
    # not hardcoded in the agent config.
    data = _load()
    assert data["model"] is None


def test_mcp_servers_empty() -> None:
    data = _load()
    assert data["mcpServers"] == {}


def test_include_mcp_json_is_false() -> None:
    data = _load()
    assert data["includeMcpJson"] is False
