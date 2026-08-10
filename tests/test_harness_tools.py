"""The Sonic tool registry over the intents spool (task t9).

Every test here runs against a temporary state dir (``REACHY_STATE_DIR``) with
NO behavior engine present, so the two halves of the contract are both
exercised: a fake "engine" (a helper thread that drains the commands dir and
writes a result file) proves the confirmed path, and its absence proves the
degraded "submitted but unconfirmed" path — including that the command file
STAYS on disk so a later-started engine still applies it.

Nothing here imports ``reachy`` (the reachy-mini-cli package) or
``reachy_mini``: file paths are the entire wire contract, and a test that
imported the peer would stop testing the contract and start testing the peer.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time

import pytest

from reachy_nova.harness import statedir
from reachy_nova.harness.tools import (
    DEGRADED_NOTE,
    TOOL_SPECS,
    IntentTools,
)

# The exact tool set — a seventh tool is a deliberate widening, never incidental.
EXPECTED_TOOLS = (
    "run_behavior",
    "declare_goal",
    "set_mode",
    "set_inhibition",
    "goto",
    "create_rule",
)

#: ``<time.time_ns()>-<uuid4.hex>.json``
SPOOL_NAME_RE = re.compile(r"^\d{10,}-[0-9a-f]{32}\.json$")


# --------------------------------------------------------------------------- #
# Fixtures + the fake engine                                                  #
# --------------------------------------------------------------------------- #


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    return tmp_path


@pytest.fixture
def tools(state_dir):
    # A short await timeout keeps the "no engine" tests fast; the fake engine
    # answers well inside it.
    return IntentTools(await_timeout=0.15)


class FakeEngine:
    """Drain the intents (or reload) spool and answer, like the real engine.

    Used as a context manager so the thread's lifetime is bounded by the test.
    """

    def __init__(self, commands_dir, results_dir, response=None):
        self.commands_dir = commands_dir
        self.results_dir = results_dir
        self.response = response
        self.seen: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.commands_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)
        return False

    def _answer(self, payload: dict) -> dict:
        if self.response is not None:
            return {**self.response, "cmd_id": payload.get("cmd_id")}
        out = {"ok": True, "cmd_id": payload.get("cmd_id")}
        if "op" in payload:
            out["op"] = payload["op"]
        return out

    def _run(self):
        while not self._stop.is_set():
            for path in sorted(self.commands_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                path.unlink(missing_ok=True)
                self.seen.append(payload)
                cmd_id = payload.get("cmd_id")
                if cmd_id:
                    (self.results_dir / f"{cmd_id}.json").write_text(
                        json.dumps(self._answer(payload)), encoding="utf-8"
                    )
            self._stop.wait(0.005)


def intents_engine(**kwargs):
    return FakeEngine(statedir.intents_commands_dir(), statedir.intents_results_dir(), **kwargs)


def reload_engine(**kwargs):
    return FakeEngine(statedir.reload_commands_dir(), statedir.reload_results_dir(), **kwargs)


def spooled():
    d = statedir.intents_commands_dir()
    return sorted(d.glob("*.json")) if d.is_dir() else []


def spool_payloads():
    return [json.loads(p.read_text(encoding="utf-8")) for p in spooled()]


#: One valid arguments dict per spool-backed tool.
VALID_ARGS = {
    "run_behavior": {"name": "nod", "duration": 2.0},
    "declare_goal": {"goal": "feel-alive"},
    "set_mode": {"mode": "calm"},
    "set_inhibition": {"behaviors": ["nod"]},
    "goto": {"head": {"pitch": 5.0}, "duration": 1.0},
}


# --------------------------------------------------------------------------- #
# Tool specs                                                                  #
# --------------------------------------------------------------------------- #


def test_tool_specs_are_exactly_the_six_tools():
    names = tuple(spec["toolSpec"]["name"] for spec in TOOL_SPECS)
    assert names == EXPECTED_TOOLS


def test_tool_specs_are_sonic_shaped():
    for spec in TOOL_SPECS:
        assert set(spec) == {"toolSpec"}
        tool = spec["toolSpec"]
        assert set(tool) == {"name", "description", "inputSchema"}
        assert tool["description"].strip()
        # Sonic's toolConfiguration carries the schema as a JSON *string*.
        assert set(tool["inputSchema"]) == {"json"}
        schema = json.loads(tool["inputSchema"]["json"])
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict)


# --------------------------------------------------------------------------- #
# The confirmed path — a fake engine answers                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tool_name", sorted(VALID_ARGS))
def test_execute_returns_engine_confirmation(tools, tool_name):
    with intents_engine() as engine:
        result = json.loads(tools.execute(tool_name, VALID_ARGS[tool_name]))
    assert result["ok"] is True
    assert result["op"] == tool_name
    assert [p["op"] for p in engine.seen] == [tool_name]


@pytest.mark.parametrize("tool_name", sorted(VALID_ARGS))
def test_execute_surfaces_an_engine_rejection(tools, tool_name):
    with intents_engine(response={"ok": False, "error": "engine says no"}):
        result = json.loads(tools.execute(tool_name, VALID_ARGS[tool_name]))
    assert result["ok"] is False
    assert result["error"] == "engine says no"


def test_await_result_deletes_the_result_file(tools):
    with intents_engine():
        cmd_id = tools.submit({"op": "set_mode", "mode": None})
        assert tools.await_result(cmd_id) is not None
    assert not (statedir.intents_results_dir() / f"{cmd_id}.json").exists()


# --------------------------------------------------------------------------- #
# The degraded path — no engine at all                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tool_name", sorted(VALID_ARGS))
def test_execute_degrades_when_no_engine_confirms(tools, tool_name):
    result = json.loads(tools.execute(tool_name, VALID_ARGS[tool_name]))
    assert result["ok"] is None
    assert result["note"] == DEGRADED_NOTE
    # The command stays spooled: a later engine still applies it.
    payloads = spool_payloads()
    assert [p["cmd_id"] for p in payloads] == [result["submitted"]]


def test_await_result_returns_none_without_an_engine(tools):
    assert tools.await_result("deadbeef" * 4) is None


# --------------------------------------------------------------------------- #
# Spool file format                                                           #
# --------------------------------------------------------------------------- #


def test_spool_file_name_is_ns_prefixed_and_ordered(tools):
    tools.execute("set_mode", {"mode": "one"})
    tools.execute("set_mode", {"mode": "two"})
    names = [p.name for p in spooled()]
    assert len(names) == 2
    for name in names:
        assert SPOOL_NAME_RE.match(name), name
    # sorted() over the spool == submission order (the ns prefix's whole job).
    assert [json.loads(p.read_text())["mode"] for p in spooled()] == ["one", "two"]


def test_spool_write_leaves_no_partial_files(tools):
    tools.execute("set_mode", {"mode": "calm"})
    leftovers = [p.name for p in statedir.intents_commands_dir().iterdir() if ".tmp." in p.name]
    assert leftovers == []


def test_run_behavior_payload_fields_exact(tools):
    tools.execute(
        "run_behavior", {"name": "nod", "params": {"amp": 8}, "duration": 2, "loop": True}
    )
    (payload,) = spool_payloads()
    assert set(payload) == {"cmd_id", "op", "name", "params", "lifetime"}
    assert payload["op"] == "run_behavior"
    assert payload["name"] == "nod"
    assert payload["params"] == {"amp": 8.0}
    assert payload["lifetime"] == {"looping": True, "duration": 2.0}


def test_run_behavior_omits_unspecified_lifetime_keys(tools):
    """An omitted loop/duration must fall through to the library entry default.

    The engine resolves a missing key against the behavior's own default; a key
    we invent here (``looping: false``) would silently override that default and
    turn every unqualified call into a one-shot with no duration, which the
    engine then refuses.
    """
    tools.execute("run_behavior", {"name": "nod"})
    (payload,) = spool_payloads()
    assert payload["lifetime"] == {}


def test_declare_goal_payload_fields_exact(tools):
    tools.execute("declare_goal", {"goal": "gaze-hold", "params": {"yaw": 10}})
    (payload,) = spool_payloads()
    assert set(payload) == {"cmd_id", "op", "goal", "params"}
    assert payload["goal"] == "gaze-hold"
    assert payload["params"] == {"yaw": 10.0}


def test_declare_goal_clears_with_a_null_goal(tools):
    tools.execute("declare_goal", {})
    (payload,) = spool_payloads()
    assert payload["goal"] is None
    assert payload["params"] == {}


def test_set_mode_clears_with_a_null_mode(tools):
    tools.execute("set_mode", {})
    (payload,) = spool_payloads()
    assert set(payload) == {"cmd_id", "op", "mode"}
    assert payload["mode"] is None


def test_set_inhibition_replaces_the_whole_set(tools):
    tools.execute("set_inhibition", {"behaviors": []})
    (payload,) = spool_payloads()
    assert set(payload) == {"cmd_id", "op", "behaviors"}
    assert payload["behaviors"] == []


def test_goto_payload_carries_only_the_named_channels(tools):
    tools.execute(
        "goto",
        {"head": {"pitch": 5}, "antennas": [10, -10], "duration": 1.5, "label": "peek"},
    )
    (payload,) = spool_payloads()
    assert set(payload) == {"cmd_id", "op", "label", "head", "antennas", "duration"}
    assert payload["head"] == {"pitch": 5.0}
    assert payload["antennas"] == [10.0, -10.0]
    assert payload["duration"] == 1.5
    assert payload["label"] == "peek"


def test_submit_accepts_a_raw_op_payload(tools):
    cmd_id = tools.submit({"op": "goto", "body_yaw": 3.0, "duration": 1.0})
    (payload,) = spool_payloads()
    assert payload == {"cmd_id": cmd_id, "op": "goto", "body_yaw": 3.0, "duration": 1.0}


def test_explicit_dirs_override_the_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path / "unused"))
    cmds = tmp_path / "c"
    results = tmp_path / "r"
    tools = IntentTools(commands_dir=cmds, results_dir=results, await_timeout=0.05)
    tools.execute("set_mode", {"mode": "calm"})
    assert len(list(cmds.glob("*.json"))) == 1
    assert not (tmp_path / "unused").exists()


# --------------------------------------------------------------------------- #
# Fail-closed refusals — nothing reaches the spool                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("duration", [0, -1, 11, 10.5])
def test_goto_duration_out_of_bounds_is_refused_without_spooling(tools, duration):
    result = json.loads(tools.execute("goto", {"head": {"pitch": 5.0}, "duration": duration}))
    assert result["ok"] is False
    assert "duration" in result["error"]
    assert spooled() == []


def test_goto_accepts_the_upper_duration_bound(tools):
    tools.execute("goto", {"head": {"pitch": 5.0}, "duration": 10.0})
    assert len(spooled()) == 1


REFUSED_CALLS = [
    ("goto", {"duration": 1.0}),  # names no channel at all
    ("goto", {"head": {"nose": 1.0}}),  # unknown head axis
    ("goto", {"head": {"pitch": "far"}}),  # non-numeric axis
    ("goto", {"head": {"pitch": 1.0}, "wiggle": 3}),  # unknown field
    ("goto", {"antennas": [1.0]}),  # not a [right, left] pair
    ("goto", {"body_yaw": True}),  # bool is not a number
    ("run_behavior", {}),  # no behavior name
    ("run_behavior", {"name": "nod", "params": {"amp": "big"}}),
    ("run_behavior", {"name": "nod", "params": []}),
    ("run_behavior", {"name": "nod", "duration": "long"}),
    ("declare_goal", {"goal": 7}),
    ("set_mode", {"mode": 7}),
    ("set_inhibition", {}),  # 'behaviors' is required
    ("set_inhibition", {"behaviors": "nod"}),
    ("set_inhibition", {"behaviors": [3]}),
]


@pytest.mark.parametrize("tool_name,args", REFUSED_CALLS)
def test_bad_arguments_are_refused_without_spooling(tools, tool_name, args):
    result = json.loads(tools.execute(tool_name, args))
    assert result["ok"] is False
    assert result["error"]
    assert spooled() == []


def test_unknown_tool_is_an_error_payload(tools):
    result = json.loads(tools.execute("self_destruct", {"now": True}))
    assert result["ok"] is False
    assert "self_destruct" in result["error"]
    assert spooled() == []


def test_non_dict_params_are_refused(tools):
    result = json.loads(tools.execute("set_mode", ["calm"]))
    assert result["ok"] is False
    assert spooled() == []


# --------------------------------------------------------------------------- #
# Observability — one SENSE line per execute, whatever the outcome            #
# --------------------------------------------------------------------------- #


def _sense_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "nova.sensory"]


def test_confirmed_execute_logs_one_sense_line(tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        with intents_engine():
            tools.execute("set_mode", {"mode": "calm"})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=set_mode]" in line
    assert "confirmed" in line


def test_degraded_execute_logs_one_sense_line(tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        tools.execute("set_mode", {"mode": "calm"})
    (line,) = _sense_lines(caplog)
    assert "degraded" in line
    assert "submitted=" in line


def test_refused_execute_logs_one_sense_line(tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        tools.execute("goto", {"head": {"pitch": 1.0}, "duration": 99})
    (line,) = _sense_lines(caplog)
    assert "refused" in line
    assert "duration" in line


def test_unknown_tool_logs_one_sense_line(tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        tools.execute("nope", {})
    (line,) = _sense_lines(caplog)
    assert "refused" in line


# --------------------------------------------------------------------------- #
# create_rule routes through the overlay, not the intents spool               #
# --------------------------------------------------------------------------- #

RULE_ARGS = {
    "id": "nova-pat-nod",
    "when": {"field": "pat", "op": "gt", "value": 0.5},
    "run": "nod",
    "duration_s": 2.0,
}


def test_create_rule_writes_the_overlay_and_reports_the_reload_verdict(tools):
    with reload_engine() as engine:
        result = json.loads(tools.execute("create_rule", RULE_ARGS))
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["rule_id"] == "nova-pat-nod"
    assert "confirmed" in result["reload"]
    assert len(engine.seen) == 1
    assert "nova-pat-nod" in statedir.rules_overlay_path().read_text()
    # Never the intents spool.
    assert spooled() == []


def test_create_rule_degrades_when_no_engine_reloads(tools):
    result = json.loads(tools.execute("create_rule", RULE_ARGS))
    assert result["ok"] is True
    assert "not confirmed" in result["reload"]
    assert statedir.rules_overlay_path().is_file()


def test_create_rule_refusal_is_an_error_payload(tools):
    result = json.loads(tools.execute("create_rule", {**RULE_ARGS, "id": "sneaky"}))
    assert result["ok"] is False
    assert "nova-" in result["error"]
    assert not statedir.rules_overlay_path().exists()


def test_create_rule_is_idempotent_through_execute(tools):
    first = json.loads(tools.execute("create_rule", RULE_ARGS))
    second = json.loads(tools.execute("create_rule", RULE_ARGS))
    assert first["changed"] is True
    assert second["changed"] is False
    assert statedir.rules_overlay_path().read_text().count("[[react]]") == 1


# --------------------------------------------------------------------------- #
# Timeout behaviour                                                           #
# --------------------------------------------------------------------------- #


def test_await_timeout_is_bounded(state_dir):
    tools = IntentTools(await_timeout=0.1)
    started = time.monotonic()
    tools.execute("set_mode", {"mode": "calm"})
    assert time.monotonic() - started < 1.0
