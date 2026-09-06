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
import sys
import threading
import time

import pytest

from reachy_nova.harness import statedir
from reachy_nova.harness import tools as tools_module
from reachy_nova.harness.daemon_client import restore_volume
from reachy_nova.harness.quiet import QuietState
from reachy_nova.harness.attention import AttentionState
from reachy_nova.harness.tools import (
    BROWSE_DISABLED_REASON,
    BROWSE_NOT_WIRED_REASON,
    COLD_REFUSED_TOOLS,
    DEFAULT_RECALL_SENSES_N,
    DEFAULT_VOICE_STEP,
    DEGRADED_NOTE,
    ENROLL_FACE_TARGET,
    ENROLL_NAME_INVALID_REASON,
    ENROLL_NAME_TOO_LONG_REASON,
    ENROLL_NAME_UNPRINTABLE_REASON,
    ENROLL_OP,
    HISTORY_NOT_WIRED_REASON,
    MAX_ENROLL_NAME_LEN,
    MAX_QUIET_MINUTES,
    MAX_RECALL_SENSES_N,
    MAX_VOICE_LEVEL,
    MIN_QUIET_MINUTES,
    MIN_RECALL_SENSES_N,
    MIN_VOICE_LEVEL,
    NOT_ADDRESSED_REASON,
    QUIET_NOT_WIRED_REASON,
    THINK_BEHAVIOR,
    THINK_DURATION_S,
    THINK_SIDE_INVALID_REASON,
    THINK_YAW_LEFT,
    THINK_YAW_RIGHT,
    TOOL_SPECS,
    VOICE_NOT_WIRED_REASON,
    IntentTools,
)

# The exact tool set — each addition past the original six is a deliberate
# widening, never incidental. ``browse`` (task t4) is the first, ``enroll_face``
# (task t8) the second.
EXPECTED_TOOLS = (
    "run_behavior",
    "declare_goal",
    "set_mode",
    "set_inhibition",
    "goto",
    "create_rule",
    "browse",
    "enroll_face",
    # the gaze-lock pair (task t9) — spool-backed like the original five,
    # riding the runtime's lock_face/release_face intent kinds.
    "lock_face",
    "release_face",
    # the gaze ALIAS pair (live finding L4) — thin run_behavior aliases under
    # the underscored names Sonic actually reached for on the robot.
    "look_at_face",
    "look_at_sound",
    # think (task t7) — published like look_at_face for the same reason
    # (finding L4): a thin run_behavior alias over the runtime's 'thoughtful'.
    "think",
    # the kiro-writer pair (deviation d1) — refused unless a ForgeLeg is wired
    "forge",
    "use_skill",
    # author_rule (qodo review comment 3812045168): the kiro-authored-rule
    # pipeline's production caller, refused the same way as forge/use_skill.
    "author_rule",
    # voice-level tools (task t10) — dispatched locally against the daemon
    # client, never through the intents spool.
    "raise_voice",
    "lower_voice",
    "set_voice_level",
    # recall_senses (task t8) — reads the SenseHistory ring buffer directly,
    # never through the intents spool.
    "recall_senses",
    # the timed-quiet pair (task t12) — mind-side quiet (QuietState) AND a
    # merged set_inhibition that closes the body's own 'speak' mouth.
    "stay_silent",
    "end_silence",
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


#: One valid arguments dict per spool-backed tool whose wire ``op`` equals its
#: tool name. ``enroll_face`` is spool-backed too but rides the runtime's
#: ``enroll`` op under a different tool name, so it gets its own section below.
VALID_ARGS = {
    "run_behavior": {"name": "nod", "duration": 2.0},
    "declare_goal": {"goal": "feel-alive"},
    "set_mode": {"mode": "calm"},
    "set_inhibition": {"behaviors": ["nod"]},
    "goto": {"head": {"pitch": 5.0}, "duration": 1.0},
    "lock_face": {},
    "release_face": {},
}


# --------------------------------------------------------------------------- #
# Tool specs                                                                  #
# --------------------------------------------------------------------------- #


def test_tool_specs_are_exactly_the_expected_tools():
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


# --------------------------------------------------------------------------- #
# browse — drives NovaBrowser directly, not the intents spool (task t4)       #
# --------------------------------------------------------------------------- #


class FakeBrowser:
    """A minimal ``queue_task``/``on_progress`` double.

    Proves ``IntentTools`` drives NovaBrowser's own interface without ever
    needing a real (nova_act/playwright-backed) instance in these tests.
    """

    def __init__(self):
        self.queued: list[tuple[str, str | None]] = []
        self.on_progress = None

    def queue_task(self, instruction, url=None):
        self.queued.append((instruction, url))


def test_importing_tools_does_not_import_nova_act_or_playwright():
    # reachy_nova.harness.tools is already imported (module-level, above);
    # confirm that alone never dragged in nova_act/playwright. tools.py only
    # takes nova_browser's flag-checking act_enabled() — never a code path
    # that imports the automation libraries themselves.
    assert "nova_act" not in sys.modules
    assert "playwright" not in sys.modules


def test_browse_tool_spec_requires_instruction():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "browse"]
    schema = json.loads(spec["toolSpec"]["inputSchema"]["json"])
    assert schema["required"] == ["instruction"]
    assert set(schema["properties"]) == {"instruction", "url"}


def test_browse_is_refused_when_nova_act_is_disabled(tools, monkeypatch):
    monkeypatch.delenv("NOVA_ACT_ENABLED", raising=False)
    result = json.loads(tools.execute("browse", {"instruction": "find a recipe"}))
    assert result["ok"] is False
    assert result["error"] == BROWSE_DISABLED_REASON
    assert spooled() == []


def test_browse_is_refused_when_enabled_but_no_browser_wired(state_dir, monkeypatch):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    tools = IntentTools(await_timeout=0.05)
    result = json.loads(tools.execute("browse", {"instruction": "find a recipe"}))
    assert result["ok"] is False
    assert result["error"] == BROWSE_NOT_WIRED_REASON


def test_browse_queues_the_instruction_when_enabled_and_wired(state_dir, monkeypatch):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    browser = FakeBrowser()
    tools = IntentTools(await_timeout=0.05, browser=browser)

    result = json.loads(
        tools.execute(
            "browse", {"instruction": "find a recipe", "url": "https://example.com"}
        )
    )

    assert result == {
        "ok": True,
        "queued": True,
        "instruction": "find a recipe",
        "url": "https://example.com",
    }
    assert browser.queued == [("find a recipe", "https://example.com")]
    # Never spooled — browse talks to NovaBrowser directly, not the engine.
    assert spooled() == []


def test_browse_queues_without_a_url(state_dir, monkeypatch):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "yes")
    browser = FakeBrowser()
    tools = IntentTools(await_timeout=0.05, browser=browser)

    result = json.loads(tools.execute("browse", {"instruction": "find a recipe"}))

    assert result["ok"] is True
    assert result["url"] is None
    assert browser.queued == [("find a recipe", None)]


def test_browse_progress_callback_is_invocable(state_dir, monkeypatch):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    browser = FakeBrowser()
    seen = []
    tools = IntentTools(await_timeout=0.05, browser=browser, on_browse_progress=seen.append)

    # IntentTools wires the callback onto the browser handle at construction —
    # simulate NovaBrowser itself invoking it mid-task. (Bound methods aren't
    # identity-stable across attribute access, so compare by equality.)
    assert browser.on_progress == seen.append
    browser.on_progress("Opening browser...")
    assert seen == ["Opening browser..."]


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"instruction": ""},
        {"instruction": "   "},
        {"instruction": 5},
        {"instruction": "go", "url": 5},
    ],
)
def test_browse_bad_arguments_are_refused_without_touching_the_browser(
    state_dir, monkeypatch, args
):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    browser = FakeBrowser()
    tools = IntentTools(await_timeout=0.05, browser=browser)

    result = json.loads(tools.execute("browse", args))

    assert result["ok"] is False
    assert result["error"]
    assert browser.queued == []


def test_browse_logs_one_sense_line_on_refusal(tools, caplog, monkeypatch):
    monkeypatch.delenv("NOVA_ACT_ENABLED", raising=False)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        tools.execute("browse", {"instruction": "find a recipe"})
    (line,) = _sense_lines(caplog)
    assert "refused" in line
    assert "browsing is disabled" in line


# --------------------------------------------------------------------------- #
# enroll_face — rides the runtime's enroll seam over the intents spool (t8)    #
# --------------------------------------------------------------------------- #
#
# The runtime (reachy-mini-cli) owns face recognition and the FaceStore; the
# wire contract requested in agentculture/reachy-mini-cli#166 is one more op on
# the SAME intents spool. These tests therefore assert two separable things:
# the exact bytes we put on the spool (which is the whole ask of the issue), and
# that whatever comes back — success, a typed refusal, an unknown-op refusal
# from a runtime that predates the seam, or nothing at all — reaches the model
# unchanged. Nothing here assumes the seam has shipped.


ENROLL_ARGS = {"name": "Ori"}


def test_enroll_face_tool_spec_takes_only_a_required_name():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "enroll_face"]
    schema = json.loads(spec["toolSpec"]["inputSchema"]["json"])
    assert schema["required"] == ["name"]
    assert set(schema["properties"]) == {"name"}
    assert schema["properties"]["name"]["type"] == "string"


def test_enroll_face_tool_spec_tells_nova_when_to_use_it():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "enroll_face"]
    description = spec["toolSpec"]["description"].lower()
    # The description is the ONLY thing that makes the model reach for this on
    # "I'm Ori" — it must name both halves of the trigger (being told a name,
    # while a face is visible), not just describe enrollment mechanics.
    assert "name" in description
    assert "face" in description


def test_enroll_face_spools_the_agreed_enroll_command(tools):
    tools.execute("enroll_face", ENROLL_ARGS)
    (payload,) = spool_payloads()
    assert set(payload) == {"cmd_id", "op", "name", "face"}
    assert payload["op"] == ENROLL_OP == "enroll"
    assert payload["name"] == "Ori"
    assert payload["face"] == ENROLL_FACE_TARGET == "current"


def test_enroll_face_returns_the_engines_success_verbatim(tools):
    answer = {"ok": True, "id": "face-7", "name": "Ori"}
    with intents_engine(response=answer) as engine:
        result = json.loads(tools.execute("enroll_face", ENROLL_ARGS))
    assert result["ok"] is True
    assert result["id"] == "face-7"
    assert result["name"] == "Ori"
    (seen,) = engine.seen
    assert seen["op"] == "enroll"
    assert seen["name"] == "Ori"
    assert seen["face"] == "current"


@pytest.mark.parametrize(
    "error",
    [
        "no-recent-unknown-face",
        "vision-unavailable",
        "invalid name",
        # A runtime that predates the seam (issue #166 unshipped) answers with
        # its standard unknown-op refusal — which must reach the model as a
        # refusal, never as a success and never as a harness crash.
        "unknown op 'enroll'",
    ],
)
def test_enroll_face_surfaces_the_engines_typed_refusal(tools, error):
    with intents_engine(response={"ok": False, "error": error}):
        result = json.loads(tools.execute("enroll_face", ENROLL_ARGS))
    assert result["ok"] is False
    assert result["error"] == error


def test_enroll_face_degrades_when_no_engine_confirms(tools):
    """No seam, no engine, no answer — the not-confirmed shape, command retained."""
    result = json.loads(tools.execute("enroll_face", ENROLL_ARGS))
    assert result["ok"] is None
    assert result["note"] == DEGRADED_NOTE
    payloads = spool_payloads()
    assert [p["cmd_id"] for p in payloads] == [result["submitted"]]
    assert payloads[0]["op"] == "enroll"


def test_enroll_face_strips_surrounding_whitespace(tools):
    tools.execute("enroll_face", {"name": "  Ori  "})
    (payload,) = spool_payloads()
    assert payload["name"] == "Ori"


def test_enroll_face_accepts_the_upper_name_length_bound(tools):
    name = "o" * MAX_ENROLL_NAME_LEN
    tools.execute("enroll_face", {"name": name})
    (payload,) = spool_payloads()
    assert payload["name"] == name


REFUSED_ENROLL_CALLS = [
    ({}, ENROLL_NAME_INVALID_REASON),
    ({"name": ""}, ENROLL_NAME_INVALID_REASON),
    ({"name": "   "}, ENROLL_NAME_INVALID_REASON),
    ({"name": None}, ENROLL_NAME_INVALID_REASON),
    ({"name": 5}, ENROLL_NAME_INVALID_REASON),
    ({"name": ["Ori"]}, ENROLL_NAME_INVALID_REASON),
    ({"name": "o" * (MAX_ENROLL_NAME_LEN + 1)}, ENROLL_NAME_TOO_LONG_REASON),
    ({"name": "Ori\nDROP TABLE faces"}, ENROLL_NAME_UNPRINTABLE_REASON),
    ({"name": "Ori\x00"}, ENROLL_NAME_UNPRINTABLE_REASON),
]


@pytest.mark.parametrize("args,reason", REFUSED_ENROLL_CALLS)
def test_enroll_face_bad_names_are_refused_without_spooling(tools, args, reason):
    result = json.loads(tools.execute("enroll_face", args))
    assert result["ok"] is False
    assert result["error"] == reason
    # A pre-flight refusal never reaches the spool, so no engine can act on it.
    assert spooled() == []


def test_enroll_face_refusal_reasons_are_named_constants():
    """Each pre-flight refusal is a distinct, importable, non-empty constant."""
    reasons = (
        ENROLL_NAME_INVALID_REASON,
        ENROLL_NAME_TOO_LONG_REASON,
        ENROLL_NAME_UNPRINTABLE_REASON,
    )
    assert all(isinstance(r, str) and r.strip() for r in reasons)
    assert len(set(reasons)) == 3


def test_enroll_face_logs_one_sense_line_on_confirmation(tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        with intents_engine(response={"ok": True, "id": "face-7", "name": "Ori"}):
            tools.execute("enroll_face", ENROLL_ARGS)
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=enroll_face]" in line
    assert "confirmed" in line


def test_enroll_face_logs_one_sense_line_on_refusal(tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        tools.execute("enroll_face", {"name": ""})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=enroll_face]" in line
    assert "refused" in line


# --------------------------------------------------------------------------- #
# lock_face / release_face — spool-backed gaze lock (task t9)                 #
# --------------------------------------------------------------------------- #
#
# The runtime (reachy-mini-cli) owns the actual gaze lock; these two tools are
# plain no-argument ops on the SAME intents spool as run_behavior/goto/etc.
# Whatever the engine answers — a typed success, a named no-op note, or an
# older runtime's unknown-kind refusal — reaches the model unchanged.


def test_lock_face_tool_spec_takes_no_arguments():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "lock_face"]
    schema = json.loads(spec["toolSpec"]["inputSchema"]["json"])
    assert schema["required"] == []
    assert schema["properties"] == {}


def test_release_face_tool_spec_takes_no_arguments():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "release_face"]
    schema = json.loads(spec["toolSpec"]["inputSchema"]["json"])
    assert schema["required"] == []
    assert schema["properties"] == {}


def test_lock_face_tool_spec_describes_keeping_the_gaze():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "lock_face"]
    description = spec["toolSpec"]["description"].lower()
    assert "look" in description
    assert "stop" in description


def test_release_face_tool_spec_describes_looking_away():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "release_face"]
    description = spec["toolSpec"]["description"].lower()
    assert "look away" in description or "stop" in description


def test_lock_face_spools_the_agreed_op(tools):
    tools.execute("lock_face", {})
    (payload,) = spool_payloads()
    assert payload["op"] == "lock_face"


def test_release_face_spools_the_agreed_op(tools):
    tools.execute("release_face", {})
    (payload,) = spool_payloads()
    assert payload["op"] == "release_face"


def _assert_verbatim(result: dict, answer: dict) -> None:
    """The engine's typed result reaches the model verbatim (plus its own cmd_id)."""
    assert answer.items() <= result.items()


def test_lock_face_returns_the_engines_success_verbatim(tools):
    answer = {"ok": True, "op": "lock_face", "locked": True}
    with intents_engine(response=answer):
        result = json.loads(tools.execute("lock_face", {}))
    _assert_verbatim(result, answer)


def test_lock_face_already_locked_note_passes_through(tools):
    answer = {"ok": True, "op": "lock_face", "locked": True, "note": "already locked"}
    with intents_engine(response=answer):
        result = json.loads(tools.execute("lock_face", {}))
    _assert_verbatim(result, answer)


def test_lock_face_surfaces_no_face_known_refusal(tools):
    answer = {"ok": False, "error": "no face known"}
    with intents_engine(response=answer):
        result = json.loads(tools.execute("lock_face", {}))
    _assert_verbatim(result, answer)


def test_release_face_returns_the_engines_success_verbatim(tools):
    answer = {"ok": True, "op": "release_face", "released": True}
    with intents_engine(response=answer):
        result = json.loads(tools.execute("release_face", {}))
    _assert_verbatim(result, answer)


def test_release_face_inactive_lock_no_op_passes_through_unchanged(tools):
    """Releasing when nothing is locked is a named no-op, not a refusal."""
    answer = {"ok": True, "op": "release_face", "released": False, "note": "not locked"}
    with intents_engine(response=answer):
        result = json.loads(tools.execute("release_face", {}))
    _assert_verbatim(result, answer)


def test_lock_face_surfaces_an_older_runtimes_unknown_kind_refusal(tools):
    """An older runtime that predates the seam answers its unknown-kind refusal."""
    answer = {"ok": False, "error": "unknown kind 'lock_face'"}
    with intents_engine(response=answer):
        result = json.loads(tools.execute("lock_face", {}))
    _assert_verbatim(result, answer)


def test_release_face_surfaces_an_older_runtimes_unknown_kind_refusal(tools):
    answer = {"ok": False, "error": "unknown kind 'release_face'"}
    with intents_engine(response=answer):
        result = json.loads(tools.execute("release_face", {}))
    _assert_verbatim(result, answer)


def test_lock_face_degrades_when_no_engine_confirms(tools):
    result = json.loads(tools.execute("lock_face", {}))
    assert result["ok"] is None
    assert result["note"] == DEGRADED_NOTE
    payloads = spool_payloads()
    assert [p["cmd_id"] for p in payloads] == [result["submitted"]]
    assert payloads[0]["op"] == "lock_face"


def test_release_face_degrades_when_no_engine_confirms(tools):
    result = json.loads(tools.execute("release_face", {}))
    assert result["ok"] is None
    assert result["note"] == DEGRADED_NOTE
    payloads = spool_payloads()
    assert [p["cmd_id"] for p in payloads] == [result["submitted"]]
    assert payloads[0]["op"] == "release_face"


def test_lock_face_logs_one_sense_line_on_confirmation(tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        with intents_engine(response={"ok": True, "op": "lock_face", "locked": True}):
            tools.execute("lock_face", {})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=lock_face]" in line
    assert "confirmed" in line


def test_release_face_logs_one_sense_line_on_confirmation(tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        with intents_engine(response={"ok": True, "op": "release_face", "released": True}):
            tools.execute("release_face", {})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=release_face]" in line
    assert "confirmed" in line


# --------------------------------------------------------------------------- #
# lock_face/release_face mirror the confirmed verdict into LockState (t13)    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tools_with_lock(state_dir):
    from reachy_nova.harness.lock_state import LockState

    lock_state = LockState()
    return IntentTools(await_timeout=0.15, lock_state=lock_state), lock_state


def test_confirmed_lock_face_marks_the_belief_locked(tools_with_lock):
    tools, lock_state = tools_with_lock
    with intents_engine(response={"ok": True, "op": "lock_face", "locked": True}):
        tools.execute("lock_face", {})
    assert lock_state.locked is True


def test_confirmed_release_face_marks_the_belief_released(tools_with_lock):
    tools, lock_state = tools_with_lock
    lock_state.mark_locked()
    with intents_engine(response={"ok": True, "op": "release_face", "released": True}):
        tools.execute("release_face", {})
    assert lock_state.locked is False


def test_refused_lock_face_does_not_touch_the_belief(tools_with_lock):
    tools, lock_state = tools_with_lock
    with intents_engine(response={"ok": False, "error": "no face known"}):
        tools.execute("lock_face", {})
    assert lock_state.locked is None


def test_degraded_lock_face_does_not_touch_the_belief(tools_with_lock):
    tools, lock_state = tools_with_lock
    tools.execute("lock_face", {})  # no engine — degrades to ok: null
    assert lock_state.locked is None


def test_lock_face_works_normally_with_no_lock_state_wired(tools):
    # IntentTools() built by the plain `tools` fixture has no lock_state at all.
    with intents_engine(response={"ok": True, "op": "lock_face", "locked": True}):
        result = json.loads(tools.execute("lock_face", {}))
    assert result["ok"] is True


# --------------------------------------------------------------------------- #
# run_behavior names the gaze one-shots reachable through it (task t9)        #
# --------------------------------------------------------------------------- #


def test_run_behavior_description_names_the_gaze_one_shots():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "run_behavior"]
    description = spec["toolSpec"]["description"]
    assert "look-at-sound" in description
    assert "look-at-face" in description
    assert "2" in description  # the 2 s default duration is named


# --------------------------------------------------------------------------- #
# Voice level tools (task t10) — dispatched locally, not through the spool    #
# --------------------------------------------------------------------------- #


class FakeDaemonClient:
    """Stub of ``daemon_client.DaemonClient`` — no network, records calls."""

    def __init__(self, volume: int = 50, get_error: Exception | None = None,
                 set_error: Exception | None = None):
        self.volume = volume
        self.get_error = get_error
        self.set_error = set_error
        self.set_calls: list[int] = []

    def get_volume(self) -> int:
        if self.get_error is not None:
            raise self.get_error
        return self.volume

    def set_volume(self, volume: int) -> int:
        if self.set_error is not None:
            raise self.set_error
        self.set_calls.append(volume)
        self.volume = volume
        return volume


@pytest.fixture
def daemon_client():
    return FakeDaemonClient(volume=50)


@pytest.fixture
def voice_tools(state_dir, daemon_client):
    return IntentTools(await_timeout=0.15, daemon_client=daemon_client)


def test_raise_voice_normal_step(voice_tools, daemon_client):
    daemon_client.volume = 50
    result = json.loads(voice_tools.execute("raise_voice", {}))
    assert result == {"ok": True, "volume": 60}
    assert daemon_client.set_calls == [60]


def test_raise_voice_clamps_to_maximum(voice_tools, daemon_client):
    daemon_client.volume = 95
    result = json.loads(voice_tools.execute("raise_voice", {}))
    assert result == {"ok": True, "volume": 100, "note": "at maximum"}


def test_lower_voice_normal_step(voice_tools, daemon_client):
    daemon_client.volume = 50
    result = json.loads(voice_tools.execute("lower_voice", {}))
    assert result == {"ok": True, "volume": 40}
    assert daemon_client.set_calls == [40]


def test_lower_voice_clamps_to_minimum(voice_tools, daemon_client):
    daemon_client.volume = 15
    result = json.loads(voice_tools.execute("lower_voice", {}))
    assert result == {"ok": True, "volume": 10, "note": "at minimum"}


def test_raise_voice_with_explicit_step(voice_tools, daemon_client):
    daemon_client.volume = 30
    result = json.loads(voice_tools.execute("raise_voice", {"step": 25}))
    assert result == {"ok": True, "volume": 55}


def test_set_voice_level_absolute(voice_tools, daemon_client):
    daemon_client.volume = 20
    result = json.loads(voice_tools.execute("set_voice_level", {"volume": 77}))
    assert result == {"ok": True, "volume": 77}
    assert daemon_client.set_calls == [77]


def test_set_voice_level_clamps_above_maximum(voice_tools, daemon_client):
    daemon_client.volume = 20
    result = json.loads(voice_tools.execute("set_voice_level", {"volume": 150}))
    assert result == {"ok": True, "volume": 100, "note": "at maximum"}


def test_set_voice_level_clamps_below_minimum(voice_tools, daemon_client):
    daemon_client.volume = 20
    result = json.loads(voice_tools.execute("set_voice_level", {"volume": -5}))
    assert result == {"ok": True, "volume": 10, "note": "at minimum"}


def test_set_voice_level_no_op_when_already_at_target(voice_tools, daemon_client):
    daemon_client.volume = 50
    result = json.loads(voice_tools.execute("set_voice_level", {"volume": 50}))
    assert result == {"ok": True, "volume": 50}
    # No daemon set call — avoids the (unsuppressible) confirmation sound.
    assert daemon_client.set_calls == []


def test_set_voice_level_rejects_non_numeric(voice_tools, daemon_client):
    result = json.loads(voice_tools.execute("set_voice_level", {"volume": "loud"}))
    assert result["ok"] is False
    assert "volume" in result["error"]
    assert daemon_client.set_calls == []


def test_voice_tool_surfaces_get_volume_failure(voice_tools, daemon_client):
    daemon_client.get_error = ConnectionError("no route to daemon")
    result = json.loads(voice_tools.execute("raise_voice", {}))
    assert result["ok"] is False
    assert "no route to daemon" in result["error"]


def test_voice_tool_surfaces_set_volume_failure(voice_tools, daemon_client):
    daemon_client.set_error = TimeoutError("daemon timed out")
    result = json.loads(voice_tools.execute("raise_voice", {}))
    assert result["ok"] is False
    assert "daemon timed out" in result["error"]


def test_voice_tool_refused_when_client_not_wired(state_dir):
    tools = IntentTools(await_timeout=0.15)
    result = json.loads(tools.execute("raise_voice", {}))
    assert result == {"ok": False, "error": VOICE_NOT_WIRED_REASON}


def test_voice_tools_never_touch_the_spool(voice_tools, daemon_client):
    voice_tools.execute("raise_voice", {})
    voice_tools.execute("lower_voice", {})
    voice_tools.execute("set_voice_level", {"volume": 33})
    assert spooled() == []


def test_voice_tool_logs_exactly_one_sense_line_on_success(voice_tools, daemon_client, caplog):
    daemon_client.volume = 50
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        voice_tools.execute("raise_voice", {})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=volume]" in line
    assert "old=50 new=60" in line


def test_voice_tool_logs_exactly_one_sense_line_on_failure(voice_tools, daemon_client, caplog):
    daemon_client.get_error = ConnectionError("boom")
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        voice_tools.execute("lower_voice", {})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=volume]" in line
    assert "refused" in line


def test_voice_tool_logs_exactly_one_sense_line_when_not_wired(state_dir, caplog):
    tools = IntentTools(await_timeout=0.15)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        tools.execute("set_voice_level", {"volume": 50})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=volume]" in line


def test_voice_level_bounds_are_sane():
    assert MIN_VOICE_LEVEL == 10
    assert MAX_VOICE_LEVEL == 100
    assert DEFAULT_VOICE_STEP == 10


# --------------------------------------------------------------------------- #
# recall_senses (task t8) — reads SenseHistory directly, never the spool      #
# --------------------------------------------------------------------------- #


class FakeSenseHistory:
    """Stub of ``sense_history.SenseHistory`` — no clock, just records calls."""

    def __init__(self, entries=None):
        self._entries = list(entries or [])
        self.recent_calls: list[int] = []

    def recent(self, n=5):
        self.recent_calls.append(n)
        return self._entries[:n]


@pytest.fixture
def history():
    return FakeSenseHistory(
        entries=[
            {"t": 3.0, "age_s": 0.5, "source": "rule", "type": "fire", "rule": "hear",
             "text": "third", "sense_class": None, "voice": None},
            {"t": 2.0, "age_s": 1.5, "source": "face", "type": "recognized", "rule": None,
             "text": "second", "sense_class": None, "voice": None},
            {"t": 1.0, "age_s": 2.5, "source": "pat", "type": "level1", "rule": "pat-acknowledge",
             "text": "first", "sense_class": "touch", "voice": "brief"},
        ]
    )


@pytest.fixture
def recall_tools(state_dir, history):
    return IntentTools(await_timeout=0.15, history=history)


def test_recall_senses_returns_the_wired_historys_entries(recall_tools, history):
    result = json.loads(recall_tools.execute("recall_senses", {}))
    assert result["ok"] is True
    assert result["senses"] == history._entries
    assert history.recent_calls == [DEFAULT_RECALL_SENSES_N]


def test_recall_senses_passes_n_through(recall_tools, history):
    json.loads(recall_tools.execute("recall_senses", {"n": 2}))
    assert history.recent_calls == [2]


@pytest.mark.parametrize("n,expected", [(0, MIN_RECALL_SENSES_N), (-5, MIN_RECALL_SENSES_N),
                                         (1, 1), (20, 20), (21, MAX_RECALL_SENSES_N),
                                         (999, MAX_RECALL_SENSES_N)])
def test_recall_senses_clamps_n_into_bounds(recall_tools, history, n, expected):
    json.loads(recall_tools.execute("recall_senses", {"n": n}))
    assert history.recent_calls == [expected]


def test_recall_senses_refused_when_not_wired(tools):
    result = json.loads(tools.execute("recall_senses", {}))
    assert result == {"ok": False, "error": HISTORY_NOT_WIRED_REASON}


def test_recall_senses_never_touches_the_spool(recall_tools):
    recall_tools.execute("recall_senses", {"n": 3})
    assert spooled() == []


def test_recall_senses_logs_one_sense_line_on_success(recall_tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        recall_tools.execute("recall_senses", {})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=recall_senses]" in line
    assert "confirmed" in line


def test_recall_senses_logs_one_sense_line_when_not_wired(tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        tools.execute("recall_senses", {})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=recall_senses]" in line
    assert "refused" in line


def test_recall_senses_tool_spec_has_no_required_fields():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "recall_senses"]
    schema = json.loads(spec["toolSpec"]["inputSchema"]["json"])
    assert schema["required"] == []
    assert set(schema["properties"]) == {"n"}


def test_recall_senses_tool_spec_names_the_trigger_and_forbids_mechanism_talk():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "recall_senses"]
    description = spec["toolSpec"]["description"].lower()
    assert "why" in description
    assert "felt" in description or "feel" in description
    assert "happened" in description


def test_recall_senses_bounds_are_sane():
    assert MIN_RECALL_SENSES_N == 1
    assert MAX_RECALL_SENSES_N == 20
    assert DEFAULT_RECALL_SENSES_N == 5


# --------------------------------------------------------------------------- #
# stay_silent / end_silence (task t12)                                        #
# --------------------------------------------------------------------------- #


class FakeClock:
    """A wall clock a test can move by hand — quiet is a deadline, not a mode."""

    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def quiet(state_dir, clock):
    return QuietState(clock=clock, path=state_dir / "nova-quiet.json")


@pytest.fixture
def quiet_tools(state_dir, quiet):
    return IntentTools(await_timeout=0.5, quiet=quiet)


def _write_inhibitions(names):
    """Publish an engine ``intents`` view — the merge source for set_inhibition."""
    path = statedir.state_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"intents": {"inhibitions": list(names)}}), encoding="utf-8")


def _inhibition_payloads(engine):
    return [p for p in engine.seen if p.get("op") == "set_inhibition"]


def test_stay_silent_arms_the_deadline_and_reports_it(quiet_tools, quiet):
    with intents_engine():
        result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 10}))
    assert result["ok"] is True
    assert result["note"] == "armed"
    assert result["until"] == quiet.until_iso()
    assert quiet.active() is True


def test_stay_silent_leaves_one_acknowledgement_utterance_pending(quiet_tools, quiet):
    with intents_engine():
        quiet_tools.execute("stay_silent", {"minutes": 10})
    assert quiet.pending_first_utterance is True


def test_stay_silent_mutes_the_body_voice_by_adding_speak_to_the_inhibitions(quiet_tools):
    with intents_engine() as engine:
        result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 5}))
    (payload,) = _inhibition_payloads(engine)
    assert payload["behaviors"] == ["speak"]
    assert result["body_muted"] is True


def test_stay_silent_merges_speak_into_the_runtimes_current_inhibitions(quiet_tools):
    _write_inhibitions(["nod"])
    with intents_engine() as engine:
        quiet_tools.execute("stay_silent", {"minutes": 5})
    (payload,) = _inhibition_payloads(engine)
    assert payload["behaviors"] == ["nod", "speak"]


def test_stay_silent_spools_exactly_one_set_inhibition(quiet_tools):
    with intents_engine() as engine:
        quiet_tools.execute("stay_silent", {"minutes": 5})
    assert len(_inhibition_payloads(engine)) == 1


@pytest.mark.parametrize("minutes", [0, 0.5, 181, 10_000, -5])
def test_stay_silent_out_of_bounds_is_refused_without_arming(quiet_tools, quiet, minutes):
    result = json.loads(quiet_tools.execute("stay_silent", {"minutes": minutes}))
    assert result["ok"] is False
    assert result["error"]
    assert quiet.active() is False
    assert spooled() == []


def test_stay_silent_non_numeric_minutes_is_refused(quiet_tools, quiet):
    result = json.loads(quiet_tools.execute("stay_silent", {"minutes": "ten"}))
    assert result["ok"] is False
    assert quiet.active() is False


def test_stay_silent_at_the_bounds_is_accepted(quiet_tools, quiet):
    with intents_engine():
        low = json.loads(quiet_tools.execute("stay_silent", {"minutes": MIN_QUIET_MINUTES}))
        quiet_tools.execute("end_silence", {})
        high = json.loads(quiet_tools.execute("stay_silent", {"minutes": MAX_QUIET_MINUTES}))
    assert low["ok"] is True
    assert high["ok"] is True


def test_stay_silent_again_with_a_longer_duration_extends(quiet_tools):
    with intents_engine():
        quiet_tools.execute("stay_silent", {"minutes": 5})
        result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 20}))
    assert result["note"] == "extended"


def test_stay_silent_again_with_a_shorter_duration_keeps_the_longer_one(quiet_tools):
    with intents_engine():
        quiet_tools.execute("stay_silent", {"minutes": 30})
        result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 5}))
    assert result["note"] == "kept"


def test_stay_silent_holds_the_mind_side_quiet_even_when_the_body_mute_degrades(
    quiet_tools, quiet
):
    # No engine at all: the set_inhibition degrades to submitted-but-unconfirmed.
    result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 10}))
    assert result["ok"] is True
    assert result["body_muted"] is False
    assert quiet.active() is True


def test_stay_silent_without_a_wired_quiet_state_is_refused(tools):
    result = json.loads(tools.execute("stay_silent", {"minutes": 10}))
    assert result["ok"] is False
    assert result["error"] == QUIET_NOT_WIRED_REASON


def test_stay_silent_logs_exactly_one_sense_line(quiet_tools, caplog):
    with intents_engine(), caplog.at_level(logging.INFO, logger="nova.sensory"):
        quiet_tools.execute("stay_silent", {"minutes": 10})
    lines = [ln for ln in _sense_lines(caplog) if "event=stay_silent" in ln]
    assert len(lines) == 1
    assert "[SENSE stage=act source=nova event=stay_silent]" in lines[0]


def test_end_silence_when_nothing_is_armed_is_accepted_and_named(quiet_tools):
    result = json.loads(quiet_tools.execute("end_silence", {}))
    assert result["ok"] is True
    assert result["note"] == "not silent"
    assert spooled() == []


def test_end_silence_ends_an_armed_quiet(quiet_tools, quiet):
    with intents_engine():
        quiet_tools.execute("stay_silent", {"minutes": 10})
        result = json.loads(quiet_tools.execute("end_silence", {}))
    assert result["ok"] is True
    assert result["note"] == "ended"
    assert quiet.active() is False


def test_end_silence_restores_the_body_voice(quiet_tools):
    with intents_engine() as engine:
        quiet_tools.execute("stay_silent", {"minutes": 10})
        result = json.loads(quiet_tools.execute("end_silence", {}))
    first, second = _inhibition_payloads(engine)
    assert first["behaviors"] == ["speak"]
    assert second["behaviors"] == []
    assert result["body_restored"] is True


def test_end_silence_leaves_an_inhibition_we_did_not_add_alone(quiet_tools):
    _write_inhibitions(["speak"])
    with intents_engine() as engine:
        quiet_tools.execute("stay_silent", {"minutes": 10})
        result = json.loads(quiet_tools.execute("end_silence", {}))
    # One spool for the (idempotent) arm; NONE for the release — somebody else
    # is holding 'speak' down and un-muting them would be a silent override.
    assert len(_inhibition_payloads(engine)) == 1
    assert result["body_restored"] is True  # the voice unmute is still ours to send


def test_end_silence_without_a_wired_quiet_state_is_refused(tools):
    result = json.loads(tools.execute("end_silence", {}))
    assert result["ok"] is False
    assert result["error"] == QUIET_NOT_WIRED_REASON


def test_end_silence_logs_exactly_one_sense_line(quiet_tools, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        quiet_tools.execute("end_silence", {})
    lines = [ln for ln in _sense_lines(caplog) if "event=end_silence" in ln]
    assert len(lines) == 1


def test_tick_restores_the_body_voice_when_the_quiet_expires(quiet_tools, quiet, clock):
    with intents_engine() as engine:
        quiet_tools.execute("stay_silent", {"minutes": 10})
        clock.advance(11 * 60)
        quiet_tools.tick()
    first, second = _inhibition_payloads(engine)
    assert first["behaviors"] == ["speak"]
    assert second["behaviors"] == []
    assert quiet.active() is False


def test_tick_while_still_quiet_changes_nothing(quiet_tools, clock):
    with intents_engine() as engine:
        quiet_tools.execute("stay_silent", {"minutes": 10})
        clock.advance(60)
        quiet_tools.tick()
    assert len(_inhibition_payloads(engine)) == 1


def test_tick_restores_only_once_after_an_expiry(quiet_tools, clock):
    with intents_engine() as engine:
        quiet_tools.execute("stay_silent", {"minutes": 10})
        clock.advance(11 * 60)
        quiet_tools.tick()
        quiet_tools.tick()
        quiet_tools.tick()
    assert len(_inhibition_payloads(engine)) == 2


def test_tick_is_a_no_op_without_a_wired_quiet_state(tools):
    tools.tick()
    assert spooled() == []


def test_quiet_tools_tool_specs_are_shaped_for_the_model():
    (stay,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "stay_silent"]
    schema = json.loads(stay["toolSpec"]["inputSchema"]["json"])
    assert schema["required"] == ["minutes"]
    assert set(schema["properties"]) == {"minutes"}
    (end,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "end_silence"]
    end_schema = json.loads(end["toolSpec"]["inputSchema"]["json"])
    assert end_schema["required"] == []


def test_stay_silent_description_asks_for_one_brief_acknowledgement():
    (stay,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "stay_silent"]
    description = stay["toolSpec"]["description"].lower()
    assert "quiet" in description
    assert "once" in description


def test_end_silence_description_says_leaving_is_silent():
    (end,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "end_silence"]
    description = end["toolSpec"]["description"].lower()
    assert "talk again" in description
    assert "announce" in description or "silent" in description


def test_quiet_bounds_are_sane():
    assert MIN_QUIET_MINUTES == 1
    assert MAX_QUIET_MINUTES == 180


def _ops(engine, op):
    return [c for c in engine.seen if c.get("op") == op]


def test_stay_silent_also_spools_the_runtime_mute_intent(quiet_tools):
    with intents_engine() as engine:
        result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 5}))
    assert result["body_muted"] is True
    assert len(_ops(engine, "mute")) == 1
    assert _ops(engine, "unmute") == []


def test_end_silence_spools_unmute_after_a_mute(quiet_tools):
    with intents_engine() as engine:
        quiet_tools.execute("stay_silent", {"minutes": 5})
        result = json.loads(quiet_tools.execute("end_silence", {}))
    assert result["body_restored"] is True
    assert len(_ops(engine, "unmute")) == 1


# --------------------------------------------------------------------------- #
# Voice level: persistence (finding F4) + concurrency (finding F7)            #
#                                                                             #
# The daemon forgets its volume across a restart, so ``restore_volume`` reads #
# ``<state>/nova-volume.json`` on harness start — which is only worth reading #
# if the tools that CHANGE the level also write it.                           #
# --------------------------------------------------------------------------- #


def _volume_file():
    return statedir.volume_state_path()


def _persisted_volume():
    return json.loads(_volume_file().read_text(encoding="utf-8"))


def test_raise_voice_persists_the_new_level(voice_tools, daemon_client):
    daemon_client.volume = 50
    result = json.loads(voice_tools.execute("raise_voice", {}))
    assert result == {"ok": True, "volume": 60}
    assert _persisted_volume() == {"volume": 60}


def test_set_voice_level_persists_the_new_level(voice_tools, daemon_client):
    daemon_client.volume = 20
    voice_tools.execute("set_voice_level", {"volume": 77})
    assert _persisted_volume() == {"volume": 77}


def test_a_no_op_set_still_creates_an_absent_persisted_file(voice_tools, daemon_client):
    daemon_client.volume = 50
    assert not _volume_file().exists()
    result = json.loads(voice_tools.execute("set_voice_level", {"volume": 50}))
    assert result == {"ok": True, "volume": 50}
    # Still no daemon call (the confirmation sound is unsuppressible)...
    assert daemon_client.set_calls == []
    # ...but the level the person asked for is now durable.
    assert _persisted_volume() == {"volume": 50}


def test_a_no_op_set_refreshes_a_stale_persisted_file(voice_tools, daemon_client):
    _volume_file().parent.mkdir(parents=True, exist_ok=True)
    _volume_file().write_text(json.dumps({"volume": 42}), encoding="utf-8")
    daemon_client.volume = 50
    voice_tools.execute("set_voice_level", {"volume": 50})
    assert daemon_client.set_calls == []
    assert _persisted_volume() == {"volume": 50}


def test_a_failed_set_leaves_the_persisted_level_untouched(voice_tools, daemon_client):
    _volume_file().parent.mkdir(parents=True, exist_ok=True)
    _volume_file().write_text(json.dumps({"volume": 42}), encoding="utf-8")
    daemon_client.set_error = TimeoutError("daemon timed out")
    result = json.loads(voice_tools.execute("raise_voice", {}))
    assert result["ok"] is False
    assert _persisted_volume() == {"volume": 42}


def test_a_failed_get_never_writes_a_persisted_level(voice_tools, daemon_client):
    daemon_client.get_error = ConnectionError("no route to daemon")
    result = json.loads(voice_tools.execute("raise_voice", {}))
    assert result["ok"] is False
    assert not _volume_file().exists()


def test_a_persistence_failure_is_reported_not_claimed(
    voice_tools, daemon_client, monkeypatch
):
    def boom(path, text):
        raise OSError("read-only state dir")

    monkeypatch.setattr(tools_module, "_atomic_write", boom)
    daemon_client.volume = 50
    result = json.loads(voice_tools.execute("raise_voice", {}))
    # The daemon really did apply it, so ok stays True — but the caller is
    # told, in the payload, that it will not survive a restart.
    assert result["ok"] is True
    assert result["volume"] == 60
    assert result["persisted"] is False
    assert tools_module.VOLUME_NOT_PERSISTED_NOTE in result["note"]


def test_a_persistence_failure_keeps_a_clamp_note_visible(
    voice_tools, daemon_client, monkeypatch
):
    def boom(path, text):
        raise OSError("read-only state dir")

    monkeypatch.setattr(tools_module, "_atomic_write", boom)
    daemon_client.volume = 95
    result = json.loads(voice_tools.execute("raise_voice", {}))
    assert result["volume"] == 100
    assert "at maximum" in result["note"]
    assert tools_module.VOLUME_NOT_PERSISTED_NOTE in result["note"]


def test_a_successful_set_does_not_advertise_persistence(voice_tools, daemon_client):
    daemon_client.volume = 50
    result = json.loads(voice_tools.execute("lower_voice", {}))
    # Durable is the contract, not a field: only a FAILURE is announced.
    assert result == {"ok": True, "volume": 40}
    assert "persisted" not in result


def test_the_persisted_shape_is_what_restore_volume_reads(voice_tools, daemon_client):
    daemon_client.volume = 50
    voice_tools.execute("set_voice_level", {"volume": 33})
    # A "restart": the daemon came back at its own default.
    fresh = FakeDaemonClient(volume=50)
    assert restore_volume(_volume_file(), fresh) == 33
    assert fresh.set_calls == [33]


class SlowReadDaemonClient(FakeDaemonClient):
    """A client whose read is slow enough for two threads to interleave.

    Tool calls arrive on their own threads (``app.py``'s ``_on_tool_use``), so
    a relative volume change is a read-compute-set transaction two callers can
    genuinely be inside at once.
    """

    def get_volume(self) -> int:
        time.sleep(0.02)
        return super().get_volume()


def test_concurrent_raises_do_not_overwrite_each_other(state_dir):
    client = SlowReadDaemonClient(volume=50)
    tools = IntentTools(await_timeout=0.15, daemon_client=client)
    threads = [
        threading.Thread(target=tools.execute, args=("raise_voice", {}))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    # Two +10 steps from 50 is 70. An unlocked read-compute-set gives 60:
    # both threads read 50 and both write 60, and one request vanishes.
    assert client.set_calls == [60, 70]
    assert client.volume == 70
    assert _persisted_volume() == {"volume": 70}


def test_concurrent_raise_and_lower_serialize(state_dir):
    client = SlowReadDaemonClient(volume=50)
    tools = IntentTools(await_timeout=0.15, daemon_client=client)
    threads = [
        threading.Thread(target=tools.execute, args=("raise_voice", {})),
        threading.Thread(target=tools.execute, args=("lower_voice", {})),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    # Whichever order they run in, +10 and -10 from 50 must land back on 50.
    assert client.volume == 50


# --------------------------------------------------------------------------- #
# Quiet: expiry, acknowledgement timing and restart ownership (F2, F5, F1)    #
# --------------------------------------------------------------------------- #


class PerOpEngine(FakeEngine):
    """A fake engine that answers differently per ``op``.

    Needed for the half-degraded cases: a runtime that refuses the inhibition
    but accepts the mute (or the other way round) is exactly where the
    harness's ownership bookkeeping gets it wrong.
    """

    def __init__(self, commands_dir, results_dir, answers):
        super().__init__(commands_dir, results_dir)
        self.answers = answers

    def _answer(self, payload: dict) -> dict:
        op = payload.get("op")
        base = self.answers.get(op, {"ok": True})
        return {**base, "cmd_id": payload.get("cmd_id"), "op": op}


def per_op_engine(answers):
    return PerOpEngine(
        statedir.intents_commands_dir(), statedir.intents_results_dir(), answers
    )


# -- F2: a mute-only quiet must still expire -------------------------------- #


def test_expiry_unmutes_when_only_the_mute_latch_is_owned(quiet_tools, quiet, clock):
    answers = {"set_inhibition": {"ok": False, "error": "refused"}}
    with per_op_engine(answers) as engine:
        result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 10}))
        assert result["body_muted"] is False  # the inhibition half was refused
        assert len(_ops(engine, "mute")) == 1  # ...but the mute half landed
        clock.advance(11 * 60)
        quiet_tools.tick()
        quiet_tools.tick()
    # The mute is ours and only we can lift it — exactly once.
    assert len(_ops(engine, "unmute")) == 1


def test_expiry_unmutes_when_only_the_inhibition_latch_is_owned(
    quiet_tools, quiet, clock
):
    answers = {"mute": {"ok": False, "error": "unknown kind"}}
    with per_op_engine(answers) as engine:
        quiet_tools.execute("stay_silent", {"minutes": 10})
        clock.advance(11 * 60)
        quiet_tools.tick()
    first, second = _inhibition_payloads(engine)
    assert first["behaviors"] == ["speak"]
    assert second["behaviors"] == []
    assert _ops(engine, "unmute") == []  # never ours to undo


# -- F5: the acknowledgement must outlive a slow body mute ------------------ #


def test_the_acknowledgement_survives_a_slow_body_mute(quiet_tools, quiet, clock):
    def slow_submit(payload):
        # Each body round-trip eats most of the 2 s acknowledgement grace.
        clock.advance(1.5)
        return {"ok": True, "op": payload.get("op")}

    quiet_tools.submit_and_await = slow_submit
    result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 10}))
    assert result["ok"] is True
    assert result["body_muted"] is True
    # "okay, quiet for ten minutes" must still be allowed out.
    assert quiet.pending_first_utterance is True
    assert quiet.allow_utterance() is True


def test_stay_silent_result_field_order_is_unchanged(quiet_tools):
    with intents_engine():
        result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 10}))
    assert list(result) == ["ok", "until", "note", "body_muted"]


def test_the_mind_side_quiet_holds_when_the_body_mute_raises(quiet_tools, quiet):
    def boom(payload):
        raise OSError("spool is gone")

    quiet_tools.submit_and_await = boom
    result = json.loads(quiet_tools.execute("stay_silent", {"minutes": 10}))
    assert result["ok"] is True
    assert result["body_muted"] is False
    assert quiet.active() is True


# -- F1: ownership survives a harness restart ------------------------------- #


def _write_quiet_file(path, until, latches=None):
    payload = {"until": until}
    if latches is not None:
        payload["body"] = {"added_speak": latches[0], "muted_voice": latches[1]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _restarted_tools(state_dir, clock):
    """A brand-new IntentTools over the persisted quiet — a harness restart."""
    restored = QuietState(clock=clock, path=state_dir / "nova-quiet.json")
    return IntentTools(await_timeout=0.5, quiet=restored), restored


def test_stay_silent_persists_the_body_ownership_latches(quiet_tools, state_dir):
    with intents_engine():
        quiet_tools.execute("stay_silent", {"minutes": 10})
    payload = json.loads((state_dir / "nova-quiet.json").read_text(encoding="utf-8"))
    assert payload["body"] == {"added_speak": True, "muted_voice": True}


def test_end_silence_clears_the_persisted_latches(quiet_tools, state_dir):
    with intents_engine():
        quiet_tools.execute("stay_silent", {"minutes": 10})
        quiet_tools.execute("end_silence", {})
    assert not (state_dir / "nova-quiet.json").exists()


def test_a_restart_inside_a_quiet_re_mutes_and_reclaims_ownership(state_dir, clock):
    _write_quiet_file(state_dir / "nova-quiet.json", clock.now + 600, (True, True))
    _write_inhibitions(["speak"])  # the runtime kept the mute across our restart
    tools, restored = _restarted_tools(state_dir, clock)
    with intents_engine() as engine:
        tools.tick()
    # Idempotent re-issue: one set_inhibition (still just 'speak') and one mute.
    (payload,) = _inhibition_payloads(engine)
    assert payload["behaviors"] == ["speak"]
    assert len(_ops(engine, "mute")) == 1
    # Ownership is ours again, so the expiry can undo it.
    clock.advance(11 * 60)
    with intents_engine() as engine2:
        tools.tick()
    assert _inhibition_payloads(engine2)[0]["behaviors"] == []
    assert len(_ops(engine2, "unmute")) == 1
    assert restored.active() is False


def test_a_restart_re_mutes_a_runtime_that_came_back_talking(state_dir, clock):
    _write_quiet_file(state_dir / "nova-quiet.json", clock.now + 600, (True, True))
    _write_inhibitions([])  # the runtime restarted too: nothing held back
    tools, _restored = _restarted_tools(state_dir, clock)
    with per_op_engine({"mute": {"ok": True, "note": "not muted"}}) as engine:
        tools.tick()
    (payload,) = _inhibition_payloads(engine)
    assert payload["behaviors"] == ["speak"]
    assert len(_ops(engine, "mute")) == 1


def test_a_restart_after_the_deadline_mutes_nothing(state_dir, clock):
    _write_quiet_file(state_dir / "nova-quiet.json", clock.now - 1, (True, True))
    tools, restored = _restarted_tools(state_dir, clock)
    with intents_engine() as engine:
        tools.tick()
    assert restored.active() is False
    assert engine.seen == []


def test_a_restart_over_an_old_shape_file_still_mutes_the_body(state_dir, clock):
    # No latches recorded (pre-F1 file): quiet is still on, so the body must be.
    _write_quiet_file(state_dir / "nova-quiet.json", clock.now + 600, None)
    tools, restored = _restarted_tools(state_dir, clock)
    with intents_engine() as engine:
        tools.tick()
    assert restored.active() is True
    assert len(_inhibition_payloads(engine)) == 1
    assert len(_ops(engine, "mute")) == 1


def test_the_restart_re_mute_happens_only_once(state_dir, clock):
    _write_quiet_file(state_dir / "nova-quiet.json", clock.now + 600, (True, True))
    tools, _restored = _restarted_tools(state_dir, clock)
    with intents_engine() as engine:
        tools.tick()
        tools.tick()
        tools.tick()
    assert len(_ops(engine, "mute")) == 1


# --------------------------------------------------------------------------- #
# Live findings L4/L5 (on-device run 2026-08-26 18:26–18:31)                  #
#                                                                             #
# L4: Sonic reached for a tool named `look_at_face`, was pre-flight refused   #
#     ("unknown tool"), and gave up instead of calling                        #
#     run_behavior(name="look-at-face").                                      #
# L5: Sonic never called release_face on "you can look away", never called    #
#     lower_voice on "a bit quieter", and never called recall_senses on "why  #
#     did you do that?" — the descriptions never named those phrases.         #
# --------------------------------------------------------------------------- #

GAZE_ALIASES = {
    "look_at_face": "look-at-face",
    "look_at_sound": "look-at-sound",
}


def _spec_for(name: str) -> dict:
    return next(s["toolSpec"] for s in TOOL_SPECS if s["toolSpec"]["name"] == name)


@pytest.mark.parametrize("tool_name,behavior", sorted(GAZE_ALIASES.items()))
def test_l4_a_gaze_alias_spools_the_run_behavior_op(tools, tool_name, behavior):
    with intents_engine() as engine:
        result = json.loads(tools.execute(tool_name, {}))
    assert result["ok"] is True
    assert [p["op"] for p in engine.seen] == ["run_behavior"]
    payload = engine.seen[0]
    assert payload["name"] == behavior
    assert payload["lifetime"]["duration"] == pytest.approx(2.0)


@pytest.mark.parametrize("tool_name,behavior", sorted(GAZE_ALIASES.items()))
def test_l4_a_gaze_alias_honours_an_explicit_duration(tools, tool_name, behavior):
    with intents_engine() as engine:
        json.loads(tools.execute(tool_name, {"duration": 4.5}))
    assert engine.seen[0]["name"] == behavior
    assert engine.seen[0]["lifetime"]["duration"] == pytest.approx(4.5)


@pytest.mark.parametrize("tool_name", sorted(GAZE_ALIASES))
def test_l4_a_gaze_alias_refuses_a_bad_duration_before_the_spool_write(tools, tool_name):
    result = json.loads(tools.execute(tool_name, {"duration": 0}))
    assert result["ok"] is False
    assert "duration" in result["error"]
    assert spooled() == []


@pytest.mark.parametrize("tool_name", sorted(GAZE_ALIASES))
def test_l4_a_gaze_alias_takes_no_required_arguments(tool_name):
    schema = json.loads(_spec_for(tool_name)["inputSchema"]["json"])
    assert schema["required"] == []
    assert set(schema["properties"]) == {"duration"}


def test_l4_the_gaze_alias_descriptions_say_what_they_do():
    assert (
        "glance at the person in front of you"
        in _spec_for("look_at_face")["description"].lower()
    )
    assert "turn toward the last sound" in _spec_for("look_at_sound")["description"].lower()


@pytest.mark.parametrize(
    "tool_name,phrases",
    [
        ("release_face", ("look away", "stop following", "you can stop looking at me")),
        ("lower_voice", ("quieter", "softer", "turn it down")),
        ("raise_voice", ("speak up", "louder")),
        ("recall_senses", ("why did you do that", "what did you feel", "what just happened")),
    ],
)
def test_l5_descriptions_name_the_phrases_people_actually_say(tool_name, phrases):
    description = _spec_for(tool_name)["description"].lower()
    for phrase in phrases:
        assert phrase in description, f"{tool_name} description never names {phrase!r}"


# --------------------------------------------------------------------------- #
# think (task t7) — a run_behavior alias over the runtime's 'thoughtful'      #
# --------------------------------------------------------------------------- #


def test_think_tool_spec_is_published():
    (spec,) = [s for s in TOOL_SPECS if s["toolSpec"]["name"] == "think"]
    schema = json.loads(spec["toolSpec"]["inputSchema"]["json"])
    assert schema["required"] == []
    assert set(schema["properties"]) == {"side"}
    assert schema["properties"]["side"]["enum"] == ["left", "right"]
    assert "thinking" in spec["toolSpec"]["description"].lower()


def test_think_left_spools_the_thoughtful_behavior_with_positive_yaw(tools):
    with intents_engine() as engine:
        result = json.loads(tools.execute("think", {"side": "left"}))
    assert result["ok"] is True
    (payload,) = engine.seen
    assert payload["op"] == "run_behavior"
    assert payload["name"] == THINK_BEHAVIOR
    assert payload["params"] == {"yaw": THINK_YAW_LEFT}
    assert payload["lifetime"] == {"duration": THINK_DURATION_S}
    assert THINK_YAW_LEFT > 0


def test_think_right_uses_the_opposite_sign(tools):
    with intents_engine() as engine:
        tools.execute("think", {"side": "right"})
    payload = engine.seen[0]
    assert payload["params"]["yaw"] == THINK_YAW_RIGHT
    assert THINK_YAW_RIGHT == -THINK_YAW_LEFT


def test_think_without_a_side_alternates_across_calls(tools):
    with intents_engine() as engine:
        tools.execute("think", {})
        tools.execute("think", {})
        tools.execute("think", {})
    yaws = [p["params"]["yaw"] for p in engine.seen]
    assert yaws[0] != yaws[1]
    assert yaws[0] == yaws[2]
    assert set(yaws) == {THINK_YAW_LEFT, THINK_YAW_RIGHT}


def test_think_rejects_a_bad_side_before_the_spool_write(tools):
    result = json.loads(tools.execute("think", {"side": "up"}))
    assert result["ok"] is False
    assert result["error"] == THINK_SIDE_INVALID_REASON
    assert spooled() == []


def test_think_is_in_the_action_set_right_after_look_at_sound():
    names = tuple(spec["toolSpec"]["name"] for spec in TOOL_SPECS)
    assert names[names.index("look_at_sound") + 1] == "think"


# --------------------------------------------------------------------------- #
# browse (task t7) — a typed queue_task result (including a duplicate) passes #
# through unchanged; an older None-returning stub still gets the old shape   #
# --------------------------------------------------------------------------- #


class TypedFakeBrowser:
    """A ``queue_task`` double that returns NovaBrowser's own typed dict."""

    def __init__(self, response):
        self.response = response
        self.calls: list[tuple[str, str | None]] = []
        self.on_progress = None

    def queue_task(self, instruction, url=None):
        self.calls.append((instruction, url))
        return self.response


def test_browse_duplicate_is_passed_through_unchanged(state_dir, monkeypatch):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    duplicate = {
        "ok": True,
        "queued": False,
        "duplicate": True,
        "instruction": "find a recipe",
        "url": None,
    }
    browser = TypedFakeBrowser(duplicate)
    tools = IntentTools(await_timeout=0.05, browser=browser)

    result = json.loads(tools.execute("browse", {"instruction": "find a recipe"}))

    assert result == duplicate
    assert browser.calls == [("find a recipe", None)]


def test_browse_disabled_result_from_the_browser_is_passed_through(state_dir, monkeypatch):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    disabled = {"ok": False, "queued": False, "reason": "nova-act-disabled"}
    browser = TypedFakeBrowser(disabled)
    tools = IntentTools(await_timeout=0.05, browser=browser)

    result = json.loads(tools.execute("browse", {"instruction": "find a recipe"}))

    assert result == disabled


def test_browse_none_returning_browser_still_gets_the_old_queued_dict(state_dir, monkeypatch):
    # Mirrors test_browse_queues_the_instruction_when_enabled_and_wired's
    # FakeBrowser (queue_task returns None implicitly) — the pre-typed-return
    # shape an older NovaBrowser stub would still produce.
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    browser = FakeBrowser()
    tools = IntentTools(await_timeout=0.05, browser=browser)

    result = json.loads(
        tools.execute("browse", {"instruction": "find a recipe", "url": "https://example.com"})
    )

    assert result == {
        "ok": True,
        "queued": True,
        "instruction": "find a recipe",
        "url": "https://example.com",
    }


# --------------------------------------------------------------------------- #
# lock_face marks the belief owner "model" (task t7)                          #
# --------------------------------------------------------------------------- #


def test_confirmed_lock_face_marks_the_belief_owner_model(tools_with_lock):
    tools, lock_state = tools_with_lock
    with intents_engine(response={"ok": True, "op": "lock_face", "locked": True}):
        tools.execute("lock_face", {})
    assert lock_state.locked is True
    assert lock_state.owner == "model"


# --------------------------------------------------------------------------- #
# Cold refusal of effectful tools (task t7)                                   #
# --------------------------------------------------------------------------- #


class _StepClock:
    """A tiny monotonic-seconds clock a test can move by hand."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


@pytest.fixture
def cold_clock():
    return _StepClock()


@pytest.fixture
def cold_attention(cold_clock):
    return AttentionState(clock=cold_clock)


def test_cold_and_nameless_browse_is_refused_and_never_touches_the_browser(
    state_dir, cold_attention
):
    assert cold_attention.note_transcript("what time is it") == "ignored"
    browser = FakeBrowser()
    tools = IntentTools(await_timeout=0.05, browser=browser, attention=cold_attention)

    result = json.loads(tools.execute("browse", {"instruction": "find a recipe"}))

    assert result["ok"] is False
    assert "not addressed" in result["error"]
    assert result["error"] == NOT_ADDRESSED_REASON
    assert browser.queued == []


def test_cold_and_nameless_run_behavior_is_refused_without_spooling(state_dir, cold_attention):
    cold_attention.note_transcript("what time is it")
    tools = IntentTools(await_timeout=0.05, attention=cold_attention)

    result = json.loads(tools.execute("run_behavior", {"name": "nod"}))

    assert result["ok"] is False
    assert result["error"] == NOT_ADDRESSED_REASON
    assert spooled() == []


def test_cold_and_nameless_reflex_run_behavior_is_not_refused(state_dir, cold_attention):
    """The gaze stack's own ops (lock, sway, base-layer revive) bypass the
    cold refusal: it exists for the MODEL acting on overheard speech, and
    gating the body's reflexes left the robot rigid and lock-less for as long
    as people talked near it without naming it (live, 2026-09-06 12:28 BST).
    """
    cold_attention.note_transcript("what time is it")
    tools = IntentTools(await_timeout=0.05, attention=cold_attention)

    refused = json.loads(tools.execute("run_behavior", {"name": "nod"}))
    assert refused["error"] == NOT_ADDRESSED_REASON
    assert spooled() == []

    result = json.loads(tools.execute("run_behavior", {"name": "nod"}, reflex=True))

    assert "error" not in result or result["error"] != NOT_ADDRESSED_REASON
    assert len(spooled()) == 1
    assert spool_payloads()[0]["op"] == "run_behavior"


def test_cold_and_nameless_reflex_lock_face_is_not_refused(state_dir, cold_attention):
    cold_attention.note_transcript("what time is it")
    tools = IntentTools(await_timeout=0.05, attention=cold_attention)

    result = json.loads(tools.execute("lock_face", {}, reflex=True))

    assert result.get("error") != NOT_ADDRESSED_REASON
    assert len(spooled()) == 1
    assert spool_payloads()[0]["op"] == "lock_face"


def test_cold_and_nameless_recall_senses_still_executes(state_dir, cold_attention):
    cold_attention.note_transcript("what time is it")
    history = FakeSenseHistory(entries=[])
    tools = IntentTools(await_timeout=0.05, attention=cold_attention, history=history)

    result = json.loads(tools.execute("recall_senses", {}))

    assert result["ok"] is True
    assert history.recent_calls == [DEFAULT_RECALL_SENSES_N]


def test_warm_after_a_named_transcript_allows_browse(state_dir, monkeypatch, cold_attention):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    assert cold_attention.note_transcript("nova, look it up") == "opened"
    browser = FakeBrowser()
    tools = IntentTools(await_timeout=0.05, browser=browser, attention=cold_attention)

    result = json.loads(tools.execute("browse", {"instruction": "find a recipe"}))

    assert result["ok"] is True
    assert browser.queued == [("find a recipe", None)]


def test_cold_with_an_inject_after_the_nameless_transcript_is_allowed(
    state_dir, monkeypatch, cold_attention
):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    cold_attention.note_transcript("what time is it")
    cold_attention.note_inject()
    browser = FakeBrowser()
    tools = IntentTools(await_timeout=0.05, browser=browser, attention=cold_attention)

    result = json.loads(tools.execute("browse", {"instruction": "find a recipe"}))

    assert result["ok"] is True
    assert browser.queued == [("find a recipe", None)]


def test_no_attention_wired_allows_everything_as_before(tools):
    # The plain `tools` fixture builds IntentTools() with attention=None.
    with intents_engine() as engine:
        result = json.loads(tools.execute("run_behavior", {"name": "nod"}))
    assert result["ok"] is True
    assert engine.seen


def test_cold_refused_tools_are_exactly_the_effectful_set():
    assert COLD_REFUSED_TOOLS == {
        "browse",
        "forge",
        "use_skill",
        "author_rule",
        "goto",
        "run_behavior",
        "declare_goal",
        "set_mode",
        "set_inhibition",
        "create_rule",
        "enroll_face",
        "lock_face",
        "look_at_face",
        "look_at_sound",
        "think",
    }


def test_cold_refusal_logs_the_same_one_sense_line_shape(state_dir, cold_attention, caplog):
    cold_attention.note_transcript("what time is it")
    tools = IntentTools(await_timeout=0.05, attention=cold_attention)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        tools.execute("run_behavior", {"name": "nod"})
    (line,) = _sense_lines(caplog)
    assert "[SENSE stage=act source=nova event=run_behavior]" in line
    assert "refused" in line
    assert NOT_ADDRESSED_REASON in line
