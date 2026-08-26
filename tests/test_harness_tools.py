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
from reachy_nova.harness.quiet import QuietState
from reachy_nova.harness.tools import (
    BROWSE_DISABLED_REASON,
    BROWSE_NOT_WIRED_REASON,
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
    QUIET_NOT_WIRED_REASON,
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
    assert result["body_restored"] is False


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
