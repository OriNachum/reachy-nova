"""The forge leg — the harness's kiro-writer wiring (plan deviation d1).

The kiro-writer arc's components (kiro_acp t3, kiro_session t4, the
FORGE_WRITER seam t6, kiro_rules t7) each shipped with their own tests; this
file pins the COMPOSITION: a goal handed to :class:`ForgeLeg` flows through a
(fake) standing session into SkillForge's kiro path, stages, validates,
auto-activates WITHOUT any Sonic restart, and the activated skill is callable
through the generic ``use_skill`` tool; and ``IntentTools`` exposes
``forge``/``use_skill`` only when a leg is wired, refusing with a named
reason otherwise.
"""

from __future__ import annotations

import time

import pytest

from reachy_nova.harness import forge_leg as forge_leg_module
from reachy_nova.harness.forge_leg import ForgeLeg
from reachy_nova.harness.tools import (
    AUTHOR_RULE,
    FORGE,
    FORGE_NOT_WIRED_REASON,
    USE_SKILL,
    IntentTools,
    TOOL_SPECS,
)

GOOD_REPLY_CONTENT = (
    "```SKILL.md\n"
    "---\n"
    "name: chirp-twice\n"
    "description: Chirp a greeting twice.\n"
    "---\n"
    "\n"
    "# Chirp Twice\n"
    "\n"
    "Chirps twice.\n"
    "```\n"
    "\n"
    "```executor.py\n"
    "def execute(params, ctx):\n"
    "    ctx.say('chirp chirp')\n"
    "    return '[chirped]'\n"
    "```\n"
)


class _FakeSession:
    """Stands in for the KiroSessionUnit: .prompt returns a scripted reply."""

    def __init__(self, reply: str | Exception = GOOD_REPLY_CONTENT):
        self.reply = reply
        self.prompts: list[str] = []

    def prompt(self, text: str, timeout: float = 120.0) -> str:
        self.prompts.append(text)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _FakeSonic:
    def __init__(self):
        self.injected: list[str] = []

    def inject_text(self, text: str, force: bool = False) -> None:
        self.injected.append(text)


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition never became true")


@pytest.fixture
def roots(tmp_path):
    return {"staging_root": tmp_path / "staged", "active_root": tmp_path / "active"}


@pytest.fixture(autouse=True)
def _kiro_writer_env(monkeypatch):
    monkeypatch.setenv("FORGE_WRITER", "kiro")


def test_forge_stages_activates_and_use_skill_runs_it(roots):
    sonic = _FakeSonic()
    leg = ForgeLeg(sonic, _FakeSession(), **roots)

    result = leg.forge("chirp a greeting twice")
    assert result["ok"] is True

    _wait_until(lambda: "chirp-twice" in leg.known_skills())
    # activation announced through the same seam the direct-SDK path uses
    assert any("chirp-twice" in text for text in sonic.injected)
    # activated WITHOUT a restart callable: the skill lives in the active root
    assert (roots["active_root"] / "chirp-twice" / "executor.py").is_file()

    outcome = leg.use_skill("chirp-twice", {})
    assert outcome["ok"] is True
    assert outcome["result"] == "[chirped]"
    # the forged ctx delegated say -> sonic.inject_text (defensive path)
    assert any("chirp chirp" in text for text in sonic.injected)


def test_forge_rejection_is_announced_not_silent(roots):
    sonic = _FakeSonic()
    leg = ForgeLeg(sonic, _FakeSession(reply="no fences here at all"), **roots)

    leg.forge("do something")
    _wait_until(lambda: any("safety check" in text for text in sonic.injected))
    assert leg.known_skills() == []


def test_forge_dead_session_rejects(roots):
    sonic = _FakeSonic()
    leg = ForgeLeg(sonic, _FakeSession(reply=RuntimeError("session dead")), **roots)

    leg.forge("do something")
    _wait_until(lambda: any("safety check" in text for text in sonic.injected))
    assert leg.known_skills() == []


def test_forge_empty_goal_refused_synchronously(roots):
    leg = ForgeLeg(_FakeSonic(), _FakeSession(), **roots)
    assert leg.forge("   ")["ok"] is False


def test_use_skill_unknown_name_lists_available(roots):
    sonic = _FakeSonic()
    leg = ForgeLeg(sonic, _FakeSession(), **roots)
    leg.forge("chirp a greeting twice")
    _wait_until(lambda: "chirp-twice" in leg.known_skills())

    outcome = leg.use_skill("no-such-skill")
    assert outcome["ok"] is False
    assert outcome["available"] == ["chirp-twice"]


def test_author_rule_delegates_to_kiro_rules_with_the_same_session(monkeypatch, roots):
    """ForgeLeg.author_rule delegates to kiro_rules.author_rule using the
    SAME session handle the leg was constructed with (qodo review comment
    3812045168)."""
    calls = []

    def fake_author_rule(goal, session, **kwargs):
        calls.append((goal, session))
        return {"ok": True, "rule_id": "nova-pat-nod", "verdict": "reload confirmed: 1", "reason": None}

    monkeypatch.setattr(forge_leg_module.kiro_rules, "author_rule", fake_author_rule)
    session = _FakeSession()
    leg = ForgeLeg(_FakeSonic(), session, **roots)

    result = leg.author_rule("nod when patted")

    assert result == {
        "ok": True,
        "rule_id": "nova-pat-nod",
        "verdict": "reload confirmed: 1",
        "reason": None,
    }
    assert calls == [("nod when patted", session)]


def test_author_rule_failure_dict_passes_through(monkeypatch, roots):
    def fake_author_rule(goal, session, **kwargs):
        return {
            "ok": False,
            "rule_id": None,
            "verdict": None,
            "reason": "no fenced rule object found in kiro's reply",
        }

    monkeypatch.setattr(forge_leg_module.kiro_rules, "author_rule", fake_author_rule)
    leg = ForgeLeg(_FakeSonic(), _FakeSession(), **roots)

    result = leg.author_rule("do a thing")

    assert result["ok"] is False
    assert result["rule_id"] is None
    assert "no fenced" in result["reason"]


def test_startup_reregisters_previously_forged_skills(roots):
    sonic = _FakeSonic()
    first = ForgeLeg(sonic, _FakeSession(), **roots)
    first.forge("chirp a greeting twice")
    _wait_until(lambda: "chirp-twice" in first.known_skills())

    reborn = ForgeLeg(_FakeSonic(), _FakeSession(), **roots)
    assert "chirp-twice" in reborn.known_skills()


# --------------------------------------------------------------------------- #
# The tool surface                                                            #
# --------------------------------------------------------------------------- #


def test_tools_refuse_forge_without_a_wired_leg():
    import json

    tools = IntentTools()
    for name, params in ((FORGE, {"goal": "x"}), (USE_SKILL, {"name": "x"})):
        payload = json.loads(tools.execute(name, params))
        assert payload["ok"] is False
        assert FORGE_NOT_WIRED_REASON in payload["error"]


def test_tools_delegate_to_the_wired_leg(roots):
    import json

    sonic = _FakeSonic()
    leg = ForgeLeg(sonic, _FakeSession(), **roots)
    tools = IntentTools(forge_leg=leg)

    queued = json.loads(tools.execute(FORGE, {"goal": "chirp a greeting twice"}))
    assert queued["ok"] is True
    _wait_until(lambda: "chirp-twice" in leg.known_skills())

    ran = json.loads(tools.execute(USE_SKILL, {"name": "chirp-twice"}))
    assert ran["ok"] is True
    assert ran["result"] == "[chirped]"


def test_tools_validate_forge_arguments(roots):
    import json

    tools = IntentTools(forge_leg=ForgeLeg(_FakeSonic(), _FakeSession(), **roots))
    bad_goal = json.loads(tools.execute(FORGE, {"goal": 7}))
    assert bad_goal["ok"] is False
    bad_name = json.loads(tools.execute(USE_SKILL, {"name": None}))
    assert bad_name["ok"] is False


def test_forge_and_use_skill_are_published_tool_specs():
    names = {spec["toolSpec"]["name"] if "toolSpec" in spec else spec.get("name") for spec in TOOL_SPECS}
    assert FORGE in names or any(FORGE in str(spec) for spec in TOOL_SPECS)
    assert USE_SKILL in names or any(USE_SKILL in str(spec) for spec in TOOL_SPECS)


# --------------------------------------------------------------------------- #
# author_rule — the same wired/unwired/validation contract as forge/use_skill #
# --------------------------------------------------------------------------- #


def test_tools_refuse_author_rule_without_a_wired_leg():
    import json

    tools = IntentTools()
    payload = json.loads(tools.execute(AUTHOR_RULE, {"goal": "nod when patted"}))
    assert payload["ok"] is False
    assert FORGE_NOT_WIRED_REASON in payload["error"]


def test_tools_delegate_author_rule_to_the_wired_leg(monkeypatch, roots):
    import json

    def fake_author_rule(goal, session, **kwargs):
        return {"ok": True, "rule_id": "nova-pat-nod", "verdict": "reload confirmed: 1", "reason": None}

    monkeypatch.setattr(forge_leg_module.kiro_rules, "author_rule", fake_author_rule)
    leg = ForgeLeg(_FakeSonic(), _FakeSession(), **roots)
    tools = IntentTools(forge_leg=leg)

    result = json.loads(tools.execute(AUTHOR_RULE, {"goal": "nod when patted"}))
    assert result["ok"] is True
    assert result["rule_id"] == "nova-pat-nod"


def test_tools_delegate_author_rule_failure_dict_through(monkeypatch, roots):
    import json

    def fake_author_rule(goal, session, **kwargs):
        return {"ok": False, "rule_id": None, "verdict": None, "reason": "kiro session.prompt() failed: boom"}

    monkeypatch.setattr(forge_leg_module.kiro_rules, "author_rule", fake_author_rule)
    leg = ForgeLeg(_FakeSonic(), _FakeSession(), **roots)
    tools = IntentTools(forge_leg=leg)

    result = json.loads(tools.execute(AUTHOR_RULE, {"goal": "nod when patted"}))
    assert result["ok"] is False
    assert "boom" in result["reason"]


def test_tools_validate_author_rule_arguments(roots):
    import json

    tools = IntentTools(forge_leg=ForgeLeg(_FakeSonic(), _FakeSession(), **roots))
    missing = json.loads(tools.execute(AUTHOR_RULE, {}))
    assert missing["ok"] is False
    not_a_string = json.loads(tools.execute(AUTHOR_RULE, {"goal": 7}))
    assert not_a_string["ok"] is False
    blank = json.loads(tools.execute(AUTHOR_RULE, {"goal": "   "}))
    assert blank["ok"] is False


def test_author_rule_is_a_published_tool_spec():
    names = {spec["toolSpec"]["name"] for spec in TOOL_SPECS}
    assert AUTHOR_RULE in names
