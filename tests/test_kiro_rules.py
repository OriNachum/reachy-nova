"""Kiro rule authoring: goal -> fenced rule -> rules_overlay (task t7).

``kiro_rules.author_rule`` is the ONLY thing this module tests end to end: a
fake ``session.prompt(...)`` stands in for a real Kiro ACP session (see
``reachy_nova/kiro_acp.py``), and every landing goes through the REAL
``reachy_nova.harness.rules_overlay`` — the byte-compare tests below reuse
the exact operator-content fixtures ``test_harness_rules_overlay.py`` uses,
because rules_overlay's write/merge/verbatim-preservation behaviour is not
this module's to re-test; only that ``author_rule`` calls it correctly and
surfaces what it reports.

Nothing here imports ``reachy_mini``: the file layout IS the contract (see
``tests/test_harness_boundary.py``).
"""

from __future__ import annotations

import json
import threading

import pytest

from reachy_nova.harness import kiro_rules, statedir
from reachy_nova.harness.rules_overlay import (
    MANAGED_BEGIN,
    MANAGED_END,
    RuleRefused,
)

GOAL = "nod when someone pats my head"

RULE = {
    "id": "nova-pat-nod",
    "when": {"field": "pat", "op": "gt", "value": 0.5},
    "run": "nod",
    "duration_s": 2.0,
}


def fenced(obj, label="json") -> str:
    return f"```{label}\n{json.dumps(obj)}\n```"


# --------------------------------------------------------------------------- #
# Fixtures shared with test_harness_rules_overlay.py                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    return tmp_path


@pytest.fixture
def overlay_path(state_dir):
    return statedir.rules_overlay_path()


class ReloadEngine:
    """Answer reload commands like a running engine's ReloadDriver."""

    def __init__(self, response=None):
        self.response = response or {"ok": True, "rules": 1}
        self.seen: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        statedir.reload_commands_dir().mkdir(parents=True, exist_ok=True)
        statedir.reload_results_dir().mkdir(parents=True, exist_ok=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)
        return False

    def _run(self):
        while not self._stop.is_set():
            for path in sorted(statedir.reload_commands_dir().glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                path.unlink(missing_ok=True)
                self.seen.append(payload)
                cmd_id = payload.get("cmd_id")
                if cmd_id:
                    (statedir.reload_results_dir() / f"{cmd_id}.json").write_text(
                        json.dumps(self.response), encoding="utf-8"
                    )
            self._stop.wait(0.005)


class FakeSession:
    """Stands in for a ``KiroAcpSession``: same ``.prompt(text, timeout=...)`` surface."""

    def __init__(self, reply: str | None = None, raises: Exception | None = None):
        self.reply = reply
        self.raises = raises
        self.calls: list[tuple[str, float]] = []

    def prompt(self, text: str, timeout: float = 90.0) -> str:
        self.calls.append((text, timeout))
        if self.raises is not None:
            raise self.raises
        assert self.reply is not None
        return self.reply


def author(goal=GOAL, session=None, **kwargs):
    kwargs.setdefault("reload_timeout", 0.15)
    return kiro_rules.author_rule(goal, session, **kwargs)


# --------------------------------------------------------------------------- #
# The authoring prompt                                                        #
# --------------------------------------------------------------------------- #


def test_build_prompt_restates_the_schema_and_the_goal():
    text = kiro_rules.build_prompt(GOAL)
    assert GOAL in text
    assert "nova-" in text
    assert "500" in text  # MAX_SAY_CHARS
    assert "pat" in text and "face" in text and "transcript" in text  # sense fields
    assert "gt" in text and "is_true" in text and "absent_for" in text  # comparators
    assert "```json" in text


# --------------------------------------------------------------------------- #
# The happy path — byte-compare against operator content                      #
# --------------------------------------------------------------------------- #

OPERATOR_HEAD = (
    "# my own rules\n"
    'active_mode = "calm"\n'
    "\n"
    "[[react]]\n"
    'id = "op-face"\n'
    'when = { field = "face", op = "is_true" }\n'
    'run = "nod"\n'
    "duration_s = 1.0\n"
)
OPERATOR_TAIL = (
    "\n"
    "[[inhibit]]\n"
    'id = "op-quiet"\n'
    'when = { field = "speech", op = "is_true" }\n'
    'disable = ["antenna-sway"]\n'
)


def test_a_goal_lands_a_rule_and_preserves_operator_bytes_verbatim(overlay_path):
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(OPERATOR_HEAD, encoding="utf-8")
    session = FakeSession(reply=f"Sure, here you go:\n\n{fenced(RULE)}\n\nHope that helps!")

    with ReloadEngine() as engine:
        result = author(session=session)

    assert result["ok"] is True
    assert result["rule_id"] == "nova-pat-nod"
    assert result["reason"] is None
    assert "confirmed" in result["verdict"]
    assert len(engine.seen) == 1
    assert len(session.calls) == 1
    assert GOAL in session.calls[0][0]

    text = overlay_path.read_text(encoding="utf-8")
    assert text.startswith(OPERATOR_HEAD)

    # Splice in an operator-authored tail by hand (as if the operator hand-
    # edits around nova's block), author a second rule, and prove every byte
    # on both sides of the sentinel block survives verbatim.
    head, _, rest = text.partition(MANAGED_BEGIN)
    managed, _, tail = rest.partition(MANAGED_END)
    overlay_path.write_text(
        OPERATOR_HEAD + MANAGED_BEGIN + managed + MANAGED_END + OPERATOR_TAIL,
        encoding="utf-8",
    )
    session2 = FakeSession(reply=fenced({**RULE, "id": "nova-second"}))
    result2 = author(session=session2)
    assert result2["ok"] is True
    assert result2["rule_id"] == "nova-second"

    after = overlay_path.read_text(encoding="utf-8")
    new_head, _, new_rest = after.partition(MANAGED_BEGIN)
    _, _, new_tail = new_rest.partition(MANAGED_END)
    assert new_head == OPERATOR_HEAD
    assert new_tail == OPERATOR_TAIL
    assert "nova-pat-nod" in after and "nova-second" in after


def test_re_authoring_the_same_rule_merges_by_id_not_duplicates(overlay_path):
    session = FakeSession(reply=fenced(RULE))
    author(session=session)
    session2 = FakeSession(reply=fenced({**RULE, "run": "shake"}))
    result = author(session=session2)
    assert result["ok"] is True
    assert result["rule_id"] == "nova-pat-nod"
    text = overlay_path.read_text(encoding="utf-8")
    assert text.count("[[react]]") == 1
    assert 'run = "shake"' in text


# --------------------------------------------------------------------------- #
# The reload verdict is surfaced, never swallowed                             #
# --------------------------------------------------------------------------- #


def test_a_rejected_reload_is_surfaced_as_a_failed_result(overlay_path):
    session = FakeSession(reply=fenced(RULE))
    with ReloadEngine(response={"ok": False, "error": "rules did not load"}):
        result = author(session=session)
    assert result["ok"] is False
    assert result["rule_id"] == "nova-pat-nod"
    assert "rejected" in result["verdict"]
    assert "rules did not load" in result["reason"]
    # The rule still landed on disk — a REJECTED reload is a running-engine
    # fact, not a write failure.
    assert overlay_path.is_file()
    assert "nova-pat-nod" in overlay_path.read_text(encoding="utf-8")


def test_an_unconfirmed_reload_is_still_ok_the_file_is_on_disk(overlay_path):
    session = FakeSession(reply=fenced(RULE))
    result = author(session=session)
    assert result["ok"] is True
    assert "not confirmed" in result["verdict"]
    assert result["reason"] is None


def test_an_unchanged_upsert_is_ok_and_reports_unchanged(overlay_path):
    session = FakeSession(reply=fenced(RULE))
    author(session=session)
    session2 = FakeSession(reply=fenced(RULE))
    result = author(session=session2)
    assert result["ok"] is True
    assert "unchanged" in result["verdict"]


# --------------------------------------------------------------------------- #
# Failure modes — every one comes back structured, never an exception         #
# --------------------------------------------------------------------------- #


def test_a_raising_session_produces_a_structured_failure(overlay_path):
    session = FakeSession(raises=RuntimeError("kiro-cli process is not running"))
    result = author(session=session)
    assert set(result) == {"ok", "rule_id", "verdict", "reason"}
    assert result["ok"] is False
    assert result["rule_id"] is None
    assert result["verdict"] is None
    assert "kiro-cli process is not running" in result["reason"]
    assert not overlay_path.exists()


def test_a_reply_with_no_fence_produces_a_structured_failure(overlay_path):
    session = FakeSession(reply="Sure! Here's a rule: id nova-pat-nod, run nod when patted.")
    result = author(session=session)
    assert result["ok"] is False
    assert result["rule_id"] is None
    assert result["verdict"] is None
    assert "no fenced" in result["reason"]
    assert not overlay_path.exists()


def test_a_fence_with_unparseable_json_produces_a_structured_failure(overlay_path):
    session = FakeSession(reply="```json\n{not: valid, json,,\n```")
    result = author(session=session)
    assert result["ok"] is False
    assert "not valid JSON" in result["reason"]
    assert not overlay_path.exists()


def test_a_fenced_json_array_is_not_an_object_and_fails_structured(overlay_path):
    session = FakeSession(reply=fenced([RULE]))
    result = author(session=session)
    assert result["ok"] is False
    assert "not a JSON object" in result["reason"]


def test_overlay_refusal_produces_a_structured_failure_with_rule_id(overlay_path):
    bad = {**RULE, "id": "pat-nod"}  # outside the nova- namespace
    session = FakeSession(reply=fenced(bad))
    result = author(session=session)
    assert result["ok"] is False
    assert result["rule_id"] == "pat-nod"
    assert result["verdict"] is None
    assert "refused" in result["reason"]
    assert not overlay_path.exists()


def test_a_say_over_the_cap_is_refused_via_the_overlays_own_validator(overlay_path):
    bad = {**RULE, "say": "a" * 501}
    session = FakeSession(reply=fenced(bad))
    result = author(session=session)
    assert result["ok"] is False
    assert "500" in result["reason"]
    assert not overlay_path.exists()


def test_overlay_refusal_raises_ruleRefused_internally_but_never_escapes(overlay_path, monkeypatch):
    # Belt-and-suspenders: confirm the specific exception type rules_overlay
    # raises for a bad rule is exactly what gets caught (not some broader
    # accidental swallow of an unrelated bug).
    from reachy_nova.harness import rules_overlay

    with pytest.raises(RuleRefused):
        rules_overlay.validate_rule({**RULE, "id": "pat-nod"})


# --------------------------------------------------------------------------- #
# Fenced-object extraction — liberal find, strict parse                       #
# --------------------------------------------------------------------------- #


def test_extraction_rejects_a_reply_with_more_than_one_fence(overlay_path):
    """The protocol requires exactly one fence — a second one is refused, not
    silently resolved by preferring the json-labeled one (qodo review comment
    3812045206)."""
    reply = (
        "Let me think about this...\n"
        "```\n"
        "not the rule, just some scratch notes\n"
        "```\n"
        "Here is the rule:\n"
        f"{fenced(RULE, label='json')}\n"
    )
    session = FakeSession(reply=reply)
    result = author(session=session)
    assert result["ok"] is False
    assert result["rule_id"] is None
    assert result["verdict"] is None
    assert "multiple fences" in result["reason"]
    assert "2 fenced" in result["reason"]
    assert not overlay_path.exists()


def test_extraction_accepts_a_single_fence_with_any_label(overlay_path):
    reply = f"```\n{json.dumps(RULE)}\n```"
    session = FakeSession(reply=reply)
    result = author(session=session)
    assert result["ok"] is True
    assert result["rule_id"] == "nova-pat-nod"


def test_extraction_rejects_a_single_but_empty_fence(overlay_path):
    session = FakeSession(reply="```json\n\n```")
    result = author(session=session)
    assert result["ok"] is False
    assert result["rule_id"] is None
    assert "empty" in result["reason"]
    assert not overlay_path.exists()


# --------------------------------------------------------------------------- #
# Custom / overriding overlay injection point                                 #
# --------------------------------------------------------------------------- #


def test_a_stand_in_overlay_is_used_instead_of_the_real_one(overlay_path):
    calls = []

    class FakeOverlay:
        RuleRefused = RuleRefused

        def upsert_rule(self, rule, *, path=None, reload_timeout=1.0):
            calls.append((dict(rule), path, reload_timeout))
            return True, "reload confirmed: {}"

    session = FakeSession(reply=fenced(RULE))
    result = author(session=session, overlay=FakeOverlay())
    assert result["ok"] is True
    assert result["rule_id"] == "nova-pat-nod"
    assert len(calls) == 1
    assert calls[0][0]["id"] == "nova-pat-nod"
    # The real overlay was never touched.
    assert not overlay_path.exists()


def test_a_stand_in_overlay_reporting_a_rejected_verdict_surfaces_it(overlay_path):
    class FakeOverlay:
        def upsert_rule(self, rule, *, path=None, reload_timeout=1.0):
            return True, "reload rejected: engine says no"

    session = FakeSession(reply=fenced(RULE))
    result = author(session=session, overlay=FakeOverlay())
    assert result["ok"] is False
    assert result["verdict"] == "reload rejected: engine says no"
    assert result["reason"] == "reload rejected: engine says no"
