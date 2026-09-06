"""The nova-managed block of the behavior rules overlay (task t9).

The overlay (``<state>/behavior/rules.toml``) is the OPERATOR's file. Nova only
owns a sentinel-delimited block inside it, and every byte outside that block
must survive a write untouched — that is the property most of these tests are
about. The rest pin the fail-closed validator (a rules file the engine would
reject is never installed, not even momentarily at the real path) and the
reload handshake.

Nothing here imports ``reachy`` or ``reachy_mini``: the file layout IS the
contract.
"""

from __future__ import annotations

import json
import threading
import tomllib

import pytest

from reachy_nova.harness import statedir
from reachy_nova.harness.rules_overlay import (
    MANAGED_BEGIN,
    MANAGED_END,
    MAX_SAY_CHARS,
    RULE_ID_PREFIX,
    RuleRefused,
    list_rules,
    retire_rule,
    upsert_rule,
)

RULE = {
    "id": "nova-pat-nod",
    "when": {"field": "pat", "op": "gt", "value": 0.5},
    "run": "nod",
    "duration_s": 2.0,
}


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    return tmp_path


@pytest.fixture
def overlay(state_dir):
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


def upsert(rule, **kwargs):
    kwargs.setdefault("reload_timeout", 0.15)
    return upsert_rule(rule, **kwargs)


def retire(rule_id, **kwargs):
    kwargs.setdefault("reload_timeout", 0.15)
    return retire_rule(rule_id, **kwargs)


# --------------------------------------------------------------------------- #
# Writing the managed block                                                   #
# --------------------------------------------------------------------------- #


def test_fresh_overlay_gets_sentinels_and_the_rule(overlay):
    changed, verdict = upsert(RULE)
    assert changed is True
    text = overlay.read_text(encoding="utf-8")
    assert MANAGED_BEGIN in text
    assert MANAGED_END in text
    assert text.index(MANAGED_BEGIN) < text.index("[[react]]") < text.index(MANAGED_END)
    assert verdict
    assert list_rules() == ("nova-pat-nod",)


def test_rendered_rule_parses_as_toml_with_the_right_values(overlay):
    upsert({**RULE, "params": {"amp": 8.0}, "cooldown_s": 3, "say": "hello"})
    data = tomllib.loads(overlay.read_text(encoding="utf-8"))
    (entry,) = data["react"]
    assert entry == {
        "id": "nova-pat-nod",
        "when": {"field": "pat", "op": "gt", "value": 0.5},
        "run": "nod",
        "params": {"amp": 8.0},
        "duration_s": 2.0,
        "cooldown_s": 3.0,
        "say": "hello",
    }


def test_rendered_key_order_is_the_documented_one(overlay):
    upsert(
        {
            "say": "hi",
            "hysteresis": 0.1,
            "cooldown_s": 3.0,
            "duration_s": 2.0,
            "params": {"amp": 8.0},
            "run": "nod",
            "when": {"field": "pat", "op": "gt", "value": 0.5},
            "id": "nova-pat-nod",
        }
    )
    body = overlay.read_text(encoding="utf-8")
    keys = [
        line.split("=", 1)[0].strip()
        for line in body.splitlines()
        if "=" in line and not line.startswith("#")
    ]
    assert keys == [
        "id",
        "when",
        "run",
        "params",
        "duration_s",
        "cooldown_s",
        "hysteresis",
        "say",
    ]


def test_upserting_the_same_id_twice_keeps_one_entry(overlay):
    upsert(RULE)
    changed, verdict = upsert(RULE)
    assert changed is False
    assert "unchanged" in verdict
    assert overlay.read_text(encoding="utf-8").count("[[react]]") == 1
    assert list_rules() == ("nova-pat-nod",)


def test_upserting_the_same_id_with_new_content_replaces_it(overlay):
    upsert(RULE)
    changed, _ = upsert({**RULE, "run": "shake"})
    assert changed is True
    data = tomllib.loads(overlay.read_text(encoding="utf-8"))
    assert len(data["react"]) == 1
    assert data["react"][0]["run"] == "shake"


def test_two_ids_coexist_sorted(overlay):
    upsert({**RULE, "id": "nova-zeta"})
    upsert({**RULE, "id": "nova-alpha"})
    assert list_rules() == ("nova-alpha", "nova-zeta")
    data = tomllib.loads(overlay.read_text(encoding="utf-8"))
    assert [e["id"] for e in data["react"]] == ["nova-alpha", "nova-zeta"]


def test_list_rules_on_a_missing_overlay_is_empty(state_dir):
    assert list_rules() == ()


def test_no_temp_files_are_left_behind(overlay):
    upsert(RULE)
    leftovers = [p.name for p in overlay.parent.iterdir() if ".tmp." in p.name]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# The operator's bytes                                                        #
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


def test_operator_text_is_preserved_byte_exact(overlay):
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(OPERATOR_HEAD, encoding="utf-8")
    upsert(RULE)
    text = overlay.read_text(encoding="utf-8")
    assert text.startswith(OPERATOR_HEAD)


def test_operator_head_and_tail_around_the_block_are_preserved(overlay):
    overlay.parent.mkdir(parents=True, exist_ok=True)
    upsert(RULE)
    # The operator hand-edits around the managed block.
    text = overlay.read_text(encoding="utf-8")
    head, _, rest = text.partition(MANAGED_BEGIN)
    managed, _, tail = rest.partition(MANAGED_END)
    overlay.write_text(
        OPERATOR_HEAD + MANAGED_BEGIN + managed + MANAGED_END + OPERATOR_TAIL,
        encoding="utf-8",
    )
    upsert({**RULE, "id": "nova-second"})
    after = overlay.read_text(encoding="utf-8")
    new_head, _, new_rest = after.partition(MANAGED_BEGIN)
    _, _, new_tail = new_rest.partition(MANAGED_END)
    assert new_head == OPERATOR_HEAD
    assert new_tail == OPERATOR_TAIL
    assert sorted(e["id"] for e in tomllib.loads(after)["react"]) == [
        "nova-pat-nod",
        "nova-second",
        "op-face",
    ]


def test_repeated_writes_do_not_grow_the_file(overlay):
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(OPERATOR_HEAD, encoding="utf-8")
    upsert(RULE)
    once = overlay.read_text(encoding="utf-8")
    upsert({**RULE, "run": "shake"})
    upsert(RULE)
    assert overlay.read_text(encoding="utf-8") == once


def test_list_rules_ignores_operator_ids(overlay):
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(OPERATOR_HEAD, encoding="utf-8")
    upsert(RULE)
    assert list_rules() == ("nova-pat-nod",)


# --------------------------------------------------------------------------- #
# Fail-closed refusals — nothing is written                                   #
# --------------------------------------------------------------------------- #

BAD_RULES = [
    ("id", {**RULE, "id": "pat-nod"}),  # outside the nova- namespace
    ("id", {**RULE, "id": ""}),
    ("id", {k: v for k, v in RULE.items() if k != "id"}),
    ("field", {**RULE, "when": {"field": "mood", "op": "gt", "value": 1}}),
    ("op", {**RULE, "when": {"field": "pat", "op": "explodes", "value": 1}}),
    ("value", {**RULE, "when": {"field": "pat", "op": "gt"}}),  # ordered op needs a value
    # a boolean op takes no value at all
    ("value", {**RULE, "when": {"field": "pat", "op": "is_true", "value": 1}}),
    ("when", {**RULE, "when": {"field": "pat", "op": "gt", "value": 1, "extra": 2}}),
    ("when", {**RULE, "when": "pat > 1"}),
    ("when", {k: v for k, v in RULE.items() if k != "when"}),
    ("run", {k: v for k, v in RULE.items() if k != "run"}),
    ("run", {**RULE, "run": 7}),
    ("params", {**RULE, "params": {"amp": "big"}}),
    ("duration_s", {**RULE, "duration_s": 0}),
    ("cooldown_s", {**RULE, "cooldown_s": -1}),
    ("hysteresis", {**RULE, "hysteresis": "wide"}),
    ("unexpected", {**RULE, "exec": "rm -rf /"}),
]


_BAD_RULE_IDS = [f"{hint}-{i}" for i, (hint, _) in enumerate(BAD_RULES)]


@pytest.mark.parametrize("hint,rule", BAD_RULES, ids=_BAD_RULE_IDS)
def test_bad_rules_are_refused_before_any_file_write(overlay, hint, rule):
    with pytest.raises(RuleRefused) as excinfo:
        upsert(rule)
    assert hint in str(excinfo.value)
    assert not overlay.exists()
    assert list(statedir.reload_commands_dir().glob("*.json")) == []


def test_a_say_over_the_cap_is_refused_never_truncated(overlay):
    with pytest.raises(RuleRefused) as excinfo:
        upsert({**RULE, "say": "a" * (MAX_SAY_CHARS + 1)})
    assert str(MAX_SAY_CHARS) in str(excinfo.value)
    assert not overlay.exists()


def test_a_say_at_the_cap_is_accepted(overlay):
    upsert({**RULE, "say": "a" * MAX_SAY_CHARS})
    assert list_rules() == ("nova-pat-nod",)


def test_a_non_dict_rule_is_refused(overlay):
    with pytest.raises(RuleRefused):
        upsert(["nova-x"])
    assert not overlay.exists()


def test_an_unparseable_operator_file_is_refused_not_overwritten(overlay):
    overlay.parent.mkdir(parents=True, exist_ok=True)
    broken = "this is not = = toml\n"
    overlay.write_text(broken, encoding="utf-8")
    with pytest.raises(RuleRefused):
        upsert(RULE)
    assert overlay.read_text(encoding="utf-8") == broken


def test_the_rule_id_prefix_is_the_documented_one():
    assert RULE_ID_PREFIX == "nova-"


# --------------------------------------------------------------------------- #
# The reload handshake                                                        #
# --------------------------------------------------------------------------- #


def test_reload_command_is_written_and_confirmed(overlay):
    with ReloadEngine() as engine:
        _, verdict = upsert(RULE)
    assert len(engine.seen) == 1
    assert set(engine.seen[0]) == {"cmd_id"}
    assert "confirmed" in verdict
    assert "not confirmed" not in verdict


def test_reload_command_file_name_is_ns_prefixed(overlay):
    upsert(RULE)
    (path,) = list(statedir.reload_commands_dir().glob("*.json"))
    stamp, _, rest = path.name.partition("-")
    assert stamp.isdigit()
    assert rest.endswith(".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"cmd_id"}
    assert path.name == f"{stamp}-{payload['cmd_id']}.json"


def test_an_unconfirmed_reload_is_reported_as_such(overlay):
    _, verdict = upsert(RULE)
    assert "not confirmed" in verdict
    # The overlay is still written — a later engine picks it up on its own reload.
    assert overlay.is_file()


def test_a_rejected_reload_is_reported_and_leaves_the_old_rules_running(overlay):
    with ReloadEngine(response={"ok": False, "error": "rules did not load"}):
        _, verdict = upsert(RULE)
    assert "rejected" in verdict
    assert "rules did not load" in verdict


def test_an_unchanged_upsert_submits_no_reload(overlay):
    upsert(RULE)
    for path in statedir.reload_commands_dir().glob("*.json"):
        path.unlink()
    upsert(RULE)
    assert list(statedir.reload_commands_dir().glob("*.json")) == []


# --------------------------------------------------------------------------- #
# retire_rule — tombstones (task t10)                                         #
# --------------------------------------------------------------------------- #

FACE_NOTICED = {
    "id": "nova-face-noticed",
    "when": {"field": "face", "op": "is_true"},
    "run": "nod",
    "duration_s": 2.0,
    "cooldown_s": 30.0,
}


def test_retiring_a_rule_in_the_block_leaves_only_id_and_enabled_false(overlay):
    upsert(FACE_NOTICED)
    for path in statedir.reload_commands_dir().glob("*.json"):
        path.unlink()
    with ReloadEngine() as engine:
        result = retire(FACE_NOTICED["id"])
    assert result["changed"] is True
    assert "confirmed" in result["verdict"]
    assert len(engine.seen) == 1

    data = tomllib.loads(overlay.read_text(encoding="utf-8"))
    (entry,) = data["react"]
    assert entry == {"id": "nova-face-noticed", "enabled": False}


def test_retiring_preserves_operator_text_byte_identical(overlay):
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(OPERATOR_HEAD, encoding="utf-8")
    upsert(FACE_NOTICED)
    before_head = overlay.read_text(encoding="utf-8")
    assert before_head.startswith(OPERATOR_HEAD)

    retire(FACE_NOTICED["id"])
    after = overlay.read_text(encoding="utf-8")
    assert after.startswith(OPERATOR_HEAD)
    head, _, rest = after.partition(MANAGED_BEGIN)
    assert head == before_head[: before_head.index(MANAGED_BEGIN)]


def test_a_retired_rule_file_re_parses_under_the_module_validator(overlay):
    upsert(FACE_NOTICED)
    retire(FACE_NOTICED["id"])
    # validate_rules_document is exercised by _install on every write; a
    # second, independent re-parse here proves the file it left behind is
    # still valid on its own.
    data = tomllib.loads(overlay.read_text(encoding="utf-8"))
    from reachy_nova.harness.rules_overlay import validate_rules_document

    validate_rules_document(data)


def test_a_second_retire_of_the_same_id_is_a_no_op(overlay):
    upsert(FACE_NOTICED)
    retire(FACE_NOTICED["id"])
    before = overlay.read_text(encoding="utf-8")
    before_mtime = overlay.stat().st_mtime_ns

    for path in statedir.reload_commands_dir().glob("*.json"):
        path.unlink()
    result = retire(FACE_NOTICED["id"])

    assert result == {"changed": False, "verdict": None}
    assert overlay.read_text(encoding="utf-8") == before
    assert overlay.stat().st_mtime_ns == before_mtime
    assert list(statedir.reload_commands_dir().glob("*.json")) == []


def test_retiring_an_id_not_in_the_block_still_writes_the_tombstone(overlay):
    # No prior upsert: 'nova-face-noticed' is not in the managed block at
    # all — it may be a SHIPPED or operator rule of that id, and the tombstone
    # must still disable it.
    result = retire("nova-face-noticed")
    assert result["changed"] is True
    data = tomllib.loads(overlay.read_text(encoding="utf-8"))
    (entry,) = data["react"]
    assert entry == {"id": "nova-face-noticed", "enabled": False}


def test_retiring_a_non_nova_prefixed_id_is_accepted(overlay):
    """A tombstone may target an operator's own (unprefixed) rule id."""
    result = retire("op-face")
    assert result["changed"] is True
    data = tomllib.loads(overlay.read_text(encoding="utf-8"))
    (entry,) = data["react"]
    assert entry == {"id": "op-face", "enabled": False}


def test_retire_with_reload_false_submits_no_reload(overlay):
    result = retire(FACE_NOTICED["id"], reload=False)
    assert result == {"changed": True, "verdict": None}
    assert list(statedir.reload_commands_dir().glob("*.json")) == []
