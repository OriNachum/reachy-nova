"""t6: quiet inject templates + the ``voice``/``sense`` rules.yaml fields.

Covers the two things test_rules_coverage.py / test_harness_bus.py don't:

1. The reflex-adjacent inject templates (``rule/fire``, its two per-rule
   overrides, and the three ``pat/*`` entries) read as quiet situational
   context, not narration of the robot's own reflex mechanism — the pre-t6
   ``rule/fire`` text ("Your body just reacted on its own — the '{rule}'
   reflex fired.", observed live 2026-08-26) is exactly the kind of
   self-narration this guards against.
2. ``route_event`` appends the right voice marker (silent/brief/free/absent)
   to the rendered inject text, per rule entry.

Runs against the real ``config/nervous-system/rules.yaml`` and the real
``reachy_nova.harness.bus`` module — no broker, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reachy_nova.harness import bus

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "config" / "nervous-system" / "rules.yaml"

# Substrings (case-insensitive) that would mean the inject text is narrating
# the reflex mechanism itself rather than giving quiet situational context.
FORBIDDEN_SUBSTRINGS = ("reflex", "rule", "reacted on its own", "leaning", "antenna")

# The entries this task's acceptance criterion 1 requires to be quiet.
QUIET_ENTRIES = (
    "rule/fire",
    "rule/fire:pat-acknowledge",
    "rule/fire:nova-face-noticed",
    "pat/level1",
    "pat/level2",
    "pat/detected",
)


@pytest.fixture(scope="module")
def rules_cfg() -> dict:
    return bus.load_rules(RULES_PATH)


@pytest.fixture(scope="module")
def raw_rules() -> dict:
    with open(RULES_PATH) as f:
        return yaml.safe_load(f)


def _split_key(key: str) -> tuple[str, str, str | None]:
    """"source/type" or "source/type:rule" -> (source, type, rule|None)."""
    base, _, rule_name = key.partition(":")
    source, _, event_type = base.partition("/")
    return source, event_type, (rule_name or None)


@pytest.mark.parametrize("key", QUIET_ENTRIES)
def test_quiet_entries_render_without_reflex_narration(rules_cfg, key):
    source, event_type, rule_name = _split_key(key)
    payload = {"rule": rule_name} if rule_name else {}
    text, reason = bus.route_event(rules_cfg, source, event_type, payload)
    assert reason == bus.REASON_INJECT
    assert text
    lowered = text.lower()
    for forbidden in FORBIDDEN_SUBSTRINGS:
        assert forbidden not in lowered, (
            f"{key} inject text still narrates the reflex mechanism "
            f"(found {forbidden!r} in {text!r})"
        )


def test_generic_rule_fire_still_produces_an_inject(rules_cfg):
    """Quiet != dropped: the generic rule/fire entry still always injects."""
    text, reason = bus.route_event(rules_cfg, "rule", "fire", {"rule": "some-unmapped-rule"})
    assert reason == bus.REASON_INJECT
    assert text
    assert text.endswith(bus.VOICE_MARKERS["silent"])


@pytest.mark.parametrize(
    "key,expected_voice",
    [
        ("rule/fire", "silent"),
        ("rule/fire:pat-acknowledge", "brief"),
        ("rule/fire:nova-face-noticed", "brief"),
        ("pat/level1", "brief"),
        ("pat/level2", "brief"),
        ("pat/detected", "brief"),
    ],
)
def test_entries_carry_the_expected_voice_field(raw_rules, key, expected_voice):
    assert raw_rules["rules"][key].get("voice") == expected_voice


@pytest.mark.parametrize(
    "key,expected_sense",
    [
        ("rule/fire:pat-acknowledge", "pat"),
        ("pat/level1", "pat"),
        ("pat/level2", "pat"),
        ("pat/detected", "pat"),
        ("rule/fire:nova-face-noticed", "face"),
    ],
)
def test_entries_carry_the_expected_sense_field(raw_rules, key, expected_sense):
    assert raw_rules["rules"][key].get("sense") == expected_sense


def test_default_block_is_voice_free(raw_rules):
    assert raw_rules["default"].get("voice") == "free"


def test_route_event_appends_silent_marker():
    cfg = {
        "rules": {"src/type": {"inject_template": "context", "voice": "silent"}},
        "default": {},
    }
    text, reason = bus.route_event(cfg, "src", "type", {})
    assert reason == bus.REASON_INJECT
    assert text == "context (quiet: do not speak about this)"


def test_route_event_appends_brief_marker():
    cfg = {
        "rules": {"src/type": {"inject_template": "context", "voice": "brief"}},
        "default": {},
    }
    text, reason = bus.route_event(cfg, "src", "type", {})
    assert reason == bus.REASON_INJECT
    assert text == "context (react briefly if at all)"


def test_route_event_free_voice_appends_nothing():
    cfg = {
        "rules": {"src/type": {"inject_template": "context", "voice": "free"}},
        "default": {},
    }
    text, reason = bus.route_event(cfg, "src", "type", {})
    assert reason == bus.REASON_INJECT
    assert text == "context"


def test_route_event_absent_voice_defaults_to_free():
    cfg = {
        "rules": {"src/type": {"inject_template": "context"}},
        "default": {},
    }
    text, reason = bus.route_event(cfg, "src", "type", {})
    assert reason == bus.REASON_INJECT
    assert text == "context"
