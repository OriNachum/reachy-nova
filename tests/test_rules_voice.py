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


def test_face_lost_is_a_quiet_brief_face_sense(rules_cfg, raw_rules):
    """t13: face-lost is worth a quiet cue (the lock persists), not narration."""
    assert raw_rules["rules"]["motion/face-lost"].get("voice") == "brief"
    assert raw_rules["rules"]["motion/face-lost"].get("sense") == "face"

    text, reason = bus.route_event(rules_cfg, "motion", "face-lost", {"id": "p1", "absent_s": 4})
    assert reason == bus.REASON_INJECT
    assert text.endswith(bus.VOICE_MARKERS["brief"])
    lowered = text.lower()
    for forbidden in FORBIDDEN_SUBSTRINGS:
        assert forbidden not in lowered


def test_lock_released_is_silent_and_names_the_reason_in_plain_words(rules_cfg, raw_rules):
    """t13: lock-released is bookkeeping (voice: silent) but still says WHY,
    in words a person understands — never the raw reason token alone."""
    assert raw_rules["rules"]["motion/lock-released"].get("voice") == "silent"

    text, reason = bus.route_event(
        rules_cfg, "motion", "lock-released", {"id": "p1", "reason": "max-hold"}
    )
    assert reason == bus.REASON_INJECT
    assert "max-hold" in text
    assert text.endswith(bus.VOICE_MARKERS["silent"])
    lowered = text.lower()
    for forbidden in FORBIDDEN_SUBSTRINGS:
        assert forbidden not in lowered


def test_route_event_absent_voice_defaults_to_free():
    cfg = {
        "rules": {"src/type": {"inject_template": "context"}},
        "default": {},
    }
    text, reason = bus.route_event(cfg, "src", "type", {})
    assert reason == bus.REASON_INJECT
    assert text == "context"


# --------------------------------------------------------------------------- #
# Live finding L3 (on-device run 2026-08-26 18:26–18:31)                      #
#                                                                             #
# While a face lock was held, the runtime's own `look-toward-sound` rule kept  #
# re-admitting `orient-to-sound` into the standing inhibition and being        #
# refused. Every ~10 s the harness spoke "The runtime blocked your intention   #
# 'set_inhibition'." — the body's reflex being held back exactly as asked,     #
# narrated as Nova's own request failing. `intent/applied` was the same story  #
# on the other edge: "Your standing intention 'set_inhibition' is now in       #
# effect." after every quiet arm.                                             #
# --------------------------------------------------------------------------- #


def test_l3_intent_blocked_is_a_silent_deduped_body_cue(rules_cfg, raw_rules):
    entry = raw_rules["rules"]["intent/blocked"]
    assert entry.get("voice") == "silent"
    assert entry.get("dedupe") == "reflex-held"

    text, reason = bus.route_event(
        rules_cfg, "intent", "blocked", {"name": "set_inhibition", "reason": "inhibited"}
    )
    assert reason == bus.REASON_INJECT
    assert text.endswith(bus.VOICE_MARKERS["silent"])
    # It no longer accuses Nova of having its own intention refused.
    assert "set_inhibition" not in text
    assert "intention" not in text.lower()
    assert "held back" in text.lower()


def test_l3_intent_blocked_renders_no_empty_placeholder(rules_cfg):
    """The runtime's wire payload carries no top-level behaviour name, so the
    template must not leave a dangling empty field behind."""
    text, _ = bus.route_event(rules_cfg, "intent", "blocked", {"name": "set_inhibition"})
    assert "{" not in text
    assert ": )" not in text and "()" not in text
    assert text.strip() == text


def test_l3_intent_blocked_repeats_collapse_onto_one_dedupe_key(rules_cfg):
    """The ~10 s repeat is one bucket: the dedupe identity ignores the payload."""
    rule = bus.rule_for(rules_cfg, "intent", "blocked", {"name": "set_inhibition"})
    first = bus.dedupe_key_for("intent", "blocked", {"name": "set_inhibition"}, rule)
    second = bus.dedupe_key_for("intent", "blocked", {"name": "declare_goal"}, rule)
    assert first == second == "reflex-held"


def test_l3_intent_applied_is_context_only_never_spoken(rules_cfg, raw_rules):
    assert raw_rules["rules"]["intent/applied"].get("voice") == "silent"
    text, reason = bus.route_event(rules_cfg, "intent", "applied", {"name": "set_inhibition"})
    assert reason == bus.REASON_INJECT  # still context — Nova learns it landed
    assert text.endswith(bus.VOICE_MARKERS["silent"])
