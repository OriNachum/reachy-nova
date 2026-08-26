"""Rules-coverage test: every event source/type pair a publisher can emit
must have an explicit config/nervous-system/rules.yaml entry — no sense
should silently ride the ``default`` rule (see docs/plans/2026-07-17-
event-based-senses-seamless-reactions.md, task t6, covers c5/h5).

Two things are verified here:

1. Every (source, type) pair *discovered by scanning the codebase* for
   ``publish_event(`` / ``_fire_event(`` call sites has an explicit
   ``rules.yaml`` entry — publishers is a subset of rules.
2. The required-new pairs named in this task's acceptance criteria
   (face/face_recognized, tracking/pat_level1, tracking/pat_level2,
   tracking/audio_direction, speech/speech_detected, and
   forge/staged|activated|rejected) are present with priority/urgency/
   llm_evaluate, and the three that need a warm inject_template
   (face_recognized, speech_detected, forge/rejected) have one.

This test is pure file parsing (``re`` + ``pathlib`` + ``yaml``) — it
never imports ``reachy_nova``, so it stays import-light like the rest of
the suite and needs no hardware/cloud stubs.

Special cases handled explicitly while scanning:

* ``main.py`` republishes every ``tracking._fire_event(...)`` literal
  under source ``"tracking"`` via a *variable* ``event_type``
  (``mqtt.publish_event("tracking", event_type, data)``) — so
  ``tracking/<type>`` pairs are derived from the literal event-type
  strings passed to ``_fire_event(...)`` in ``tracking.py``, not from
  main.py's variable call site.
* ``main.py``'s Slack publisher uses a dynamic type,
  ``f"slack_{event.type}"``. That can't be resolved statically (note c
  in the working instruction): it is mapped to the fixed, already-covered
  set of Slack event types (slack_mention/slack_message/slack_dm/
  slack_reaction) and documented here rather than silently skipped.
* ``tracking/audio_direction``, ``speech/speech_detected`` and
  ``forge/staged|activated|rejected`` have no publisher yet in *this*
  worktree — they land in sibling tasks of the same build plan (t5, t9,
  t7). They are still required by this task's acceptance criteria, so
  they're asserted directly rather than discovered by the scan.

If a future change adds a genuinely new dynamic (non-literal) source or
type, the scanner raises loudly instead of silently under-counting, so
this test fails closed rather than open.

t6 note — before/after state of the generic ``rule/fire`` inject_template:
before this task, the rendered text narrated the reflex mechanism itself:
"Your body just reacted on its own — the '{rule}' reflex fired." (observed
live 2026-08-26). t6 replaced it with a quiet, situational template and
added optional ``voice``/``sense`` fields to the rules.yaml schema (see
tests/test_rules_voice.py for the quiet-wording and voice-marker coverage);
this test additionally pins that ``rule/fire`` carries ``voice: silent``.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "reachy_nova"
RULES_PATH = REPO_ROOT / "config" / "nervous-system" / "rules.yaml"

# `<obj>.publish_event("source", "type", ...)` — literal source + literal type.
_PUBLISH_EVENT_LITERAL_RE = re.compile(r'\.publish_event\(\s*"([^"]+)"\s*,\s*"([^"]+)"')
# `<obj>.publish_event("source", f"...", ...)` — literal source, f-string type.
_PUBLISH_EVENT_FSTRING_RE = re.compile(r'\.publish_event\(\s*"([^"]+)"\s*,\s*f"([^"]+)"')
# `<obj>.publish_event("source", some_variable, ...)` — literal source, variable type.
_PUBLISH_EVENT_VAR_RE = re.compile(r'\.publish_event\(\s*"([^"]+)"\s*,\s*(\w+)\s*,')
# `self._fire_event("type", ...)` in tracking.py.
_FIRE_EVENT_RE = re.compile(r'\._fire_event\(\s*"([^"]+)"')

# Dynamic Slack type `f"slack_{event.type}"` (main.py) can't be resolved
# statically. This is the fixed, documented set of event.type values
# rules.yaml already covers — the honest workaround for note (c).
KNOWN_SLACK_TYPES = {"slack_mention", "slack_message", "slack_dm", "slack_reaction"}

# The pairs this task's acceptance criteria require explicitly, regardless
# of whether a publisher exists yet in this worktree.
REQUIRED_NEW_PAIRS = {
    "face/face_recognized",
    "tracking/pat_level1",
    "tracking/pat_level2",
    "tracking/audio_direction",
    "speech/speech_detected",
    "forge/staged",
    "forge/activated",
    "forge/rejected",
}

# Of the required-new pairs, these need a warm inject_template per the
# working instruction's note (e).
REQUIRED_TEMPLATED_PAIRS = {
    "face/face_recognized",
    "speech/speech_detected",
    "forge/rejected",
}


def _load_rules_section() -> dict:
    with open(RULES_PATH) as f:
        raw = yaml.safe_load(f)
    return raw.get("rules", {})


def _discover_publisher_pairs() -> set[str]:
    """Scan reachy_nova/ for publish_event/_fire_event call sites and
    return the "source/type" pairs they can emit.
    """
    pairs: set[str] = set()
    tracking_fire_event_types: set[str] = set()

    for py_file in sorted(PACKAGE_DIR.rglob("*.py")):
        text = py_file.read_text()

        for source, event_type in _PUBLISH_EVENT_LITERAL_RE.findall(text):
            pairs.add(f"{source}/{event_type}")

        for source, template in _PUBLISH_EVENT_FSTRING_RE.findall(text):
            if source == "slack" and template == "slack_{event.type}":
                pairs.update(f"slack/{t}" for t in KNOWN_SLACK_TYPES)
            else:
                raise AssertionError(
                    f"Unhandled dynamic publish_event f-string in {py_file.name}: "
                    f'publish_event("{source}", f"{template}", ...) — teach this '
                    "test how to resolve it (or map it to a known fixed set, as "
                    "done for slack)."
                )

        for source, var_name in _PUBLISH_EVENT_VAR_RE.findall(text):
            if source == "tracking" and var_name == "event_type":
                # main.py republishes every tracking._fire_event(...) literal
                # under source "tracking" via this variable — resolved below
                # from tracking.py's _fire_event literals instead.
                continue
            raise AssertionError(
                f"Unhandled dynamic publish_event call in {py_file.name}: "
                f'publish_event("{source}", {var_name}, ...) — teach this test '
                "how to resolve it."
            )

        if py_file.name == "tracking.py":
            tracking_fire_event_types.update(_FIRE_EVENT_RE.findall(text))

    assert tracking_fire_event_types, "Expected _fire_event(...) literals in tracking.py"
    pairs.update(f"tracking/{t}" for t in tracking_fire_event_types)
    return pairs


def test_every_discovered_publisher_pair_has_an_explicit_rule():
    """Publishers is a subset of rules: no sense rides the default rule."""
    discovered = _discover_publisher_pairs()
    assert discovered, "Expected to discover at least one publish_event pair"

    rules = _load_rules_section()
    missing = sorted(discovered - rules.keys())
    assert not missing, (
        "config/nervous-system/rules.yaml is missing explicit entries for "
        f"discovered publish_event/_fire_event pairs: {missing}"
    )


def test_required_new_pairs_are_covered_with_templates_where_required():
    """Acceptance criterion 1: the required-new pairs are explicit, carry
    priority/urgency/llm_evaluate, and the three that need a warm
    inject_template have one.
    """
    rules = _load_rules_section()

    missing = sorted(REQUIRED_NEW_PAIRS - rules.keys())
    assert not missing, f"rules.yaml is missing required new entries: {missing}"

    for key in sorted(REQUIRED_NEW_PAIRS):
        entry = rules[key]
        assert "priority" in entry, f"{key} is missing priority"
        assert "urgency" in entry, f"{key} is missing urgency"
        assert "llm_evaluate" in entry, f"{key} is missing llm_evaluate"

    for key in sorted(REQUIRED_TEMPLATED_PAIRS):
        assert rules[key].get("inject_template"), (
            f"{key} is expected to carry a sensible inject_template, found none"
        )


def test_default_rule_still_exists_as_the_fallback():
    """Sanity: the default rule the rules engine falls back to is intact."""
    with open(RULES_PATH) as f:
        raw = yaml.safe_load(f)
    assert "default" in raw
    assert raw["default"].get("llm_evaluate") is True


def test_generic_rule_fire_is_voice_silent():
    """t6: the generic ``rule/fire`` entry (the reflex-fired fallback) is
    marked ``voice: silent`` — Nova gets the situational context but is
    told not to narrate it, per bus.py's VOICE_MARKERS.
    """
    rules = _load_rules_section()
    assert rules["rule/fire"].get("voice") == "silent"
