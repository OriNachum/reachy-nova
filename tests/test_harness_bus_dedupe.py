"""t7: per-key sense-class dedupe in NovaBus.

Live bug this guards against (observed 2026-08-26): one physical pat
produced two Sonic injects ~8s apart — the runtime's shipped
``rule/fire:pat-acknowledge`` and the Kiro-authored overlay's own
``rule/fire:nova-pat-cheer``, each arriving as its own ``rule/fire`` event.
Sonic's only prior dedupe was a global 3s throttle, which an 8s gap sails
past.

NovaBus now sits a per-key last-inject dedupe between ``route_event`` and
``self._on_inject``: keyed by the matched rule entry's ``sense`` class when
one is set, else the exact resolved key (``"<source>/<type>:<rule>"`` or
``"<source>/<type>"``) — never a guess from the rule's name. A clock is
injectable so the window is deterministic under test, exactly like the
fake paho client in ``test_harness_bus.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reachy_nova.harness import bus

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "config" / "nervous-system" / "rules.yaml"


def fake_msg(topic: str, payload) -> SimpleNamespace:
    """A stand-in for paho's MQTTMessage — same shape as test_harness_bus.py's."""
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, separators=(",", ":"))
    if isinstance(payload, str):
        payload = payload.encode()
    return SimpleNamespace(topic=topic, payload=payload)


class Recorder:
    """Collects injects from the bus — mirrors test_harness_bus.py's Recorder."""

    def __init__(self) -> None:
        self.injects: list[str] = []

    def on_inject(self, text: str) -> None:
        self.injects.append(text)


def make_bus(recorder: Recorder, **kwargs) -> bus.NovaBus:
    return bus.NovaBus(on_inject=recorder.on_inject, **kwargs)


def make_clock(times: list[float]):
    """A zero-arg clock that yields *times* in order, then repeats the last."""
    state = {"i": 0}

    def clock() -> float:
        i = min(state["i"], len(times) - 1)
        state["i"] += 1
        return times[i]

    return clock


def pat_msg(rule_name: str, ts: float = 0.0):
    return fake_msg(
        "reachy/events/rule/fire",
        {"t": "rule", "ts": ts, "rule": rule_name},
    )


# --------------------------------------------------------------------------- #
# Acceptance criteria 2: pat-acknowledge + nova-pat-cheer collapse            #
# --------------------------------------------------------------------------- #


def test_pat_acknowledge_and_pat_cheer_within_window_collapse_to_one_inject():
    rec = Recorder()
    clock = make_clock([0.0, 8.0])
    nb = make_bus(rec, sources="rule", rules_path=RULES_PATH, clock=clock)

    nb.on_message(None, None, pat_msg("pat-acknowledge"))
    nb.on_message(None, None, pat_msg("nova-pat-cheer"))

    assert len(rec.injects) == 1


def test_a_third_pat_after_the_window_produces_a_fresh_inject():
    rec = Recorder()
    # first pat at t=0, second (dup) at t=8 (suppressed), third at t=21 — 21s
    # past the first actual inject at t=0, well outside the 10s default window.
    clock = make_clock([0.0, 8.0, 21.0])
    nb = make_bus(rec, sources="rule", rules_path=RULES_PATH, clock=clock)

    nb.on_message(None, None, pat_msg("pat-acknowledge"))
    nb.on_message(None, None, pat_msg("nova-pat-cheer"))
    nb.on_message(None, None, pat_msg("pat-acknowledge"))

    assert len(rec.injects) == 2


# --------------------------------------------------------------------------- #
# F3: sense-class dedupe must not drop an escalation                         #
# --------------------------------------------------------------------------- #


def pat_level_msg(level: int, ts: float = 0.0):
    return fake_msg(
        f"reachy/events/pat/level{level}",
        {"t": "pat", "ts": ts},
    )


def test_pat_level1_then_level2_within_the_window_both_inject():
    """F3: pat/level1 and pat/level2 share `sense: pat` in rules.yaml, but a
    level1 -> level2 escalation is a distinct sensory event and must not be
    suppressed by the pat sense-class dedupe."""
    rec = Recorder()
    clock = make_clock([0.0, 1.0])
    nb = make_bus(rec, sources="pat", rules_path=RULES_PATH, clock=clock)

    nb.on_message(None, None, pat_level_msg(1))
    nb.on_message(None, None, pat_level_msg(2))

    assert len(rec.injects) == 2


def test_pat_level1_twice_within_the_window_still_dedupes():
    """The exact-key fallback still collapses two IDENTICAL discrete pat
    events (same level) within the window — only the cross-rule
    pat-acknowledge/nova-pat-cheer pair, and now the pat/level* escalation,
    are meant to change behavior."""
    rec = Recorder()
    clock = make_clock([0.0, 1.0])
    nb = make_bus(rec, sources="pat", rules_path=RULES_PATH, clock=clock)

    nb.on_message(None, None, pat_level_msg(1))
    nb.on_message(None, None, pat_level_msg(1))

    assert len(rec.injects) == 1


# --------------------------------------------------------------------------- #
# Acceptance criteria 3: never guess a class from the rule name              #
# --------------------------------------------------------------------------- #


def test_two_unclassed_rule_fire_events_with_different_names_both_inject():
    """Neither rule name has a per-rule override or a `sense` field, so each
    resolves to the generic rule/fire entry — but the dedupe key still
    includes the literal rule name, so two DIFFERENT rule names never
    collapse just because both are unclassed.

    Uses "nod-yes"/"wave-hello", NOT "look-toward-sound" — t14 gave that one
    its own `rule/fire:look-toward-sound` override with `voice: none`, so it
    no longer falls through to the generic (inject-producing) entry here.
    """
    rec = Recorder()
    clock = make_clock([0.0, 1.0])
    nb = make_bus(rec, sources="rule", rules_path=RULES_PATH, clock=clock)

    nb.on_message(None, None, pat_msg("nod-yes"))
    nb.on_message(None, None, pat_msg("wave-hello"))

    assert len(rec.injects) == 2


def test_same_unclassed_rule_name_twice_within_window_still_dedupes():
    rec = Recorder()
    clock = make_clock([0.0, 1.0])
    nb = make_bus(rec, sources="rule", rules_path=RULES_PATH, clock=clock)

    nb.on_message(None, None, pat_msg("nod-yes"))
    nb.on_message(None, None, pat_msg("nod-yes"))

    assert len(rec.injects) == 1


# --------------------------------------------------------------------------- #
# Acceptance criteria 4: exactly one senselog line per suppressed duplicate  #
# --------------------------------------------------------------------------- #


def test_a_suppressed_duplicate_logs_exactly_one_dedupe_line(caplog):
    rec = Recorder()
    clock = make_clock([0.0, 8.0])
    nb = make_bus(rec, sources="rule", rules_path=RULES_PATH, clock=clock)

    with caplog.at_level("INFO", logger="nova.sensory"):
        nb.on_message(None, None, pat_msg("pat-acknowledge"))
        nb.on_message(None, None, pat_msg("nova-pat-cheer"))

    dedupe_lines = [
        r.getMessage()
        for r in caplog.records
        if "[SENSE stage=inject source=nova event=dedupe]" in r.getMessage()
    ]
    assert len(dedupe_lines) == 1
    assert "pat" in dedupe_lines[0]
    assert "age=" in dedupe_lines[0]


# --------------------------------------------------------------------------- #
# Acceptance criteria 1: dedupe_key_for + dedupe_window_s (pure helpers)     #
# --------------------------------------------------------------------------- #


def test_dedupe_key_for_prefers_the_dedupe_field_over_the_resolved_key():
    key = bus.dedupe_key_for(
        "rule", "fire", {"rule": "pat-acknowledge"}, {"sense": "pat", "dedupe": "pat-touch"}
    )
    assert key == "pat-touch"


def test_dedupe_key_for_ignores_sense_and_falls_back_to_the_resolved_key():
    """F3: `sense` alone (no `dedupe` field) must NOT collapse the dedupe
    identity — it is SenseHistory's semantic class only."""
    key = bus.dedupe_key_for("rule", "fire", {"rule": "pat-acknowledge"}, {"sense": "pat"})
    assert key == "rule/fire:pat-acknowledge"


def test_dedupe_key_for_falls_back_to_the_resolved_rule_key():
    key = bus.dedupe_key_for("rule", "fire", {"rule": "look-toward-sound"}, {})
    assert key == "rule/fire:look-toward-sound"


def test_dedupe_key_for_falls_back_to_the_generic_key_with_no_rule_name():
    assert bus.dedupe_key_for("motion", "goto", {}, {}) == "motion/goto"
    assert bus.dedupe_key_for("motion", "goto", None, {}) == "motion/goto"


def test_dedupe_window_s_default_is_ten_seconds(monkeypatch):
    monkeypatch.delenv("NOVA_SENSE_DEDUPE_S", raising=False)
    assert bus.dedupe_window_s() == 10.0 == bus.DEFAULT_DEDUPE_WINDOW_S


@pytest.mark.parametrize("raw", ["not-a-number", "", "   ", "0", "-5"])
def test_dedupe_window_s_defensive_parsing_never_raises(raw):
    assert bus.dedupe_window_s({"NOVA_SENSE_DEDUPE_S": raw}) == bus.DEFAULT_DEDUPE_WINDOW_S


def test_dedupe_window_s_reads_a_valid_override():
    assert bus.dedupe_window_s({"NOVA_SENSE_DEDUPE_S": "2.5"}) == 2.5


def test_dedupe_window_env_var_is_honored_by_the_bus(monkeypatch):
    """A short configured window lets the second pat through, unlike the
    10s default exercised above."""
    monkeypatch.setenv("NOVA_SENSE_DEDUPE_S", "1")
    rec = Recorder()
    clock = make_clock([0.0, 8.0])
    nb = make_bus(rec, sources="rule", rules_path=RULES_PATH, clock=clock)
    assert nb._dedupe_window_s == 1.0

    nb.on_message(None, None, pat_msg("pat-acknowledge"))
    nb.on_message(None, None, pat_msg("nova-pat-cheer"))

    assert len(rec.injects) == 2
