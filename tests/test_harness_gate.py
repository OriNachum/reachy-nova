"""The echo gate and its POLICY — ``reachy_nova/harness/gate.py`` (task t2).

One :class:`EchoGate` object serves two legs with different rules:

* the speaking leg always uses it for one-speaker-at-a-time discipline;
* the hearing leg reads it under an env-selectable policy, ``NOVA_ECHO_GATE``.

The policy exists because the premise that justified blanket suppression is
gone: on 2026-08-10 the XVF3800's hardware AEC was verified ACTIVE live on the
wireless capture path, so the default is ``off`` (hear while speaking, which is
what barge-in needs) and ``half-duplex`` is the opt-in for hardware without a
verified AEC.
"""

from __future__ import annotations

import time

from reachy_nova.harness import gate as gate_mod
from reachy_nova.harness import hearing as hearing_mod
from reachy_nova.harness.gate import EchoGate


# --------------------------------------------------------------------------- #
# 1. The window itself is unchanged (both legs still depend on it).           #
# --------------------------------------------------------------------------- #


def test_arm_for_opens_a_window_and_clear_drops_it() -> None:
    gate = EchoGate(margin_s=0.0)
    assert not gate.active
    gate.arm_for(60.0)
    assert gate.active
    assert 59.0 < gate.remaining() <= 60.0
    gate.clear()
    assert not gate.active
    assert gate.remaining() == 0.0


def test_a_window_expires_on_its_own() -> None:
    gate = EchoGate(margin_s=0.0)
    gate.arm_for(0.02)
    time.sleep(0.05)
    assert not gate.active


def test_arming_never_shortens_a_live_window() -> None:
    gate = EchoGate(margin_s=0.0)
    gate.arm_for(60.0)
    gate.arm_for(0.01)  # a short utterance must not cut the long one short
    assert gate.remaining() > 1.0


# --------------------------------------------------------------------------- #
# 2. The policy resolver                                                      #
# --------------------------------------------------------------------------- #


def test_the_policy_names_are_off_and_half_duplex() -> None:
    assert gate_mod.ECHO_GATE_ENV == "NOVA_ECHO_GATE"
    assert gate_mod.POLICY_OFF == "off"
    assert gate_mod.POLICY_HALF_DUPLEX == "half-duplex"
    assert set(gate_mod.ECHO_GATE_POLICIES) == {"off", "half-duplex"}
    assert gate_mod.DEFAULT_ECHO_GATE_POLICY == "off"


def test_an_unset_env_resolves_to_off(monkeypatch) -> None:
    monkeypatch.delenv(gate_mod.ECHO_GATE_ENV, raising=False)
    assert gate_mod.resolve_policy() == "off"


def test_the_env_selects_half_duplex(monkeypatch) -> None:
    monkeypatch.setenv(gate_mod.ECHO_GATE_ENV, "half-duplex")
    assert gate_mod.resolve_policy() == "half-duplex"


def test_the_env_value_is_case_and_whitespace_tolerant(monkeypatch) -> None:
    monkeypatch.setenv(gate_mod.ECHO_GATE_ENV, "  Half-Duplex \n")
    assert gate_mod.resolve_policy() == "half-duplex"


def test_an_unrecognised_value_falls_back_to_off(monkeypatch) -> None:
    """A typo must never silently deafen the robot — fail OPEN, to ``off``."""
    for raw in ("", "   ", "on", "full-duplex", "halfduplex", "1", "true"):
        monkeypatch.setenv(gate_mod.ECHO_GATE_ENV, raw)
        assert gate_mod.resolve_policy() == "off", raw


def test_an_explicit_argument_beats_the_env(monkeypatch) -> None:
    monkeypatch.setenv(gate_mod.ECHO_GATE_ENV, "half-duplex")
    assert gate_mod.resolve_policy("off") == "off"
    monkeypatch.delenv(gate_mod.ECHO_GATE_ENV, raising=False)
    assert gate_mod.resolve_policy("half-duplex") == "half-duplex"


# --------------------------------------------------------------------------- #
# 3. The docstrings carry the live finding, not the disproven premise.        #
# --------------------------------------------------------------------------- #


def test_gate_docstring_states_the_policy_and_cites_the_live_aec_check() -> None:
    doc = gate_mod.__doc__ or ""
    assert "no verified hardware AEC" not in doc, "the disproven premise is still cited"
    assert "NOVA_ECHO_GATE" in doc
    assert "half-duplex" in doc
    assert "2026-08-10" in doc
    assert "AEC" in doc


def test_hearing_docstring_states_the_policy_and_cites_the_live_aec_check() -> None:
    doc = hearing_mod.__doc__ or ""
    assert "no verified hardware AEC" not in doc, "the disproven premise is still cited"
    assert "NOVA_ECHO_GATE" in doc
    assert "half-duplex" in doc
    assert "2026-08-10" in doc
