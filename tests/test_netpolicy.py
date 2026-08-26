"""Tests for the pure dual-network failover policy (task t2).

Covers every rule in ``reachy_nova/netpolicy.py``'s docstring plus the edge
cases called out in the plan (docs/plans/2026-08-26-dual-network-never-downtime.md
t2) and spec (docs/specs/2026-08-26-dual-network-never-downtime.md c22/c26,
s12/s15): link-loss activation, fallback re-evaluation, storm control
("attempt-gap"), signal ties, unknown SSIDs, ``should_log`` latching, and an
AST-based boundary test mirroring ``tests/test_harness_boundary.py`` — the
module must be pure stdlib, no I/O.
"""

from __future__ import annotations

import ast
from pathlib import Path

from reachy_nova.netpolicy import Decision, Observation, Policy

PREFERRED = "iPhone (5)"
FALLBACK = "bar-nachum"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "reachy_nova" / "netpolicy.py"

#: The only modules netpolicy.py may import from — pure stdlib, and only
#: what it actually needs (no ``time`` — decisions are derived from
#: ``obs.now``, never read from the wall clock directly).
_ALLOWED_IMPORT_ROOTS = {"__future__", "dataclasses", "typing"}

_FORBIDDEN_NAMES = {"subprocess", "system"}


def _policy(**overrides) -> Policy:
    kwargs = dict(preferred=PREFERRED, fallback=FALLBACK)
    kwargs.update(overrides)
    return Policy(**kwargs)


# --------------------------------------------------------------------------- #
# Link lost + a candidate visible -> activate.                                #
# --------------------------------------------------------------------------- #


def test_link_lost_preferred_visible_activates_preferred() -> None:
    policy = _policy()
    obs = Observation(
        now=1000.0,
        active_ssid=None,
        has_route=False,
        visible={PREFERRED: 80, FALLBACK: 40},
    )
    decision = policy.decide(obs)
    assert decision.action == "activate"
    assert decision.target == PREFERRED
    assert decision.reason == "link-lost-hotspot-visible"
    assert decision.next_check_s <= 30.0


def test_link_lost_only_fallback_visible_activates_fallback() -> None:
    policy = _policy()
    obs = Observation(
        now=1000.0,
        active_ssid=None,
        has_route=False,
        visible={FALLBACK: 55},
    )
    decision = policy.decide(obs)
    assert decision.action == "activate"
    assert decision.target == FALLBACK
    assert decision.reason == "link-lost-fallback-visible"
    assert decision.next_check_s <= 30.0


def test_link_lost_nothing_visible_waits_with_bounded_recheck() -> None:
    policy = _policy()
    obs = Observation(now=1000.0, active_ssid=None, has_route=False, visible={})
    decision = policy.decide(obs)
    assert decision.action == "none"
    assert decision.target is None
    assert decision.reason == "waiting-no-candidate"
    assert decision.next_check_s <= 30.0


def test_link_lost_preferred_beats_fallback_when_both_visible() -> None:
    """Preferred always wins over fallback, regardless of relative signal."""
    policy = _policy()
    obs = Observation(
        now=1000.0,
        active_ssid=None,
        has_route=False,
        # fallback has a much stronger signal — preferred must still win.
        visible={PREFERRED: 10, FALLBACK: 99},
    )
    decision = policy.decide(obs)
    assert decision.action == "activate"
    assert decision.target == PREFERRED


def test_signal_tie_between_preferred_and_fallback_still_prefers_preferred() -> None:
    policy = _policy()
    obs = Observation(
        now=1000.0,
        active_ssid=None,
        has_route=False,
        visible={PREFERRED: 50, FALLBACK: 50},
    )
    decision = policy.decide(obs)
    assert decision.target == PREFERRED


def test_unknown_ssids_in_scan_are_ignored() -> None:
    policy = _policy()
    obs = Observation(
        now=1000.0,
        active_ssid=None,
        has_route=False,
        visible={"CoffeeShopWifi": 90, "NeighborNet": 70},
    )
    decision = policy.decide(obs)
    assert decision.action == "none"
    assert decision.reason == "waiting-no-candidate"


# --------------------------------------------------------------------------- #
# On the fallback: re-evaluate for the preferred network reappearing.         #
# --------------------------------------------------------------------------- #


def test_on_fallback_preferred_visible_activates_preferred() -> None:
    policy = _policy()
    obs = Observation(
        now=2000.0,
        active_ssid=FALLBACK,
        has_route=True,
        visible={FALLBACK: 60, PREFERRED: 70},
    )
    decision = policy.decide(obs)
    assert decision.action == "activate"
    assert decision.target == PREFERRED
    assert decision.reason == "on-fallback-hotspot-visible"
    assert decision.next_check_s <= 60.0


def test_on_fallback_preferred_absent_does_nothing_and_rechecks_within_60s() -> None:
    policy = _policy()
    obs = Observation(
        now=2000.0,
        active_ssid=FALLBACK,
        has_route=True,
        visible={FALLBACK: 60},
    )
    decision = policy.decide(obs)
    assert decision.action == "none"
    assert decision.reason == "on-fallback-hotspot-absent"
    assert decision.next_check_s <= 60.0


# --------------------------------------------------------------------------- #
# On the preferred network: settle, long recheck.                             #
# --------------------------------------------------------------------------- #


def test_on_preferred_does_nothing_with_long_recheck() -> None:
    policy = _policy()
    obs = Observation(
        now=3000.0,
        active_ssid=PREFERRED,
        has_route=True,
        visible={PREFERRED: 90, FALLBACK: 20},
    )
    decision = policy.decide(obs)
    assert decision.action == "none"
    assert decision.target is None
    assert decision.reason == "on-preferred"
    # long recheck: strictly greater than the fallback/disconnected bounds.
    assert decision.next_check_s > 60.0


def test_connected_to_unrecognized_network_leaves_it_alone() -> None:
    policy = _policy()
    obs = Observation(
        now=3000.0,
        active_ssid="SomeOtherNetwork",
        has_route=True,
        visible={"SomeOtherNetwork": 80},
    )
    decision = policy.decide(obs)
    assert decision.action == "none"
    assert decision.reason == "on-unknown-network"
    assert decision.next_check_s <= 60.0


# --------------------------------------------------------------------------- #
# Storm control: never two attempts against the same target within the gap.   #
# --------------------------------------------------------------------------- #


def test_attempt_gap_blocks_repeat_activation_of_same_target() -> None:
    policy = _policy(min_attempt_gap_s=30.0)
    obs = Observation(
        now=1010.0,
        active_ssid=None,
        has_route=False,
        visible={PREFERRED: 80},
        last_attempt_at={PREFERRED: 1000.0},  # 10s ago, within the 30s gap
    )
    decision = policy.decide(obs)
    assert decision.action == "none"
    assert decision.reason == "attempt-gap"
    assert decision.target == PREFERRED


def test_attempt_gap_expires_and_allows_retry() -> None:
    policy = _policy(min_attempt_gap_s=30.0)
    obs = Observation(
        now=1031.0,
        active_ssid=None,
        has_route=False,
        visible={PREFERRED: 80},
        last_attempt_at={PREFERRED: 1000.0},  # 31s ago, gap has expired
    )
    decision = policy.decide(obs)
    assert decision.action == "activate"
    assert decision.target == PREFERRED


def test_attempt_gap_is_per_target_not_global() -> None:
    """A recent attempt against the fallback must not block activating the
    preferred network, and vice versa."""
    policy = _policy(min_attempt_gap_s=30.0)
    obs = Observation(
        now=1010.0,
        active_ssid=None,
        has_route=False,
        visible={PREFERRED: 80},
        last_attempt_at={FALLBACK: 1000.0},  # a different target's attempt
    )
    decision = policy.decide(obs)
    assert decision.action == "activate"
    assert decision.target == PREFERRED


def test_attempt_gap_applies_on_fallback_reevaluation_too() -> None:
    policy = _policy(min_attempt_gap_s=30.0)
    obs = Observation(
        now=2010.0,
        active_ssid=FALLBACK,
        has_route=True,
        visible={FALLBACK: 60, PREFERRED: 70},
        last_attempt_at={PREFERRED: 2000.0},  # 10s ago, within the gap
    )
    decision = policy.decide(obs)
    assert decision.action == "none"
    assert decision.reason == "attempt-gap"
    assert decision.target == PREFERRED


# --------------------------------------------------------------------------- #
# should_log latching.                                                        #
# --------------------------------------------------------------------------- #


def test_should_log_true_when_no_previous_decision() -> None:
    decision = Decision(action="none", target=None, reason="waiting-no-candidate", next_check_s=30.0, ts=100.0)
    assert decision.should_log(None) is True


def test_should_log_false_when_same_reason_and_target_within_a_minute() -> None:
    prev = Decision(action="none", target=None, reason="waiting-no-candidate", next_check_s=30.0, ts=100.0)
    decision = Decision(action="none", target=None, reason="waiting-no-candidate", next_check_s=30.0, ts=129.0)
    assert decision.should_log(prev) is False


def test_should_log_true_when_a_minute_has_elapsed() -> None:
    prev = Decision(action="none", target=None, reason="waiting-no-candidate", next_check_s=30.0, ts=100.0)
    decision = Decision(action="none", target=None, reason="waiting-no-candidate", next_check_s=30.0, ts=160.0)
    assert decision.should_log(prev) is True


def test_should_log_true_when_reason_changes() -> None:
    prev = Decision(action="none", target=None, reason="waiting-no-candidate", next_check_s=30.0, ts=100.0)
    decision = Decision(action="activate", target=PREFERRED, reason="link-lost-hotspot-visible", next_check_s=30.0, ts=105.0)
    assert decision.should_log(prev) is True


def test_should_log_true_when_target_changes_but_reason_matches() -> None:
    prev = Decision(action="none", target=FALLBACK, reason="attempt-gap", next_check_s=30.0, ts=100.0)
    decision = Decision(action="none", target=PREFERRED, reason="attempt-gap", next_check_s=30.0, ts=105.0)
    assert decision.should_log(prev) is True


def test_waiting_state_costs_at_most_one_log_line_per_minute_over_many_ticks() -> None:
    """Simulate ~3 minutes of 5s-interval polling with nothing visible; at
    most one log-worthy decision should occur per 60s window."""
    policy = _policy()
    prev: Decision | None = None
    logged_at: list[float] = []
    t = 0.0
    while t <= 180.0:
        obs = Observation(now=t, active_ssid=None, has_route=False, visible={})
        decision = policy.decide(obs)
        if decision.should_log(prev):
            logged_at.append(t)
            prev = decision
        t += 5.0
    # ~180s of ticks -> at most 4 log lines (t=0, ~60, ~120, ~180)
    assert len(logged_at) <= 4
    for a, b in zip(logged_at, logged_at[1:]):
        assert (b - a) >= 60.0


# --------------------------------------------------------------------------- #
# to_record().                                                                 #
# --------------------------------------------------------------------------- #


def test_to_record_contains_expected_fields() -> None:
    decision = Decision(
        action="activate",
        target=PREFERRED,
        reason="link-lost-hotspot-visible",
        next_check_s=30.0,
        ts=42.0,
    )
    record = decision.to_record()
    assert record == {
        "ts": 42.0,
        "action": "activate",
        "target": PREFERRED,
        "reason": "link-lost-hotspot-visible",
        "next_check_s": 30.0,
    }


# --------------------------------------------------------------------------- #
# Purity / boundary: stdlib only, no I/O, no subprocess/nmcli.                #
# --------------------------------------------------------------------------- #


def test_module_file_exists() -> None:
    assert _MODULE_PATH.is_file(), f"expected {_MODULE_PATH} to exist"


def test_module_imports_only_from_stdlib_allow_list() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"), filename=str(_MODULE_PATH))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    offenders.append(f"from {node.module} import ...")
    assert not offenders, f"netpolicy.py imported outside the stdlib allow-list: {offenders}"


def test_module_contains_no_subprocess_or_os_system_calls() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_MODULE_PATH))
    names_used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names_used.add(node.id)
        elif isinstance(node, ast.Attribute):
            names_used.add(node.attr)
    offenders = names_used & _FORBIDDEN_NAMES
    assert not offenders, f"netpolicy.py references forbidden names: {offenders}"
    assert "open(" not in source, "netpolicy.py must not open files — no I/O"
    # nmcli may be *mentioned* in docstrings/comments (explaining the
    # caller's responsibility) but must never appear as executable code —
    # i.e. never as a Name or Attribute node in the AST.
    assert "nmcli" not in names_used, "netpolicy.py must never call nmcli directly"


def test_module_never_reads_the_wall_clock_itself() -> None:
    """decide() must derive everything from obs.now — no time.time()/monotonic()."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "import time" not in source
    assert "time.time(" not in source
    assert "time.monotonic(" not in source
