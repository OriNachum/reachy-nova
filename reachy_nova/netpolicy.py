"""Pure failover policy for the dual-network never-downtime harness (task t2).

This module decides *what to do*, never *how to do it*. It has no I/O, no
``subprocess``, no ``nmcli`` calls, and it never reads the clock itself —
every call to :meth:`Policy.decide` is handed an :class:`Observation` built
by the caller (a scan snapshot, the currently active SSID, whether a route
exists, and timestamps), and it returns a :class:`Decision` describing the
action to take and when to check again. The caller (task t3's dispatcher
hook) is the only place nmcli is ever invoked.

Why this split matters (see ``docs/specs/2026-08-26-dual-network-never-downtime.md``
c22/c26 and ``docs/architecture.md`` §8): NetworkManager retries the
*current* profile with wpa_supplicant backoffs before it even attempts a
fallback, it does not roam to a higher-priority profile on its own while
connected, and the iPhone hotspot only beacons while its Hotspot screen is
open or a client is attached. The policy below encodes the rules the
dispatcher must apply on top of that observed behaviour:

- link lost (no route) + preferred visible -> activate preferred immediately
- link lost + only fallback visible -> activate fallback
- link lost + nothing visible -> no action, re-check within
  ``disconnected_interval_s``
- on the fallback + preferred visible -> activate preferred (re-evaluated at
  least every ``fallback_recheck_interval_s``, since NM will not roam there
  on its own)
- on the preferred network -> no action, long re-check interval
- never re-attempt the same target within ``min_attempt_gap_s`` of the last
  attempt against it (storm control)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Action = Literal["activate", "none"]


@dataclass(frozen=True)
class Observation:
    """A snapshot of the world the policy reasons about.

    :param now: monotonic-or-wall clock reading, in seconds, chosen by the
        caller. The policy never reads the clock itself.
    :param active_ssid: the SSID NetworkManager currently reports as
        connected, or ``None`` if nothing is active.
    :param has_route: whether the active connection currently has a usable
        default route (this is what "link lost" means here — a connection
        can be nominally "active" with no route during a supplicant retry).
    :param visible: scan results as ``{ssid: signal}``, signal in 0-100.
        SSIDs not present are treated as not currently visible.
    :param last_attempt_at: ``{ssid: timestamp}`` of the last time an
        activation of that SSID was attempted (by the caller, after acting
        on a previous ``Decision``), used for storm control.
    :param last_change_at: timestamp of the last actual network change
        (join/drop), informational — not currently consulted by
        :meth:`Policy.decide` but carried for callers/logging.
    """

    now: float
    active_ssid: str | None
    has_route: bool
    visible: dict[str, int] = field(default_factory=dict)
    last_attempt_at: dict[str, float] = field(default_factory=dict)
    last_change_at: float = 0.0


@dataclass(frozen=True)
class Policy:
    """Tunable thresholds for the failover policy.

    :param preferred: SSID of the preferred network (the iPhone hotspot).
    :param fallback: SSID of the fallback network (home Wi-Fi).
    :param loss_bound_s: documented only — the c22 requirement that a link
        loss be acted on within this bound is met by the caller invoking
        :meth:`decide` at least this often while disconnected (see
        ``next_check_s`` below), not by anything decide() measures itself.
    :param disconnected_interval_s: max re-check interval while
        disconnected (no route), regardless of whether a candidate was
        found this round.
    :param fallback_recheck_interval_s: max re-check interval while sitting
        on the fallback network, since NM will not roam to the preferred
        network on its own even when it becomes visible.
    :param min_attempt_gap_s: minimum time between two activation attempts
        against the same target SSID (storm control).
    """

    preferred: str
    fallback: str
    loss_bound_s: float = 30.0
    disconnected_interval_s: float = 30.0
    fallback_recheck_interval_s: float = 60.0
    min_attempt_gap_s: float = 30.0

    #: re-check interval used while sitting happily on the preferred
    #: network — long, since nothing needs to happen there.
    preferred_recheck_interval_s: float = 300.0

    def decide(self, obs: Observation) -> "Decision":
        """Decide what (if anything) to do given *obs*.

        Pure function of *obs* and the policy's own thresholds: no I/O, no
        clock reads, no mutation.
        """
        disconnected = not obs.has_route

        if disconnected:
            preferred_visible = self.preferred in obs.visible
            fallback_visible = self.fallback in obs.visible

            if preferred_visible:
                target = self.preferred
                reason = "link-lost-hotspot-visible"
            elif fallback_visible:
                target = self.fallback
                reason = "link-lost-fallback-visible"
            else:
                return Decision(
                    action="none",
                    target=None,
                    reason="waiting-no-candidate",
                    next_check_s=self.disconnected_interval_s,
                    ts=obs.now,
                )

            gapped = self._within_attempt_gap(obs, target)
            if gapped:
                return Decision(
                    action="none",
                    target=target,
                    reason="attempt-gap",
                    next_check_s=self.disconnected_interval_s,
                    ts=obs.now,
                )

            return Decision(
                action="activate",
                target=target,
                reason=reason,
                next_check_s=self.disconnected_interval_s,
                ts=obs.now,
            )

        # Connected. On the fallback, watch for the preferred network
        # reappearing — NM will never roam to it unassisted.
        if obs.active_ssid == self.fallback:
            if self.preferred in obs.visible:
                if self._within_attempt_gap(obs, self.preferred):
                    return Decision(
                        action="none",
                        target=self.preferred,
                        reason="attempt-gap",
                        next_check_s=self.fallback_recheck_interval_s,
                        ts=obs.now,
                    )
                return Decision(
                    action="activate",
                    target=self.preferred,
                    reason="on-fallback-hotspot-visible",
                    next_check_s=self.fallback_recheck_interval_s,
                    ts=obs.now,
                )
            return Decision(
                action="none",
                target=None,
                reason="on-fallback-hotspot-absent",
                next_check_s=self.fallback_recheck_interval_s,
                ts=obs.now,
            )

        if obs.active_ssid == self.preferred:
            return Decision(
                action="none",
                target=None,
                reason="on-preferred",
                next_check_s=self.preferred_recheck_interval_s,
                ts=obs.now,
            )

        # Connected to something the policy doesn't recognise (neither
        # preferred nor fallback) — leave it alone but keep watching at the
        # cautious (fallback) cadence.
        return Decision(
            action="none",
            target=None,
            reason="on-unknown-network",
            next_check_s=self.fallback_recheck_interval_s,
            ts=obs.now,
        )

    def _within_attempt_gap(self, obs: Observation, target: str) -> bool:
        last = obs.last_attempt_at.get(target)
        if last is None:
            return False
        return (obs.now - last) < self.min_attempt_gap_s


@dataclass(frozen=True)
class Decision:
    """The outcome of :meth:`Policy.decide`.

    :param action: ``"activate"`` (caller should nmcli-activate ``target``)
        or ``"none"`` (do nothing this round).
    :param target: the SSID to activate, or the SSID a pending/blocked
        decision concerns (e.g. during "attempt-gap"); ``None`` when no
        SSID is relevant.
    :param reason: a short, grep-able machine token describing why.
    :param next_check_s: how soon (in seconds from ``ts``) the caller
        should invoke :meth:`Policy.decide` again.
    :param ts: the ``Observation.now`` this decision was computed from —
        carried so :meth:`should_log` can measure elapsed time without the
        caller re-threading a clock through it.
    """

    action: Action
    target: str | None
    reason: str
    next_check_s: float
    ts: float = 0.0

    def should_log(self, prev: "Decision | None") -> bool:
        """Whether this decision is worth a new log line.

        Latches on ``(reason, target)`` so a caller decide()-ing on a tight
        loop while nothing changes emits at most one line per
        ``log_interval_s`` (60s) — never a line per tick.
        """
        log_interval_s = 60.0
        if prev is None:
            return True
        if prev.reason != self.reason or prev.target != self.target:
            return True
        return (self.ts - prev.ts) >= log_interval_s

    def to_record(self) -> dict:
        """Structured form for logging: ``{ts, action, target, reason, next_check_s}``."""
        return {
            "ts": self.ts,
            "action": self.action,
            "target": self.target,
            "reason": self.reason,
            "next_check_s": self.next_check_s,
        }
