"""The speaking window shared by the speak and hear legs, and its policy.

One :class:`EchoGate` is armed by ``speaking.py`` around each playback
(duration + a tail margin) and read by BOTH legs — for two different reasons,
under two different rules:

* **Speaking leg — always, unconditionally.** The worker waits out the
  previous window before posting the next utterance, so two utterances never
  mix at the device (reachy-mini-cli 0.48.0 has no speaker arbitration:
  concurrent plays literally overlap). Overlapping playback is a device-level
  defect, not a preference, so this leg is NOT policy-selectable.
* **Hearing leg — under the ``NOVA_ECHO_GATE`` policy.** ``hearing.py``
  consults :func:`resolve_policy`::

      NOVA_ECHO_GATE=off          # DEFAULT: keep hearing while the robot speaks
      NOVA_ECHO_GATE=half-duplex  # suppress the mic feed for the whole window

  Anything else — unset, empty, misspelt — resolves to ``off``: a typo must
  never silently deafen the robot.

Why ``off`` is the default: on **2026-08-10** the XVF3800's hardware AEC was
verified ACTIVE live on the wireless capture path — during robot playback the
robot's own speaker barely registers on the mic (human speech RMS ~0.09 against
a ~0.002 floor). The earlier premise that this path has no verified hardware
AEC (spec c19) is therefore disproven **for this device**, and blanket
suppression bought nothing but a robot that is deaf while it speaks — which is
also why barge-in could never fire: Sonic never received user speech during
robot speech. ``half-duplex`` stays one env flip away (and is the documented
rollback) for hardware whose AEC is absent or unproven.
"""

from __future__ import annotations

import os
import threading
import time

# --------------------------------------------------------------------------- #
# The hearing-side policy                                                      #
# --------------------------------------------------------------------------- #

#: Environment variable selecting what the HEARING leg does with the window.
ECHO_GATE_ENV = "NOVA_ECHO_GATE"
#: Keep feeding the mic to Sonic while the robot speaks (hardware AEC does the
#: echo work). The default, and what makes barge-in possible.
POLICY_OFF = "off"
#: Drop every mic chunk that lands inside a playback window.
POLICY_HALF_DUPLEX = "half-duplex"
#: Every policy this harness understands; anything else resolves to the default.
ECHO_GATE_POLICIES = (POLICY_OFF, POLICY_HALF_DUPLEX)
DEFAULT_ECHO_GATE_POLICY = POLICY_OFF


def resolve_policy(explicit: str | None = None) -> str:
    """Resolve the hearing-side echo policy: *explicit* wins, else the env.

    Fails OPEN — an unset, empty or unrecognised value resolves to
    :data:`POLICY_OFF` rather than suppressing the mic on a guess.
    """
    raw = explicit if explicit is not None else os.environ.get(ECHO_GATE_ENV)
    if raw is None:
        return DEFAULT_ECHO_GATE_POLICY
    candidate = raw.strip().lower()
    if candidate in ECHO_GATE_POLICIES:
        return candidate
    return DEFAULT_ECHO_GATE_POLICY


class EchoGate:
    """Thread-safe speaking window: armed for a playback duration + margin."""

    def __init__(self, margin_s: float = 1.0):
        self.margin_s = margin_s
        self._until = 0.0
        self._lock = threading.Lock()

    def arm_for(self, duration_s: float) -> None:
        """Arm the gate for a playback of ``duration_s`` seconds from now."""
        with self._lock:
            self._until = max(self._until, time.monotonic() + duration_s + self.margin_s)

    def clear(self) -> None:
        """Drop the gate immediately (playback failed or was preempted)."""
        with self._lock:
            self._until = 0.0

    @property
    def active(self) -> bool:
        with self._lock:
            return time.monotonic() < self._until

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self._until - time.monotonic())
