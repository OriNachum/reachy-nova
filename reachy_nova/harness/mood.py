"""Mood: a small decaying state rendered as one context sentence (t6).

Ports the *idea* behind ``config/emotions.yaml`` -- a handful of bounded
dimensions, each with a resting baseline and a per-second decay back toward
it, nudged by a small per-event delta table -- without loading that YAML file
or the legacy ``emotions.py`` module (that machinery assumes the retired
direct-SDK app's event loop; see the plan's t6 instruction and spec
assumption c9). The same pat should read differently depending on how the
rest of the day has gone: a pat right after a long lonely silence and a pat
two minutes after the last one are different events, and this is the whole
state that makes them different.

Four dimensions, each clamped to ``[0.0, 1.0]``:

* **playful** -- baseline low, spikes on a pat, decays over a couple of
  minutes.
* **calm** -- baseline high (the resting state is calm, not neutral-zero),
  dips on excitement and recovers slowly.
* **lonely** -- baseline moderate-high (idle time drifts *toward* lonely,
  not away from it), pushed down by any sign someone is there (a pat, a
  recognised face, a conversational turn).
* **cheeky** -- baseline low, the sharpest and fastest-decaying spike: the
  immediate "just been petted" reaction, gone within a minute or two so it
  does not linger and blur into general playfulness.

Decay is computed functionally rather than by mutating a running value on a
timer: each dimension stores the level *recorded at* its last :meth:`Mood.note`
call and the time it was recorded, and the current value at any ``now`` is an
exponential blend of that recorded level toward the baseline,
``baseline + (recorded - baseline) * exp(-elapsed / tau)``. Two things fall
out of that for free: the value can never leave ``[baseline, recorded]``'s
convex hull (both already in ``[0, 1]``, so the blend never needs clamping),
and reading the state twice without an intervening :meth:`note` costs nothing
and needs no background timer -- the same "compute the age at read time"
shape as :mod:`reachy_nova.harness.sense_history`.

``silence`` is deliberately NOT one of the levels nudged by an event delta.
It is *derived*: the time elapsed since the last real :meth:`note` call, used
directly by :meth:`Mood.render` to choose the lonely/bored sentences and to
override the level-driven ones once it gets long enough (nothing a moment-old
playful spike says matters if nobody has spoken in ten minutes). An explicit
``note("silence")`` call is accepted as a harmless no-op — it carries no
delta and, notably, does *not* itself count as activity, or the silence clock
could never advance.

Thread-safety: one lock guards every read and write of the recorded levels
and the last-activity timestamp, matching the pattern in
:mod:`reachy_nova.harness.quiet` and :mod:`reachy_nova.harness.sense_history`.

The clock is injectable monotonic seconds (``time.monotonic`` by default) --
mood is process-lifetime state, never persisted, so wall-clock/restart
concerns (see ``quiet.py``) do not apply here.

stdlib only; never imports ``reachy_mini`` (``tests/test_harness_boundary.py``).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

#: name -> (baseline, decay time-constant in seconds). Larger tau = slower
#: return to baseline. ``lonely``'s baseline sits well above zero because the
#: resting state, with nobody around, drifts toward lonely rather than away
#: from it; interaction pulls it down and it climbs back over minutes.
DIMENSIONS: dict[str, tuple[float, float]] = {
    "playful": (0.2, 120.0),
    "calm": (0.6, 240.0),
    "lonely": (0.5, 420.0),
    "cheeky": (0.1, 90.0),
}

#: Per-event-class deltas applied to each dimension by :meth:`Mood.note`,
#: scaled by the caller's ``intensity``. A small table, ported from the shape
#: of ``config/emotions.yaml``'s ``events:`` section (deltas per emotion per
#: event), not its content or its module.
#:
#: ``silence`` is listed here for documentation -- it is one of the five
#: event classes the plan names -- but carries no delta: see the module
#: docstring for why it is derived rather than applied.
EVENT_DELTAS: dict[str, dict[str, float]] = {
    "pat": {"cheeky": 0.55, "playful": 0.35, "calm": -0.1, "lonely": -0.4},
    "face": {"lonely": -0.5, "calm": 0.15, "playful": 0.05},
    "user_turn": {"lonely": -0.3, "calm": 0.05},
    "assistant_turn": {"lonely": -0.15, "calm": 0.05},
    "silence": {},
}

# -- render() thresholds ----------------------------------------------------

#: Cheeky is the sharpest, most specific reaction -- checked first.
CHEEKY_HI = 0.5
#: Playful is a softer, longer-lingering elevation than cheeky.
PLAYFUL_HI = 0.35
#: Below this many seconds of silence, level-driven sentences win.
LONELY_SILENCE_S = 180.0
#: Past this many seconds of silence, silence dominates regardless of levels.
BORED_SILENCE_S = 600.0

NEUTRAL_SENTENCE = "You are calm and easy right now."
LONELY_SENTENCE = "It has been quiet for a while; you feel a little lonely."
CHEEKY_SENTENCE = "You have just been petted and feel cheeky."
PLAYFUL_SENTENCE = "You are in a playful mood."


class Mood:
    """Small decaying mood state, rendered as one second-person sentence.

    Parameters
    ----------
    clock:
        Zero-arg monotonic-seconds source, injectable for tests. ``None``
        uses :func:`time.monotonic`.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        now = self._clock()
        # The level recorded at the last note() for each dimension, and when.
        self._levels: dict[str, float] = {
            name: baseline for name, (baseline, _tau) in DIMENSIONS.items()
        }
        self._recorded_at: dict[str, float] = dict.fromkeys(DIMENSIONS, now)
        # When any REAL activity (a non-"silence" note) last happened.
        self._last_activity = now

    def _value_at_locked(self, name: str, now: float) -> float:
        """Current value of dimension *name* at time *now*. Caller holds the lock."""
        baseline, tau = DIMENSIONS[name]
        recorded = self._levels[name]
        elapsed = max(0.0, now - self._recorded_at[name])
        weight = math.exp(-elapsed / tau) if tau > 0.0 else 0.0
        return baseline + (recorded - baseline) * weight

    def note(self, event: str, intensity: float = 1.0) -> None:
        """Apply *event*'s deltas (scaled by *intensity*) to every dimension.

        An event class absent from :data:`EVENT_DELTAS` is a safe no-op --
        an unrecognised sense class must never raise out of the mood state.
        """
        deltas = EVENT_DELTAS.get(event)
        if deltas is None:
            return
        with self._lock:
            now = self._clock()
            for name in DIMENSIONS:
                current = self._value_at_locked(name, now)
                delta = deltas.get(name, 0.0) * intensity
                self._levels[name] = min(1.0, max(0.0, current + delta))
                self._recorded_at[name] = now
            if event != "silence":
                self._last_activity = now

    def state(self, now: float | None = None) -> dict[str, float]:
        """Current levels plus ``silence_s``, for tests/observability."""
        moment = self._clock() if now is None else now
        with self._lock:
            levels = {
                name: self._value_at_locked(name, moment) for name in DIMENSIONS
            }
            silence_s = max(0.0, moment - self._last_activity)
        levels["silence_s"] = silence_s
        return levels

    def render(self, now: float | None = None) -> str:
        """One short second-person sentence describing the current mood.

        Priority (first match wins): a long silence dominates regardless of
        levels (nothing recent matters after ten quiet minutes); then the
        sharp, specific cheeky reaction; then the softer playful one; then a
        moderate silence reads as lonely; otherwise the neutral sentence.
        """
        snapshot = self.state(now)
        silence_s = snapshot["silence_s"]
        if silence_s >= BORED_SILENCE_S:
            minutes = int(silence_s // 60)
            return (
                f"Nobody has spoken to you for {minutes} minutes; "
                "you are a little bored."
            )
        if snapshot["cheeky"] >= CHEEKY_HI:
            return CHEEKY_SENTENCE
        if snapshot["playful"] >= PLAYFUL_HI:
            return PLAYFUL_SENTENCE
        if silence_s >= LONELY_SILENCE_S:
            return LONELY_SENTENCE
        return NEUTRAL_SENTENCE
