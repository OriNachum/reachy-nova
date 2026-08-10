"""Half-duplex echo gate shared between the speak and hear legs.

The wireless capture path has no verified hardware AEC (spec c19), so the
harness enforces half-duplex: while robot speech is playing on the speaker,
the mic feed to Sonic is suppressed. ``speaking.py`` arms the gate around each
playback window (duration + a tail margin); ``hearing.py`` checks it per chunk.
"""

from __future__ import annotations

import threading
import time


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
