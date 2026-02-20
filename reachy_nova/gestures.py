"""Expressive gesture animations for Reachy Nova.

Defines 8 named gestures (yes, no, curious, pondering, boredom, nuzzle, purr, enjoy)
as parameterized animation curves driven by envelope patterns.
"""

import logging
import threading
import time

import numpy as np
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)

VALID_GESTURES = ("yes", "no", "curious", "pondering", "boredom", "nuzzle", "purr", "enjoy")
RAMP_TIME = 0.2  # seconds to ease into full amplitude
DT = 0.02  # 50Hz animation loop


def clamp_pose(yaw: float, pitch: float) -> tuple[float, float]:
    """Clamp head yaw/pitch to safe limits."""
    return max(-45.0, min(45.0, yaw)), max(-15.0, min(25.0, pitch))


class GestureEngine:
    """Executes named gesture animations on the robot head."""

    def __init__(self, reachy_mini, state, emotional_state, gesture_cancel_event: threading.Event):
        self._reachy_mini = reachy_mini
        self._state = state
        self._emotional_state = emotional_state
        self._cancel = gesture_cancel_event

    def execute(self, gesture_name: str) -> str:
        """Run a gesture animation. Blocks until complete or cancelled."""
        gesture = gesture_name.lower()
        if gesture not in VALID_GESTURES:
            return f"[Unknown gesture '{gesture}'. Available: {', '.join(VALID_GESTURES)}]"

        if not self._state.get("movement_enabled"):
            return "[Movement disabled (presentation mode). Cannot perform gesture.]"

        # Cancel any running gesture
        self._cancel.set()
        time.sleep(0.05)
        self._cancel.clear()

        # Capture current head position as animation center
        center_yaw = self._state.get("smooth_yaw")
        center_pitch = self._state.get("smooth_pitch")

        self._state.update(
            gesture_active=True,
            gesture_name=gesture,
            head_override=None,
        )

        try:
            method = getattr(self, f"_gesture_{gesture}")
            method(center_yaw, center_pitch)
            return f"[Gesture '{gesture}' completed.]"
        finally:
            self._state.update(
                smooth_yaw=center_yaw,
                smooth_pitch=center_pitch,
                gesture_active=False,
                gesture_name="",
            )

    def cancel(self) -> None:
        """Cancel any running gesture."""
        self._cancel.set()

    def _send_pose(self, yaw: float, pitch: float, roll: float = 0.0) -> None:
        pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
        self._reachy_mini.set_target(head=pose)

    def _cancelled(self) -> bool:
        return self._cancel.is_set()

    # --- Gesture implementations ---

    def _gesture_yes(self, center_yaw: float, center_pitch: float) -> None:
        duration, freq, amp = 1.2, 2.5, 12.0
        elapsed = 0.0
        while elapsed < duration and not self._cancelled():
            ramp = 0.5 * (1.0 - np.cos(np.pi * min(elapsed / RAMP_TIME, 1.0)))
            decay = 1.0 - 0.7 * (elapsed / duration)
            envelope = ramp * decay
            pitch = center_pitch + amp * envelope * np.sin(2.0 * np.pi * freq * elapsed)
            yaw, pitch = clamp_pose(center_yaw, pitch)
            self._send_pose(yaw, pitch)
            time.sleep(DT)
            elapsed += DT

    def _gesture_no(self, center_yaw: float, center_pitch: float) -> None:
        duration, freq, amp = 1.5, 2.0, 15.0
        elapsed = 0.0
        while elapsed < duration and not self._cancelled():
            ramp = 0.5 * (1.0 - np.cos(np.pi * min(elapsed / RAMP_TIME, 1.0)))
            decay = 1.0 - 0.65 * (elapsed / duration)
            envelope = ramp * decay
            yaw = center_yaw + amp * envelope * np.sin(2.0 * np.pi * freq * elapsed)
            yaw, pitch = clamp_pose(yaw, center_pitch)
            self._send_pose(yaw, pitch)
            time.sleep(DT)
            elapsed += DT

    def _gesture_curious(self, center_yaw: float, center_pitch: float) -> None:
        self._emotional_state.set_mood_override("curious", duration=10.0)
        target_roll = 15.0

        # Phase 1: ease into tilt
        elapsed = 0.0
        while elapsed < 0.4 and not self._cancelled():
            alpha = 0.5 * (1.0 - np.cos(np.pi * elapsed / 0.4))
            self._send_pose(center_yaw, center_pitch, roll=alpha * target_roll)
            time.sleep(DT)
            elapsed += DT

        # Phase 2: hold with subtle oscillation
        elapsed = 0.0
        while elapsed < 1.0 and not self._cancelled():
            osc = 2.0 * np.sin(2.0 * np.pi * 0.5 * elapsed)
            self._send_pose(center_yaw, center_pitch, roll=target_roll + osc)
            time.sleep(DT)
            elapsed += DT

        # Phase 3: ease back to neutral
        elapsed = 0.0
        while elapsed < 0.4 and not self._cancelled():
            alpha = 0.5 * (1.0 - np.cos(np.pi * elapsed / 0.4))
            self._send_pose(center_yaw, center_pitch, roll=target_roll * (1.0 - alpha))
            time.sleep(DT)
            elapsed += DT

    def _gesture_pondering(self, center_yaw: float, center_pitch: float) -> None:
        self._emotional_state.set_mood_override("thinking", duration=10.0)
        target_yaw, target_pitch = clamp_pose(center_yaw - 20.0, center_pitch - 12.0)

        # Phase 1: ease into pondering pose
        elapsed = 0.0
        while elapsed < 0.5 and not self._cancelled():
            alpha = 0.5 * (1.0 - np.cos(np.pi * elapsed / 0.5))
            yaw = center_yaw + alpha * (target_yaw - center_yaw)
            pitch = center_pitch + alpha * (target_pitch - center_pitch)
            self._send_pose(yaw, pitch)
            time.sleep(DT)
            elapsed += DT

        # Phase 2: hold with subtle drift
        elapsed = 0.0
        while elapsed < 1.3 and not self._cancelled():
            drift_yaw = 3.0 * np.sin(2.0 * np.pi * 0.3 * elapsed)
            drift_pitch = 2.0 * np.sin(2.0 * np.pi * 0.2 * elapsed)
            yaw, pitch = clamp_pose(target_yaw + drift_yaw, target_pitch + drift_pitch)
            self._send_pose(yaw, pitch)
            time.sleep(DT)
            elapsed += DT

        # Phase 3: ease back to center
        elapsed = 0.0
        while elapsed < 0.5 and not self._cancelled():
            alpha = 0.5 * (1.0 - np.cos(np.pi * elapsed / 0.5))
            yaw = target_yaw + alpha * (center_yaw - target_yaw)
            pitch = target_pitch + alpha * (center_pitch - target_pitch)
            self._send_pose(yaw, pitch)
            time.sleep(DT)
            elapsed += DT

    def _gesture_boredom(self, center_yaw: float, center_pitch: float) -> None:
        self._emotional_state.set_mood_override("sleepy", duration=10.0)
        direction = -1.0 if center_yaw > 0 else 1.0
        target_yaw = max(-45.0, min(45.0, center_yaw + direction * 30.0))
        target_pitch = -25.0

        # Phase 1: slow slide into bored pose
        elapsed = 0.0
        while elapsed < 2.0 and not self._cancelled():
            alpha = 0.5 * (1.0 - np.cos(np.pi * elapsed / 2.0))
            yaw = center_yaw + alpha * (target_yaw - center_yaw)
            pitch = center_pitch + alpha * (target_pitch - center_pitch)
            self._send_pose(yaw, pitch)
            time.sleep(DT)
            elapsed += DT

        # Phase 2: sigh/sag
        elapsed = 0.0
        while elapsed < 0.5 and not self._cancelled():
            dip = 5.0 * np.sin(np.pi * elapsed / 0.5)
            self._send_pose(target_yaw, target_pitch - dip)
            time.sleep(DT)
            elapsed += DT

        # Phase 3: hold while drifting
        elapsed = 0.0
        while elapsed < 1.5 and not self._cancelled():
            drift = 2.0 * np.sin(2.0 * np.pi * 0.2 * elapsed)
            self._send_pose(target_yaw + drift, target_pitch)
            time.sleep(DT)
            elapsed += DT

        # Phase 4: slow recovery
        elapsed = 0.0
        while elapsed < 1.5 and not self._cancelled():
            alpha = 0.5 * (1.0 - np.cos(np.pi * elapsed / 1.5))
            yaw = target_yaw + alpha * (center_yaw - target_yaw)
            pitch = target_pitch + alpha * (center_pitch - target_pitch)
            self._send_pose(yaw, pitch)
            time.sleep(DT)
            elapsed += DT

    def _gesture_nuzzle(self, center_yaw: float, center_pitch: float) -> None:
        self._emotional_state.set_mood_override("excited", duration=5.0)
        duration, freq, yaw_amp, roll_amp = 2.5, 1.8, 12.0, 5.0
        elapsed = 0.0
        while elapsed < duration and not self._cancelled():
            ramp = 0.5 * (1.0 - np.cos(np.pi * min(elapsed / RAMP_TIME, 1.0)))
            decay = 1.0 - 0.6 * (elapsed / duration)
            envelope = ramp * decay
            phase = 2.0 * np.pi * freq * elapsed
            yaw = max(-45.0, min(45.0, center_yaw + yaw_amp * envelope * np.sin(phase)))
            roll = roll_amp * envelope * np.sin(phase + 0.5)
            _, pitch = clamp_pose(0, center_pitch)
            self._send_pose(yaw, pitch, roll=roll)
            time.sleep(DT)
            elapsed += DT

    def _gesture_purr(self, center_yaw: float, center_pitch: float) -> None:
        self._emotional_state.set_mood_override("happy", duration=8.0)
        duration = 3.0
        lean_pitch = min(25.0, center_pitch + 8.0)
        elapsed = 0.0
        while elapsed < duration and not self._cancelled():
            ramp = 0.5 * (1.0 - np.cos(np.pi * min(elapsed / 0.4, 1.0)))
            decay = 1.0 - 0.3 * (elapsed / duration)
            envelope = ramp * decay
            pitch = center_pitch + (lean_pitch - center_pitch) * envelope
            roll = 4.0 * envelope * np.sin(2.0 * np.pi * 0.8 * elapsed)
            yaw = max(-45.0, min(45.0, center_yaw + 3.0 * envelope * np.sin(2.0 * np.pi * 0.6 * elapsed)))
            pitch = max(-15.0, min(25.0, pitch))
            self._send_pose(yaw, pitch, roll=roll)
            time.sleep(DT)
            elapsed += DT

    def _gesture_enjoy(self, center_yaw: float, center_pitch: float) -> None:
        self._emotional_state.set_mood_override("happy", duration=5.0)
        lean_pitch = min(25.0, center_pitch + 10.0)

        # Phase 1: lean into touch
        elapsed = 0.0
        while elapsed < 0.5 and not self._cancelled():
            alpha = 0.5 * (1.0 - np.cos(np.pi * elapsed / 0.5))
            pitch = center_pitch + alpha * (lean_pitch - center_pitch)
            self._send_pose(center_yaw, pitch)
            time.sleep(DT)
            elapsed += DT

        # Phase 2: short nuzzle while leaned
        elapsed = 0.0
        while elapsed < 1.0 and not self._cancelled():
            decay = 1.0 - 0.4 * elapsed
            yaw = max(-45.0, min(45.0, center_yaw + 8.0 * decay * np.sin(2.0 * np.pi * 2.0 * elapsed)))
            roll = 3.0 * decay * np.sin(2.0 * np.pi * 2.0 * elapsed + 0.5)
            self._send_pose(yaw, lean_pitch, roll=roll)
            time.sleep(DT)
            elapsed += DT

        # Phase 3: settle back
        elapsed = 0.0
        while elapsed < 0.5 and not self._cancelled():
            alpha = 0.5 * (1.0 - np.cos(np.pi * elapsed / 0.5))
            pitch = lean_pitch + alpha * (center_pitch - lean_pitch)
            self._send_pose(center_yaw, pitch)
            time.sleep(DT)
            elapsed += DT
