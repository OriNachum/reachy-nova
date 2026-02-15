"""Safety Manager for Reachy Nova.

Enforces head-body collision avoidance with head-priority resolution.
The head is treated as "self" — the body adjusts to accommodate head movements,
creating organic coordinated motion (body follows head at extremes).

Simplified from conversation_app's SafetyManager: no roll handling,
only yaw/pitch + body_yaw.
"""

import logging

import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Safety limits matching tracking.py conventions."""

    HEAD_YAW_LIMIT: float = 45.0       # ±45° (tracking.py MAX_YAW)
    HEAD_PITCH_MAX: float = 25.0       # max upward tilt (tracking.py MAX_PITCH)
    HEAD_PITCH_MIN: float = -15.0      # max downward tilt (tracking.py MIN_PITCH)
    BODY_YAW_LIMIT: float = 25.0       # ±25°
    MAX_YAW_DIFFERENCE: float = 30.0   # max |head_yaw - body_yaw|
    BODY_FOLLOW_THRESHOLD: float = 30.0  # head yaw beyond which body starts following
    BODY_FOLLOW_RATIO: float = 0.4     # how much body follows (0=none, 1=full match)
    SAFE_MARGINS: float = 5.0          # buffer zone in degrees


class SafetyManager:
    """Head-priority collision avoidance that creates organic movement.

    Runs every frame (~50Hz). When the head turns far to one side,
    the body automatically follows partway, making it look like the
    robot naturally turns to look rather than just rotating its head.
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        logger.info(
            f"SafetyManager initialized: HEAD_YAW=±{self.config.HEAD_YAW_LIMIT}°, "
            f"PITCH=[{self.config.HEAD_PITCH_MIN}°..{self.config.HEAD_PITCH_MAX}°], "
            f"BODY_YAW=±{self.config.BODY_YAW_LIMIT}°, "
            f"MAX_DIFF={self.config.MAX_YAW_DIFFERENCE}°"
        )

    def validate(
        self, yaw: float, pitch: float, body_yaw: float
    ) -> tuple[float, float, float]:
        """Validate and adjust angles for safety + organic movement.

        Args:
            yaw: Head yaw in degrees (positive = left).
            pitch: Head pitch in degrees (positive = up).
            body_yaw: Body yaw in degrees.

        Returns:
            (safe_yaw, safe_pitch, safe_body_yaw) in degrees.
        """
        cfg = self.config

        # 1. Clamp head angles
        safe_yaw = float(np.clip(yaw, -cfg.HEAD_YAW_LIMIT, cfg.HEAD_YAW_LIMIT))
        safe_pitch = float(np.clip(pitch, cfg.HEAD_PITCH_MIN, cfg.HEAD_PITCH_MAX))

        # 2. Clamp body yaw
        safe_body_yaw = float(np.clip(body_yaw, -cfg.BODY_YAW_LIMIT, cfg.BODY_YAW_LIMIT))

        # 3. Organic body follow: when head yaw exceeds threshold, body follows
        if abs(safe_yaw) > cfg.BODY_FOLLOW_THRESHOLD:
            # How far past the threshold the head is
            excess = abs(safe_yaw) - cfg.BODY_FOLLOW_THRESHOLD
            # Body follows proportionally in the same direction
            follow_target = np.sign(safe_yaw) * excess * cfg.BODY_FOLLOW_RATIO
            # Blend toward follow target (don't jump — caller uses EMA on body_yaw)
            safe_body_yaw = follow_target
            safe_body_yaw = float(
                np.clip(safe_body_yaw, -cfg.BODY_YAW_LIMIT, cfg.BODY_YAW_LIMIT)
            )

        # 4. Yaw difference enforcement: if |head - body| > limit, adjust body
        yaw_diff = safe_yaw - safe_body_yaw
        if abs(yaw_diff) > cfg.MAX_YAW_DIFFERENCE:
            # Head has priority — move body to reduce difference
            safe_body_yaw = safe_yaw - np.sign(yaw_diff) * cfg.MAX_YAW_DIFFERENCE
            safe_body_yaw = float(
                np.clip(safe_body_yaw, -cfg.BODY_YAW_LIMIT, cfg.BODY_YAW_LIMIT)
            )

        return safe_yaw, safe_pitch, safe_body_yaw
