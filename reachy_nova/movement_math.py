"""Shared movement math utilities for Reachy Mini motion animation."""

import numpy as np


def ease_sin(t: float, freq: float) -> float:
    """Pure sine wave — naturally smooth, decelerates at peaks."""
    return np.sin(2.0 * np.pi * freq * t)


def ease_sin_soft(t: float, freq: float) -> float:
    """Sine-of-sine — zero velocity AND acceleration at extremes.

    Dwells at each peak with ultra-smooth starts and stops. Traversal
    through center is 1.57× faster than plain sine, giving an organic,
    breathing quality.
    """
    return float(np.sin(0.5 * np.pi * np.sin(2.0 * np.pi * freq * t)))
