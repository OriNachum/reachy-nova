"""Antenna animation pipeline for Reachy Mini.

Computes antenna positions based on mood oscillation profiles, pat vibration
overlays, double-EMA smoothing, and velocity clamping.  Called every main-loop
frame (~50 Hz) and returns a (2,) ndarray in radians ready for set_target().
"""

import logging

import numpy as np

from .movement_math import ease_sin, ease_sin_soft


# ---------------------------------------------------------------------------
# Mood antenna profiles
# ---------------------------------------------------------------------------

MOOD_ANTENNAS = {
    "happy": {
        "freq": 0.25, "amp": 18.0, "phase": "oppose",
        "ease": ease_sin,
    },
    "excited": {
        "freq": 0.7, "amp": 30.0, "phase": "oppose",
        "ease": ease_sin,
    },
    "curious": {
        "freq": 0.35, "amp": 18.0, "phase": "sync",
        "ease": ease_sin,
        "offset": 10.0,
    },
    "thinking": {
        "freq": 0.12, "amp": 15.0, "phase": "custom",
        "ease": ease_sin_soft,
    },
    "sad": {
        "freq": 0.08, "amp": 8.0, "phase": "sync",
        "ease": ease_sin_soft,
        "offset": -25.0,
    },
    "disappointed": {
        "freq": 0.06, "amp": 5.0, "phase": "sync",
        "ease": ease_sin_soft,
        "offset": -20.0,
    },
    "surprised": {
        "freq": 0.5, "amp": 25.0, "phase": "oppose",
        "ease": ease_sin,
        "offset": 15.0,
    },
    "sleepy": {
        "freq": 0.04, "amp": 4.0, "phase": "sync",
        "ease": ease_sin_soft,
        "offset": -18.0,
    },
    "proud": {
        "freq": 0.15, "amp": 6.0, "phase": "oppose",
        "ease": ease_sin_soft,
        "offset": 20.0,
    },
    "calm": {
        "freq": 0.12, "amp": 10.0, "phase": "oppose",
        "ease": ease_sin_soft,
    },
}

MOOD_BLEND_TIME = 1.5
ANTENNA_SMOOTH_TAU = 0.10   # seconds — EMA time constant (double-pass)
ANTENNA_MAX_SLEW = 80.0     # deg/s — velocity clamp to prevent servo stutter
ANTENNA_DEBUG = False        # set True to log to /tmp/antenna_diag.log

# Pat vibration overlay constants
_PAT_ANT_DUR = 2.0
_PAT_ANT_FREQ = 3.5
_PAT_ANT_AMP = 10.0


# ---------------------------------------------------------------------------
# AntennaAnimator
# ---------------------------------------------------------------------------

class AntennaAnimator:
    """Stateful antenna animation pipeline.

    Tracks internal time accumulator, EMA filter state, previous mood for
    blending, and velocity-clamp history.  Each call to ``update()`` advances
    one frame and returns antenna positions in radians.
    """

    def __init__(self, debug: bool = False):
        # Decoupled time accumulator — never jumps on loop stalls
        self._antenna_t: float = 0.0

        # Double-EMA filter state
        self._smooth_antennas = np.array([0.0, 0.0])
        self._smooth_antennas_2 = np.array([0.0, 0.0])

        # Velocity clamp history
        self._prev_output = np.array([0.0, 0.0])

        # Mood blending state
        self._prev_antennas = np.array([0.0, 0.0])
        self._prev_mood: str = "happy"
        self._mood_change_time: float = 0.0

        # Diagnostic logger (gated by *debug*)
        self._diag_logger = None
        if debug:
            self._diag_logger = logging.getLogger("antenna_diag")
            self._diag_logger.setLevel(logging.DEBUG)
            self._diag_logger.propagate = False
            fh = logging.FileHandler("/tmp/antenna_diag.log", mode="w")
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self._diag_logger.addHandler(fh)

    # ------------------------------------------------------------------
    def update(
        self,
        mood: str,
        dt: float,
        now: float,
        antenna_mode: str,
        pat_time: float,
    ) -> np.ndarray:
        """Advance one frame and return antenna positions in **radians**.

        Parameters
        ----------
        mood : str
            Current mood name (must be a key in ``MOOD_ANTENNAS``).
        dt : float
            Wall-clock delta since previous frame (seconds).
        now : float
            Current ``time.time()`` value (used for pat elapsed calc).
        antenna_mode : str
            ``"auto"`` for normal animation, ``"off"`` to zero antennas.
        pat_time : float
            ``time.time()`` of the most recent pat event (0 if none).

        Returns
        -------
        np.ndarray
            Shape ``(2,)`` in radians, ready for ``set_target(antennas=…)``.
        """
        ant_dt = min(dt, 0.03)  # cap so loop stalls don't cause target jumps
        self._antenna_t += ant_dt

        # --- Mood oscillation ---
        if antenna_mode == "off":
            antennas_deg = np.array([0.0, 0.0])
        else:
            if mood != self._prev_mood:
                self._prev_mood = mood
                self._mood_change_time = self._antenna_t

            profile = MOOD_ANTENNAS.get(mood, MOOD_ANTENNAS["happy"])
            ease_fn = profile["ease"]
            freq = profile["freq"]
            amp = profile["amp"]
            offset = profile.get("offset", 0.0)

            if profile["phase"] == "custom" and mood == "thinking":
                a1 = offset + amp * ease_fn(self._antenna_t, freq)
                a2 = -10.0 + 5.0 * ease_fn(self._antenna_t, freq * 1.5)
                target_antennas = np.array([a1, a2])
            elif profile["phase"] == "sync":
                a = offset + amp * ease_fn(self._antenna_t, freq)
                target_antennas = np.array([a, a])
            else:
                a = amp * ease_fn(self._antenna_t, freq)
                target_antennas = np.array([offset + a, offset - a])

            elapsed = self._antenna_t - self._mood_change_time
            if elapsed < MOOD_BLEND_TIME:
                alpha = elapsed / MOOD_BLEND_TIME
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                antennas_deg = self._prev_antennas * (1.0 - alpha) + target_antennas * alpha
            else:
                antennas_deg = target_antennas

            self._prev_antennas = antennas_deg

        # --- Pat vibration overlay ---
        pat_elapsed = now - pat_time
        if 0 < pat_elapsed < _PAT_ANT_DUR:
            env = (1.0 - pat_elapsed / _PAT_ANT_DUR) ** 2
            vib = _PAT_ANT_AMP * env * np.sin(2 * np.pi * _PAT_ANT_FREQ * pat_elapsed)
            antennas_deg = antennas_deg + np.array([vib, vib])

        # --- Double EMA smoothing ---
        ant_alpha = 1.0 - np.exp(-ant_dt / ANTENNA_SMOOTH_TAU)
        self._smooth_antennas += ant_alpha * (antennas_deg - self._smooth_antennas)
        self._smooth_antennas_2 += ant_alpha * (self._smooth_antennas - self._smooth_antennas_2)
        antennas_deg = self._smooth_antennas_2.copy()

        # --- Velocity clamp ---
        max_step = ANTENNA_MAX_SLEW * dt
        delta = antennas_deg - self._prev_output
        delta = np.clip(delta, -max_step, max_step)
        antennas_deg = self._prev_output + delta

        # --- Diagnostic logging ---
        if self._diag_logger is not None:
            jump = np.max(np.abs(delta))
            if dt > 0.04:
                self._diag_logger.debug(
                    "DT_SPIKE dt=%.4f jump=%.2f pos=[%.1f,%.1f]",
                    dt, jump, antennas_deg[0], antennas_deg[1],
                )
            if jump >= max_step * 0.99:
                self._diag_logger.debug(
                    "CLAMP    dt=%.4f step=%.2f max=%.2f pos=[%.1f,%.1f]",
                    dt, jump, max_step, antennas_deg[0], antennas_deg[1],
                )
        self._prev_output = antennas_deg.copy()

        return np.deg2rad(antennas_deg)
