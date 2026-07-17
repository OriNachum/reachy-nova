"""Additive harmonic synthesis for expressive non-speech robot vocalizations.

Pure-numpy synthesis of short (0.3-1.5s) expressive sounds a robot can
"vocalize" instead of speaking — a rising chirp, a warbling trill, a low
purring rumble. Each sound is built the same way: a fundamental sine plus
2-4 harmonics (additive synthesis), driven by a per-kind pitch envelope
(rising / falling / oscillating), shaped by an amplitude envelope that
always fades to (near) zero at both ends so the sound can be queued into a
speaker buffer next to other chunks without producing an audible click at
the seam.

No new dependencies — numpy only, matching the rest of the audio pipeline.
Sample rate defaults to 24000 Hz to match ``NovaSonic.OUTPUT_SAMPLE_RATE``
(see ``reachy_nova/nova_sonic.py``), since synthesized audio rides the same
buffer/resample path as Sonic's speech output (see ``handle_audio_output``
and the playback loop in ``reachy_nova/main.py``). The constant is kept
independent here (not imported) so this module stays a standalone,
dependency-light leaf that doesn't pull in the AWS SDK just to synthesize a
waveform.
"""

from __future__ import annotations

import numpy as np

# Expressive vocalization kinds this module knows how to synthesize.
# chirp_up = rising pitch envelope, purr_tone = falling-then-settled pitch
# envelope with a purr-rate tremolo, trill = oscillating (up/down) pitch
# envelope.
VALID_KINDS = ("chirp_up", "trill", "purr_tone")

DEFAULT_SAMPLE_RATE = 24000  # matches NovaSonic.OUTPUT_SAMPLE_RATE
MIN_DURATION = 0.3
MAX_DURATION = 1.5

# Per-kind default duration (seconds) when the caller doesn't pass one.
_DEFAULT_DURATIONS = {
    "chirp_up": 0.45,
    "trill": 0.65,
    "purr_tone": 1.0,
}

# Fixed short fade applied at both ends of every sound, regardless of kind,
# so the amplitude envelope always reaches (near) zero at the boundaries —
# this is what makes back-to-back queued chunks click-free.
_FADE_SECONDS = 0.02

# Harmonics above this fraction of Nyquist are dropped sample-by-sample to
# avoid aliasing at low requested sample rates; well below Nyquist for the
# default 24kHz rate so it never affects normal playback.
_NYQUIST_GUARD = 0.45


def _clamp_duration(duration: float) -> float:
    return float(min(MAX_DURATION, max(MIN_DURATION, duration)))


def _clamp_intensity(intensity: float) -> float:
    return float(min(1.0, max(0.0, intensity)))


def _pitch_envelope(kind: str, t: np.ndarray, duration: float, intensity: float) -> np.ndarray:
    """Instantaneous fundamental frequency (Hz) at each sample time."""
    frac = t / duration if duration > 0 else np.zeros_like(t)

    if kind == "chirp_up":
        # Rising pitch envelope: a bright, alert upward chirp.
        f_start = 320.0
        f_end = f_start + 480.0 * intensity
        return f_start + (f_end - f_start) * frac

    if kind == "trill":
        # Oscillating pitch envelope: several up/down warble cycles.
        center = 440.0
        depth = 70.0 * intensity
        rate = 12.0  # Hz — warble rate
        return center + depth * np.sin(2 * np.pi * rate * t)

    if kind == "purr_tone":
        # Falling pitch envelope: starts a bit higher, settles exponentially
        # down to a low resting purr pitch, then a tremolo (amplitude, not
        # pitch) carries the rumble for the rest of the sound.
        f_rest = 55.0
        f_start = f_rest + 35.0 * intensity
        decay_rate = 4.0
        return f_rest + (f_start - f_rest) * np.exp(-decay_rate * frac)

    raise ValueError(f"Unknown vocalize kind: {kind!r}. Valid kinds: {', '.join(VALID_KINDS)}")


def _amplitude_envelope(kind: str, t: np.ndarray, sample_rate: int) -> np.ndarray:
    """Overall amplitude shape: always fades in/out to (near) zero."""
    n = len(t)
    fade_n = max(1, int(round(_FADE_SECONDS * sample_rate)))
    fade_n = min(fade_n, n // 2) if n >= 2 else 0

    env = np.ones(n, dtype=np.float64)
    if fade_n > 0:
        # Raised-cosine (Hann-style) ramp: exactly 0 at the boundary sample,
        # exactly 1 once fully ramped in/out — no discontinuity, no click.
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_n) / fade_n))
        env[:fade_n] *= ramp
        env[n - fade_n:] *= ramp[::-1]

    if kind == "purr_tone":
        # Slow tremolo on top of the fade — the "rumble" of a purr. Kept
        # above a floor so it reads as one sustained purr, not several
        # separate blips.
        tremolo_rate = 24.0  # Hz, roughly a cat's purr rate
        env = env * (0.7 + 0.3 * np.sin(2 * np.pi * tremolo_rate * t))

    return env


def synthesize(
    kind: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration: float | None = None,
    intensity: float = 1.0,
    num_harmonics: int = 3,
) -> np.ndarray:
    """Synthesize a short expressive non-speech vocalization.

    Additive harmonic synthesis: a fundamental sine plus ``num_harmonics``
    additional harmonics (clamped to 2-4, so 3-5 partials total), each
    decaying in amplitude as ``1/n``, driven by a per-``kind`` pitch
    envelope and shaped by an amplitude envelope that fades to (near) zero
    at both ends.

    Args:
        kind: One of ``VALID_KINDS`` — ``"chirp_up"``, ``"trill"``, or
            ``"purr_tone"``.
        sample_rate: Output sample rate in Hz.
        duration: Length in seconds, clamped to ``[MIN_DURATION,
            MAX_DURATION]`` (0.3-1.5s). Defaults to a per-kind value when
            omitted.
        intensity: Expressiveness, clamped to ``[0.0, 1.0]`` — scales pitch
            excursion and overall loudness.
        num_harmonics: Number of harmonics above the fundamental, clamped
            to ``[2, 4]``.

    Returns:
        A 1-D ``float32`` numpy array, mono, with every sample in
        ``[-1.0, 1.0]``.

    Raises:
        ValueError: If ``kind`` isn't one of ``VALID_KINDS`` or
            ``sample_rate`` isn't positive.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Unknown vocalize kind: {kind!r}. Valid kinds: {', '.join(VALID_KINDS)}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")

    intensity = _clamp_intensity(intensity)
    num_harmonics = int(min(4, max(2, num_harmonics)))

    if duration is None:
        duration = _DEFAULT_DURATIONS[kind]
    duration = _clamp_duration(duration)

    n_samples = max(1, int(round(duration * sample_rate)))
    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    f0 = _pitch_envelope(kind, t, duration, intensity)

    # Integrate instantaneous fundamental frequency into a continuous phase
    # (rectangle-rule cumulative sum) so a moving pitch never introduces a
    # phase discontinuity — harmonic h's phase is simply h * phase0, since
    # its instantaneous frequency is h * f0(t).
    dt = 1.0 / sample_rate
    phase0 = 2.0 * np.pi * np.cumsum(f0) * dt

    nyquist_limit = _NYQUIST_GUARD * sample_rate
    partials = range(1, num_harmonics + 2)  # fundamental (1) + num_harmonics
    raw_amps = [1.0 / h for h in partials]
    norm = sum(raw_amps)

    signal = np.zeros(n_samples, dtype=np.float64)
    for h, raw_amp in zip(partials, raw_amps):
        # Drop a harmonic sample-by-sample once it would alias, so this
        # stays well-behaved at low requested sample rates too.
        below_nyquist = (h * f0) < nyquist_limit
        signal += (raw_amp / norm) * below_nyquist * np.sin(h * phase0)

    env = _amplitude_envelope(kind, t, sample_rate)
    # Overall gain tracks intensity but never fully mutes — intensity shapes
    # pitch excursion/loudness, it isn't an on/off switch.
    overall_gain = 0.3 + 0.7 * intensity
    signal = signal * env * overall_gain

    signal = np.clip(signal, -1.0, 1.0)
    return signal.astype(np.float32)
