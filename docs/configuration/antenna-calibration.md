# Antenna Calibration

Guide to tuning antenna animation parameters for Reachy Mini running Reachy Nova.

**File:** `reachy_nova/main.py`

## How Antenna Animation Works

Antennas are driven by a three-stage pipeline that runs every frame (~50Hz):

1. **Mood profile** — a sine-wave oscillation with mood-specific frequency, amplitude, phase, and offset.
2. **Overlays** — transient effects like pat-vibration added on top.
3. **EMA smoothing** — a time-aware low-pass filter that removes servo jitter before the final `set_target()` call.

Positions are computed in degrees, then converted to radians for the hardware.

## Mood Profiles

Each mood defines an oscillation profile in `MOOD_ANTENNAS`:

| Mood | Freq (Hz) | Amp (deg) | Phase | Offset (deg) | Ease |
| :--- | :--- | :--- | :--- | :--- | :--- |
| happy | 0.25 | 18 | oppose | 0 | sin |
| excited | 0.70 | 30 | oppose | 0 | sin |
| curious | 0.35 | 18 | sync | +10 | sin |
| thinking | 0.12 | 15 | custom | 0 | sin_soft |
| sad | 0.08 | 8 | sync | -25 | sin_soft |
| disappointed | 0.06 | 5 | sync | -20 | sin_soft |
| surprised | 0.50 | 25 | oppose | +15 | sin |
| sleepy | 0.04 | 4 | sync | -18 | sin_soft |
| proud | 0.15 | 6 | oppose | +20 | sin_soft |
| calm | 0.12 | 10 | oppose | 0 | sin_soft |

### Parameters

- **freq** — oscillation rate in Hz. Higher = faster movement.
- **amp** — peak displacement in degrees from center (or from offset).
- **phase** — `oppose` means left/right antennas mirror each other, `sync` means they move together, `custom` allows per-antenna expressions.
- **offset** — baseline angle shift in degrees. Negative = droopy, positive = perky.
- **ease** — `sin` is a pure sine wave; `sin_soft` applies sine-of-sine for extra dwell at extremes.

### Tuning Tips

- Keep `freq` below ~0.8 Hz. The servos can track faster, but the motion stops looking organic.
- `amp` above ~35 deg risks mechanical contact depending on antenna length.
- Use `sin_soft` easing for moods that should feel languid (sad, sleepy, thinking). Use plain `sin` for energetic moods.
- `offset` controls "resting posture". Ears-up moods use positive offsets; ears-down moods use negative.

## Mood Blending

When the mood changes, antennas don't snap to the new profile. A Hermite smooth-step blend runs over `MOOD_BLEND_TIME` (default **1.5 s**):

```
alpha = t^2 * (3 - 2t)    where t = elapsed / MOOD_BLEND_TIME
antennas = prev * (1 - alpha) + target * alpha
```

- Increase `MOOD_BLEND_TIME` for slower, more gradual mood transitions.
- Decrease it for snappier reactions (minimum ~0.3 s before blends become noticeable as jumps).

## EMA Smoothing

A time-aware exponential moving average is applied as the final step before sending positions to the servos. This eliminates frame-to-frame jitter caused by variable loop timing and abrupt value changes.

```
alpha = 1 - exp(-dt / ANTENNA_SMOOTH_TAU)
output = prev_output + alpha * (input - prev_output)
```

### `ANTENNA_SMOOTH_TAU`

Default: **0.08 s**

| tau | Effect |
| :--- | :--- |
| 0.04 s | Minimal smoothing — almost raw. Pat vibration at full strength. |
| 0.08 s | Default. Removes jitter, preserves all mood oscillations. Pat vibration slightly softened (~40%). |
| 0.15 s | Noticeably smoothed. Fast moods (excited at 0.7 Hz) start to feel damped. |
| 0.25 s | Heavy filtering. Only slow moods look normal. Pat vibration nearly invisible. |

**Rule of thumb:** `tau` should be at most ~1/(4 * max_freq). For excited mood at 0.7 Hz, that gives ~0.36 s as the upper limit before the oscillation visibly lags.

### Interaction with Pat Vibration

The pat overlay runs at 3.5 Hz. At `tau = 0.08`, the EMA attenuates it by roughly 40%. If pat feedback feels too subtle, either:

- Lower `ANTENNA_SMOOTH_TAU` (e.g., 0.05), or
- Increase `_PAT_ANT_AMP` (default 6 deg) to compensate.

## Pat Vibration Overlay

When the robot detects a head pat (Level 1), antennas vibrate briefly:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `_PAT_ANT_DUR` | 2.0 s | Total vibration duration |
| `_PAT_ANT_FREQ` | 3.5 Hz | Vibration frequency |
| `_PAT_ANT_AMP` | 6.0 deg | Peak vibration amplitude |

The envelope decays quadratically: `(1 - t/dur)^2`, so the vibration fades naturally.

## Sleep Breathing

During sleep mode, antennas use a separate animation path (not affected by mood profiles or EMA smoothing):

- **Frequency:** 0.05 Hz
- **Amplitude:** 1.5 deg (radians: ~0.026)
- **Base position:** `SLEEP_ANTENNAS_JOINT_POSITIONS` from `reachy_mini`

This runs in the sleep short-circuit branch of the main loop and bypasses all post-processing.

## Quick Reference

| Constant | Default | Location |
| :--- | :--- | :--- |
| `MOOD_ANTENNAS` | (table above) | `main.py` top-level dict |
| `MOOD_BLEND_TIME` | 1.5 s | `main.py` line ~124 |
| `ANTENNA_SMOOTH_TAU` | 0.08 s | `main.py` line ~125 |
| `_PAT_ANT_DUR` | 2.0 s | `main.py` main loop |
| `_PAT_ANT_FREQ` | 3.5 Hz | `main.py` main loop |
| `_PAT_ANT_AMP` | 6.0 deg | `main.py` main loop |

## Calibration Procedure

1. Start the app: `uv run python -m reachy_nova.main`
2. Set mood to `calm` via API: `curl -X POST http://localhost:8042/api/mood -H 'Content-Type: application/json' -d '{"mood":"calm"}'`
3. Watch the antennas. If movement is jerky, increase `ANTENNA_SMOOTH_TAU` by 0.02 increments.
4. Switch to `excited` mood. If the oscillation looks sluggish or damped, decrease `ANTENNA_SMOOTH_TAU`.
5. Pat the robot's head. If the vibration is too faint, increase `_PAT_ANT_AMP` or decrease `ANTENNA_SMOOTH_TAU`.
6. Toggle antenna modes via API to verify transitions:
   - `curl -X POST http://localhost:8042/api/antenna/mode -H 'Content-Type: application/json' -d '{"mode":"off"}'`
   - `curl -X POST http://localhost:8042/api/antenna/mode -H 'Content-Type: application/json' -d '{"mode":"auto"}'`
