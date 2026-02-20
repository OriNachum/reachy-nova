# Antenna Calibration

Guide to tuning antenna animation parameters for Reachy Mini running Reachy Nova.

**File:** `reachy_nova/main.py`

## How Antenna Animation Works

Antennas are driven by a four-stage pipeline that runs every frame (~30Hz):

1. **Mood profile** — a sine-wave oscillation with mood-specific frequency, amplitude, phase, and offset. Uses a decoupled time accumulator (`_antenna_t`) so loop stalls never cause the target to jump.
2. **Overlays** — transient effects like pat-vibration added on top.
3. **Double EMA smoothing** — a cascaded low-pass filter that smooths both position and velocity.
4. **Velocity clamp** — caps the maximum deg/frame to prevent servo stutter.

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

## Double EMA Smoothing

A cascaded (double) exponential moving average smooths both position and velocity. Two identical EMA passes run in sequence, eliminating frame-to-frame jitter and stutter artifacts caused by variable `dt`.

```
alpha = 1 - exp(-dt / ANTENNA_SMOOTH_TAU)
stage1 = stage1 + alpha * (input - stage1)       # smooths position
stage2 = stage2 + alpha * (stage1 - stage2)       # smooths velocity
output = stage2
```

The `dt` used for antenna computation is capped at 30ms so loop stalls don't cause large filter steps.

### `ANTENNA_SMOOTH_TAU`

Default: **0.10 s** (per stage; effective smoothing similar to a single EMA at ~0.20 s but with continuous velocity)

| tau | Effect |
| :--- | :--- |
| 0.05 s | Light double smoothing. Pat vibration clearly visible. |
| 0.10 s | Default. Eliminates dt-jitter stutter, preserves all mood oscillations. Pat vibration softened (~60%). |
| 0.15 s | Heavy smoothing. Fast moods (excited at 0.7 Hz) start to feel damped. |
| 0.20 s | Very heavy. Only slow moods look normal. Pat vibration nearly invisible. |

**Rule of thumb:** `tau` should be at most ~1/(4 * max_freq). For excited mood at 0.7 Hz, that gives ~0.36 s as the upper limit per stage before the oscillation visibly lags.

### Interaction with Pat Vibration

The pat overlay runs at 3.5 Hz. The double EMA attenuates high frequencies more than a single pass. `_PAT_ANT_AMP` is set to **10.0 deg** (up from 6.0) to compensate. If pat feedback feels too subtle, either:

- Lower `ANTENNA_SMOOTH_TAU` (e.g., 0.07), or
- Increase `_PAT_ANT_AMP` further.

## Velocity Clamp

After the double EMA, a velocity clamp limits the maximum position change per frame. This prevents the servos from receiving steps too large to track smoothly.

```
max_step = ANTENNA_MAX_SLEW * dt
delta = clamp(output - prev_output, -max_step, max_step)
output = prev_output + delta
```

### `ANTENNA_MAX_SLEW`

Default: **80.0 deg/s**

| Slew | Effect |
| :--- | :--- |
| 60 deg/s | Conservative. Very smooth but excited mood visibly slower. |
| 80 deg/s | Default. Excited mood looks responsive, servos track cleanly. |
| 120 deg/s | Light clamping. Only active during fast mood transitions. |

At 30Hz frame rate, 80 deg/s = ~2.7 deg/frame max step. Excited mood peak velocity (132 deg/s) is clamped to this limit.

## Pat Vibration Overlay

When the robot detects a head pat (Level 1), antennas vibrate briefly:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `_PAT_ANT_DUR` | 2.0 s | Total vibration duration |
| `_PAT_ANT_FREQ` | 3.5 Hz | Vibration frequency |
| `_PAT_ANT_AMP` | 10.0 deg | Peak vibration amplitude |

The envelope decays quadratically: `(1 - t/dur)^2`, so the vibration fades naturally.

## Sleep Breathing

During sleep mode, antennas use a separate animation path (not affected by mood profiles or EMA smoothing):

- **Frequency:** 0.05 Hz
- **Amplitude:** 1.5 deg (radians: ~0.026)
- **Base position:** `SLEEP_ANTENNAS_JOINT_POSITIONS` from `reachy_mini`

This runs in the sleep short-circuit branch of the main loop and bypasses all post-processing.

## Debug Logging

Set `ANTENNA_DEBUG = True` in `main.py` to enable diagnostic logging to `/tmp/antenna_diag.log`. Events logged:

- `DT_SPIKE` — frames where `dt > 40ms` (loop stall detected)
- `CLAMP` — frames where the velocity clamp is active

## Quick Reference

| Constant | Default | Location |
| :--- | :--- | :--- |
| `MOOD_ANTENNAS` | (table above) | `main.py` top-level dict |
| `MOOD_BLEND_TIME` | 1.5 s | `main.py` line ~130 |
| `ANTENNA_SMOOTH_TAU` | 0.10 s | `main.py` line ~131 |
| `ANTENNA_MAX_SLEW` | 80.0 deg/s | `main.py` line ~132 |
| `ANTENNA_DEBUG` | False | `main.py` line ~133 |
| `_PAT_ANT_DUR` | 2.0 s | `main.py` main loop |
| `_PAT_ANT_FREQ` | 3.5 Hz | `main.py` main loop |
| `_PAT_ANT_AMP` | 10.0 deg | `main.py` main loop |

## Calibration Procedure

1. Start the app: `uv run python -m reachy_nova.main`
2. Set mood to `calm` via API: `curl -X POST http://localhost:8042/api/mood -H 'Content-Type: application/json' -d '{"mood":"calm"}'`
3. Watch the antennas. If movement is jerky, increase `ANTENNA_SMOOTH_TAU` by 0.02 increments.
4. Switch to `excited` mood. If the oscillation looks sluggish or damped, decrease `ANTENNA_SMOOTH_TAU` or increase `ANTENNA_MAX_SLEW`.
5. Pat the robot's head. If the vibration is too faint, increase `_PAT_ANT_AMP` or decrease `ANTENNA_SMOOTH_TAU`.
6. If stuttering persists despite tuning, enable `ANTENNA_DEBUG = True` and check `/tmp/antenna_diag.log` to distinguish software-side issues from driver/SDK-side issues.
7. Toggle antenna modes via API to verify transitions:
   - `curl -X POST http://localhost:8042/api/antenna/mode -H 'Content-Type: application/json' -d '{"mode":"off"}'`
   - `curl -X POST http://localhost:8042/api/antenna/mode -H 'Content-Type: application/json' -d '{"mode":"auto"}'`
