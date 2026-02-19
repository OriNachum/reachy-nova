# Patting Detection Documentation

This documentation covers the two-level pat detection system, which allows the robot to sense and react to being patted on the head with escalating responses.

## Overview

Pat detection is a heuristic-based system that uses the difference between **commanded servo position** and **actual servo position** to detect physical interactions. The system has two escalation levels:

- **Level 1** (initial pat): Subtle antenna vibration + mild emotion. No head movement, no voice.
- **Level 2** (sustained scratching): Full reaction — head nuzzle, voice response, significant emotion boost.

**File:** `reachy_nova/tracking.py` (specifically `PatDetector` class)

## Concept: Deviation-Based Sensing

The system does not use touch sensors. Instead, it monitors the **pitch (up/down)** joint of the head.

1.  **Commanded Pitch**: Where the code told the head to be.
2.  **Actual Pitch**: Where the servo encoders say the head actually is.
3.  **Deviation**: `Actual - Commanded`.

-   **Normal Operation**: Deviation is small (near 0), mostly due to gravity or friction.
-   **Pat Event**: A downward force (pat) causes the head to dip *below* the commanded angle, resulting in a significant negative deviation.

## `PatDetector` State Machine

The `PatDetector` uses a three-state machine: `idle` -> `level1` -> `level2_cooldown` -> `idle`.

### States

| State | Description |
| :--- | :--- |
| `idle` | Waiting for initial presses. 2+ presses in 2.5s triggers Level 1. |
| `level1` | Tracking sustained interaction. If presses continue for 5-15s (random), triggers Level 2. Resets to idle if no presses for 3s. |
| `level2_cooldown` | Suppresses all detection for 5s after Level 2, then resets to idle. |

### Interaction Flow

```
User taps head 2-3x quickly
    -> Level 1 fires (antenna vibration, mild joy +0.05)
    -> PatDetector enters "level1" state
    -> Random threshold generated (5-15s)

User keeps scratching/tapping...
    -> After threshold elapsed: Level 2 fires
       (head nuzzle + voice + joy +0.30, sadness -0.20, fear -0.15)
    -> PatDetector enters "level2_cooldown" (5s)

User stops for 3+ seconds during level1...
    -> Interaction resets to idle (no Level 2)
    -> Next taps will trigger Level 1 again
```

### Detection Logic

1.  **Baseline Correction**:
    -   Uses an EMA (Exponential Moving Average) to track the "resting" deviation (e.g., constant gravity sag).
    -   Subtracts this baseline to get a clean signal.
2.  **Press Detection**:
    -   Threshold: Checks if `deviation < -1.5` degrees (`press_threshold`).
    -   This indicates a "press" (head pushed down).
3.  **Pattern Recognition** (idle state):
    -   Counts distinct "press" events within 2.5s window.
    -   Requires 2+ presses to trigger Level 1.
4.  **Sustained Interaction** (level1 state):
    -   Tracks `_last_press_time` for gap detection.
    -   If gap > 3s, resets to idle.
    -   If elapsed time exceeds random threshold (5-15s), triggers Level 2.

### Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `press_threshold` | `1.5` deg | Minimum deviation to count as a press. |
| `release_threshold` | `0.6` deg | Deviation must return above this to "release" the press. |
| `min_presses` | `2` | Number of taps required to trigger Level 1. |
| `pat_window` | `2.5` s | Time window to accumulate taps. |
| `pat_cooldown` | `3.0` s | Minimum time between separate pat events. |
| `_interaction_gap_timeout` | `3.0` s | No presses for this long resets from level1 to idle. |
| `_level2_cooldown` | `5.0` s | Cooldown after Level 2 before new detection. |
| `_level2_threshold` | `5-15` s | Random threshold for sustained interaction (per session). |

## Reactions

### Level 1: Antenna Vibration

When Level 1 fires, `main.py` sets `pat_antenna_time` which triggers an antenna vibration overlay:

- **Frequency**: 3.5 Hz (distinctly faster than any mood animation)
- **Amplitude**: 6 degrees
- **Duration**: 2 seconds
- **Envelope**: Squared decay `(1 - t/dur)^2` for natural fade-out
- **Both antennas vibrate in sync** (same phase)

### Level 2: Full Reaction

When Level 2 fires:

1.  **Head Nuzzle**: Sinusoidal side-to-side motion overlay.
    -   Amplitude: 8.0 degrees, Frequency: 2.5 Hz, Duration: 1.5s.
    -   Linear decay envelope.
2.  **Voice**: "Someone is scratching your head and it feels wonderful."
3.  **Emotion**: Significant joy boost (+0.30), sadness/fear reduction, wound healing (0.08).

## Emotion Events

| Event | Severity | Joy | Sadness | Fear | Wound Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pat_level1` | mild | +0.05 | — | — | 0.0 |
| `pat_level2` | healing | +0.30 | -0.20 | -0.15 | 0.08 |

## Usage Example

```python
tracker = TrackingManager()

# In main loop (running at ~50Hz)
track_target = tracker.get_head_target(...)
robot.look_at(*track_target)

# Read feedback
actual_pose = robot.get_current_head_pose()
commanded_pose = robot.get_commanded_pose()

# Update detector — events fire via on_event callback
tracker.detect_pat(commanded_pose, actual_pose)
# "pat_level1" -> antenna vibration, mild emotion
# "pat_level2" -> head nuzzle + voice + strong emotion
```
