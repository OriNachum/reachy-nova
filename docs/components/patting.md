# Patting Detection Documentation

This documentation covers the two-level pat detection system, which allows the robot to sense and react to being patted on the head with escalating responses.

## Overview

Pat detection is a heuristic-based system that uses the difference between **commanded servo position** and **actual servo position** to detect physical interactions. The system uses a **two-tier response architecture**:

- **Reflex tier** (immediate, hardcoded): Antenna vibration + emotion state changes. These happen within one frame, like spinal reflexes.
- **Deliberate tier** (model-driven, 1-2s latency): The AI model receives a sensory notification via `inject_text` and decides how to react — speech, gestures (nuzzle, purr, enjoy), or both.

**File:** `reachy_nova/tracking.py` (specifically `PatDetector` class)

## Concept: Deviation-Based Sensing

The system monitors both **pitch (up/down)** and **yaw (left/right)** joints of the head.

1.  **Commanded Pose**: Where the code told the head to be (pitch + yaw).
2.  **Actual Pose**: Where the servo encoders say the head actually is.
3.  **Deviation**: `Actual - Commanded` for each axis.

-   **Normal Operation**: Deviation is small (near 0), mostly due to gravity or friction.
-   **Scratch Event**: A downward force causes the head to dip *below* the commanded pitch, resulting in a significant negative pitch deviation.
-   **Side Pat Event**: A lateral nudge causes the head yaw to deviate from the commanded yaw, resulting in a significant absolute yaw deviation.

## Touch Type Classification

The system differentiates between two touch types based on which axis accumulates more press events:

| Touch Type | Axis | Detection | Description |
| :--- | :--- | :--- | :--- |
| `scratch` | pitch | `pitch_dev < -threshold` | Head pushed forward/down — typical head scratching or tapping from above |
| `side_pat` | yaw | `abs(yaw_dev) > threshold` | Head nudged side-to-side — gentle caressing of the side of the head |

Classification uses a simple majority vote: count pitch vs yaw presses in the recent window. If yaw presses outnumber pitch presses, the touch is classified as `side_pat`; otherwise `scratch`. The touch type is determined at Level 1 and carried through to Level 2.

## `PatDetector` State Machine

The `PatDetector` uses a three-state machine: `idle` -> `level1` -> `level2_cooldown` -> `idle`.

### States

| State | Description |
| :--- | :--- |
| `idle` | Waiting for initial presses. 2+ presses in 3.0s triggers Level 1. |
| `level1` | Tracking sustained interaction. If presses continue for 4-8s (random), triggers Level 2. Resets to idle if no presses for 5s. |
| `level2_cooldown` | Suppresses all detection for 5s after Level 2, then resets to idle. |

### Interaction Flow

```
User taps head 2-3x quickly (from above or from the side)
    -> Level 1 fires with touch_type ("scratch" or "side_pat")
       Reflex: antenna vibration, mild joy +0.05
       Deliberate (scratch): inject_text("You feel a gentle tap on the top of your head.")
       Deliberate (side_pat): inject_text("You feel a gentle touch on the side of your head.")
       Model may respond verbally or with a gesture
    -> PatDetector enters "level1" state
    -> Random threshold generated (4-8s)

User keeps scratching/tapping...
    -> After threshold elapsed: Level 2 fires (same touch_type as level 1)
       Reflex: emotion boost (joy +0.30, sadness -0.20, fear -0.15)
       Deliberate (scratch): inject_text("Someone is scratching your head...")
       Deliberate (side_pat): inject_text("Someone is caressing the side of your head...")
       Model decides reaction: nuzzle, purr, enjoy gesture, speech, or combination
    -> PatDetector enters "level2_cooldown" (5s)

User stops for 5+ seconds during level1...
    -> Interaction resets to idle (no Level 2)
    -> Next taps will trigger Level 1 again
```

### Detection Logic

1.  **Baseline Correction**:
    -   Uses an EMA (Exponential Moving Average, alpha=0.003) to track the "resting" deviation for each axis.
    -   Subtracts this baseline to get a clean signal.
2.  **Pitch Press Detection**:
    -   Threshold: Checks if `pitch_dev < -0.8` degrees (`press_threshold`).
    -   This indicates a "press" (head pushed down).
    -   Tagged as `("timestamp", "pitch")` in press history.
3.  **Yaw Press Detection**:
    -   Threshold: Checks if `abs(yaw_dev) > 0.8` degrees (`yaw_press_threshold`).
    -   This indicates a lateral nudge.
    -   Tagged as `("timestamp", "yaw")` in press history.
4.  **Pattern Recognition** (idle state):
    -   Counts distinct press events (both axes) within 3.0s window.
    -   Requires 2+ presses to trigger Level 1.
    -   Classifies touch type by majority axis at trigger time.
5.  **Sustained Interaction** (level1 state):
    -   Tracks `_last_press_time` for gap detection.
    -   If gap > 5s, resets to idle.
    -   If elapsed time exceeds random threshold (4-8s), triggers Level 2.

### Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `press_threshold` | `0.8` deg | Pitch deviation to count as a press (downward). |
| `release_threshold` | `0.3` deg | Pitch deviation below this to "release" the press. |
| `yaw_press_threshold` | `0.8` deg | Yaw deviation (absolute) to count as a press. |
| `yaw_release_threshold` | `0.3` deg | Yaw deviation below this to "release" the press. |
| `min_presses` | `2` | Number of taps required to trigger Level 1. |
| `pat_window` | `3.0` s | Time window to accumulate taps. |
| `pat_cooldown` | `2.0` s | Minimum time between separate pat events. |
| `_baseline_alpha` | `0.003` | EMA alpha for baseline tracking (both axes). |
| `_interaction_gap_timeout` | `5.0` s | No presses for this long resets from level1 to idle. |
| `_level2_cooldown` | `5.0` s | Cooldown after Level 2 before new detection. |
| `_level2_threshold` | `4-8` s | Random threshold for sustained interaction (per session). |

## Reactions

### Level 1: Reflex + Sensory Notification

When Level 1 fires:

1.  **Reflex — Antenna Vibration**: `main.py` sets `pat_antenna_time` which triggers an antenna vibration overlay:
    - **Frequency**: 3.5 Hz (distinctly faster than any mood animation)
    - **Amplitude**: 6 degrees
    - **Duration**: 2 seconds
    - **Envelope**: Squared decay `(1 - t/dur)^2` for natural fade-out
    - **Both antennas vibrate in sync** (same phase)
2.  **Reflex — Emotion**: Mild joy boost (+0.05)
3.  **Deliberate — Sensory inject** (touch-type dependent):
    - `scratch`: `inject_text("You feel a gentle tap on the top of your head.")`
    - `side_pat`: `inject_text("You feel a gentle touch on the side of your head.")`
    - Model decides response

### Level 2: Reflex + Sensory Notification

When Level 2 fires:

1.  **Reflex — Emotion**: Significant joy boost (+0.30), sadness/fear reduction, wound healing (0.08)
2.  **Deliberate — Sensory inject** (touch-type dependent):
    - `scratch`: `inject_text("Someone is scratching your head and it feels wonderful...")`
    - `side_pat`: `inject_text("Someone is caressing the side of your head and it feels wonderful...")`
    - Model decides response
3.  **Model may use gesture skill**: `nuzzle`, `purr`, or `enjoy` gestures are available for physical reactions

Note: Head nuzzle is no longer hardcoded in `tracking.py`. The model decides whether and how to physically react using the gesture skill.

## Emotion Events

| Event | Severity | Joy | Sadness | Fear | Wound Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pat_level1` | mild | +0.05 | — | — | 0.0 |
| `pat_level2` | healing | +0.30 | -0.20 | -0.15 | 0.08 |

## Available Touch Gestures

The model can use these gestures from the gesture skill to react to touch:

| Gesture | Description | Duration | Mood Override |
| :--- | :--- | :--- | :--- |
| `nuzzle` | Side-to-side yaw oscillation with subtle roll | ~2.5s | excited (5s) |
| `purr` | Slow pitch+roll wobble, leaning slightly down | ~3s | happy (8s) |
| `enjoy` | Brief lean into touch, short nuzzle, settle back | ~2s | happy (5s) |

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
# "pat_level1" -> data includes touch_type ("scratch" or "side_pat")
# "pat_level2" -> data includes touch_type (same as level 1)
```
