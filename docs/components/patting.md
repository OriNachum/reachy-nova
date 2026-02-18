# Patting Detection Documentation

This documentation covers the pat detection mechanism, which allows the robot to sense and react to being patted on the head.

## Overview

Pat detection is a heuristic-based system that uses the difference between **commanded servo position** and **actual servo position** to detect physical interactions. When a user pats the robot's head, the external force causes the servos to deviate from their target position (compliance/load error).

**File:** `reachy_nova/tracking.py` (specifically `PatDetector` class)

## Concept: Deviation-Based Sensing

The system does not use touch sensors. Instead, it monitors the **pitch (up/down)** joint of the head.

1.  **Commanded Pitch**: Where the code told the head to be.
2.  **Actual Pitch**: Where the servo encoders say the head actually is.
3.  **Deviation**: `Actual - Commanded`.

-   **Normal Operation**: Deviation is small (near 0), mostly due to gravity or friction.
-   **Pat Event**: A downward force (pat) causes the head to dip *below* the commanded angle, resulting in a significant negative deviation.

## `PatDetector` Class

The `PatDetector` class encapsulates the logic for identifying these events while filtering out noise and gravity bias.

### Logic Flow

1.  **Baseline Correction**:
    -   Uses an EMA (Exponential Moving Average) to track the "resting" deviation (e.g., constant gravity sag).
    -   Subtracts this baseline to get a clean signal.
2.  **Press Detection**:
    -   Threshold: Checks if `deviation < -1.5` degrees (`press_threshold`).
    -   This indicates a "press" (head pushed down).
3.  **Pattern Recognition**:
    -   Counts distinct "press" events.
    -   Requires **2 or more** presses (`min_presses`) within a **2.5s** window (`pat_window`).
    -   This distinguishes a deliberate patting motion from a single accidental bump.

### Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `press_threshold` | `1.5` deg | Minimum deviation to count as a press. |
| `release_threshold` | `0.6` deg | Deviation must return above this to "release" the press. |
| `min_presses` | `2` | Number of taps required to trigger a pat event. |
| `pat_window` | `2.5` s | Time window to accumulate taps. |
| `pat_cooldown` | `3.0` s | Minimum time between separate pat events. |

## Integration in `TrackingManager`

The `TrackingManager` uses the `PatDetector` to trigger a specific reaction:

### The "Nuzzle" Reaction
When a pat is detected:
1.  **Event Fired**: `pat_detected` event is sent.
2.  **Motion Overlay**:
    -   A sinusoidal "nuzzle" motion is added to the head's yaw (side-to-side).
    -   **Amplitude**: 8.0 degrees.
    -   **Frequency**: 2.5 Hz.
    -   **Duration**: 1.5 seconds.
    -   **Envelope**: Decays linearly from full strength to zero.

This gives the robot a lively, responsive feel, as if it is enjoying the pat.

## Usage Example

```python
tracker = TrackingManager()

# In main loop (running at ~60Hz)
# proper head pose must be retrieved from the robot hardware
track_target = tracker.get_head_target(...)
robot.look_at(*track_target)

# Read feedback
actual_pose = robot.get_current_head_pose()
commanded_pose = robot.get_commanded_pose() 

# Update detector
tracker.detect_pat(commanded_pose, actual_pose)
```
