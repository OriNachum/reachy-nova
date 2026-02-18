# Head Gestures Documentation

This documentation covers the Head Gestures component, which enables the robot to perform expressive head movements like nodding, shaking, or emoting.

## Overview

The gesture system allows Reachy Nova to temporarily override the standard head tracking behavior to perform pre-defined animations. These gestures are used to non-verbally communicate intent, agreement, or emotional state.

**File:** `reachy_nova/main.py` (implemented within `gesture_executor`)

## Supported Gestures

The system currently supports the following gestures:

| Gesture | Description | Duration |
| :--- | :--- | :--- |
| **Yes** | Nod the head up and down (pitch oscillation). Used for agreement or confirmation. | ~1.2s |
| **No** | Shake the head side to side (yaw oscillation). Used for disagreement or denial. | ~1.5s |
| **Curious** | Tilt/roll the head to the side like a curious dog. Used when asking questions or expressing interest. | ~1.8s |
| **Pondering** | Look up and to the side diagonally. Used when thinking or processing a request. | ~2.3s |
| **Boredom** | Slow look away and down, with a sighing motion. Used when disengaged or inactive. | ~5.5s |

## Implementation Details

### `gesture_executor`

The gestures are managed by the `gesture_executor` function in `main.py`, which is registered as a skill.

#### Logic Flow

1.  **Request**: The skill manager receives a request with a `gesture` parameter.
2.  **Validation**: Checks if the gesture is valid (yes, no, curious, pondering, boredom).
3.  **Override**:
    *   Sets `app_state["gesture_active"] = True`.
    *   Takes control of the head via `reachy_mini.set_target(head=pose)`.
    *   Tracking continues in the background but its output is ignored during the gesture.
4.  **Animation**:
    *   Uses mathematical functions (sine waves, cosine easing) to generate smooth, organic trajectories.
    *   Gestures are often multi-phased (e.g., ease in -> hold/oscillate -> ease out).
5.  **Completion**:
    *   Resets `gesture_active` to `False`.
    *   Syncs the smoothing state (`_smooth_yaw`, `_smooth_pitch`) to the final gesture position to prevent sudden jumps when tracking resumes.

### Mood Integration

Some gestures automatically trigger mood overrides to match the physical expression:
*   **Curious**: Sets mood to `"curious"`.
*   **Pondering**: Sets mood to `"thinking"`.
*   **Boredom**: Sets mood to `"sleepy"`.

## Usage

Gestures can be triggered via the `gesture` skill.

### Parameters

*   `gesture` (string, required): The name of the gesture to perform.

### Examples

```json
{
  "gesture": "yes"
}
```

```json
{
  "gesture": "boredom"
}
```
