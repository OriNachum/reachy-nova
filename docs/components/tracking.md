# Tracking Manager Documentation

This documentation covers the `TrackingManager` component, which enables active tracking behaviors for the robot.

## Overview

The `TrackingManager` is responsible for fusing multiple sensor inputs to control the robot's head movement intelligently. It enables behaviors like:
-   **Looking at a speaking person** using sound direction (DoA).
-   **Tracking a face** using computer vision.
-   **Reacting to sudden sounds** like snaps or claps.
-   **Responding to physical pats** on the head.
-   **Idle animation** when no active target is present.

See also:
-   [Face Recognition](face_recognition.md): Background identification of specific people.
-   [Pat Detection](patting.md): Detailed mechanics of the physical interaction sensing.

**File:** `reachy_nova/tracking.py`

## Class Structure

### `TrackingManager`

The main class responsible for fusing sensor data and computing target angles.

#### Constructor

```python
TrackingManager(on_event: Callable[[str, dict], None] | None = None)
```

-   **on_event**: Callback for tracking events (e.g., `person_detected`, `snap_detected`).

#### Key Methods

-   `update_doa(doa_result: tuple)`: Updates the speaker direction from the microphone array.
-   `update_vision(frame: np.ndarray, t: float)`: Sends a camera frame to the background thread for face detection.
-   `detect_snap(audio_chunk: np.ndarray)`: Detects sharp audio transients (snaps/claps) for immediate reaction.
-   `get_head_target(t: float, voice_state: str, mood: str)`: Returns the target `(yaw, pitch)` angles.

### Events

The `TrackingManager` fires events to the `on_event` callback to notify the main application of significant changes:

| Event Type | Data | Description |
| :--- | :--- | :--- |
| `person_detected` | `{"bbox": (x1,y1,x2,y2)}` | Fired when a person is first detected after a period of absence. |
| `person_lost` | `{}` | Fired when a tracked person is lost (after hold duration). |
| `snap_detected` | `{"rms": float, "target_yaw": float}` | Fired when a snap/clap is detected. |
| `mode_changed` | `{"from": str, "to": str}` | Fired when the tracking priority mode changes (e.g., `idle` -> `face`). |

### Tracking Logic

The core logic uses a priority system to determine where to look:

1.  **Snap (Highest Priority)**:
    -   Triggered by sudden loud noises relative to background level.
    -   Overrides all other tracking for `1.5s`.
    -   Looks toward the sound source using the latest DoA angle.
    -   Uses fast movement smoothing.

2.  **Face Tracking**:
    -   Triggered by YOLO face detection.
    -   Active whenever a person is detected in the frame.
    -   Uses a proportional controller (`face_kp=30.0`) to center the face in the view.
    -   Maintains target for `2.0s` after losing sight of the face.

3.  **Speaker Tracking (DoA)**:
    -   Triggered by speech activity on the microphone array.
    -   Looks toward the sound source.
    -   Maintains target for `3.0s` after speech stops.
    -   Uses moderate smoothing (`alpha=0.2`).

4.  **Pat Reaction**:
    -   Triggered by physical pats on the head (detected via servo load).
    -   Adds a "nuzzle" overlay (side-to-side motion) to the current movement.
    -   Decaying amplitude over `1.5s`.
    -   See [Pat Detection](patting.md) for algorithm details.

5.  **Idle Animation (Lowest Priority)**:
    -   Triggered when no active target exists.
    -   Uses sinusoidal motion patterns based on the robot's state:
        -   `listening`: Small, attentive movements.
        -   `speaking`: Active, expressive movements.
        -   `thinking`: Slow, ponderous movements.
        -   `idle`: Gentle, wide scanning.

### Vision Pipeline

-   **Background Thread**: YOLO detection runs in a separate thread to avoid blocking the main loop.
-   **Optimization**: Only runs detection every `200ms`. Uses `ultralytics` YOLOv8n model.
-   **Lazy Loading**: The model is loaded only on the first use.

### Snap Detection Algorithm

-   Calculates RMS energy of incoming audio chunks.
-   Maintains a rolling average of past energy levels.
-   Triggers if `current_rms > 5.0 * rolling_avg` AND the previous chunk was quiet.
-   Prevents false triggers from continuous loud noises.

## Usage Example

```python
tracker = TrackingManager()

# In main loop
doa = robot.get_doa()
tracker.update_doa(doa)

audio = robot.get_audio()
tracker.detect_snap(audio)

if vision_enabled:
    frame = robot.get_frame()
    tracker.update_vision(frame, time.time())

yaw, pitch = tracker.get_head_target(
    time.time(), 
    voice_state="listening", 
    mood="curious"
)
robot.look_at(yaw, pitch)
```
