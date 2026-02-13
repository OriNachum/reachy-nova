# Nova Vision Documentation

This documentation covers the `NovaVision` component, which enables the robot to understand its environment using its camera.

## Overview

`NovaVision` periodically captures frames from the robot's camera and sends them to Amazon Bedrock's **Nova 2 Lite** model for analysis. This allows the robot to "see" and describe its surroundings.

**File:** `reachy_nova/nova_vision.py`

## Class Structure

### `NovaVision`

The main class responsible for vision processing.

#### Constructor

```python
NovaVision(
    region: str = "us-east-1",
    model_id: str = "us.amazon.nova-2-lite-v1:0",
    system_prompt: str = "...",
    analyze_interval: float = 5.0,
    on_description: Callable[[str], None] | None = None,
)
```

-   **region**: AWS region for Bedrock.
-   **model_id**: Bedrock model ID for Nova 2 Lite.
-   **system_prompt**: Instructions for the vision model (e.g., "Describe what you see").
-   **analyze_interval**: Time between automatic frame analysis (seconds).
-   **on_description**: Callback with the text description of the scene.

#### Key Methods

-   `start(stop_event: threading.Event)`: Starts the background thread for periodic analysis.
-   `update_frame(frame: np.ndarray)`: Stores the latest camera frame (BGR from OpenCV) in a ring buffer (size 3) for processing.
-   `trigger_analyze()`: Forces an immediate analysis via the event path (`on_tracking_event`), resetting the periodic timer.
-   `analyze_latest(prompt: str)`: Analyzes the most recent buffered frame for a specific tool query (e.g., "Look Skill"). Returns the result directly without triggering the general `on_description` callback.
-   `reset_timer()`: Resets the fallback analysis countdown.

### Analysis Process

1.  **Frame Capture**: The main application calls `update_frame()` with new camera frames, which are stored in a ring buffer.
2.  **Trigger**: The background thread waits for `analyze_interval` seconds (default: 30s) or a manual/event trigger.
3.  **Preprocessing**: Frames are resized (max width 1280px) and encoded as JPEG.
4.  **Bedrock Call**:
    -   Uses `boto3.client("bedrock-runtime").invoke_model()`.
    -   Sends the JPEG image and a text prompt ("What do you see?" or custom query).
    -   Uses the `messages-v1` schema with `maxTokens=256`.
5.  **Output**:
    -   Periodic/Event Trigger: Result passed to `on_description` callback.
    -   Tool Trigger (`analyze_latest`): Result returned directly to caller.

### Frame Buffering

`NovaVision` maintains a `deque` of the last 3 frames to ensure that when a tool is called (e.g., "What is that?"), it has access to recent visual context even if the camera feed is slightly delayed.

### Timer Logic

-   **Periodic Fallback**: If no events occur for `analyze_interval` seconds, a general analysis runs.
-   **Event Reset**: Any analysis (periodic, manual, or tool-based) resets the timer via `reset_timer()`.

### Performance Considerations

-   **Resolution**: Frames are downscaled if wider than 1280px to stay within model limits and improve speed.
-   **Concurrency**: Analysis happens in a daemon thread so it doesn't block the robot's movement or audio processing.
-   **Locking**: Uses a `threading.Lock` to safely access the latest frame.

## Usage Example

```python
def handle_vision(description):
    print(f"Robot sees: {description}")
    # Inject into conversation
    sonic.inject_text(f"[Camera sees: {description}] React to this.")

vision = NovaVision(
    analyze_interval=10.0,
    on_description=handle_vision
)
vision.start(stop_event)

# In main loop
frame = robot.get_frame()
vision.update_frame(frame)
```
