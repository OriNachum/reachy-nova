# Nova Vision Documentation

This documentation covers the `NovaVision` component, which enables the robot to understand its environment using its camera.

## Overview

`NovaVision` periodically captures frames from the robot's camera and sends them to Amazon Bedrock's **Nova Pro** model for analysis. This allows the robot to "see" and describe its surroundings.

**File:** `reachy_nova/nova_vision.py`

## Class Structure

### `NovaVision`

The main class responsible for vision processing.

#### Constructor

```python
NovaVision(
    region: str = "us-east-1",
    model_id: str = "us.amazon.nova-pro-v1:0",
    system_prompt: str = "...",
    analyze_interval: float = 5.0,
    on_description: Callable[[str], None] | None = None,
)
```

-   **region**: AWS region for Bedrock.
-   **model_id**: Bedrock model ID for Nova Pro.
-   **system_prompt**: Instructions for the vision model (e.g., "Describe what you see").
-   **analyze_interval**: Time between automatic frame analysis (seconds).
-   **on_description**: Callback with the text description of the scene.

#### Key Methods

-   `start(stop_event: threading.Event)`: Starts the background thread for periodic analysis.
-   `update_frame(frame: np.ndarray)`: Stores the latest camera frame (BGR from OpenCV) for processing.
-   `trigger_analyze()`: Forces an immediate analysis, interrupting the interval timer.
-   `analyze_frame(frame: np.ndarray, prompt: str)`: Sends a single frame to Bedrock and returns the description.

### Analysis Process

1.  **Frame Capture**: The main application continuously calls `update_frame()` with new camera frames.
2.  **Trigger**: The background thread waits for `analyze_interval` seconds or a manual trigger (e.g., user asks "What do you see?").
3.  **Preprocessing**: The frame is resized (max width 1280px) and encoded as JPEG to optimize latency and cost.
4.  **Bedrock Call**:
    -   Uses `boto3.client("bedrock-runtime").invoke_model()`.
    -   Sends the JPEG image and a text prompt ("What do you see?").
    -   Uses the `messages-v1` schema with `maxTokens=256` for concise descriptions.
5.  **Output**: The model returns a text description, which is passed to the `on_description` callback.

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
