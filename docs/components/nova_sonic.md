# Nova Sonic Documentation

This documentation covers the `NovaSonic` component, which handles the voice conversational interface for Reachy Nova.

## Overview

`NovaSonic` connects the robot's microphone and speaker to Amazon Bedrock's **Nova Sonic** model. It uses a bidirectional stream to achieve real-time, low-latency speech-to-speech interaction.

**File:** `reachy_nova/nova_sonic.py`

## Class Structure

### `NovaSonic`

The main class responsible for managing the conversation session.

#### Constructor

```python
NovaSonic(
    region: str = "us-east-1",
    model_id: str = "amazon.nova-sonic-v1:0",
    voice_id: str = "matthew",
    system_prompt: str = "...",
    on_transcript: Callable[[str, str], None] | None = None,
    on_audio_output: Callable[[np.ndarray], None] | None = None,
    on_state_change: Callable[[str], None] | None = None,
    tools: list[dict] | None = None,
    on_tool_use: Callable[[str, str, dict], None] | None = None,
)
```

-   **region**: AWS region for Bedrock.
-   **model_id**: Bedrock model ID for Nova Sonic.
-   **voice_id**: The voice personality (e.g., "matthew").
-   **system_prompt**: Instructions defining the robot's persona.
-   **on_transcript**: Callback when the model converts speech to text.
-   **on_audio_output**: Callback when the model sends audio to be played.
-   **on_state_change**: Callback when the system state (listening, thinking, speaking) changes.
-   **tools**: List of tool specifications (JSON schema) for function calling.
-   **on_tool_use**: Callback when the model requests to use a tool.

#### Key Methods

-   `start(stop_event: threading.Event)`: Launches the background thread that maintains the connection loop.
-   `feed_audio(samples: np.ndarray)`: Sends raw microphone audio (float32, 16kHz) to the model. Automatically handles mono conversion and int16 PCM encoding.
-   `inject_text(text: str)`: Allows the system to inject non-voice context (like "I see a cat") into the conversation stream as if it were a user message.
-   `send_tool_result(tool_use_id: str, result: str)`: Sends the output of a tool execution back to the model.

### Connection Protocol

The component uses `Boto3`'s `invoke_model_with_bidirectional_stream` to establish a persistent connection. The protocol involves sending JSON events:

1.  **sessionStart**: Configures inference parameters (maxTokens, temperature).
2.  **promptStart**: Defines output audio format (24kHz, 16-bit PCM) and **tool specifications**.
3.  **contentStart/textInput**: Sends the system prompt.
4.  **audioInput**: Streams user audio chunks.

The model responds with events containing:
-   **textOutput**: Partial or complete transcripts.
-   **audioOutput**: Base64-encoded PCM audio chunks.
-   **toolUse**: Requests to execute a function (name, ID, arguments).

### Implementation Details

-   **Tool Use Handling**:
    -   Listens for `contentStart` (type `TOOL`) and accumulates `toolUse` events.
    -   On `contentEnd` (type `TOOL`), validates JSON arguments and triggers `on_tool_use`.
    -   Expects the main application to execute the tool and return data via `send_tool_result`.
-   **Audio Format**:
    -   Input: 16kHz, mono, 16-bit PCM.
    -   Output: 24kHz, mono, 16-bit PCM.
-   **Threading**: Running in a daemon thread with an `asyncio` loop to handle the asynchronous stream.
-   **State Machine**:
    -   `idle`: Not connected.
    -   `listening`: Waiting for user input.
    -   `thinking`: Processing user input.
    -   `speaking`: Playing model response audio.

## Usage Example

```python
def handle_audio(samples):
    robot.play(samples)

sonic = NovaSonic(
    on_audio_output=handle_audio
)
sonic.start(stop_event)

# In main loop
mic_data = robot.get_audio()
sonic.feed_audio(mic_data)
```
