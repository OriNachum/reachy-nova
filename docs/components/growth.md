# Growth (RLHF Feedback Capture)

Nova learns from reactions. When someone scratches her head, praises her, or corrects her, she records a rich multimodal feedback package — conversation history, camera footage, and audio — for future reinforcement learning from human feedback (RLHF).

## Overview

The growth system adds two skills (`remember_positively` and `remember_negatively`) backed by a `NovaFeedback` class that maintains three rolling buffers. When feedback is triggered, the buffers are snapshotted and written to disk in a background thread, producing a self-contained training example.

**File:** `reachy_nova/nova_feedback.py`

## How It Works

1.  **Continuous capture**: The main loop feeds every conversation turn, camera frame, and audio chunk into `NovaFeedback` buffers.
2.  **Trigger**: The AI model calls `remember_positively` or `remember_negatively` with a description of what it did.
3.  **Snapshot**: All three buffers are copied under a single lock (~instant).
4.  **Background save**: A daemon thread writes the package to disk (I/O heavy but non-blocking).

### Trigger Flow

```
User scratches head
    -> pat_level2 fires
    -> inject_text("...they liked what you just did.")
    -> Model calls remember_positively(what="told a funny joke", trigger="head scratch")
    -> NovaFeedback.record() snapshots buffers, spawns save thread
    -> Package written to ~/reachy_nova_data/feedback/
```

## Rolling Buffers

| Buffer | Type | Capacity | Feed Point |
| :--- | :--- | :--- | :--- |
| Conversation | `deque(maxlen=50)` of `{role, text, timestamp}` | 50 messages | `on_transcript` callback |
| Frames | `deque(maxlen=240)` of `(timestamp, jpeg_bytes)` | ~240 JPEGs (2 min at 2 Hz) | Main loop after `vision.update_frame()` |
| Audio | `deque` of float32 numpy chunks | ~2 min at 16 kHz (~1.92M samples) | Main loop before `sonic.feed_audio()` |

### Memory Footprint

-   **Frames**: ~240 JPEGs at ~50 KB each = ~12 MB
-   **Audio**: ~1.92M float32 samples = ~7.7 MB
-   **Total**: ~20 MB steady-state

### Throttling

-   **Frames**: `update_frame()` checks `time.time() - _last_frame_time >= 0.5` before encoding. Frames are JPEG-compressed on capture (`cv2.imencode`, quality 85).
-   **Audio**: After each `update_audio()` call, oldest chunks are trimmed when total samples exceed 1,920,000.

## Storage Format

Each feedback event creates a timestamped folder:

```
~/reachy_nova_data/feedback/
  2026-02-19T15-30-00_positive_fb_a1b2c3/
    feedback.json      # metadata
    messages.json      # last 50 conversation turns
    frames/            # ~240 JPEG images
      000000.jpg
      000001.jpg
      ...
    audio.wav          # 2 min of 16kHz mono PCM
```

### feedback.json

```json
{
  "id": "a1b2c3",
  "timestamp": 1740000600.0,
  "timestamp_str": "2026-02-19T15-30-00",
  "sentiment": "positive",
  "what": "told a funny joke about cats",
  "trigger": "head scratch",
  "num_messages": 42,
  "num_frames": 238,
  "audio_chunks": 150
}
```

### messages.json

```json
[
  {"role": "USER", "text": "Tell me a joke", "timestamp": 1740000500.0},
  {"role": "ASSISTANT", "text": "Why did the cat sit on the computer?", "timestamp": 1740000502.0}
]
```

## Skills

### `remember_positively`

Record what you did that earned a positive reaction.

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `what` | string | yes | What you did that was liked |
| `trigger` | string | no | What triggered the feedback (e.g. "head scratch") |

### `remember_negatively`

Record what you did that earned a negative reaction.

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `what` | string | yes | What you did that was disliked |
| `trigger` | string | no | What triggered the feedback (e.g. "verbal correction") |

## System Prompt Integration

The system prompt includes:

> "You learn from reactions. When someone scratches your head, remember positively what you just did. When corrected or told to stop, remember negatively."

The pat_level2 injection also hints the model should reflect:

> "This probably means they liked what you just did."

## Thread Safety

-   A single `threading.Lock` protects all buffer reads and snapshots.
-   Buffer feeds (`update_*`) acquire the lock briefly to append.
-   `record()` acquires the lock once to copy all three buffers, then releases it.
-   File I/O runs in a background daemon thread after the snapshot — no lock contention with the main loop.

## API

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/feedback/stats` | GET | Returns current buffer sizes (messages, frames, audio samples/seconds) |

## Usage

### In Code

```python
from reachy_nova.nova_feedback import NovaFeedback

feedback = NovaFeedback()

# Feed buffers (called from main loop)
feedback.update_conversation("USER", "Tell me a joke")
feedback.update_frame(camera_frame)
feedback.update_audio(audio_chunk)

# Record feedback (returns immediately)
feedback.record(sentiment="positive", what="told a joke", trigger="head scratch")

# Check buffer stats
stats = feedback.get_stats()
# {'messages': 42, 'frames': 238, 'audio_samples': 1920000, 'audio_seconds': 120.0}
```

### As a Skill

The feedback system is exposed as two tools to the LLM:

-   **`remember_positively(what="...", trigger="...")`**: Record positive feedback.
-   **`remember_negatively(what="...", trigger="...")`**: Record negative feedback.

## Related

-   [Patting Detection](patting.md) — triggers that lead to positive feedback
-   [Nova Sonic](nova_sonic.md) — voice conversation captured in message buffer
-   [Nova Vision](nova_vision.md) — camera frames captured in frame buffer
-   [Skills System](skills.md) — skill discovery and registration
