# Session Persistence

Nova remembers. When she restarts — whether from a crash, a quick reboot, or a long power-off — she picks up where she left off. Conversation history, emotional state, mood, and wounds survive across sessions. Crash recoveries are silent, short breaks are briefly acknowledged, and long absences get a warm welcome back.

## Overview

The session persistence system saves a snapshot of Nova's internal state to disk every 30 seconds and on clean shutdown. On startup it loads the previous session, classifies the restart type, restores emotional state with time-adjusted decay, and generates context injection text so the voice model can continue naturally.

The system also upgrades the existing [Growth](growth.md) feedback pipeline with S3 upload support, so RLHF training packages reach the cloud automatically.

**Files:**
-   `reachy_nova/session_state.py` — session save/load, restart classification, context generation
-   `reachy_nova/emotions.py` — `get_serializable_state()` and `restore_state()` methods
-   `reachy_nova/nova_feedback.py` — `get_recent_messages()` and S3 upload modes

## How It Works

### Startup Sequence

```
App starts
  -> SessionState.load()                  # read ~/.reachy_nova/session/session.json
  -> SessionState.classify_restart()      # fresh_start | crash_recovery | short_break | long_absence
  -> EmotionalState.restore_state()       # restore levels + wounds with time decay
  -> SessionState.mark_started()          # write shutdown_clean=false (crash marker)
  -> _inject_startup_context()            # combine session context + memory context into one inject
```

### Main Loop

```
Every frame (~50Hz):
  -> session.update_heartbeat()           # writes timestamp every 10s (internally throttled)
  -> session.save(emotions, conversation) # writes full state every 30s (internally throttled)
```

### Shutdown

```
stop_event fires
  -> session.save_shutdown()              # writes shutdown_clean=true immediately
  -> normal cleanup
```

## Restart Classification

| Type | Condition | Behavior |
| :--- | :--- | :--- |
| `fresh_start` | No previous session file | Standard startup, no session context injected |
| `crash_recovery` | `shutdown_clean=false` AND elapsed < 30s | Silent — last 10 messages injected, model told to continue naturally |
| `short_break` | elapsed < 1 hour | Brief acknowledgment — last 15 messages + mood + wounds injected |
| `long_absence` | elapsed >= 1 hour | Warm welcome — topic summary + time passed, model told to greet warmly |

### Context Injection

Session context and memory context are combined into a single `inject_text()` call to avoid hitting Sonic's 3-second throttle. The session portion varies by restart type:

-   **Crash recovery**: Recent conversation transcript, current mood. Instruction: "Continue seamlessly as if nothing happened."
-   **Short break**: Recent conversation, mood, active wounds, last person seen. Instruction: "Greet naturally, briefly acknowledge you're back."
-   **Long absence**: Topic snippets from user messages, time elapsed, session count, last person. Instruction: "Welcome them back warmly."

## Storage

### Session File

**Path:** `~/.reachy_nova/session/session.json`

Follows the same `~/.reachy_nova/` convention as [Face Recognition](face_recognition.md) (`~/.reachy_nova/faces/`).

```json
{
  "version": 1,
  "heartbeat": 1708300000.0,
  "shutdown_clean": true,
  "session_start": 1708290000.0,
  "session_id": "a1b2c3d4",
  "uptime_seconds": 10000.0,
  "total_sessions": 42,
  "emotions": {
    "levels": {"joy": 0.35, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "disgust": 0.0},
    "boredom": 0.12,
    "current_mood": "happy",
    "wounds": [
      {
        "id": "e5f6g7h8",
        "event": "harsh_criticism",
        "floors": {"sadness": 0.4},
        "duration": 300,
        "heal_rate": 0.001,
        "created_at": 1708299500.0,
        "remaining_fixed_seconds": 120.0
      }
    ]
  },
  "conversation": [
    {"role": "USER", "text": "Tell me about space", "timestamp": 1708299800.0},
    {"role": "ASSISTANT", "text": "Oh, space is incredible!", "timestamp": 1708299802.0}
  ],
  "sleep_state": "awake",
  "last_person_seen": "Ori",
  "last_person_seen_time": 1708299700.0
}
```

### Atomic Writes

Session state is written atomically to prevent corruption on power loss:

1.  Write to `session.json.tmp`
2.  Backup current `session.json` to `session.json.bak`
3.  `os.replace()` the tmp file over the main file (atomic on POSIX)
4.  On load failure, fall back to `.bak`

## Emotional Continuity

### Serialization

`EmotionalState.get_serializable_state()` extends the existing `get_full_state()` with wound reconstruction data: `created_at`, `heal_rate`, `duration`, and `remaining_fixed_seconds`.

### Restoration

`EmotionalState.restore_state(data, elapsed_seconds)` performs:

1.  **Level decay**: Each emotion decays toward its baseline at its configured rate for the elapsed time.
2.  **Boredom reset**: If elapsed > 60s, boredom resets to 0 (person returning = not boring).
3.  **Wound reconstruction**:
    -   Wounds still in their fixed phase: `created_at` adjusted so remaining time accounts for elapsed.
    -   Wounds past fixed phase: `heal_rate * healing_time` applied to reduce floors.
    -   Fully healed wounds: discarded.

## S3 Feedback Upload

### Storage Modes

Configured via `FEEDBACK_STORAGE` environment variable:

| Mode | Behavior |
| :--- | :--- |
| `local` | Save to disk only (`~/reachy_nova_data/feedback/`). Fallback if no AWS credentials. |
| `local+s3` (default) | Save to disk, then upload folder to S3 in the same background thread. |
| `s3` | Upload directly from memory buffers to S3. No local disk writes. Ideal for low-storage devices. |

### S3 Key Structure

```
s3://<FEEDBACK_S3_BUCKET>/
  feedback/<session_id>/<timestamp>_<sentiment>_fb_<id>/
    feedback.json
    messages.json
    frames/
      000000.jpg
      ...
    audio.wav
```

### Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FEEDBACK_S3_BUCKET` | `reachy-nova-feedback` | S3 bucket name (auto-created if it doesn't exist) |
| `FEEDBACK_STORAGE` | `local+s3` | Storage mode: `local`, `local+s3`, or `s3` |

### Failure Handling

-   S3 client is lazy-initialized on first use (same pattern as `nova_memory.py`).
-   Bucket is auto-created with `CreateBucketConfiguration` for the configured region.
-   If S3 upload fails in `local+s3` mode: local save already succeeded, warning logged.
-   If S3 upload fails in `s3` mode: falls back to local save so data is never lost silently.

### In-Memory WAV

In `s3` mode, audio is assembled entirely in memory — float32 chunks are concatenated, converted to int16 PCM, packed with a WAV header via `struct.pack`, and uploaded as bytes. No temporary files touch disk.

## API

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/session` | GET | Returns session metadata (session_id, restart_type, elapsed, uptime, total_sessions) |
| `/api/feedback/stats` | GET | Returns buffer sizes + storage mode + S3 bucket |

## Thread Safety

-   `SessionState` performs all I/O in the main thread but throttled to 10s (heartbeat) and 30s (save) intervals. Actual disk I/O takes ~1ms.
-   Conversation snapshots for session save use `NovaFeedback.get_recent_messages()` which copies under the existing feedback lock.
-   S3 uploads run in the existing `feedback-save-{id}` background threads. No new threading model.

## Usage

### In Code

```python
from reachy_nova.session_state import SessionState
from reachy_nova.emotions import EmotionalState
from reachy_nova.nova_feedback import NovaFeedback

# Startup
session = SessionState()
previous = session.load()
restart_type, elapsed = session.classify_restart(previous)

emotional_state = EmotionalState()
if previous.get("emotions"):
    emotional_state.restore_state(previous["emotions"], elapsed)

session.mark_started()

# Feedback with S3
feedback = NovaFeedback(session_id=session.session_id)

# Main loop
session.update_heartbeat()
session.save(
    emotions=emotional_state.get_serializable_state(),
    conversation=feedback.get_recent_messages(50),
)

# Shutdown
session.save_shutdown(
    emotions=emotional_state.get_serializable_state(),
    conversation=feedback.get_recent_messages(50),
)
```

## Related

-   [Growth](growth.md) — RLHF feedback capture (extended with S3 upload)
-   [Nova Sonic](nova_sonic.md) — voice model that receives session context injection
-   [Nova Memory](nova_memory.md) — long-term memory (combined with session context on startup)
-   [Face Recognition](face_recognition.md) — provides `last_person_seen` for session state
