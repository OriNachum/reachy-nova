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
    model_id: str = "amazon.nova-2-sonic-v1:0",
    voice_id: str = "matthew",
    system_prompt: str = "...",
    on_transcript: Callable[[str, str], None] | None = None,
    on_audio_output: Callable[[np.ndarray], None] | None = None,
    on_state_change: Callable[[str], None] | None = None,
    tools: list[dict] | None = None,
    on_tool_use: Callable[[str, str, dict], None] | None = None,
)
```

- **region**: AWS region for Bedrock.
- **model_id**: Bedrock model ID for Nova Sonic.
- **voice_id**: The voice personality (e.g., "matthew").
- **system_prompt**: Instructions defining the robot's persona.
- **on_transcript**: Callback when the model converts speech to text.
- **on_audio_output**: Callback when the model sends audio to be played.
- **on_state_change**: Callback when the system state (listening, thinking, speaking) changes.
- **tools**: List of tool specifications (JSON schema) for function calling.
- **on_tool_use**: Callback when the model requests to use a tool.

#### Key Methods

- `start(stop_event: threading.Event)`: Launches the background thread that maintains the connection loop.
- `feed_audio(samples: np.ndarray)`: Sends raw microphone audio (float32, 16kHz) to the model. Automatically handles mono conversion and int16 PCM encoding.
- `inject_text(text: str, force=False, sense_class=None, must_deliver=False)`: Allows the system to inject non-voice context (like "I see a cat") into the conversation stream as if it were a user message. See [Must-deliver injects](#must-deliver-injects).
- `send_tool_result(tool_use_id: str, result: str)`: Sends the output of a tool execution back to the model.

### Connection Protocol

The component uses `Boto3`'s `invoke_model_with_bidirectional_stream` to establish a persistent connection. The protocol involves sending JSON events:

1. **sessionStart**: Configures inference parameters (maxTokens, temperature).
2. **promptStart**: Defines output audio format (24kHz, 16-bit PCM) and **tool specifications**.
3. **contentStart/textInput**: Sends the system prompt.
4. **audioInput**: Streams user audio chunks.

The model responds with events containing:

- **textOutput**: Partial or complete transcripts.
- **audioOutput**: Base64-encoded PCM audio chunks.
- **toolUse**: Requests to execute a function (name, ID, arguments).

### Implementation Details

- **Tool Use Handling**:
  - Listens for `contentStart` (type `TOOL`) and accumulates `toolUse` events.
  - On `contentEnd` (type `TOOL`), validates JSON arguments and triggers `on_tool_use`.
  - Expects the main application to execute the tool and return data via `send_tool_result`.
- **Audio Format**:
  - Input: 16kHz, mono, 16-bit PCM.
  - Output: 24kHz, mono, 16-bit PCM.
- **Threading**: Running in a daemon thread with an `asyncio` loop to handle the asynchronous stream.
- **State Machine**:
  - `idle`: Not connected.
  - `listening`: Waiting for user input.
  - `thinking`: Processing user input.
  - `speaking`: Playing model response audio.

## Chunked Playback (Harness)

The harness's speaker leg, `reachy_nova/harness/speaking.py`'s
`SonicSpeaker`, replaces the "buffer the whole utterance, then play it"
shape above with chunked playback (task t8, spec c2): Sonic's 24 kHz output
is cut into pieces and posted to the daemon as soon as each piece is ready,
so first audio is audible about a second after Sonic's first sample instead
of after the whole reply finishes generating.

### Why the state transition alone was not enough

The `speaking` state machine above still exists — `on_state_change`'s
transition out of `"speaking"` is still observed — but it is no longer what
*starts* playback, only the final sweep of whatever tail is left. Measured
on the robot on 2026-09-05/06, that transition is produced by a 4 s speaking
watchdog rather than an end-of-turn event: five short replies (0.84-1.28 s
of audio) were each queued 4.3-4.6 s after their first audio chunk, and a
12.5 s reply 9.9 s after. Buffering per-utterance bought a flat 4 s of
silence on every single reply, regardless of length.

### The chunker

`on_audio_chunk` buffers every callback and flushes to the playback queue on
whichever of three triggers fires first:

1. **Size** — the buffer reaches `chunk_s` (~1 s), split at the lowest-RMS
   50 ms window inside the last 200 ms before the target so the cut lands in
   a pause rather than mid-word.
2. **Inactivity** — no new audio for `inactivity_s` (~300 ms) — this is what
   ends a SHORT reply, whose buffer never reaches the size target.
3. **State change** — the transition out of `"speaking"` sweeps whatever
   sub-chunk tail remains.

### Per-chunk files, deleted on a window

Each chunk uploads under its own `nova-<utt>-<seq>.wav` (a single reused
filename was only ever safe because the next post waited out the previous
playback window) and is removed via `DELETE /api/media/sounds/{filename}`
once its playback window elapses, capped at `max_outstanding_files` (default
8) undeleted at a time — the robot's root disk runs close to full. Deletes
run in the slack after a post, never before one, so cleanup never sits on
the latency path; a failed delete is one latched senselog line, never a
lost voice.

### No pre-roll, seamless on the daemon

Chunks post with no pre-roll: a playback probe on the robot (2026-09-06) put
two 1 s tones back to back, posting the second exactly when the first's
window ended, and the join was seamless to the ear — no gap, no click —
with `play_sound` returning in 29-73 ms. The one-speaker discipline
(`EchoGate`) still serialises chunks, but waits out the audio itself rather
than the gate's full ear-side margin between chunks of the same utterance,
since paying that margin mid-sentence would reinsert the delay chunking
removes.

### The 4 s watchdog today

The 4 s speaking watchdog survives only as the safety net for a stuck
generation (Sonic's ASSISTANT `contentEnd` not arriving, or arriving with an
ignored `stopReason` — logged at INFO on every `contentEnd` since this
round, see `nova_sonic.py`) — it is no longer on the latency path for an
ordinary reply.

### Kill switch

`NOVA_CHUNKED_PLAYBACK=0` (`chunked=False`) restores whole-utterance
playback exactly as it shipped before this round: one buffer per utterance,
the single reused `tts_synth.wav`, no per-chunk cleanup, and the gate wait
includes its full margin again.

## Must-deliver injects

Most injects are *body cues* — a pat, a face, a scene description — and the
anti-flood machinery around them (a 3 s throttle, the speaking guard, the
"no session, no inject" rule) is right for that traffic. It is wrong for an
**answer**: on 2026-09-06 the browser's result inject ("Your web browsing
finished. Tell the user what you found: ...") landed 0 ms after its own
progress inject and the journal logged it away three times as
`dropped reason=throttled interval=0.0s`, and a result that arrived during a
1-3 s session rotation was answered `dropped-inactive` and lost outright.

`inject_text(..., must_deliver=True)` marks a text as an answer:

- **Throttle-exempt.** The 3 s interval is skipped, then re-armed, so the
  plain injects that follow are spaced exactly as before.
- **Queued, not dropped, across a session gap.** With no live session the
  text goes into a bounded FIFO (`MUST_DELIVER_QUEUE_MAX = 8`, oldest
  evicted) and `inject_text` returns `"queued-inactive"`. `_start_session`
  drains it — in order, exactly once, through the same `_send_user_text`
  path — right after the session starts listening, appending each item's age
  (`... (this arrived 4s ago)`) so a late answer is delivered as a late
  answer. Each drained item gets one `drained-queued age=...` senselog line.
- **Still subject to the speaking guard.** Injecting into a generating
  Bedrock stream can hang it, so the text is parked in the deferred slot like
  any cue. Callers should pass a distinctive `sense_class` (the browse caller
  passes `browse`) so the latest-wins slot cannot let an unrelated cue
  overwrite the answer.

A plain (non-must-deliver) inject keeps every previous behaviour, including
`dropped-inactive` and `dropped-throttled`.

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
