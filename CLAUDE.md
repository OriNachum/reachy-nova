# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reachy Nova is an AI brain for the Reachy Mini robot, integrating three Amazon Nova services: **Nova Sonic** (real-time voice), **Nova 2 Lite** (camera vision), and **Nova Act** (browser automation). It's a `ReachyMiniApp` plugin discovered via the `reachy_mini_apps` entry point. Beyond speech and vision, Nova hears speech onsets and touch/direction as logged events, has a non-speech voice (`vocalize`), and can forge new reaction skills for itself at runtime (`skill_forge.py`, auto-activated after static validation — see `docs/components/skill-forge.md`).

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python -m reachy_nova.main

# Run the Sonic voice demo tool
uv run sonic-demo

# Run the test suite
uv run pytest

# Install browser automation support (if needed)
playwright install chromium
```

## Architecture

The app runs a **multi-threaded main loop** at ~50Hz in `ReachyNova.run()` (`main.py`). Per tick: emotional-state decay + derived mood, session-state heartbeat save, a sleep-mode short-circuit (breathing animation + wake-word listening only, see `sleep_orchestrator.py`), head-position computation (focus/tracking/idle priority cascade, `tracking.py`), antenna animation, mic audio in (Sonic feed first and unconditionally, then — additively, never gating it — the speech-capture lane, snap detection, and a throttled DoA poll), camera frame in (vision + tracking + face recognition), and speaker audio out (Sonic's own output and any queued `vocalize` sounds, both drained from the same buffer).

### Threads

Each subsystem below runs its own daemon thread (or thread pool) so the main loop never blocks on it:

- **Main thread** (`main.py`): the 50Hz control loop described above.
- **Nova Sonic thread** (`nova_sonic.py`): persistent bidirectional stream to Bedrock (`amazon.nova-2-sonic-v1:0`), its own asyncio event loop. Audio in at 16kHz, out at 24kHz. `inject_text()` feeds non-audio context (vision descriptions, sensory notices, memory/session context) into the conversation; drops (speaking guard, throttle) log at INFO via `sensory_log.stage`, never silently at DEBUG.
- **Nova Vision thread** (`nova_vision.py`): periodic Bedrock Nova 2 Lite (`us.amazon.nova-2-lite-v1:0`) frame analysis; fires `on_description` which injects text into the voice conversation.
- **Nova Browser thread** (`nova_browser.py`): queue-based worker using `nova-act`, tasks triggered by voice keywords detected in `on_transcript`.
- **Face Recognition thread** (`face_recognition.py`): background YuNet+SFace detection/match dispatched from `update_frame()`, throttled to 500ms.
- **Tracking's YOLO thread** (`tracking.py`): background person detection dispatched from `update_vision()`.
- **Wake-word thread pool** (`wake_word.py`): single-worker background transcription of the sleep-mode audio ring buffer via the parakeet ASR model.
- **Speech-events thread pool** (`speech_events.py`): mirrors wake-word's pattern, sharing the SAME loaded parakeet instance (never a second load) to detect speech onset while awake; falls back to polling the XMOS `speech_detected` flag when no ASR handle is available — see `docs/components/speech-events.md`.
- **Skill-forge dispatch thread** (`skill_forge.py`): one daemon thread per `forge(goal=...)` tool call — POSTs to the configured coder endpoint and stages the result; never blocks the main loop even on an unreachable/slow rig — see `docs/components/skill-forge.md`.
- **Sleep orchestrator's SDK threads** (`sleep_orchestrator.py`): transient daemon threads for `goto_sleep()`/`wake_up()` SDK calls and deferred startup-context injection.

### Module Map

- **Voice**: `nova_sonic.py` (Bedrock stream), `speech_events.py` (onset detection + backtracked clip capture), `vocalize.py` (non-speech chirp/trill/purr synthesis — see `docs/components/vocalize.md`), `wake_word.py` (sleep-mode wake phrase), `audio_pipeline.py` (mic/speaker format conversion + resampling), `sensory_log.py` (the `[SENSE stage=... source=... event=...]` per-stage log line).
- **Vision**: `nova_vision.py` (scene description), `face_recognition.py` + `face_manager.py` (YuNet/SFace identity storage — see `docs/components/face_recognition.md`).
- **Movement**: `tracking.py` (DoA + YOLO + snap + pat fusion, priority snap > face > speaker > idle; also fires the rate-limited `audio_direction` event and `pat_level1`/`pat_level2` — see `docs/components/tracking.md`, `docs/components/patting.md`), `gestures.py` (8 named gesture animations), `antenna_animator.py` (mood/pat-driven antenna motion), `safety.py` (head-body collision avoidance), `movement_math.py` (shared easing curves).
- **Mind/state**: `emotions.py` (5-dimension emotional state + mood derivation), `state.py` (`State`/`AppState`, the thread-safe reactive state container), `session_state.py` (cross-restart persistence), `temporal.py` (vague human-like time sense), `sleep_mode.py` (pure sleep FSM) + `sleep_orchestrator.py` (coordinates Sonic/vision/gesture/session across the full sleep cycle).
- **Self-extension**: `skill_forge.py` (dispatch to the coder endpoint, stage results), `forge_validator.py` (AST-only static validator — the activation gate), `skills.py` (`SkillManager` for built-in skills; `discover_runtime`/`activate_forged`/`ForgedSkillContext` for the forged-skill runtime area) — see `docs/components/skill-forge.md`.
- **Integration**: `nova_context.py` (`NovaContext`, the DI container passed to skill executors/API routes), `skill_executors.py` (the tool-callable implementations behind Sonic's `toolConfiguration`), `api_routes.py` (the FastAPI dashboard endpoints under `/api/*`), `nova_mqtt.py` (the MQTT nervous-system bridge — publishes `nova/events/<source>/<type>`, subscribes to inject commands; every publishing source has an explicit priority/urgency rule in `config/nervous-system/rules.yaml`).
- **Harness gaze/voice**: `harness/lock_state.py` (`LockState`, the harness's own belief about a runtime gaze lock, updated from confirmed `lock_face`/`release_face` results and `motion/lock-released` events — see `docs/components/gaze.md`), `harness/quiet.py` (`QuietState`, the persisted, self-expiring timed-quiet deadline behind `stay_silent`/`end_silence` — see `docs/components/quiet-mode.md`).
- **External integrations**: `nova_memory.py` (qq knowledge system query), `nova_slack.py` (Socket Mode Slack bridge), `nova_browser.py` (browser automation), `nova_feedback.py` (RLHF feedback capture, local and/or S3 per `FEEDBACK_STORAGE`).

### Data Flow

Voice keywords (e.g., "search for", "what do you see") detected in `main.py`'s `on_transcript` trigger vision analysis or browser tasks. Sensory events (speech, touch, vision, face, forge) flow through `nova_mqtt.py` as `nova/events/<source>/<type>`, get a priority/urgency verdict from `config/nervous-system/rules.yaml`, and reach the conversation via `sonic.inject_text()`. Reactions compose freely — Sonic speech, a `vocalize` non-speech sound, and a gesture can all fire off the same event.

### Shared State

`state.py`'s `State` wraps a typed `AppState` dataclass under a lock, firing an `on_change` callback (wired to `mqtt.publish_state`) on every update. `api_routes.py` exposes it — plus vision/browser/emotion/tracking/speech/sleep controls — via FastAPI endpoints under `http://localhost:8042/api/*` for the web dashboard (`reachy_nova/static/index.html`/`style.css`).

## Environment

Requires `.env` file with AWS credentials (see `.env.sample`):

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION=us-east-1`
- Bedrock models must be enabled: `amazon.nova-2-sonic-v1:0` and `us.amazon.nova-2-lite-v1:0`
- Skill forge (optional): `FORGE_BASE_URL`, `FORGE_MODEL` (default `qwen3`), `FORGE_API_KEY` — see `docs/components/skill-forge.md`

## Key Patterns

- All subsystems take a `stop_event: threading.Event` for graceful shutdown
- Callbacks are the primary communication pattern between components (not queues, except for browser tasks and forge dispatch)
- Audio resampling is done with `np.interp` (simple linear interpolation) — no external resampling library
- YOLO and face-recognition models are lazy-loaded on first detection frame to avoid startup delay
- The `ReachyMiniApp` base class provides `self.settings_app` (FastAPI) for API endpoints and `reachy_mini.media` for hardware access
- Per-stage sensory logging (`sensory_log.stage`) gives every step of the sensory pipeline (capture/vad/event/inject/...) one grep-able `[SENSE stage=... source=... event=...]` INFO line — see `docs/components/speech-events.md`
- Tests exist and are the merge gate: `uv run pytest` (`tests/`)
