# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reachy Nova is an AI brain for the Reachy Mini robot, integrating three Amazon Nova services: **Nova Sonic** (real-time voice), **Nova 2 Lite** (camera vision), and **Nova Act** (browser automation). It's a `ReachyMiniApp` plugin discovered via the `reachy_mini_apps` entry point.

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python -m reachy_nova.main

# Run the Sonic voice demo tool
uv run sonic-demo

# Install browser automation support (if needed)
playwright install chromium
```

## Architecture

The app runs a **multi-threaded main loop** at ~50Hz in `ReachyNova.run()`. Each subsystem runs in its own daemon thread:

- **Main thread** (`main.py`): Control loop — feeds mic audio to Sonic, camera frames to Vision/Tracking, pushes speaker audio out, animates head/antennas based on mood and voice state.
- **Nova Sonic thread** (`nova_sonic.py`): Persistent bidirectional stream to Bedrock (`amazon.nova-2-sonic-v1:0`). Runs its own asyncio event loop. Audio in at 16kHz, out at 24kHz. Use `inject_text()` to feed non-audio context (vision descriptions, browser results) into the conversation.
- **Nova Vision thread** (`nova_vision.py`): Periodically analyzes camera frames via Bedrock Nova 2 Lite (`us.amazon.nova-2-lite-v1:0`). Fires `on_description` callback which injects text into the voice conversation.
- **Nova Browser thread** (`nova_browser.py`): Queue-based worker using `nova-act` library. Tasks triggered by voice keywords detected in `on_transcript`. Results injected back into voice conversation.
- **Tracking** (`tracking.py`): Fuses DoA (direction of arrival from XMOS mic), YOLO person detection (yolov8n), and snap/clap audio transients. Priority: snap > face > speaker > idle animation. YOLO runs in its own background thread to avoid blocking.

### Data Flow

Voice keywords (e.g., "search for", "what do you see") detected in `main.py:on_transcript` trigger vision analysis or browser tasks. Results flow back through callbacks → `sonic.inject_text()` → the robot speaks the result.

### Shared State

A thread-safe `app_state` dict in `main.py` tracks voice state, mood, vision/browser status, and tracking mode. Exposed via FastAPI endpoints at `http://localhost:8042/api/state` for the web dashboard (`index.html`/`style.css`).

## Environment

Requires `.env` file with AWS credentials (see `.env.sample`):
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION=us-east-1`
- Bedrock models must be enabled: `amazon.nova-2-sonic-v1:0` and `us.amazon.nova-2-lite-v1:0`

## Key Patterns

- All subsystems take a `stop_event: threading.Event` for graceful shutdown
- Callbacks are the primary communication pattern between components (not queues, except for browser tasks)
- Audio resampling is done with `np.interp` (simple linear interpolation) — no external resampling library
- YOLO model is lazy-loaded on first detection frame to avoid startup delay
- The `ReachyMiniApp` base class provides `self.settings_app` (FastAPI) for API endpoints and `reachy_mini.media` for hardware access
