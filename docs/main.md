# Main Application Documentation

This documentation covers the `ReachyNova` application logic, which orchestrates the entire system.

## Overview

The `ReachyNova` application serves as the central brain of the robot, coordinating:
-   Voice (Nova Sonic)
-   Vision (Nova Vision)
-   Browser (Nova Browser)
-   Head Tracking (Tracking Manager)
-   Robot Hardware (Reachy Mini via `reachy-mini-app`)
-   Web Dashboard (via FastAPI)

**File:** `reachy_nova/main.py`

## Class Structure

### `ReachyNova`

The application class inheriting from `ReachyMiniApp`.

#### `run(self, reachy_mini: ReachyMini, stop_event: threading.Event)`

The main execution loop.

1.  **Initialization**:
    -   Sets up shared state (`app_state`) and locks (`state_lock`, `audio_lock`).
    -   Initializes `NovaSonic`, `NovaVision`, `NovaBrowser`, and `TrackingManager`.
    -   Starts mic recording and background service threads.

2.  **API Endpoints**:
    -   Defines FastAPI routes for the web dashboard (see below).

3.  **Control Loop** (`while not stop_event.is_set()`):
    -   **Tracking**: Updates tracking logic and computes head targets.
    -   **Animation**: Controls antenna wiggles based on mood (`happy`, `thinking`, `excited`).
    -   **Audio Feed**: Pulls audio samples from the robot, processes them (stereo -> mono, resampling), and feeds them to `NovaSonic`.
    -   **Vision Feed**: Pulls camera frames and sends them to `NovaVision`.
    -   **Audio Output**: Plays buffered audio chunks from `NovaSonic` on the robot's speaker.

## State Management

The `app_state` dictionary tracks the system's status:

| Key | Description | Values |
| :--- | :--- | :--- |
| `voice_state` | Current activity of the voice model | `idle`, `listening`, `thinking`, `speaking` |
| `mood` | Emotional context for animations | `happy`, `curious`, `excited`, `thinking` |
| `vision_enabled` | Is camera analysis active? | `True`, `False` |
| `vision_analyzing`| Is a frame currently being processed?| `True`, `False` |
| `tracking_mode` | Reason for head movement | `idle`, `speaker`, `face`, `snap` |
| `antenna_mode` | Specific antenna behavior override | `auto`, `excited`, `calm`, `off` |

## API Endpoints

The application exposes a local web server (default port `8042`) for control and monitoring.

-   **GET /api/state**: Returns the current `app_state`.
-   **POST /api/vision/toggle**: Enable/disable vision analysis.
-   **POST /api/vision/analyze**: Trigger an immediate scene description.
-   **POST /api/browser/task**: Submit a browser automation task (instruction, optional URL).
-   **POST /api/antenna/mode**: Manually set antenna behavior.
-   **POST /api/mood**: Manually set the robot's mood.
-   **POST /api/tracking/toggle**: Enable/disable active tracking.

## Audio Pipeline

-   **Input**:
    -   Captures raw audio from `reachy_mini.media`.
    -   Converts to float32 `[-1.0, 1.0]`.
    -   Downmixes to mono if necessary.
    -   Resamples to 16kHz for `NovaSonic` input.
-   **Output**:
    -   Receives 24kHz audio from `NovaSonic`.
    -   Buffers chunks to avoid underruns.
    -   Resamples to the robot's native output rate (usually 44.1kHz or 48kHz).
    -   Pushes to `reachy_mini.media` for playback.

## Animation System

-   **Head**: Controlled by `TrackingManager` (see `tracking.md`).
-   **Antennas**:
    -   `excited`: Fast, large wiggles (opposing phase).
    -   `calm`: Slow, gentle wiggles.
    -   `thinking`: Asymmetric wiggles.
    -   `curious`: Parallel wiggles (same phase).
    -   `idle`: Standard breathing-like motion.

## Usage

Run the application using the `reachy-mini` launcher or directly:

```bash
uv run python -m reachy_nova.main
```
