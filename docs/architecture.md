# Architecture Overview

Reachy Nova is a Python application designed for the Reachy Mini robot. It integrates multiple Amazon Nova AI services to create an interactive, multimodal agent that can see, hear, speak, and browse the web.

## High-Level Architecture

The application is structured as a `ReachyMiniApp`, which is the standard way to build applications for Reachy Mini. The main class `ReachyNova` orchestrates the following core components:

1.  **Nova Sonic (Voice)**: Handles bidirectional speech-to-speech communication and tool use.
2.  **Nova Vision (Sight)**: Analysis camera frames (periodically or event-triggered).
3.  **Nova Act (Action)**: Performs browser automation tasks.
4.  **Skill Manager**: Discovers and executes tools (like "Look Skill").
5.  **Tracking Manager**: Fuses sensor signals and emits events (`person_detected`, `snap_detected`) to trigger system behaviors.

## Component Interaction

```mermaid
graph TD
    User((User)) <--> Robot[Reachy Mini]
    Main[ReachyNova]
    Sonic[Nova Sonic]
    Vision[Nova Vision]
    
    Robot -->|Frames| Vision
    Robot -->|Audio/DoA| Tracker[Tracking Manager]
    Tracker -->|Target Angles| Robot
    Tracker -->|Events| Main
    
    Main -->|Trigger| Vision
    Vision -->|Description| Sonic
    
    Sonic <-->|Voice Stream| AWS[Amazon Bedrock]
    Sonic -->|Tool Call| Main
    Main -->|Execute Skill| Main
    Main -->|Tool Result| Sonic
```

## State Management

The application maintains a shared state dictionary `app_state` protected by a `threading.Lock`. This state tracks:
-   **Voice State**: `idle`, `listening`, `thinking`, `speaking`.
-   **Mood**: `happy`, `curious`, `excited`, `thinking`.
-   **Vision/Browser Status**: Results, descriptions, and current activity.
-   **Tracking Mode**: `idle`, `speaker`, `face`, `snap`.

This state is exposed via a local API (`/api/state`) for the web dashboard.

## Concurrency

The application relies heavily on threading to ensure responsiveness:
-   **Main Thread**: Runs the control loop, managing robot movements and state synchronization.
-   **Nova Sonic Thread**: Manages the persistent WebSocket/HTTP2 stream to Amazon Bedrock.
-   **Nova Vision Thread**: Periodically sends frames for analysis without blocking the main loop.
-   **Nova Browser Thread**: Executes long-running browser automation tasks in the background.
-   **Tracking Listeners**: Run in background threads (via `ultralytics` or `reachy-mini` SDK) to process heavy tracking algorithms.
