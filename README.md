---
title: Reachy Nova
emoji: "\U0001F916"
colorFrom: indigo
colorTo: cyan
sdk: static
pinned: false
short_description: Voice, Vision & Browser AI for Reachy Mini
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Nova

AI-powered brain for Reachy Mini, integrating three Amazon Nova services:

- **Nova Sonic** - Real-time speech-to-speech conversation
- **Nova 2 Lite** - Camera-based vision and scene understanding
- **Nova Act** - Voice-controlled browser automation

This project enables Reachy Mini to see, hear, speak, and interact with the digital world.

<img width="1115" height="987" alt="image" src="https://github.com/user-attachments/assets/3b66f514-3384-47a8-8fd8-dc6f82faa38c" />


## Documentation

-   [**Architecture Overview**](docs/architecture.md): High-level system design.
-   [**Component Documentation**](docs/components/): Detailed docs for each module.
    -   [Nova Sonic (Voice)](docs/components/nova_sonic.md)
    -   [Nova Vision (Sight)](docs/components/nova_vision.md)
    -   [Nova Browser (Action)](docs/components/nova_browser.md)
    -   [Tracking Manager](docs/components/tracking.md)
    -   [Skills System](docs/components/skills.md)
-   [**Main Application**](docs/main.md): Core loop and state management.
-   [**Setup Guide**](docs/setup.md): detailed installation instructions.

## Quick Start

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Configure environment**:
    ```bash
    cp .env.sample .env
    # Edit .env with your AWS credentials
    ```

3.  **Run**:
    ```bash
    uv run python -m reachy_nova.main
    ```

## Features

- **Natural Conversation**: Talk to the robot naturally - it responds with voice via Nova Sonic.
- **Visual Understanding**: The robot sees through its camera and describes what it sees.
- **Browser Automation**: Say "search for..." or "look up..." to trigger Google searches and more.
- **Smart Tracking**:
    -   Tracks faces automatically.
    -   Turns to look at speakers.
    -   Reacts instantly to snaps/claps.
- **Expressive Animations**: Mood-reactive antenna and head movements.
- **Web Dashboard**: Monitor status at `http://localhost:8042`.

## Project Structure

-   `reachy_nova/`: Main package.
    -   `main.py`: Entry point and orchestration.
    -   `nova_sonic.py`: Voice interface.
    -   `nova_vision.py`: Vision interface.
    -   `nova_browser.py`: Browser automation.
    -   `tracking.py`: Head tracking logic.
-   `tools/`: Utility scripts (e.g., demos).
-   `docs/`: Detailed documentation.
