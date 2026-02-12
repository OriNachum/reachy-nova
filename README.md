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
- **Nova Pro** - Camera-based vision and scene understanding
- **Nova Act** - Voice-controlled browser automation

## Setup

```bash
# Install with uv
uv sync

# Copy .env.sample to .env and fill in your AWS credentials
cp .env.sample .env
# Edit .env with your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
# (same credentials work for Nova Sonic, Nova Pro, and Nova Act)

# Run
uv run python -m reachy_nova.main
```

## Features

- Talk to the robot naturally - it responds with voice via Nova Sonic
- The robot sees through its camera and describes what it sees
- Say "search for..." or "look up..." to trigger browser automation
- Web dashboard at `http://localhost:8042` with live status
- Mood-reactive antenna and head animations
- Quick command buttons for weather, news, jokes
