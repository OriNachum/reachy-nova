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

# Set AWS credentials for Nova Sonic & Nova Pro
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Set Nova Act API key
export NOVA_ACT_API_KEY=your_nova_act_key

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
