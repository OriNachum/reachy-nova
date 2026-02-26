# Setup Guide — Raspberry Pi CM4 (reachy-mini.local)

Run Reachy Nova directly on the Reachy Mini's onboard Raspberry Pi CM4.
Cloud services (Nova Sonic, Nova 2 Lite) run on AWS Bedrock; all local
processing is lightweight (face detection, wake word, DoA tracking).

---

## 1. Prerequisites

- Raspberry Pi CM4 with 4GB+ RAM
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### System packages

```bash
sudo apt update && sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gst-plugins-bad-1.0 \
    libcairo2-dev \
    libgirepository1.0-dev \
    portaudio19-dev
```

---

## 2. Clone & install

```bash
git clone <repo-url> ~/reachy_nova
cd ~/reachy_nova
uv sync --extra pi
```

This installs only the lightweight dependencies (no pymongo, neo4j, pyaudio,
ultralytics, nemo_toolkit, nova-act, or slack-bolt).

---

## 3. Configure environment

```bash
cp .env.sample .env
```

Edit `.env` with your AWS credentials:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
REACHY_NOVA_CONFIG=config/deployment-pi.yaml
```

---

## 4. Download models

### OpenWakeWord model

The `openwakeword` package downloads models automatically on first use.
The Pi config uses the `hey_jarvis` model by default. To pre-download:

```bash
uv run python -c "import openwakeword; openwakeword.utils.download_models()"
```

### Face recognition models (YuNet + SFace)

These are small (~37MB total) and downloaded automatically by OpenCV on first
use. No manual download needed.

---

## 5. Run

```bash
REACHY_NOVA_CONFIG=config/deployment-pi.yaml uv run python -m reachy_nova.main
```

Or set `REACHY_NOVA_CONFIG` in your `.env` file (see step 3) and just run:

```bash
uv run python -m reachy_nova.main
```

The app starts in sleep mode. Wake it with "hey Reachy" (OpenWakeWord) or a
snap/clap.

---

## 6. What works on Pi

| Feature | Backend | Notes |
|---------|---------|-------|
| Voice conversation | Nova Sonic (cloud) | Bidirectional streaming via Bedrock |
| Vision analysis | Nova 2 Lite (cloud) | Periodic frame analysis via Bedrock |
| Face recognition | YuNet + SFace (local) | Lightweight OpenCV DNN, ~37MB models |
| DoA tracking | XMOS mic array (local) | Direction of arrival for head tracking |
| Snap/pat detection | Audio transient (local) | Triggers vision, head pats detected |
| Emotions & gestures | Local state machine | Full emotional state and gesture engine |
| Sleep/wake | OpenWakeWord (local) | ~50MB model, runs on CPU |
| MQTT integration | paho-mqtt (local) | Nervous system events |

## 7. What's disabled on Pi

| Feature | Why | Re-enable |
|---------|-----|-----------|
| YOLO person tracking | Needs GPU (~200MB model) | Set `yolo_tracking: true` in config |
| Memory (MongoDB/Neo4j) | Needs database services | Install `pymongo`/`neo4j`, set `memory: true` |
| Browser automation | nova-act + Chromium ~500MB+ RAM | Install `nova-act`, set `browser: true` |
| NeMo Parakeet wake word | Needs GPU (~600MB model) | Install `nemo_toolkit[asr]`, set `backend: parakeet` |
| Slack integration | Needs `slack-bolt` | Install `slack-bolt`, set env vars |

---

## Known issues on Pi

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: openwakeword` | Pi extra not installed | `uv sync --extra pi` |
| High CPU on wake word | OpenWakeWord inference | Lower threshold or increase check interval |
| `ImportError: nova_act` with browser enabled | nova-act not installed | Set `browser: false` in config |
| Slow face recognition | CPU-only OpenCV DNN | Expected; runs at ~5 FPS which is sufficient |
