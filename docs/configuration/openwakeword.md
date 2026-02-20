# Wake Word Configuration

Guide to choosing, tuning, and training wake word models for Reachy Nova.

**Files:** `reachy_nova/wake_word.py`, `reachy_nova/sleep_orchestrator.py`

## Overview

When the robot is sleeping, it listens for a specific spoken phrase before waking up. This is handled by `WakeWordDetector`, which wraps the [openWakeWord](https://github.com/dscripka/openWakeWord) library.

**Why wake word instead of snap detection?**
Snap/clap detection is an audio energy heuristic — it fires on any loud transient: background music, dropped objects, nearby conversations. A trained wake word model is a neural classifier that only scores high for the specific phrase it was trained on, making accidental wakeups far less likely.

Snap detection is still active in awake mode for head-turning toward sounds. Only the sleep-exit path was changed.

---

## Quick Setup

Add to your `.env` file:

```ini
WAKE_WORD_MODEL=hey_jarvis
WAKE_WORD_THRESHOLD=0.5
```

Then reinstall dependencies and restart:

```bash
uv sync
uv run python -m reachy_nova.main
```

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `WAKE_WORD_MODEL` | `hey_jarvis` | Built-in model name **or** absolute path to a custom `.onnx` file |
| `WAKE_WORD_THRESHOLD` | `0.5` | Confidence threshold (0–1). Scores at or above this value trigger a wake. |

---

## Built-in Models

openWakeWord ships several pretrained models that work without any extra files. Set `WAKE_WORD_MODEL` to one of these names:

| Name | Phrase | Notes |
| :--- | :--- | :--- |
| `hey_jarvis` | "hey Jarvis" | Default. Well-tested, low false-positive rate. |
| `alexa` | "Alexa" | Short, snappy. Higher false-positive rate in noisy environments. |
| `hey_mycroft` | "hey Mycroft" | Open-source assistant phrase. |
| `hey_rhasspy` | "hey Rhasspy" | Open-source assistant phrase. |
| `ok_nabu` | "OK Nabu" | From the Nabu Casa / Home Assistant ecosystem. |
| `timer` | "set a timer" | Task-phrase model (less useful as a wake word). |

To switch models, update `.env` and restart the app — no code changes needed.

---

## Custom Models

To wake the robot with a phrase like "hey Reachy", train a custom model using the openWakeWord toolkit:

### 1. Install the training toolkit

```bash
pip install openwakeword[train]
```

### 2. Generate positive samples

```bash
# Install a TTS engine for synthetic data (e.g. pyttsx3 or edge-tts)
pip install edge-tts

# Generate ~1000 synthetic samples of your phrase
python -m openwakeword.train.generate_samples \
    --phrase "hey reachy" \
    --n_samples 1000 \
    --output_dir ./samples/hey_reachy/positive
```

### 3. Train the model

```bash
python -m openwakeword.train.train_model \
    --phrase "hey reachy" \
    --positive_samples ./samples/hey_reachy/positive \
    --output_path ./models/hey_reachy.onnx \
    --epochs 100
```

Training takes ~5–15 minutes on CPU. The output is a single `.onnx` file.

### 4. Deploy the custom model

```ini
# .env
WAKE_WORD_MODEL=/home/spark/models/hey_reachy.onnx
WAKE_WORD_THRESHOLD=0.5
```

Restart the app and say "hey Reachy" to wake the robot.

---

## Threshold Tuning

The threshold controls the tradeoff between sensitivity (fewer misses) and specificity (fewer false positives).

| Threshold | Behaviour |
| :--- | :--- |
| `0.3` | Very sensitive. Wakes on partial matches and background noise. |
| `0.5` | Default. Balanced for quiet-to-moderate environments. |
| `0.7` | Strict. Requires clear, close speech. May miss soft or accented voices. |
| `0.9` | Very strict. Effectively disabled except for ideal conditions. |

### Tuning procedure

1. Set `WAKE_WORD_THRESHOLD=0.3` to confirm the model is loading and detecting at all.
2. Speak the phrase at your normal distance and volume. The robot should wake reliably.
3. Walk away, turn on background audio, or have a conversation nearby. Note any false wakes.
4. Increase the threshold in `0.05` increments until false wakes stop while true wakes still work.
5. Write the final value to `.env`.

---

## How It Works

### Audio pipeline

1. The main loop reads a ~20ms audio chunk from the microphone at 16 kHz (float32, mono).
2. During sleep, the `SleepOrchestrator` passes each chunk to `WakeWordDetector.detect()`.
3. `detect()` converts float32 → int16 PCM and calls `oww_model.predict(audio_int16)`.
4. openWakeWord runs the audio through a mel-spectrogram frontend and a small LSTM/GRU classifier.
5. It returns a `{"model_name": score}` dict. Any score ≥ threshold → robot wakes.

### Buffer reset

`WakeWordDetector.reset()` is called every time the robot enters sleep. This clears the model's internal audio buffer so audio captured while the robot was awake (e.g. the tail of the "good night" command) cannot bleed into the sleep-state detection window.

### Detection guard

Wake word detection is suspended for the first **3 seconds** after the robot enters the sleeping state. This prevents the robot from immediately waking itself up if the wake animation triggers a loud mechanical sound.

---

## Troubleshooting

**Robot never wakes on the phrase**
- Check logs for `[WakeWord] Loaded model:` — if absent, the import failed.
- Lower `WAKE_WORD_THRESHOLD` to `0.3` temporarily to confirm the model is running.
- Make sure the phrase is spoken clearly and at close range (~1m).
- For a custom `.onnx` path, verify the path is absolute and the file exists.

**Too many accidental wakeups**
- Raise `WAKE_WORD_THRESHOLD` in `0.05` increments.
- Switch to a longer phrase (e.g. `hey_jarvis` has lower false-positive rates than `alexa`).
- Train a custom model on your specific voice and environment.

**`ModuleNotFoundError: openwakeword`**
```bash
uv sync
# or
pip install openwakeword>=0.6.0
```

**Model loads but scores are always 0**
- openWakeWord requires 16 kHz audio. Confirm the mic sample rate with logs: `Mic recording started: samplerate=16000`.
- If the mic runs at a different rate, `preprocess_mic_audio()` resamples it before detection — verify that path is working.

---

## Quick Reference

| Item | Location |
| :--- | :--- |
| `WakeWordDetector` class | `reachy_nova/wake_word.py` |
| `detect()` call site | `reachy_nova/sleep_orchestrator.py:tick_sleeping()` |
| `reset()` call site | `reachy_nova/sleep_orchestrator.py:initiate_sleep()` |
| Env var: model | `WAKE_WORD_MODEL` (default: `hey_jarvis`) |
| Env var: threshold | `WAKE_WORD_THRESHOLD` (default: `0.5`) |
| 3-second guard | `sleep_orchestrator.py` — `t_sleep > 3.0` condition |
