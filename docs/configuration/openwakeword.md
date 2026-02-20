# Wake Word Configuration

Guide to configuring the wake phrase and ASR model used to wake Reachy Nova from sleep.

**Files:** `reachy_nova/wake_word.py`, `reachy_nova/sleep_orchestrator.py`

## Overview

During sleep, Reachy Nova continuously buffers microphone audio and periodically transcribes it with **NVIDIA Parakeet TDT** — a 0.6B-parameter speech recognition model. When the transcript contains the configured wake phrase, the robot wakes up.

This approach is more reliable than energy-based snap/clap detection because it only responds to a specific spoken phrase, not any loud transient.

---

## Quick Setup

```ini
# .env
WAKE_WORD_PHRASE=hey reachy
WAKE_WORD_MODEL=nvidia/parakeet-tdt-0.6b-v2
```

```bash
uv sync
uv run python -m reachy_nova.main
```

The model is downloaded from HuggingFace on first run (~1.2 GB) and cached locally by NeMo.

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `WAKE_WORD_PHRASE` | `hey reachy` | The spoken phrase that wakes the robot. Case-insensitive; punctuation is ignored. |
| `WAKE_WORD_MODEL` | `nvidia/parakeet-tdt-0.6b-v2` | NeMo model name (HuggingFace Hub) or path to a local `.nemo` checkpoint. |

---

## Choosing a Wake Phrase

Any phrase works. Practical guidelines:

- **2–3 syllables minimum.** Single-syllable words ("wake", "go") produce too many false positives from ambient speech.
- **Uncommon sequences.** "Hey Reachy" is unlikely to appear in background TV or conversation. "OK go" would fire constantly.
- **Avoid homophones.** "Nova" sounds like "no-va" which may match fragments of other words.
- **Consistent stress pattern.** Phrases with natural stress (HEY rea-CHY) are more reliably transcribed.

To change the phrase at runtime, update `.env` and restart the app:

```ini
WAKE_WORD_PHRASE=wake up reachy
```

No retraining or model changes are needed — the phrase is matched as a substring of the ASR transcript after normalization.

---

## How It Works

### Audio pipeline

1. The main loop reads ~20ms audio chunks from the microphone at 16 kHz (float32, mono).
2. During sleep, `SleepOrchestrator.tick_sleeping()` passes each chunk to `WakeWordDetector.detect()`.
3. `detect()` appends the chunk to a **4-second rolling buffer**.
4. Every **2 seconds**, the buffer is snapshotted and submitted to a background thread for transcription.
5. The background thread calls `model.transcribe([audio_array])` — NeMo accepts float32 numpy arrays directly, no temp file needed.
6. The transcript is normalized (lowercase, punctuation stripped) and checked for the wake phrase as a substring.
7. On match, `detect()` returns `True` → `initiate_wake()` is called.

### Background thread

Transcription runs in a `ThreadPoolExecutor(max_workers=1)` so it never blocks the main sleep animation (breathing, rocking). The main loop only checks `future.done()` — it never waits.

### Buffer reset

`WakeWordDetector.reset()` is called every time the robot enters sleep. It clears the audio buffer and cancels any pending transcription, so audio captured while the robot was awake cannot bleed into the first detection window.

### Detection guard

The 3-second guard in `tick_sleeping()` (`t_sleep > 3.0`) suspends wake word detection for the first 3 seconds after entering sleep, preventing the sleep animation's own sounds from triggering an immediate re-wake.

---

## Model Details

| Property | Value |
| :--- | :--- |
| Model | Parakeet TDT 0.6B v2 |
| Parameters | 600M |
| Input | Float32 or int16 numpy array, 16 kHz mono |
| Max audio length | 24 minutes per inference call |
| WER (average) | 6.05% |
| RTFx | 3380 (batch=128 on A100) |
| Hardware | NVIDIA Ampere / Hopper / Blackwell (DGX Spark: Blackwell) |
| HuggingFace | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) |

The DGX Spark's Blackwell GPU transcribes a 4-second audio buffer in well under 100ms, making the 2-second polling interval the dominant latency rather than inference time.

---

## Using a Local Checkpoint

If you have a fine-tuned `.nemo` file or want to avoid re-downloading:

```ini
# .env
WAKE_WORD_MODEL=/home/spark/models/parakeet-finetuned.nemo
```

Any `ASRModel`-compatible NeMo checkpoint works. The model is loaded with `ASRModel.from_pretrained()`, which accepts both Hub names and local paths.

---

## Tuning Response Latency

Two constructor parameters control the detection latency (not exposed as env vars, edit `main.py` if needed):

| Parameter | Default | Effect |
| :--- | :--- | :--- |
| `transcribe_interval` | `2.0 s` | How often transcription runs. Lower = faster response, more GPU load. |
| `buffer_seconds` | `4.0 s` | Rolling audio window sent to the model. Longer = more context, more memory. |

The practical minimum for `transcribe_interval` is ~0.5s (below that, inference queues faster than it completes on typical hardware). For the DGX Spark, 1.0s is comfortable if snappier waking is wanted.

---

## Troubleshooting

**Robot never wakes**
- Check logs for `[WakeWord] Ready` at startup. If absent, the NeMo import failed — run `uv sync`.
- Check `[WakeWord] Transcript:` debug lines (set log level to DEBUG) to see what the model hears.
- Speak the phrase louder and closer (~0.5 m from the mic).
- Try a simpler phrase like `hey reachy` and confirm it appears literally in the transcript.

**Too many false wakes**
- Use a longer or more distinctive phrase.
- Check what `[WakeWord] Matched in:` logs show — if random ambient speech keeps matching, the phrase is too common.

**Slow startup**
- First run downloads ~1.2 GB. Subsequent runs use NeMo's local cache (usually `~/.cache/huggingface/`).
- Model loading (weight allocation) adds ~5–10 seconds to startup on first use after cache warmup.

**`ModuleNotFoundError: nemo`**
```bash
uv sync
# or manually:
pip install nemo_toolkit[asr]
```

**CUDA out of memory**
- Parakeet TDT requires ~2 GB VRAM. On DGX Spark this is not a concern.
- If running on a smaller GPU alongside other models, reduce `buffer_seconds` to limit inference input size.

---

## Quick Reference

| Item | Location |
| :--- | :--- |
| `WakeWordDetector` class | `reachy_nova/wake_word.py` |
| `detect()` call site | `reachy_nova/sleep_orchestrator.py:tick_sleeping()` |
| `reset()` call site | `reachy_nova/sleep_orchestrator.py:initiate_sleep()` |
| `WAKE_WORD_PHRASE` env var | default: `hey reachy` |
| `WAKE_WORD_MODEL` env var | default: `nvidia/parakeet-tdt-0.6b-v2` |
| Transcription interval | `2.0 s` (constructor param in `main.py`) |
| Rolling buffer | `4.0 s` (constructor param in `main.py`) |
| Detection guard | `t_sleep > 3.0` in `sleep_orchestrator.py` |
