# Startup Sleep Mode

Reachy Nova starts in sleep mode rather than jumping straight to full awake operation. This eliminates the visible "jump" on boot (head sweeping, antennas animating, Sonic voice active) and gives the robot a calm, living presence from the moment it powers on.

**Files:** `reachy_nova/main.py`, `reachy_nova/sleep_mode.py`, `reachy_nova/sleep_orchestrator.py`, `reachy_nova/wake_word.py`

## How It Works

On startup the application skips starting Nova Sonic and instead calls `_startup_sleep()`, which does two things:

1. Disables vision and tracking (they are harmless but unnecessary during sleep).
2. Checks whether the robot is already physically in the sleep position.

### Already in sleep position (crash recovery, restart during sleep)

If the head and antenna joints are within tolerance of the SDK's sleep constants, the system enters the breathing loop immediately — no sound, no movement, no `goto_sleep()` animation. This is handled by `SleepManager.enter_sleeping_direct()`, which bypasses the normal `awake -> falling_asleep -> sleeping` state chain.

**Tolerances:**
- Head joints: norm distance < 0.3 rad (SDK uses 0.2 internally for its own check)
- Antenna joints: max absolute difference < 0.5 rad per joint

**Log output:** `[Sleep] Already in sleep position — entering sleeping directly`

### Not in sleep position (fresh boot, manual repositioning)

If the robot is not in the sleep position, the normal `goto_sleep()` flow runs: the SDK plays the sleep sound, moves the head and antennas to the sleep pose over 2 seconds, then transitions to the breathing loop.

**Log output:** `[Sleep] Not in sleep position — running goto_sleep transition`

## Deferred Sonic & Context Injection

Nova Sonic is **not** started at boot. Instead:

- `sonic.restart()` is called inside `_initiate_wake()` when the robot wakes up.
- Startup context (session history + memory recall) is injected on the **first wake only**, controlled by the `_startup_context_injected` flag. Subsequent sleep/wake cycles do not re-inject.

This means the LLM voice stream only activates once someone actually wakes the robot, saving resources and avoiding talking to an empty room.

## Waking Up

The robot wakes from startup sleep the same way it wakes from any sleep — the **wake word** triggers `initiate_wake()` inside `SleepOrchestrator`, which:

1. Runs the SDK `wake_up()` animation (sound + head movement).
2. Starts Nova Sonic.
3. Re-enables vision and tracking.
4. Injects startup context (first wake only).

Voice commands and the `/api/sleep` endpoint also work as usual.

### Wake word

During sleep, each audio chunk is passed to `WakeWordDetector.detect()`. The default phrase is "hey Jarvis" (the `hey_jarvis` built-in model). To use a different phrase or a custom-trained model, set the environment variables:

```ini
WAKE_WORD_MODEL=hey_jarvis        # built-in name or /path/to/custom_model.onnx
WAKE_WORD_THRESHOLD=0.5           # confidence threshold (0–1)
```

The detector resets its internal buffer every time the robot enters sleep, and a 3-second guard prevents the wake animation's own sounds from re-triggering a wake.

See [Wake Word Configuration](../configuration/openwakeword.md) for full details on built-in models, custom model training, and threshold tuning.

### Snap detection

Snap/clap detection is no longer used to exit sleep. It remains active in **awake mode only**, where it turns the robot's head toward sudden sounds. See [Tracking](components/tracking.md) for details.

## State Diagram

```
                    ┌─────────────────────────────────┐
                    │           STARTUP                │
                    └───────────┬─────────────────────┘
                                │
                        _startup_sleep()
                                │
                    ┌───────────┴───────────┐
                    │                       │
              in position?            not in position?
                    │                       │
         enter_sleeping_direct()     trigger_sleep()
              (no sound)            goto_sleep() in bg
                    │                  (plays sound)
                    │                       │
                    └───────────┬───────────┘
                                │
                            sleeping
                         (breathing loop)
                                │
                          wake word
                                │
                        initiate_wake()
                                │
                        SDK wake_up()
                      sonic.restart()
                    inject startup context
                                │
                             awake
                       (normal operation)
```

## What Stays the Same

- Main loop sleep short-circuit (breathing animation) is unchanged.
- `initiate_sleep()` for runtime sleep (boredom auto-sleep, voice commands) is unchanged.
- Session persistence saves and restores `sleep_state` across restarts.
- `sonic.stop()` / `sonic.restart()` remain idempotent.
- Snap/clap detection in awake mode (head-turning) is unchanged.
