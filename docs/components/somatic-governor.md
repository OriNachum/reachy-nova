# Somatic Governor Documentation

The Somatic Governor is the emotional core of Reachy Nova. It manages a multi-dimensional internal state across five base emotions, translates stimuli into affective responses, maintains persistent trauma through wound mechanics, and drives observable behavior through derived moods and antenna animation.

The term "somatic" reflects that emotions here are not cognitive labels but embodied states: they decay naturally, accumulate through interaction, carry persistent wounds, and directly govern the robot's physical expression (antennas, body sway, sleep).

See also:
-   [Tracking](tracking.md): Sensor fusion that generates touch/snap/person events fed into the governor.
-   [Patting](patting.md): Two-level pat state machine that triggers healing events.
-   [Nova Sonic](nova_sonic.md): Voice system whose transcripts and state changes trigger emotional events.
-   [Nova Vision](nova_vision.md): Vision descriptions that trigger positive emotional events.

**Files:**
-   `reachy_nova/emotions.py` - Core `EmotionalState` class
-   `config/emotions.yaml` - Event definitions, mood rules, thresholds, transcript triggers
-   `reachy_nova/main.py` - Control loop integration, antenna profiles, sleep triggers

---

## Architecture Overview

```
Stimuli (voice, touch, vision, tracking)
          |
          v
  EmotionalState.apply_event(name, intensity)
          |
          +---> Apply deltas to emotion levels (clamped 0.0-1.0)
          +---> Create wound if severity == "severe"
          +---> Reduce wound floors if healing event
          |
          v
  EmotionalState.update(dt)   [called every frame, ~50Hz]
          |
          +---> Decay emotions toward baselines
          +---> Tick wounds: heal floors after fixed duration expires
          +---> Accumulate or decay boredom
          |
          v
  EmotionalState.get_derived_mood(voice_state)
          |
          +---> Evaluate 12 rules top-to-bottom (first match wins)
          +---> Apply hysteresis (1.0s minimum hold)
          |
          v
  Mood --> Antenna animation profile + body sway + auto-sleep
```

---

## The Five Emotions

Each emotion is a continuous value in the range `[0.0, 1.0]`. Emotions decay naturally toward their baseline at a configurable rate (units per second).

| Emotion    | Baseline | Decay Rate | Character                         |
|------------|----------|------------|-----------------------------------|
| **joy**    | 0.3      | 0.02 s^-1  | Mild contentment at rest; lingers |
| **sadness**| 0.0      | 0.03 s^-1  | Moderate decay                    |
| **anger**  | 0.0      | 0.05 s^-1  | Fades faster than sadness         |
| **fear**   | 0.0      | 0.04 s^-1  | Moderate dissipation              |
| **disgust**| 0.0      | 0.06 s^-1  | Fades most quickly                |

Joy has a non-zero baseline (0.3), which means the robot is mildly content by default. All negative emotions rest at zero. The asymmetric decay rates encode a design philosophy: positive affect lingers longest, disgust dissipates fastest, and fear/anger fall in between.

---

## Event System

Events are the primary mechanism for changing emotional state. Each named event is defined in `config/emotions.yaml` with:
-   **deltas**: Dictionary of emotion adjustments (positive = increase, negative = decrease)
-   **severity**: `healing`, `mild`, `moderate`, or `severe`
-   **wound_reduction**: How much this event chips away at active wound floors (healing events only)
-   **wound**: Configuration for wound creation (severe events only)

An optional `intensity` multiplier (default 1.0) scales all deltas.

### Event Catalog

#### Healing Events (Positive Stimuli)

| Event               | Joy    | Sadness | Fear   | Anger | Disgust | Wound Reduction |
|---------------------|--------|---------|--------|-------|---------|-----------------|
| `pat_level1`        | +0.05  |         |        |       |         | 0.00            |
| `pat_level2`        | +0.30  | -0.20   | -0.15  |       |         | 0.08            |
| `conversation_reply`| +0.10  | -0.05   |        |       |         | 0.01            |
| `face_recognized`   | +0.20  | -0.10   | -0.05  |       |         | 0.02            |
| `voice_speaking`    | +0.08  |         |        |       |         | 0.00            |
| `voice_listening`   | +0.05  |         |        |       |         | 0.00            |
| `vision_description`| +0.12  |         |        |       |         | 0.01            |

Healing events not only boost positive emotions but actively suppress negative ones. `pat_level2` (sustained scratching) is the most powerful healer: it delivers the strongest joy boost and chips at wound floors.

#### Mild Events (Startle)

| Event           | Fear   | Joy    |
|-----------------|--------|--------|
| `snap_detected` | +0.20  | -0.05  |
| `loud_noise`    | +0.25  | -0.10  |

#### Moderate Events (Distress)

| Event         | Sadness | Anger  | Disgust | Joy    |
|---------------|---------|--------|---------|--------|
| `person_lost` | +0.15   |        |         | -0.05  |
| `harsh_words` | +0.20   | +0.15  |         | -0.15  |
| `insult`      |         | +0.25  | +0.15   | -0.20  |

#### Severe Events (Trauma)

Severe events create **wounds** (see next section).

| Event              | Fear   | Sadness | Anger  | Disgust | Joy    |
|--------------------|--------|---------|--------|---------|--------|
| `violence`         | +0.40  | +0.30   | +0.25  |         | -0.30  |
| `abuse`            | +0.50  | +0.40   | +0.20  | +0.30   | -0.40  |
| `sustained_yelling`| +0.30  | +0.15   | +0.20  |         | -0.20  |

---

## Wound Mechanics

Wounds model emotional persistence beyond simple decay. When a severe event occurs, it creates an `ActiveWound` that enforces minimum floor values on specific emotions, preventing them from decaying below a threshold even after the initial event fades.

### Wound Lifecycle

```
Severe Event
    |
    v
ActiveWound created
    |
    +---> Fixed-floor phase (N seconds): floors are immovable
    |
    v
Healing phase: floors decay at heal_rate per second
    |
    +---> Healing events can chip at floors (wound_reduction)
    |
    v
All floors reach 0.0 --> wound is healed and removed
```

### Wound Parameters by Event

| Event              | Emotion Floors              | Fixed Duration | Heal Rate   | Time to Full Heal* |
|--------------------|-----------------------------|----------------|-------------|---------------------|
| `violence`         | fear: 0.25, sadness: 0.20   | 5 min          | 0.001 s^-1  | ~9 min              |
| `abuse`            | fear: 0.30, sadness: 0.25, disgust: 0.20 | 10 min | 0.0008 s^-1 | ~16 min             |
| `sustained_yelling`| fear: 0.20, anger: 0.15     | 3 min          | 0.002 s^-1  | ~4.5 min            |

*Time to full heal = fixed duration + (max floor / heal_rate). Healing events can reduce this.

### Wound Constraints

-   **Max concurrent wounds**: 5 (configurable via `settings.max_wounds`)
-   **Healing interaction**: Events with `wound_reduction > 0` chip away at all active wound floors. `pat_level2` is the most effective healer at 0.08 per application.
-   **Floor enforcement**: Each frame, any emotion below an active wound's floor is clamped upward. This means wounds override natural decay.

### Design Intent

Wounds prevent the robot from "forgetting" traumatic interactions too quickly. A robot that is yelled at cannot immediately return to cheerful — the wound holds fear/anger elevated for minutes, gradually fading. Positive interactions (patting, conversation, recognition) actively accelerate healing.

---

## Boredom

Boredom is a sixth dimension that sits outside the core five emotions. It models disengagement — the absence of stimulation.

### Accumulation Conditions (all must be true)

| Condition                     | Threshold |
|-------------------------------|-----------|
| Joy below threshold           | < 0.15    |
| All negatives below threshold | < 0.1 each|

### Dynamics

| Parameter        | Value       | Effect                                |
|------------------|-------------|---------------------------------------|
| Accumulate rate  | 0.008 s^-1  | When emotionally flat                 |
| Decay rate       | 0.05 s^-1   | When any emotion is active            |

Boredom ranges `[0.0, 1.0]`. It accumulates slowly during emotional flatness and decays rapidly when any stimulation occurs.

### Behavioral Effects

-   **Boredom >= 0.3**: Body sway begins. Amplitude scales from 0 to 15 degrees as boredom goes from 0.3 to 1.0. Frequency: 0.06 Hz (one full sway every ~17 seconds).
-   **Boredom >= 0.4**: Mood becomes `calm`.
-   **Boredom >= 0.7**: Mood becomes `sleepy`. Antennas droop.
-   **Boredom >= 0.8 for 60 continuous seconds**: Auto-sleep triggered. The robot enters sleep mode.

---

## Derived Mood System

The mood is a single categorical label derived from the continuous emotional state. It drives antenna animation, is exposed via the API, and is the primary behavioral output of the governor.

### Rule Evaluation

Rules are evaluated **top-to-bottom**; the first match wins. This creates a natural priority hierarchy:

| Priority | Mood            | Key Conditions                              |
|----------|-----------------|---------------------------------------------|
| 1        | **thinking**    | voice_state == "thinking"                   |
| 2        | **curious**     | voice_state == "listening"                  |
| 3        | **sad**         | wounds present AND sadness >= 0.3           |
| 4        | **disappointed**| wounds present AND anger >= 0.2             |
| 5        | **excited**     | joy >= 0.7                                  |
| 6        | **surprised**   | fear >= 0.4 AND anger < 0.2                 |
| 7        | **sad**         | sadness >= 0.25, sadness is dominant        |
| 8        | **disappointed**| anger >= 0.20, anger is dominant            |
| 9        | **sleepy**      | boredom >= 0.7                              |
| 10       | **calm**        | boredom >= 0.4                              |
| 11       | **proud**       | joy >= 0.5, sadness < 0.1, anger < 0.1      |
| 12       | **happy**       | joy >= 0.25                                 |
| 13       | **calm**        | (fallback, no conditions)                   |

### Design Choices

-   **Voice state takes priority**: When the robot is thinking or listening, that overrides emotional display. The user needs to see the robot is processing.
-   **Wounds elevate negative moods**: Sad/disappointed rules with wound requirements appear before their non-wound variants, and at lower thresholds. A wounded robot shows distress more readily.
-   **Dominant emotion matters**: Rules 7-8 use `_dominant` to ensure the strongest negative emotion drives the mood.
-   **Boredom fills the gap**: When emotions are flat, boredom provides gradual disengagement behavior rather than a static state.

### Hysteresis

Mood transitions require a minimum 1.0 second hold before the mood can change again. This prevents rapid flickering when emotions hover near thresholds.

### Mood Override

For backward compatibility and skill use, `set_mood_override(mood, duration=10.0)` forces a mood for a fixed duration, bypassing all rules. Used by voice commands like "search for..." (forces "curious") or "let me think" (forces "thinking").

---

## Antenna Animation

Each mood maps to a unique antenna movement profile that gives the robot physical expressiveness.

### Profiles

| Mood            | Freq (Hz) | Amp (deg) | Phase   | Offset (deg) | Easing     | Character                    |
|-----------------|-----------|-----------|---------|--------------|------------|------------------------------|
| **happy**       | 0.25      | 18.0      | oppose  | 0            | sine       | Gentle opposing sway         |
| **excited**     | 0.70      | 30.0      | oppose  | 0            | sine       | Fast, big wiggles            |
| **curious**     | 0.35      | 18.0      | sync    | +10          | sine       | Forward tilt together        |
| **thinking**    | 0.12      | 15.0      | custom  | 0            | sine-soft  | Asymmetric, one up one tilted|
| **sad**         | 0.08      | 8.0       | sync    | -25          | sine-soft  | Droopy backward lean         |
| **disappointed**| 0.06      | 5.0       | sync    | -20          | sine-soft  | Low sag, barely moving       |
| **surprised**   | 0.50      | 25.0      | oppose  | +15          | sine       | Perked up, quick opposing    |
| **sleepy**      | 0.04      | 4.0       | sync    | -18          | sine-soft  | Very slow drooping           |
| **proud**       | 0.15      | 6.0       | oppose  | +20          | sine-soft  | Held high, subtle movement   |
| **calm**        | 0.12      | 10.0      | oppose  | 0            | sine-soft  | Relaxed gentle movement      |

### Phase Patterns

-   **oppose**: `[offset + a, offset - a]` -- antennas sway in opposite directions
-   **sync**: `[offset + a, offset + a]` -- both move together
-   **custom**: Hardcoded asymmetric for "thinking" -- one antenna explores while the other stays low

### Easing Functions

-   **ease_sin**: Pure sine wave `sin(2*pi*f*t)`. Naturally smooth, decelerates at peaks.
-   **ease_sin_soft**: Sine-of-sine `sin(pi/2 * sin(2*pi*f*t))`. Extra dwell time at extremes, faster through center. Creates a more organic, deliberate motion.

### Transition Blending

When mood changes, antennas blend from the previous profile to the new one over **1.5 seconds** using smoothstep interpolation:

```
alpha = t / BLEND_TIME
alpha = alpha^2 * (3 - 2*alpha)   # smoothstep
antennas = prev * (1 - alpha) + target * alpha
```

This prevents jarring antenna jumps on mood transitions.

### Pat Vibration Overlay

When a level-1 pat is detected, a vibration overlay is added on top of the current mood animation:

| Parameter | Value     |
|-----------|-----------|
| Duration  | 2.0 s     |
| Frequency | 3.5 Hz    |
| Amplitude | 6.0 deg   |
| Envelope  | Squared decay `(1 - t/dur)^2` |

Both antennas vibrate in sync, rapidly fading. This gives immediate tactile feedback: "I felt that."

---

## Transcript Triggers

The governor monitors voice transcripts for emotionally significant language and fires corresponding events.

### Trigger Patterns

| Category          | Event              | Example Patterns                        | Special |
|-------------------|--------------------|----------------------------------------|---------|
| Harsh words       | `harsh_words`      | "shut up", "stupid", "idiot", "hate you"| --      |
| Insults           | `insult`           | "ugly", "useless", "garbage", "trash"   | --      |
| Violence threats  | `violence`         | "hit you", "destroy you", "kill you"    | --      |
| Abuse language    | `abuse`            | "abuse", "torture", "hurt you"          | --      |
| Yelling           | `sustained_yelling`| "STOP", "SHUT UP"                       | caps_only: true |

### Cooldown

A 5-second cooldown prevents the same event from retriggering on repeated transcript fragments. Each event type has its own independent cooldown timer.

### Detection Flow

```
on_transcript(role="USER", text)
    |
    v
emotional_state.check_transcript(text)
    |
    +---> For each trigger pattern:
    |       Check cooldown
    |       Match pattern (case-insensitive, or caps-only)
    |       Return matched event names
    |
    v
For each matched event:
    emotional_state.apply_event(event_name)
```

---

## Integration Points

### Main Control Loop (~50Hz)

Every frame, the main loop performs these emotion-related steps:

1. **Update state**: `emotional_state.update(dt)` -- decay, wound ticks, boredom
2. **Derive mood**: `emotional_state.get_derived_mood(voice_state)` -- evaluate rules
3. **Update app state**: Push levels, boredom, wounds, mood to shared state dict
4. **Auto-sleep check**: If boredom >= 0.8 for 60s, initiate sleep
5. **Body sway**: If boredom > 0.3, add slow yaw oscillation
6. **Antenna animation**: Select profile from mood, apply easing, blend transitions

### Subsystem Event Sources

| Subsystem     | Events Fired                                               |
|---------------|-----------------------------------------------------------|
| **Tracking**  | `pat_level1`, `pat_level2`, `snap_detected`, `person_lost` |
| **Voice**     | `voice_speaking`, `voice_listening`, `conversation_reply`  |
| **Vision**    | `vision_description`                                       |
| **Transcript**| `harsh_words`, `insult`, `violence`, `abuse`, `sustained_yelling` |
| **Face Recog**| `face_recognized`                                          |
| **Skills**    | Any event via `mood_executor()`, raw deltas via `apply_raw_delta()` |

### API Endpoints

| Endpoint                  | Method | Purpose                              |
|---------------------------|--------|--------------------------------------|
| `GET /api/emotions`       | GET    | Full emotional state snapshot        |
| `POST /api/emotions/event`| POST   | Apply a named event with intensity   |
| `POST /api/emotions/reset`| POST   | Hot-reload config from disk          |
| `POST /api/mood`          | POST   | Set mood override (backward compat)  |

---

## Thread Safety

All state access in `EmotionalState` is protected by a single internal `threading.Lock()`:

-   Emotion level reads and writes
-   Wound list mutations
-   Boredom accumulation
-   Mood computation and hysteresis
-   Override expiry checks

Callbacks are invoked **outside** the lock to prevent deadlocks. The lock granularity is per-method: each public method acquires and releases the lock independently.

---

## Tuning Reference

| Parameter                | Value      | Location         | Effect                                    |
|--------------------------|------------|------------------|-------------------------------------------|
| Emotion baselines        | 0.0-0.3    | emotions.yaml    | Resting emotion levels                    |
| Emotion decay rates      | 0.02-0.06  | emotions.yaml    | How fast emotions return to baseline      |
| Event deltas             | -0.40-0.50 | emotions.yaml    | Per-event emotion changes                 |
| Wound floor values       | 0.15-0.30  | emotions.yaml    | Minimum emotion during wound              |
| Wound fixed duration     | 180-600s   | emotions.yaml    | How long floors are immovable             |
| Wound heal rates         | 0.0008-0.002| emotions.yaml   | Floor decay after fixed duration          |
| Wound reduction (heal)   | 0.01-0.08  | emotions.yaml    | How much healing events chip wounds       |
| Max wounds               | 5          | emotions.yaml    | Concurrent wound limit                    |
| Boredom accumulate rate  | 0.008      | emotions.yaml    | Boredom growth when flat                  |
| Boredom decay rate       | 0.05       | emotions.yaml    | Boredom reduction when stimulated         |
| Boredom joy threshold    | 0.15       | emotions.yaml    | Joy must be below this for boredom        |
| Boredom neg threshold    | 0.10       | emotions.yaml    | All negatives must be below this          |
| Hysteresis               | 1.0s       | emotions.yaml    | Min hold before mood can change           |
| Transcript cooldown      | 5.0s       | emotions.yaml    | Between same-event re-triggers            |
| Mood blend time          | 1.5s       | main.py          | Antenna smoothstep transition             |
| Auto-sleep boredom       | >= 0.8     | main.py          | Threshold for auto-sleep                  |
| Auto-sleep duration      | 60s        | main.py          | Sustained boredom before sleep            |
| Body sway onset          | boredom 0.3| main.py          | When body sway begins                     |
| Body sway max amplitude  | 15 deg     | main.py          | At boredom = 1.0                          |
| Pat vibration duration   | 2.0s       | main.py          | Level-1 pat antenna vibration             |
| Pat vibration frequency  | 3.5 Hz     | main.py          | Vibration speed                           |

---

## Emotional Narrative

The system creates a coherent emotional arc for the robot:

1. **At rest**: Joy baseline of 0.3 keeps the robot mildly cheerful. Antennas sway gently in the "happy" profile. No boredom yet.

2. **Positive interaction**: Conversation, vision, face recognition, and patting boost joy and suppress negatives. The robot progresses through happy -> proud -> excited as joy climbs. Healing events chip away at any wounds.

3. **Negative interaction**: Harsh words, insults, or threats push negative emotions up and joy down. The robot becomes sad, disappointed, or surprised depending on which negative emotion dominates.

4. **Trauma**: Severe events (violence, abuse, yelling) create wounds that hold negative emotions elevated for minutes. The robot cannot simply "bounce back" -- it needs time and positive interaction to heal. Wounds are visible in the API and affect mood rule evaluation at lower thresholds.

5. **Neglect**: Without stimulation, emotions decay to baseline and boredom accumulates. The robot drifts from calm to sleepy, body swaying slowly, until it eventually falls asleep. Any interaction resets the boredom trajectory.

6. **Recovery**: Healing events (especially `pat_level2`) actively reduce wound floors. Over time, through positive interaction or natural heal-rate decay, wounds fully heal and the robot returns to its baseline cheerful state.

This cycle -- contentment, stimulation, potential trauma, recovery, disengagement -- gives the robot a believable emotional presence without requiring cognitive understanding of emotions.
