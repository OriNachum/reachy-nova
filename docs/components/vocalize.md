# Vocalize Documentation

This documentation covers the `vocalize` skill, which gives Nova a
non-speech voice — short expressive sounds it can play instead of, or
alongside, talking.

## Overview

Before this arc, Nova could react two ways: speak (Nova Sonic) or move
(the gesture skill). There was no third option — no wordless sound. Vocalize
adds one: a chirp, a trill, or a purr, synthesized on the fly and played
through the exact same speaker path Sonic's own voice output uses. Because
it rides that same path, everything that already governs what comes out of
the speaker — barge-in clearing, `speech_enabled`, and hardware AEC's
far-end reference — applies to a vocalization exactly the way it applies to
speech.

**Files:** `reachy_nova/vocalize.py` (synthesis), `reachy_nova/skills/vocalize/SKILL.md`
(tool metadata), `reachy_nova/skill_executors.py`'s `_vocalize_executor`
(the tool-callable executor).

## The Three Kinds

| Kind | Pitch envelope | Character | Default duration |
| :--- | :--- | :--- | :--- |
| `chirp_up` | Rising (320 Hz → 320+480·intensity Hz) | Bright, alert, attention-grabbing | 0.45s |
| `trill` | Oscillating warble around 440 Hz, 12 Hz rate | Playful, excited, chirpy | 0.65s |
| `purr_tone` | Falling exponential settle to 55 Hz + 24 Hz amplitude tremolo | Content, relaxed, affectionate (a cat's purr) | 1.0s |

## Synthesis Approach

Pure-numpy **additive harmonic synthesis** — a fundamental sine plus 2-4
harmonics (clamped `num_harmonics`), each decaying `1/h` in amplitude, driven
by the per-kind pitch envelope above and shaped by an amplitude envelope that
always raised-cosine fades to (near) zero at both ends. That fade-to-zero
boundary is deliberate: it's what lets a vocalization be queued into the
speaker buffer next to other chunks (Sonic's own audio, or another
vocalization) without an audible click at the seam. Harmonics above
`0.45 * sample_rate` are dropped sample-by-sample to avoid aliasing at low
sample rates. No new dependency — numpy only, matching the rest of the audio
pipeline (`reachy_nova/audio_pipeline.py`).

```python
synthesize(
    kind: str,                     # "chirp_up" | "trill" | "purr_tone"
    sample_rate: int = 24000,      # matches NovaSonic.OUTPUT_SAMPLE_RATE
    duration: float | None = None, # per-kind default if omitted, clamped [0.3, 1.5]s
    intensity: float = 1.0,        # clamped [0.0, 1.0] — pitch excursion + loudness
    num_harmonics: int = 3,        # clamped [2, 4]
) -> np.ndarray                    # float32 mono, samples in [-1.0, 1.0]
```

`DEFAULT_SAMPLE_RATE = 24000` is kept independent in `vocalize.py` (not
imported from `nova_sonic.py`) so the module stays a standalone,
dependency-light leaf that doesn't pull in the AWS SDK just to synthesize a
waveform — but it deliberately matches `NovaSonic.OUTPUT_SAMPLE_RATE` because
synthesized audio rides the same buffer/resample path as Sonic's own output.

## Executor Path: The Same Speaker Buffer As Sonic

`_vocalize_executor` (`reachy_nova/skill_executors.py`) synthesizes the
samples, then calls `ctx.audio_output(samples)`. In `main.py`, `ctx.audio_output`
is bound to `handle_audio_output` — the identical callback `NovaSonic`'s
`on_audio_output` uses to append to `audio_output_buffer`:

```python
# reachy_nova/main.py
ctx.audio_output = handle_audio_output
```

Both a vocalization and Sonic's own speech land in the same buffer, get
pulled by the same main-loop drain, resampled by the same
`audio_pipeline.resample_output`, and pushed to the speaker via the same
`reachy_mini.media.push_audio_sample`. Consequences of sharing that path:

- **Barge-in**: `handle_interruption` clears `audio_output_buffer` on
  interruption — a queued vocalization is cleared exactly like queued speech
  would be.
- **`speech_enabled`**: the main loop's drain step checks
  `if not speech_enabled: chunks = []` before pushing anything to the
  speaker — a vocalization is silenced by the same flag that silences
  speech.
- **Hardware AEC far-end treatment**: the XMOS XVF3800's echo cancellation
  treats whatever comes out of the speaker as the far-end reference signal
  it must cancel from the mic input (`enable_hardware_aec` in `main.py`).
  Because a vocalization plays through that same speaker path, AEC treats it
  the same way it treats Sonic's speech — the intent is that the robot must
  not "hear" its own chirp as user speech, the same guarantee it already
  needs for its own voice.

## `SKILL.md` Schema

`reachy_nova/skills/vocalize/SKILL.md` frontmatter:

```yaml
name: vocalize
description: >
  Make an expressive non-speech sound instead of talking — a rising chirp,
  a warbling trill, or a low purring rumble. Use this when a wordless sound
  fits better than speech.
```

Tool parameters (mirrored in `skill_executors.py`'s registered input
schema):

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `kind` | string | yes | One of `chirp_up`, `trill`, `purr_tone` |
| `intensity` | number | no | 0.0-1.0, default 1.0 — wider pitch sweep and louder at higher values |

An unknown `kind` returns a bracketed error string
(`"[Unknown vocalize kind '...'. Available: ...]"`) rather than raising, and
a missing `ctx.audio_output` (e.g. a stub `ctx` in tests) degrades to
`"[Vocalize audio path unavailable]"` — the executor never crashes the tool
loop.

## Honest Limits

- **Live no-self-hearing AEC check: PENDING.** The spec's honesty condition
  for this seam is explicit: vocalize is only honestly "done" once it
  audibly plays on the real robot speaker in a live run **and** the AEC
  reference path is confirmed — the robot must not transcribe its own chirp
  as user speech. That live check is part of the live-proof task (t11 in the
  build plan) and has not run as of this doc landing. The synthesis and
  executor-path code above is complete and unit-tested (fade-to-zero
  boundaries, envelope shapes, clamping), but "plays cleanly on hardware
  without confusing AEC" is a claim only a live run can honestly make.
