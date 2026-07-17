# Speech Events Documentation

This documentation covers the speech-capture lane, which detects utterance
onset from the continuous mic feed and emits a backtracked local clip +
transcript so Nova hears the first word of what was said, not a clipped
fragment.

## Overview

Every mic chunk `main.py` reads is fed to Nova Sonic — that never changes.
Additively, the same chunk is also fed into a `SpeechEventDetector`: an
always-on rolling ring buffer that watches for speech onset and, once it
fires, emits a clip that starts 1-2s **before** the onset was detected, not
at the moment it was detected. The gap between "speech actually started" and
"the detector noticed" is measured by scanning the buffered audio for where
its energy actually rises, then closed by backtracking — never assumed to be
some fixed offset from "now".

Onset is detected one of two mutually-exclusive ways:

- **ASR-driven** (preferred): a background thread periodically transcribes
  the rolling buffer using an **already-loaded** ASR handle — the SAME
  parakeet instance `wake_word.py` loaded for wake-word detection, never a
  second model load. Onset = the transcript becomes non-empty.
- **XMOS-flag-driven** (fallback): when no ASR handle is available, the
  detector polls a `speech_flag_provider` callable on every `feed()` call.
  Onset = the flag's rising edge — the XMOS hardware `speech_detected` flag
  surfaced by `tracking.py`'s `doa_speech_active`.

**File:** `reachy_nova/speech_events.py` (`SpeechEventDetector`); wired into
the main loop by `reachy_nova/main.py` (`forward_mic_audio`, `make_on_speech`).

## Before This Pipeline

Before this arc, `main.py`'s audio-feed section called `sonic.feed_audio(audio)`
directly — no speech events existed anywhere in the codebase (see
`docs/specs/2026-07-17-event-based-senses-seamless-reactions.md`'s
Before → After: "the mic feeds Sonic continuously with no speech events, no
pre-roll"). `forward_mic_audio` (added in commit `47a9bfd`,
`reachy_nova/main.py`) is the additive replacement: Sonic still gets
`sonic.feed_audio(audio)` first and unconditionally, and the speech lane is a
parallel observer wrapped in its own `try/except` so a lane failure can never
delay or drop Sonic's feed.

## `SpeechEventDetector` Class Structure

### Constructor

```python
SpeechEventDetector(
    asr_handle=None,
    speech_flag_provider: Callable[[], bool] | None = None,
    on_speech: Callable[[dict], None] | None = None,
    pre_roll_seconds: float = 2.0,
    sample_rate: int = 16000,
    buffer_seconds: float | None = None,       # defaults to pre_roll_seconds + 8.0
    transcribe_interval: float = 2.0,
    silence_threshold: float = 0.02,            # RMS
    clip_dir: str | Path | None = None,          # defaults to ~/.reachy_nova/speech_clips
)
```

### Ring Buffer

`feed(audio)` pushes every chunk into a rolling buffer sized
`buffer_seconds` (default: `pre_roll_seconds + 8.0`), trimmed from the front
as new chunks arrive. The buffer must stay large enough to still hold the
true onset by the time it's measured — that's why it's sized relative to
`pre_roll_seconds`, not a fixed constant.

### Measured Backtrack

Once onset fires, `_measure_onset` scans the buffered snapshot in 10ms
windows for the first one whose RMS clears `silence_threshold` — this is a
**measurement** over the actual audio, not an assumed fixed duration. The
emitted clip starts at `measured_onset - pre_roll_seconds`, clamped to what
the ring buffer still holds. If nothing clears the threshold, the scan falls
back to the start of the snapshot, so backtracking still applies
conservatively rather than emitting a zero-length clip.

## Speech Event Payload

`SpeechEventDetector.on_speech` fires with this payload (also returned from
`feed()` on the call that triggers it):

| Field | Type | Description |
| :--- | :--- | :--- |
| `clip_path` | str | Path to the local WAV file (see Clip Storage below) |
| `transcript` | str | ASR transcript at onset, or `""` when triggered by the XMOS flag fallback |
| `duration_seconds` | float | Length of the emitted clip in seconds |
| `onset_ts` | float | Unix timestamp of the **measured** onset — not the time the event fired |

`main.py`'s `on_speech` handler (built by `make_on_speech`) republishes this
as the `speech/speech_detected` MQTT event with one field added:

| Field | Type | Description |
| :--- | :--- | :--- |
| `direction` | str \| None | `"left"` / `"front"` / `"right"`, or `None` — see Direction Correlation below |

## Clip Storage

Clips are written as 16-bit PCM WAV (stdlib `wave`, no codec dependency)
under `clip_dir`, which defaults to `~/.reachy_nova/speech_clips`. Filenames
are `speech_<epoch_ms>_<counter>.wav`.

## The Four `[SENSE ...]` Stage Lines

Every speech event walks through four `sensory_stage` calls (`stage=capture`,
`vad`, `event`, `inject`) from `make_on_speech` in `reachy_nova/main.py`, each
one INFO-level and parseable — see `reachy_nova/sensory_log.py` for the fixed
line shape `[SENSE stage=<stage> source=<source> event=<event>] <detail>`.
The event id is `speech-<onset_ts_ms>`.

Example trace for one utterance heard from the left while awake:

```text
[SENSE stage=capture source=speech event=speech-1737100000000] clip=/home/user/.reachy_nova/speech_clips/speech_1737100000000_0001.wav duration=2.35s
[SENSE stage=vad source=speech event=speech-1737100000000] onset_ts=1737100000.000 transcript='hey what are you doing'
[SENSE stage=event source=speech event=speech-1737100000000] published speech/speech_detected direction=left
[SENSE stage=inject source=speech event=speech-1737100000000] attempted notice='You hear someone speaking from your left.'
```

If the robot is asleep, the `inject` stage line instead reads
`suppressed reason=sleeping` and no text is injected. If awake but Sonic
itself drops the inject (already speaking, or throttled), that drop is
logged separately — see the next section.

### A Second `inject`-Stage Line: Sonic's Own Drop Path

The `attempted` line above only records that `main.py` *tried* to inject.
`NovaSonic.inject_text` (`reachy_nova/nova_sonic.py`) can still silently skip
the send — while the model is speaking, or while throttled
(`_inject_min_interval = 3.0`s) — and now logs that drop itself, through the
same `sensory_stage` helper, at INFO:

```text
[SENSE stage=inject source=speech event=<uuid>] dropped reason=speaking text='You hear someone speaking from your left.'
[SENSE stage=inject source=speech event=<uuid>] dropped reason=throttled interval=1.2s text='You hear someone speaking from your left.'
```

Before this arc, both of these were `logger.debug(...)` calls
(`"inject_text skipped — model is speaking"` /
`"inject_text throttled (interval=...)"`) — invisible at the default INFO
level, so a dropped sensory notice was silent. They are not scoped to speech
events only — every `inject_text` caller's drops now log the same way.

## Direction Correlation

`main.py` keeps a mutable `last_direction = {"time": float, "label": str}`
dict, refreshed on every `tracking/audio_direction` event (see
`docs/components/tracking.md` and `reachy_nova/tracking.py`'s
`_maybe_fire_audio_direction`). A speech event only carries a `direction`
label when that bearing is still fresh — within
`DIRECTION_CORRELATION_WINDOW = 3.0` seconds (`reachy_nova/main.py`) of the
speech event firing. Beyond that window the bearing is considered stale and
`direction` is `None` rather than a guess.

## Config Knobs

| Knob | Where | Default | Notes |
| :--- | :--- | :--- | :--- |
| `pre_roll_seconds` | `SpeechEventDetector.__init__` | `2.0` | How far before the measured onset the clip starts |
| `buffer_seconds` | `SpeechEventDetector.__init__` | `pre_roll_seconds + 8.0` | Ring buffer size; must outlast onset-measurement latency |
| `transcribe_interval` | `SpeechEventDetector.__init__` | `2.0`s | Cadence of background ASR transcription runs (ASR path only) |
| `silence_threshold` | `SpeechEventDetector.__init__` | `0.02` RMS | Onset-measurement threshold |
| `clip_dir` | `SpeechEventDetector.__init__` | `~/.reachy_nova/speech_clips` | Local-only clip storage |
| `asr_handle` | wired in `main.py` | `getattr(wake_word, "_model", None)` | The shared parakeet instance — never a second load |
| `speech_flag_provider` | wired in `main.py` | `lambda: bool(getattr(tracker, "doa_speech_active", False))` | XMOS flag fallback when no ASR handle |
| `DIRECTION_CORRELATION_WINDOW` | `main.py` module constant | `3.0`s | See Direction Correlation above |

## Nervous System Rule

`config/nervous-system/rules.yaml` carries an explicit entry for
`speech/speech_detected` (`NORMAL` priority, `NOW` urgency,
`llm_evaluate: false`, template `"You just heard: {transcript}"`) — every
sensory source that publishes an event gets an explicit rule; none rides the
default.

## Honest Limits

- **On-device parakeet latency benchmark: PENDING.** The live-proof task
  (t11 in the build plan) measures the ASR transcribe latency over the
  rolling window plus main-loop timing under load, on the real robot. That
  measurement has not run yet as of this doc landing — there is no recorded
  number here. Per the spec's honesty condition, a PASS keeps the parakeet
  ASR path as the onset trigger; a FAIL flips the detector to the
  `speech_flag_provider` (XMOS) fallback described above, which is already
  implemented and wired, not a hypothetical fallback. This section should be
  updated with the recorded number once t11 runs, in the same PR if the two
  land together, and honestly marked "unproven" until then otherwise.
- **Clips are local-only, never uploaded.** `speech_events.py` has no
  network code — clips are written via the stdlib `wave` module directly to
  `clip_dir` and nothing in this module ever reads them back off disk to
  send anywhere. This is a deliberately different policy from
  `reachy_nova/nova_feedback.py`, which **does** upload multimodal feedback
  packages to S3 by default (`FEEDBACK_STORAGE=local+s3`, see
  `NovaFeedback`'s `_upload_folder_to_s3`) — that policy is unchanged by this
  arc; speech clips simply never enter it.
