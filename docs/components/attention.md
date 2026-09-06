# Attention Documentation

This documentation covers the attention window — the cold/warm clock that
decides whether Nova is being *addressed* — and the fuzzy name matcher
behind it (task t5).

## Overview

A robot that answers every sentence it hears is not listening, it is
interrupting. `AttentionState` is one small piece of state answering one
question: did someone talk *to* the robot, or merely near it?

- The robot is **cold** until a user transcript **names** it.
- A name **opens** a warm window (default 45 s).
- Everything inside the window **renews** it.
- A whole window of nothing and it goes **cold** again, on its own.

**Files:** `reachy_nova/harness/attention.py` (`AttentionState`,
`is_name_match`), `tests/test_harness_attention.py`.

Pure state: no threads, no network, no timer, and no I/O beyond one
sensory-log line per open and one per close. Expiry is computed lazily from
the clock on every read, so the close line comes from whichever read first
observes it.

## Two reads off one clock

| Property | Governs | True when |
| --- | --- | --- |
| `warm` | the **voice** | a NAME opened the window and it has not run out |
| `conversation_live` | the **gaze** | ANY transcript or utterance landed inside the window |

They deliberately disagree for a nameless transcript from cold: somebody in
the room is plainly talking, so the eyes follow them (`conversation_live`
True), but nobody addressed the robot, so the mouth stays shut (`warm`
False). Wiring both to a single boolean is what makes a robot either blind
or a chatterbox.

## The names

`DEFAULT_NAMES` is `("nova", "reachy", "richie", "reach", "noah")` — the two
real names plus the three mishearings the on-device ASR actually produces
for them. The mishearings are listed as first-class names rather than left
to the fuzzy matcher, because `"reach"` is a *truncation* of `"reachy"` and
the matcher's prefix guard exists precisely to reject those: if we want it,
we have to say so.

`is_name_match(text, names=DEFAULT_NAMES)` restates — deliberately does not
import — the idea in the runtime's `reachy/speech/name_match.py`. The
harness must not grow a dependency on the runtime's Python package, and the
two matchers answer different questions over different name sets. A word
matches when it equals a name, or when `difflib_ratio × length_ratio`
clears `0.50` behind five guards:

1. **Length** — a fuzzy match needs at least four letters.
2. **Prefix** — a strict prefix of a name is a truncation, not a mishearing.
3. **Superstring** — a name inside a longer word is a different word.
4. **Initial** — a fuzzy match shares the name's first letter.
5. **Phonetic** — a fuzzy match must share the name's Soundex consonant
   skeleton: it has to *sound* like the name, not merely look like it.

### The n-family

`"nova"` and `"noah"` sit in a crowded neighbourhood, and every one of these
would have opened the window without the guards above:

| Word | Rejected by | Note |
| --- | --- | --- |
| `now`, `no`, `nah`, `not` | length | `nah` scores 0.64 against `noah` |
| `know` | initial | `k` ≠ `n` |
| `novel` | phonetic | `n140` ≠ `n100` |
| `november` | phonetic | `n151` ≠ `n100` |
| `nowhere` | phonetic | `n600` ≠ `n100` |

`"nova scotia"` still matches, and that is fine — someone saying it in front
of the robot gets a warm window they can ignore, which costs nothing.

## Renewal

`note_transcript(text)` returns one of three strings:

| Return | Meaning |
| --- | --- |
| `opened` | the text named the robot and the window was cold |
| `renewed` | the window was already warm — anything said in a live conversation keeps it alive, named or not |
| `ignored` | cold and nameless (or quiet) |

Nobody says "nova" before every sentence of a conversation they are already
having, which is the entire reason renewal exists.

`note_utterance()` (Nova spoke) and `note_inject()` (a body cue or tool
result reached the model) renew a warm window but can **never open a cold
one** — otherwise the robot would warm itself up by talking to itself.
`note_inject()` renews `warm` only, not `conversation_live`: an inject is the
robot's nervous system talking to its own mind, and nobody in the room said
anything.

Each note also records `last_transcript_at`, `last_transcript_named`,
`last_utterance_at` and `last_inject_at` (monotonic floats, or `None`) for
the speaker's own verdict.

## Quiet wins

Given a duck-typed `quiet` object with an `active()` method
(`QuietState` — see [quiet-mode.md](quiet-mode.md)), an active quiet reads
cold on both properties and blocks a name from opening. A person who asked
for silence does not get an exception for saying the robot's name. The
window is *closed*, not paused: quiet ending does not resurrect the window
it suppressed (the close line says `reason=quiet`).

## Session rotation is not our clock

`on_session_rotated()` exists and deliberately does nothing. The window is
the harness's clock, not Sonic's. A session rotation is plumbing — the
model's context was replayed into a fresh stream — and the person standing
in front of the robot mid-sentence neither knows nor cares. Resetting
attention there would make the robot go cold mid-conversation every few
minutes; the no-op exists to name that bug.

## Configuration

| Env | Default | Meaning |
| --- | --- | --- |
| `NOVA_ATTENTION_WINDOW_S` | `45.0` | warm-window length, in seconds |

Parsed defensively like `lock_state.default_drop_grace_s`: unset, empty,
unparseable, `NaN` or negative all resolve to the default — a typo must
never be the reason the robot went permanently deaf. `0` **is** honoured and
means "always cold unless just named": the window closes the same instant it
opens, so every utterance has to name the robot.

The clock is `time.monotonic` (injectable). Nothing is persisted: a restart
comes back cold, which is the safe direction.

## Logging

Exactly one line per open and one per close, none per renewal:

```text
[SENSE stage=attention source=nova event=window] opened by=nova
[SENSE stage=attention source=nova event=window] closed after=45.0s reason=expired
```

`reason` is `expired` or `quiet`. `opened by=` names the word actually
matched, which is the difference between a debuggable false positive and a
mystery.

## Wiring

This module is pure state and wires itself to nothing — `app.py`, the
speaker and the gaze stack consume it in a later task.
