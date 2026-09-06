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
| `conversation_live` | the **gaze** | ANY USER transcript landed inside the window (Nova's own utterances renew it, never open it) |

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

The runtime keeps its own, separate name list and deliberately never learns
`"nova"` (operator decision on reachy-mini-cli #175): reachy-mini-cli #177
instead lets the box overlay carry a top-level `names = [...]` table of
additions (letters only, three characters or more, at most eight) and adds a
one-tick `name_mentioned` sense field. `harness/rules_overlay.py` mirrors
both in its schema copies (`NAMES_TABLE`, `MAX_CONFIGURED_NAMES`,
`MIN_NAME_LENGTH`, `validate_names`, `"name_mentioned"` in `SENSE_FIELDS`)
so an operator's names table or a rule keyed on the field never makes a
later nova write fail (issue #27). Writing `names = ["nova"]` into the
operator head and reloading is the step that follows once #177 is on the
robot; until then the table must not be written on a box the current runtime
runs, because that runtime refuses the whole file.

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

**Her own voice never opens a conversation.** `note_utterance()` renews
`conversation_live` only while it is *already* live; from cold it records
`last_utterance_at` and changes nothing. It used to push the gaze clock out
unconditionally, and the live robot showed what that costs (2026-09-06, Ori:
"It feels rigid now. No liveness."): Nova's own reactions to body cues — and
her opening line at every session start — opened a conversation nobody was
having, which raised the [gaze stack](gaze.md)'s conversation layer, which took
a face lock, which inhibits `feel-alive` and `orient-to-sound` while held; she
then renewed the whole thing by speaking into it, so the journal read
`wander -> conversation` every minute or two with nobody in the room. A USER
transcript — named or not — still opens `conversation_live` immediately, which
is the only thing that ever should: a robot cannot start a conversation by
talking to itself.

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

## Voice gate

`SonicSpeaker` (`reachy_nova/harness/speaking.py`, task t6) is the first
consumer of the window, and it gates the **mouth only**. The mic feed to
Sonic stays open in every case below — a robot that stopped listening when
it was not addressed could never hear its own name.

### The verdict

`speaker.attention_verdict(now=None)` is a **pure** function of the
attention state, returning `"allowed"` or `"not-addressed"`:

| Verdict | When |
| --- | --- |
| `allowed` | no attention gate (`attention=None`) |
| `allowed` | the window is **warm** — this is a conversation |
| `allowed` | an **inject** landed within `attention_grace_s` (default 3 s): the utterance is a reaction to a body cue or a tool result, not a reply to speech |
| `allowed` | **nothing was ever transcribed** — a greeting on boot has no misheard sentence behind it |
| `allowed` | the last transcript **named** the robot (the window may have gone cold while Sonic was thinking; the person still asked) |
| `not-addressed` | cold **and** the last transcript is nameless **and** it is more recent than any inject |

Being pure and public matters: `app.py` calls the same function when the
ASSISTANT transcript arrives, so the ledger's decision and the speaker's
decision can never disagree.

### The drop

The verdict is taken **once per utterance**, on the edge into `"speaking"` —
never per chunk, because half a sentence spoken because the window closed
between chunk 2 and chunk 3 is worse than either answering or staying quiet.
A `not-addressed` utterance then has every chunk dropped in `_enqueue`,
which is the attention gate's mirror of the quiet gate's `_quiet_blocks`:

- no post, no `play_sound`, nothing uploaded;
- the echo gate is never armed (the ear is untouched);
- the queue is never purged and `on_playback_failure` never fires — a gate
  drop is not mouth loss, and the mind must not think the mouth is gone;
- `attention.note_utterance()` is **not** called: a suppressed reply must
  not renew the window, or the robot would talk its way back into a
  conversation from silence;
- one line, whatever the chunk count, plus one summary when the utterance
  ends:

```text
[SENSE stage=speak source=nova event=attention] dropped reason=not-addressed duration=1.00s (further chunks of this utterance counted, not logged)
[SENSE stage=speak source=nova event=attention-resume] utterance dropped count=4 reason=not-addressed (cold window, nameless transcript)
```

`speaker.attention_drops` counts the dropped chunks over the speaker's whole
life (cumulatively — unlike `quiet_drops`, there is no "release" edge for a
window that goes cold on its own).

### The late transcript

Sonic sometimes emits audio *before* the transcript that provoked it reaches
the harness. The speaking edge then sees no transcript, correctly allows the
utterance, and a moment later a nameless transcript lands. `app.py` calls
`speaker.recheck_attention()` immediately after
`attention.note_transcript(...)`; if an utterance allowed **only** for want
of a transcript is still in flight and the verdict has flipped, it is cut
with a single `preempt()` and marked suppressed, so the rest of it drops.

The cost is at most **one clipped chunk** (~1 s already posted) — the price
of the race, and much cheaper than the whole unwanted sentence. The recheck
is idempotent and a no-op for an utterance allowed for any other reason.

### Memory hygiene

A reply nobody heard is not part of the conversation, so it must not be
distilled into memory: `Ledger.append(..., dropped=True)` writes no line and
counts `ledger.attention_skips` instead. `app.py` passes
`dropped=(speaker.attention_verdict() == "not-addressed")` for ASSISTANT
lines. USER lines are real speech and are never dropped this way — the robot
should remember being talked near, just not remember answering.

## Cold refusal of effectful tools

The gate is not only at the speaker. `IntentTools.execute`
(`reachy_nova/harness/tools.py`) refuses `COLD_REFUSED_TOOLS` outright — a
pre-flight `ToolRefused`, nothing submitted to the spool — whenever the model
is cold and the transcript it is acting on never named the robot
(`_is_cold_and_nameless(attention)`, `None` attention always answers `False`):

```text
browse, forge, use_skill, author_rule, goto, run_behavior, declare_goal,
set_mode, set_inhibition, create_rule, enroll_face, lock_face, look_at_face,
look_at_sound, think
```

Every one of them moves the body, spends an AgentCore session, or edits
durable state; none is worth letting ambient talk trigger. The refusal reason
is `NOT_ADDRESSED_REASON = "not addressed — the robot was not spoken to by
name"`. Deliberately excluded: `recall_senses` (read-only), `stay_silent` /
`end_silence` and the voice-level trio (shaping how Nova sounds, not what she
does), and `release_face` (giving something back is always safe) — none of
those acts on someone's behalf who never addressed the robot.

## Configuration

| Env | Default | Meaning |
| --- | --- | --- |
| `NOVA_ATTENTION_WINDOW_S` | `45.0` | warm-window length, in seconds |
| `NOVA_ATTENTION_GATE` | on | kill switch (`harness/switches.py`) — off means every utterance plays and every tool call is allowed, exactly as before this window existed |

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

This module is pure state and wires itself to nothing. `app.py` builds
exactly one `AttentionState` (quiet-wired) when `switches.attention_gate` is
on and threads it through everything that needs it:

- `SonicSpeaker(attention=...)` — the [voice gate](#voice-gate) above.
- `IntentTools(attention=...)` — the [cold tool refusal](#cold-refusal-of-effectful-tools)
  above.
- `GazeStack(attention=...)` — reads `conversation_live` to decide the
  conversation layer (see [gaze.md](gaze.md)).
- `_on_transcript`: `attention.note_transcript(text)` then
  `speaker.recheck_attention()`, in that order, on every USER/ASSISTANT
  transcript.
- every inject callback (bus cues, browse progress/result, deferred drains):
  `attention.note_inject()`.

With `NOVA_ATTENTION_GATE` off, `attention` is `None` everywhere above — the
speaker allows every utterance, `IntentTools` refuses nothing, and `GazeStack`
falls back to its OWN local "conversation live" clock instead of reading this
module at all: `on_transcript`/`on_sonic_state` feed a private
`_live_until = now + FALLBACK_LIVE_S` (45 s, the same length as
`DEFAULT_WINDOW_S` on purpose, so a degraded wiring behaves like the real
thing rather than a different policy). So `NOVA_FACE_HOLD` with the gate off
still holds a conversation-shaped gaze — it just no longer knows whether
anyone *named* the robot, only whether anyone is talking.
