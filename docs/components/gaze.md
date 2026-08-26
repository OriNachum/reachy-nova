# Gaze Documentation

This documentation covers the four gaze actions Nova has over the runtime's
lock/tracking behavior: two one-shot glances and a standing lock/release
pair (task t9/t13).

## Overview

Before this arc, the harness could only make the body glance *somewhere*
generically (`run_behavior` with a named gesture). Gaze adds two things:
one-shot behaviors that turn toward a specific stimulus, and a *standing*
lock the runtime holds on its own between ticks, so the head keeps tracking
a person as they move rather than glancing once and reverting to idle
look-around.

**Files:** `reachy_nova/harness/tools.py` (`LOCK_FACE`/`RELEASE_FACE` specs
and builders, the `RUN_BEHAVIOR` gaze-one-shot names), the runtime side
(`reachy-mini-cli`, behavior library + gaze-lock intent handling —
out of scope for this repo), `reachy_nova/harness/lock_state.py`
(`LockState`, the harness's own belief about whether the lock is held).

## The four actions

| Action | How it's requested | Spool-backed |
| :--- | :--- | :--- |
| `run_behavior("look-at-sound")` | one-shot: turn toward the last sound heard | yes — an ordinary `run_behavior` intent |
| `run_behavior("look-at-face")` | one-shot: glance at the person in front of you | yes — an ordinary `run_behavior` intent |
| `lock_face` | standing: keep the gaze on whoever the runtime already knows, until released | yes — its own intent kind |
| `release_face` | stop a standing lock, return to normal look-around | yes — its own intent kind |

The two glances are ordinary `run_behavior` calls naming a gesture the
runtime's library ships — the gaze one-shots default to about 2 seconds and
stop on their own, same as `nod` or `shake`. They are ALSO published under
their own underscored tool names, `look_at_face` and `look_at_sound`
(optional `duration`, default 2 s), which build exactly the same
`run_behavior` spool op. That alias is not a second capability: live on
2026-08-26 Sonic reached for a tool literally named `look_at_face`, got the
pre-flight "unknown tool" refusal, and abandoned the gesture rather than
retrying as `run_behavior(name="look-at-face")` (finding L4). A name the
model already reaches for is worth more than the name we would have
preferred. `lock_face`/`release_face` are
their own no-argument tool specs (`{"type": "object", "properties": {},
"required": []}`) — there is nothing to configure, only whether the lock is
on.

## Requesting and refusing

All four ride the intents spool (`<state>/behavior/intents/{commands,
results}/`, see `docs/architecture.md` §4) — the same submit/await round
trip as every other spool-backed tool, so the same three result shapes
apply: `{"ok": true, ...}` (the engine's own verdict, verbatim),
`{"ok": false, "error": ...}` (refused, here or by the engine), or
`{"ok": null, "submitted": <cmd_id>}` (degraded — on disk, unconfirmed
within the wait).

`lock_face`/`release_face` add nothing to pre-flight validation — there is
no argument to check — so every refusal is the ENGINE's own: "no face
known" when nothing is currently recognised, or (on an older runtime that
predates this feature) an `"unknown kind"` refusal naming the intent kind it
does not understand. Both surface to the model exactly as the engine phrased
them; the harness never rewrites a refusal reason.

## What comes back

A confirmed `lock_face`/`release_face` does two things: it returns the
engine's result to the model, and — **only on a confirmed `ok: true`** —
updates `LockState.locked` (`reachy_nova/harness/lock_state.py`), the
harness's own local belief about whether the gaze is held. A refusal or a
degraded (`ok: null`) result never touches the belief, because neither one
means the body's actual lock state observably changed.

Two runtime bus events (`config/nervous-system/rules.yaml`, both `sense:
face`) narrate what the *body* does with a standing lock once it's held:

- `motion/face-lost` (`voice: brief`) — the locked face has been out of view
  for a while, but the lock is still held; the runtime keeps trying. Worth a
  quiet situational cue, not a whole reaction.
- `motion/lock-released` (`voice: silent`) — the runtime actually dropped
  the lock on its own (`reason: requested|mind-offline|max-hold`). This is
  the *runtime* releasing a lock the harness believed it held — it also
  clears `LockState.locked` to `False`, the same as a confirmed
  `release_face` would.

`LockState` is also cleared to unknown (`None`) when the engine heartbeat
itself drops — a lock is a promise the engine keeps, and when the engine
goes away there is nothing left to have an opinion about. That clear waits
out a **5 s** grace (`NOVA_LOCK_DROP_GRACE_S`), and any `engine live` inside
the grace cancels it: live on 2026-08-26 the heartbeat on a loaded CM4
flapped live/lost about every 2 s while the runtime lock was held perfectly
well, and the original edge-triggered clear threw the belief away two
seconds after every lock (finding L6). The supervisor drives both edges plus
a `settle()` on every other poll tick, so an engine that stays down still
clears the belief exactly once even though no further edge ever arrives. See
`reachy_nova/harness/lock_state.py`'s module docstring for the full
reasoning; the belief is read-only local color for `supervisor.status()`'s
`locked` field (`True`/`False`/`None` — never guessed) and never gates a
tool call.

## Deploy order

**The runtime ships first.** `lock_face`/`release_face` and the two gaze
one-shots depend entirely on runtime-side behavior — the intent kinds, the
library gestures, and the standing lock loop itself all live in
`reachy-mini-cli`, not in this repo. A harness deployed ahead of that
runtime support degrades honestly (an `"unknown kind"` refusal, or a
`run_behavior` refusal naming valid behaviors that don't include the gaze
one-shots yet) rather than silently — but the intended order is runtime
first, harness second.
