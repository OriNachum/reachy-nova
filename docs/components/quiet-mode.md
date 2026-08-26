# Quiet Mode Documentation

This documentation covers timed quiet — a deadline for the robot's mouth,
not a mode — the `stay_silent`/`end_silence` tools and the state behind
them (task t11/t12).

## Overview

"Be quiet for ten minutes" is a *deadline*, not a flag someone has to
remember to clear. That difference is the whole design: a deadline expires
on its own, so the failure mode of a forgotten quiet is a robot that starts
talking again ten minutes later, never one that stays silent forever.

**Files:** `reachy_nova/harness/quiet.py` (`QuietState`, the deadline
itself), `reachy_nova/harness/speaking.py` (`SonicSpeaker`'s quiet gate),
`reachy_nova/harness/tools.py` (`stay_silent`/`end_silence` tool handlers),
`reachy_nova/harness/bus.py` (the quiet marker appended to every rendered
inject while armed).

## The deadline

`QuietState.arm(minutes)` sets (or extends) `until`, an epoch-seconds
deadline. **Later always wins**: a second `stay_silent` while quiet is
already armed can only push the deadline OUT (`note: "extended"`), never
pull it in — asking for five more minutes in the middle of a thirty-minute
quiet means "at least five more", not "cut it to five" (`note: "kept"`).
Only `end_silence` (`QuietState.release`) ends a quiet early. Requests are
bounded 1–180 minutes (`MIN_QUIET_MINUTES`/`MAX_QUIET_MINUTES` in
`tools.py`) and **refused**, not clamped, outside that range — a
mis-heard number turning into some other quiet length with nobody able to
see why is worse than a named refusal asking the person to repeat
themselves.

## Persistence

The deadline is written atomically (tmp + `os.replace`) to
`<state>/nova-quiet.json` on every arm and release, and reloaded on
`QuietState` construction. This is what makes a restart safe: a harness
that restarts at 02:05 inside a quiet armed until 02:20 comes back quiet,
rather than loudly reintroducing itself. A deadline already in the past on
load is ignored and the stale file removed — quiet can never outlive its
own wall clock. `peek_until_iso()` reads the same file side-effect-free, so
an out-of-process `status` call can report another process's quiet
(`quiet_until`) without disturbing it.

## Arming order

`stay_silent` arms two independent silences, in this order:

1. **The mind's** — `QuietState.arm()`, in-process, gates the Sonic speaker
   directly and marks every inject. This half always works; it never
   round-trips anywhere.
2. **The body's** — a merged `set_inhibition` intent that adds the runtime's
   `speak` behavior to whatever is already held back. This is a spool
   round-trip and can degrade (no engine, slow engine) like any other
   intent.

The order is deliberate: the mind-side quiet is armed **first,
unconditionally**, and the body's verdict is reported separately as
`body_muted` rather than folded into the overall `ok` — a degraded body
mute must never undo the half the person actually asked for and the half
that always works. `end_silence` unwinds in the same spirit: only the
`speak` inhibition the harness itself added is ever taken back
(`_quiet_added_speak`, latched) — an operator or rule holding `speak` down
independently survives a quiet's release untouched. A `tick()` poll
(`QUIET_TICK_S`, 1 s) also restores the body's voice on a plain *expiry*,
not just a hand-release, since `QuietState.active()` releases and logs its
own expiry but only the tool layer holds the runtime-side inhibition.

## What is and isn't silenced

Quiet gates the **speaker only** (`SonicSpeaker`'s top of `_play_one`, plus
the runtime's `speak` behavior via `set_inhibition`). The ear keeps
hearing, the mind keeps thinking, sensory events keep flowing — the robot
is quiet, not asleep and not deaf. A dropped-for-quiet utterance is a no-op
everywhere else in the speaker: no HTTP post, no gate arm, no queue purge,
no `on_playback_failure` — **quiet is not mouth loss.** Every rendered
inject also gets one visible marker appended while quiet is active
(`bus.py`'s `QUIET_MARKER`, unconditional) so the model is told, every
single time, that it is being heard but should not speak about it.

A runtime `mute` intent — the body's own equivalent, landing in
`reachy-mini-cli` in parallel with this arc — is expected to close the
loop from the other direction; this document only covers the harness's
own quiet.

## The acknowledgement

`arm()` leaves `pending_first_utterance` set, so the FIRST utterance after
arming still plays — "okay, quiet for ten minutes" — and the mouth closes
behind it. If no utterance arrives within `grace_s` (default 2 s) the mouth
closes anyway, so a Sonic turn that never produces audio can't leave the
gate open indefinitely.

## Escape by voice

`end_silence` is always accepted, even when nothing is armed — a quiet that
was never on is not an error (`{"ok": true, "note": "not silent"}`), so the
model never has to first check whether quiet is active before asking to end
it. When a quiet WAS active, the response also reports whether the body's
`speak` inhibition was successfully lifted (`body_restored`).

## Status

`supervisor.status()` surfaces `quiet_until` (an ISO timestamp, or `None`)
from either the running process's own `QuietState` or the persisted file —
so "why is it not talking?" is answerable from outside the harness process
entirely.
