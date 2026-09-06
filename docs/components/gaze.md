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
| `think` | one-shot: look up and aside for a moment, as if thinking; optional `side` ('left'/'right'), alternates by default | yes — an ordinary `run_behavior` intent (`"thoughtful"`) |

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

## Gaze stack — the posture layer

`reachy_nova/harness/gaze_stack.py` (`GazeStack`) is the one component that
owns the head between reflexes. Three parts of the harness want it at once —
the browser leg while a browse is in flight, the conversation layer while
someone is talking, the runtime's own feel-alive base the rest of the time —
so the postures are **layered, lowest first**:

| Layer | When | What the harness declares |
| :--- | :--- | :--- |
| `wander` | nothing to do | nothing — the runtime's feel-alive base owns the head |
| `browsing` | the browser is busy | a standing `declare_goal` of `gaze-hold`, `{pitch: 10, yaw: ±15}` |
| `conversation` | `AttentionState.conversation_live` | `look_at_sound`, then an `auto`-owned `lock_face`; the browsing goal is left standing beneath it |

The desired top layer is a **pure function of two inputs**:
`attention.conversation_live` and "is the browser busy". The aside yaw's sign
alternates per browse, so two browses in a row don't look identical.
`gaze-hold` claims only the head channel, so antennas and body yaw keep
feel-alive's sway underneath it.

### Single writer

Producers only ever set a flag under a short-lived lock and wake an event —
`on_browser_state(state)` (`"busy"` raises the browsing layer, `"idle"` /
`"error"` / anything else lowers it), `on_transcript(role, text)` (USER lines
only), `on_sonic_state(state)` (the rising edge into `speaking`) and
`on_speaker_idle()` (recorded for t9). ONE worker thread (`nova-gaze`)
computes the top layer on each wake or `tick_s` timeout and, on a transition,
issues **only the transition ops**, serially, each waiting out its own
`IntentTools.execute` round trip. That is what makes the op list on the spool
causally ordered rather than a race between whichever producer fired last.

Transitions issue the minimum:

- `wander -> browsing` — declare the `gaze-hold` goal.
- `browsing -> wander` — `declare_goal` with no goal (only if one is standing).
- `* -> conversation` — `look_at_sound`, then `lock_face`; the browsing goal is
  **not** cleared, because the lock owns the head by recency and the goal
  simply resumes when the lock releases.
- `conversation -> browsing` — `release_face`, then re-declare the goal only if
  it is not still standing.
- `conversation -> wander` — `release_face` and clear the goal.

Every transition costs exactly one `[SENSE stage=gaze source=nova
event=layer]` line naming `old -> new reason=...`, and every op one
`event=op` line carrying `ok=true` / `ok=false` / `ok=unknown` — the third
shape (on disk, unconfirmed) is named rather than silently read as success.
A degraded declare still counts as standing; only an explicit refusal does
not.

### `clear_for_result()` — the one synchronous op

The app calls `clear_for_result()` immediately before injecting a browse
result, so the head is demonstrably out of the thinking pose *before* Nova
starts talking about what it found. A promise like that cannot be kept by
setting a flag and hoping the worker gets there first, so this one runs on
the **caller's** thread — but through the same op lock the worker uses, so
"synchronous" never means "concurrent". It returns whether it cleared
anything, and leaves the layer alone: while the browser is still busy the
stack stays in `browsing` with nothing standing, so the later `idle`
transition issues no second clear.

### The conversation layer

When a conversation goes live — the first USER transcript, or Nova starting to
speak, whichever the wired `AttentionState` sees first — the head **turns
toward the voice and then locks on the face**: one `look_at_sound`, then
`lock_face`, in that order, within one worker tick. That is the order a person
does it in, and it also gives the runtime's face detector the best possible
frame to answer `lock_face` from.

The lock is held straight through Nova's replies and the listening gaps in
between. It is given back only when the conversation itself **fades** (the
attention window closes) — one `release_face`, and `LockState.mark_released
("auto-fade")` on the engine's `ok: true`.

**No face-presence belief lives in the harness.** The engine's own refusal,
`{"ok": false, "error": "no face known"}`, *is* the presence check. A refused
lock is simply retried while the conversation stays live, on a backoff of
**3, 6, 12, 24, 30, 30, … seconds** (`LOCK_RETRY_BACKOFF_S`, whose last value
repeats), so a conversation that starts with nobody in frame still locks on the
moment somebody leans in. A degraded `{"ok": null}` counts as **unknown**: the
belief is left exactly as it was, never read as locked, and the retry continues
on the same schedule.

Refusals cost one log line per *conversation*, not per attempt: the first logs
`[SENSE stage=gaze source=nova event=lock] no face known — retrying with
backoff`, the rest are counted, and the fade logs either
`locked after=Xs attempts=N` (it locked) or `lock never held: refusals=N` (it
never did).

**Ownership.** The automatic hold is taken as `owner="auto"`. A lock the MODEL
took (`LockState.owner == "model"`, via its own `lock_face` tool call) is never
re-taken and never released by this layer — the model asked for that lock
deliberately, and an automatic hold overruling it would be the harness
overruling the mind it serves. Entering a conversation under a model lock logs
`model lock standing — auto hold not taken` and takes no hold of its own.

**Losing the lock.** `on_lock_released(reason)` — wire it from the bus's
`motion/lock-released` tap — clears the belief, submits nothing, and drops the
retry deadline so the backoff re-arms and the hold is taken back while the
conversation is still live. The same happens when the ENGINE goes away and
`LockState.locked` falls to `None` under us. Believing a lock we no longer hold
is the one failure that cannot fix itself, so every ambiguous answer resolves
toward "not held": an *unconfirmed* `release_face` still clears the belief,
because the runtime drops a standing lock on its own max-hold timer regardless
of what the harness believes, and a stale "held" would suppress the next lock
attempt forever. The cost of being wrong the other way is one redundant
release.

### Start and stop hygiene

`start()` returns immediately and the worker submits one `release_face` and one
`declare_goal` None as its **first action**, under the op lock and before any
transition op: a harness that just restarted has no idea what the previous
process left standing on the runtime, and both ops are idempotent no-ops when
nothing is held (`release_face` answers `ok: true` "not locked"). The hygiene
runs on the worker rather than the caller because `supervisor._start_components`
starts components serially on one thread, where two synchronous spool round
trips against a stalled runtime would delay every later subsystem and the
heartbeat loop itself. `stop()` mirrors it on the caller's thread — a
`release_face` only if an `auto`-owned lock is held, a `declare_goal` None only
if a goal is standing, and a `set_inhibition` giving back exactly the names in
`_browsing_added` (the same live-set-respecting restore a transition to
`wander` does, so stopping mid-browse never leaves `orient-to-sound` and `nod`
disabled behind us) — before joining the worker. The worker's own exit path
(for a `stop_event` set from outside, where nobody ever calls `stop()`) runs the
same hygiene behind the same one-shot flag, so a stop mid-conversation costs
exactly one release either way and a second `stop()` submits nothing.

### Head reflexes under a held head (task t10)

A standing `gaze-hold` goal is not the whole story: the runtime's own
`orient-to-sound` and `nod` reflexes still compete for the head by recency,
just like the older `nova-face-noticed` rule did before a face lock existed
(it nodded on every face sighting — the wrong reflex once something can *hold*
the head by recency instead). So entering `browsing` also merges
`BROWSING_INHIBITS = ("orient-to-sound", "nod")` into the runtime's CURRENT
inhibited set via `set_inhibition` (which **replaces** the whole set — see
`tools.current_inhibitions()`), remembering exactly which names it added.
Leaving `browsing` for `wander` re-reads the live set and gives back only
those names, so a later-wins operator change (say, an inhibition the operator
added meanwhile) survives the restore. A conversation in between leaves the
names standing — the face lock re-asserts its own runtime inhibitions on
every replacement anyway — and returning from conversation to browsing
re-reads the live set again and re-adds only whatever went missing, since
`nod` is the harness's own addition and is not guaranteed to survive a lock's
replacement the way the runtime's own `orient-to-sound` is.

Retiring the old reflex is a one-line call:
`rules_overlay.retire_rule("nova-face-noticed")` writes a tombstone —
`{"id": "nova-face-noticed", "enabled": false}` — into the managed block,
through the same validated/atomic/re-parsed write path `upsert_rule` uses, and
submits a reload. A tombstone id needs no `nova-` prefix: the point is to be
able to disable a shipped or operator rule of that id too, not only one of
nova's own. A second `retire_rule` call for an already-tombstoned id is a
no-op — nothing written, no reload submitted.

### Status

`status()` reports `{"layer", "browser_busy", "conversation_live",
"goal_standing", "browsing_inhibits", "lock_held", "lock_attempts",
"next_lock_retry_s"}` — `browsing_inhibits` is the list of
`BROWSING_INHIBITS` names the stack currently believes it added itself;
`lock_attempts` and `next_lock_retry_s` are per-conversation and reset at every
fade; `next_lock_retry_s` is `None` when no retry is pending. No hook and no
worker tick ever raises — a broken `AttentionState` degrades to "not live", a
broken `LockState` to "not the model's", a failed op is logged and the loop
continues — and the worker exits within one tick of its stop event.

### Runtime facts this layer relies on (reachy-mini-cli, `face_lock.py`)

The gaze stack keeps no belief of its own about any of these — they live on
the runtime side and the harness only reacts to what the engine's result
already tells it:

- **Presence.** `lock_face` refuses `"no face known"` unless
  `face_bbox` is present AND fresher than `MAX_FACE_AGE_S = 1.5 s`. That
  refusal *is* the presence check the retry backoff above is built around.
- **Its own inhibitions.** A held lock inhibits the runtime's own
  `feel-alive` and `orient-to-sound` reflexes on its own account — the
  `BROWSING_INHIBITS` merge above is a *separate* concern (keeping those same
  reflexes off a `gaze-hold` goal, which the lock does not otherwise cover).
- **Max hold.** The runtime releases any lock on its own after
  `MAX_HOLD_S = 1800.0` (30 minutes), regardless of what the harness or the
  model believes — the ceiling `_stop_hygiene` and the `motion/lock-released`
  tap both exist to make redundant, never to replace.
- **Mind-offline release is inert on this device.** The engine also releases
  a lock after `mind_offline_grace_s` of `mind_online()` reading `False`
  (`reason: "mind-offline"`) — but on the deployed runtime that presence
  signal never fires (`"mind presence dropped reason=client-incompatible"` at
  every start), so this release path never triggers here. That is exactly why
  `GazeStack.start()`/`.stop()` run their own release/clear hygiene: the
  runtime cannot be relied on to notice a crashed harness and give the head
  back on its own, and without the harness-side hygiene a crash would leave
  the head locked until `MAX_HOLD_S`.

## Configuration

| Env | Default | Meaning |
| --- | --- | --- |
| `NOVA_FACE_HOLD` | on | kill switch (`harness/switches.py`) — the gaze stack's CONVERSATION layer, plus retiring `nova-face-noticed` (see below). Off restores the pre-round nod-on-every-face reflex and no automatic hold. |
| `NOVA_THINK_POSTURE` | on | kill switch — the gaze stack's BROWSING layer (the standing `gaze-hold` while a browse is in flight). |

Both off means no `GazeStack` is built at all (one `component absent
name=gaze reason=switch-off` line); either one on builds it, with only that
layer's producers wired — and with `NOVA_FACE_HOLD` off the stack is
constructed `conversation_enabled=False`, which makes `conversation_live()`
answer `False` whatever attention (or the local fallback clock) says, so the
CONVERSATION layer is unreachable rather than merely unwired. `app.py` only
calls `gaze.on_sonic_state` / `gaze.on_transcript` under
`switches.face_hold`, and only wires
`browser.on_state_change = gaze.on_browser_state` under
`switches.think_posture`.

## Wiring (`app.py`)

`app.py` builds `GazeStack(intents, attention=attention, lock_state=lock_state,
conversation_enabled=switches.face_hold)` whenever either switch is on, and:

- with `NOVA_FACE_HOLD` on: wires `sonic.on_state_change` to
  `gaze.on_sonic_state` and the `_on_transcript` callback to
  `gaze.on_transcript`; the runtime bus's `motion/lock-released` tap calls
  `gaze.on_lock_released(reason)`; and — since the automatic hold now owns
  the face-noticed cue — `retire_face_nod_rule()` tombstones
  `nova-face-noticed` at startup instead of `ensure_face_rule()` installing
  it (see [Retiring the old reflex](#head-reflexes-under-a-held-head-task-t10)
  above).
- with `NOVA_THINK_POSTURE` on: wires `browser.on_state_change =
  gaze.on_browser_state`.
- the browse result path calls `gaze.clear_for_result()` — see
  [nova_browser.md](nova_browser.md)'s "Browse result path, end to end".
- `gaze.start(stop_event)` / `gaze.stop()` run alongside the other
  supervisor units.
