# Memory

This documentation covers the conversation ledger, the Lite memory
compactor, and the session-start history replay that together let Nova
remember what was said today across restarts and stream rotations (tasks
t4/t10/t12, spec c11/c12).

## Overview

Before this round the harness remembered exactly one thing: a 20-entry ring
buffer of body cues (`recall_senses`). Every USER/ASSISTANT transcript
vanished the moment it was spoken, and every stream restart — proactive or
not — started the next session with the system prompt only, no recap. This
component adds two files under the state dir and one replay hook so that
"what did we talk about" survives both.

Memory here is per-household-per-day, not per-person: the face cue that
reaches the harness names the rule that fired, not the recognised identity,
so there is no way yet to key memory to a specific person (an upstream
follow-up, not solved by this round).

## Files

- `reachy_nova/harness/ledger.py` — `Ledger`, the raw half: locked NDJSON
  append of every transcript and delivered sense.
- `reachy_nova/harness/memory_compactor.py` — `MemoryCompactor`, the
  distilled half: a background thread that periodically asks Nova 2 Lite
  what the ledger was about, and the `history()` view `nova_sonic.py`
  replays at session start.
- `reachy_nova/harness/statedir.py` — `ledger_path()` /
  `memory_path()`.
- `reachy_nova/nova_sonic.py` — `_replay_history()` / `_start_session()`,
  the one place every restart path (proactive rotation, the liveness and
  clock-step watchdogs, a network change, an ordinary stream death) sends
  the replayed history.
- `reachy_nova/harness/app.py` — `build_memory()` wires the pair together
  and passes `compactor.history` to `NovaSonic(history_provider=...)`.

## The seam: two files under the state dir

| File | Shape | Written by | Read by |
| :--- | :--- | :--- | :--- |
| `<state>/nova-conversation.jsonl` | NDJSON, one `{"ts", "kind", "text", ...}` line per transcript or delivered sense | `Ledger.append`, on Sonic's transcript thread and the MQTT thread | `MemoryCompactor.compact`, `recent_exchanges()` (the Lite reactor's context) |
| `<state>/nova-memory.json` | `{"topics": [{"text", "ts"}], "items": [{"text", "ts", "kind"}]}` | `MemoryCompactor.compact`, atomic temp+`os.replace` | `MemoryCompactor.history()` (session replay), `render_memory_paragraph()` (the Lite reactor's context) |

Both are the harness's own — nothing else in the repo reads or writes them —
and both live under `REACHY_STATE_DIR` alongside the volume and quiet-deadline
files, so a `reachy_nova`/harness grep for `session_state`, `nova_feedback`
or `emotions` still finds nothing: this is a new, small ledger, not a
revival of the legacy direct-SDK app's memory.

## The raw ledger

`Ledger.append(kind, text, **fields)` writes one compact JSON line per call,
serialised through a single lock so two threads (transcripts on Sonic's
thread, senses on the MQTT thread) writing hundreds of lines each still
produce well-formed NDJSON, never an interleaved unparsable line. Three
properties are load-bearing:

- **Quiet-aware.** While a timed quiet is armed (`QuietState.active()`),
  nothing is written at all — a plaintext record of a conversation someone
  asked the robot to be quiet through is exactly the data that policy exists
  to withhold. The skip is counted (`skipped_quiet`), not logged: it is the
  expected shape of a quiet window, not a fault.
- **Never on the voice path.** A write failure (full disk, a read-only
  state dir) never raises into Sonic's thread or the MQTT thread. It
  latches to ONE senselog drop line for the whole run of failures, and the
  first success afterwards emits ONE recovery line, so the journal shows
  both edges of an outage rather than a flood or silence.
- **Atomic truncation.** `Ledger.truncate()` rewrites the file via a temp
  file + `os.replace` — the same pattern `harness/quiet.py` uses for its
  persisted deadline — so a `kill -9` mid-truncation leaves either the old
  or the new file, never a torn one. Truncation keeps only lines with
  `ts >= now - max_age_s` (24 h default) and runs at every compaction.

## The distilled memory

`MemoryCompactor` runs one background thread (`nova-memory-compactor`,
never Sonic's response loop, never the MQTT thread) that, every
`interval_s` (default 300 s), reads the ledger and asks Nova 2 Lite to
distil it into exactly two shapes a companion would be expected to
remember:

- **topics** — subjects the conversation touched.
- **items** — a request, a stated preference, a running joke, or something
  Nova was told to stop doing, each tagged with a `kind`
  (`request`/`preference`/`joke`/`stop`/`fact`).

New entries merge into the existing memory file by normalised (casefolded,
whitespace-collapsed) text — a repeat keeps the *earlier* timestamp so it
does not artificially refresh its own expiry — and anything older than
`max_age_s` (24 h) is dropped at the next successful compaction. A Lite
failure or an unparseable reply leaves the previous memory file completely
untouched and emits one named drop line; the previous memory is never lost
to a bad Lite response. `_extract_json_object` scans for the first
*balanced* `{...}` block and ignores anything the model appends after it —
a live probe on 2026-09-06 returned well-formed JSON followed by a trailing
"*Reasoning:*" paragraph even though the prompt demands JSON only.

A boot with no RTC is handled explicitly: if the injected wall clock is
*earlier* than the newest timestamp already on disk, that is the stale-boot
case, not "everything just expired" — expiry is skipped for that run (new
entries still merge in) and the condition logs once, not on every tick
until NTP steps the clock forward.

## Replay at session start

`MemoryCompactor.history(max_chars=2000)` renders the surviving memory into
the exact shape `nova_sonic.py`'s replay hook sends: one `USER`-role context
block ("earlier today we talked about: ...; things to remember: ...; do not
greet or comment on this; just carry on when spoken to.") followed by the
last three USER/ASSISTANT ledger exchanges verbatim. This is sent through
`_replay_history()` at the ONE window the Nova 2 Sonic service allows for
conversation history — after the system prompt, before audio streaming
begins — at *every* session start: a cold boot, a proactive rotation, the
liveness watchdog, the clock-step watchdog, and a network-change restart all
share this one code path, so "the robot forgot everything" stops being true
for all of them at once rather than being fixed restart-cause by
restart-cause.

The context block's explicit "do not greet or comment on this" instruction
exists because Sonic can speak unprompted on a fresh session — a rotation
every ~7 minutes replaying history must not turn into a re-greeting every
~7 minutes (spec c31). If the block plus exchanges would exceed `max_chars`,
the *oldest* exchange is dropped first, and if the context block alone still
does not fit, it is truncated rather than dropped entirely, since it is what
keeps a rotation from reading as a blank slate.

## Knobs

| Knob | Where | Default | Notes |
| :--- | :--- | :--- | :--- |
| `NOVA_MEMORY` | environment, `harness/switches.py` | on | `0`/`false`/`off`/`no` disables the ledger and the compactor entirely — `build_memory` returns `(None, None)` and `NovaSonic` gets no `history_provider`, restoring a session that always starts blank. An unrecognised value means on plus one named warning. |
| `interval_s` | `MemoryCompactor.__init__` | 300 s | How often the background thread compacts. |
| `max_age_s` | `MemoryCompactor.__init__` / `Ledger.truncate` | 86400 s (24 h) | Both the read window for compaction and the expiry age for topics/items and raw ledger lines. |
| `max_tokens` | `MemoryCompactor.__init__` | 512 | Inference cap for the compaction call — a short JSON object, not prose. |

## Log lines to grep

```text
[SENSE stage=memory source=nova event=ledger] dropped reason=ledger-write-failed: <OSError>
[SENSE stage=memory source=nova event=ledger] ledger recovered (write succeeded after a prior failure)
[SENSE stage=memory source=nova event=compact] dropped reason=memory-compaction-failed: <detail>
[SENSE stage=supervise source=nova event=component] component absent name=memory-ledger reason=<switch-off|...>
rotation delay=0 replay=<n> age=<age>s
history replayed blocks=<n>
```

## Failure modes

- **`NOVA_MEMORY=0`**: no ledger, no compactor, no replay — logged once as a
  named component-absence line, everything else in the harness unaffected.
- **Disk full / read-only state dir**: every ledger append and every memory
  write fails; latched to one drop line each; speech, hearing, and every
  other leg are untouched, since neither write path is on the voice path.
- **Lite unreachable or times out during compaction**: the previous memory
  file is left exactly as it was; one named drop line; the next scheduled
  compaction tries again.
- **Lite reply is not parseable JSON**: same as above — the balanced-brace
  scanner has to fail to find *any* `{...}` block for this to trigger, since
  trailing prose after a valid object is tolerated.
- **A stale boot clock** (no RTC, NTP has not stepped time forward yet):
  expiry is skipped for that run rather than wiping every entry as "aged
  out"; logged once.
- **History provider raises** (a caller passed a broken provider, or the
  compactor itself has a bug): `_replay_history()` catches it, logs one
  warning, and the session starts with an empty replay rather than failing
  to start at all.

## The measured facts behind the design

- The 2026-09-05 journal, at commit `f936f11` (v0.3.0), showed zero
  conversation memory: every restart was "CLEAN: fresh client, fresh UUIDs,
  system prompt only," and none of `session_state.py`, `nova_feedback.py`,
  or `emotions.py` — the legacy direct-SDK app's memory machinery — was
  imported by the harness package at all.
- A live Nova 2 Lite probe on 2026-09-06 returned well-formed compaction
  JSON with a trailing "*Reasoning:*" paragraph the prompt did not ask for
  — the reason `_extract_json_object` tolerates trailing text after a
  balanced object rather than requiring the whole reply to parse as JSON.
- The Nova 2 Sonic connection limit on this account measured 480.5 s (a
  session that started 21:51:29.93 and died 21:59:30.43 with "Model has
  timed out in processing the request") — the reason replay is exercised on
  every restart cause, not only a cold boot: a rotation every ~7 minutes
  (see `docs/components/lite-reactor.md`'s sibling, the chunked-playback
  section of `docs/components/nova_sonic.md`, and architecture.md §5.4) is
  the common case this feature has to survive, not the rare one.
