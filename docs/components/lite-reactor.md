# Lite Reactor

This documentation covers the Nova 2 Lite reaction tier: a fast side channel
that gives an opted-in body cue a one-line reaction plan instead of the
fixed template line every cue used to get (task t11/t13, spec c10).

## Overview

Before this round every rules.yaml entry rendered exactly one fixed
sentence per cue — the same pat at breakfast and at midnight produced the
same inject, word for word, because `llm_evaluate` on an entry was carried
as metadata only ("no model is called here"). Sonic itself is a big,
slow-to-turn mind: routing every cue through a live conversational turn to
get variety would cost seconds Sonic doesn't have to spare. `LiteReactor` is
the middle tier: a rules.yaml entry that opts in (`react: lite`) hands its
rendered cue, plus a little context, to Nova 2 Lite on a dedicated worker,
and gets back one short line — what to say (if anything), a vocalisation, a
gesture — within a named timeout, falling back to the plain template on any
failure.

## Files

- `reachy_nova/harness/lite_reactor.py` — `LiteReactor`, the worker,
  the request/reply parsing (`parse_plan`), and the plan-to-inject
  rendering (`render_reaction`).
- `reachy_nova/harness/bus.py` — `NovaBus`'s `route_event`/`_deliver`: the
  ONE place a `react: lite` entry is recognised and the ONE place the
  entry's voice/quiet markers get applied to whatever the reactor delivers.
- `config/nervous-system/rules.yaml` — the `react: lite` per-entry field
  (see the file's own header comment for the full field reference).
- `reachy_nova/harness/app.py` — constructs `LiteReactor` with
  `context_provider=_reaction_context` (senses, memory, mood, exchanges)
  and `on_gesture=_reaction_gesture` (submits through the same intents
  spool every tool call uses), and wires it into `NovaBus(reactor=...)`.

## The seam

A `react: lite` rules.yaml entry does not change what fires the cue or how
often (dedupe, priority, and urgency are unchanged) — only what happens
*after* the bus decides to deliver it. `NovaBus` hands the reactor the
**base** rendered text (no voice/quiet marker yet, spec c29) as both the
`cue` and the `template`; the reactor calls `react(cue, template, deliver)`
and returns immediately. Whatever text is eventually handed to `deliver` —
the Lite plan or the template fallback — is what gets the entry's own
`voice`/quiet markers applied, exactly as a plain template render would, so
the inject Sonic receives and what `SenseHistory` records are always the
same *delivered* text regardless of which tier produced it. `voice: none`
entries never reach the reactor at all — `_handle_message` routes those to
sense-history-only delivery before `react` is ever consulted.

## The worker

One dedicated thread drains a small bounded `queue.Queue` (`max_queue`,
default 4) in latest-wins order: a `react()` call arriving while the queue
is full evicts the oldest still-pending request rather than blocking or
growing unbounded — an old reaction is not worth acting on once a newer cue
has already superseded it, and the evicted request's `deliver` is simply
never called (one named `lite-evicted` line, not a fallback delivery).

Because `boto3`'s `invoke_model` is itself a blocking call, even the
worker thread does not call it directly: each attempt runs on a short-lived
daemon helper thread, and the worker `Event.wait()`s for it up to
`timeout_s`. A call that never returns is simply abandoned — nothing joins
it — so a 30 s hang in Lite costs exactly one timed-out reaction, never a
wedged worker or a delayed pat/face/vision reaction behind it.

## The reply format

Lite is asked to reply with exactly one line:

```text
say=<one short line or none> | vocalize=<chirp|trill|purr|none> | gesture=<nod|shake|antenna-sway|none>
```

`parse_plan` matches the *first* line in the reply fitting this format and
ignores everything after it — a live probe on 2026-09-06 showed Lite
occasionally appending a "*Reasoning:*" trailer even when told not to, and a
well-formed first line still parses fine. A reply with no matching line at
all is malformed and falls back to the template.

## Delivery ordering

The senselog line for a reaction (planned or fallen back) is always emitted
**before** the `deliver`/gesture callbacks fire — a test (or any caller)
that observes the delivered text can rely on the log line already being
durable, rather than racing it on another thread.

## Context provided to Lite

`app.py`'s `_reaction_context()` builds four labelled parts, always present
even when a part is empty (an absent part reads as `(none)` in the prompt
rather than silently vanishing): the last five sense-history entries, the
day's memory as one paragraph (`render_memory_paragraph`, from
`MemoryCompactor.memory()` — see `docs/components/memory.md`), the current
mood sentence (`Mood.render()` — see `reachy_nova/harness/mood.py`), and the
last four USER/ASSISTANT ledger exchanges. A raising or `None` context
provider degrades to an empty context rather than failing the worker.

## Knobs

| Knob | Where | Default | Notes |
| :--- | :--- | :--- | :--- |
| `NOVA_LITE_REACTIONS` | environment, `harness/switches.py` | on | `0`/`false`/`off`/`no` disables the whole tier — every `react: lite` entry then renders byte-identically to an entry without the field. An unrecognised value means on plus one named warning. |
| `react: lite` | per-entry in `config/nervous-system/rules.yaml` | absent | Opts one entry into the tier. Never applies to a `voice: none` entry. |
| `timeout_s` | `LiteReactor.__init__` | 2.0 s | One Lite round trip's deadline before it is abandoned and the template is delivered. Set from a live probe (2026-09-06): median ~1.0 s, max 1.34 s for a 60-token reply. |
| `max_queue` | `LiteReactor.__init__` | 4 | Bound on pending requests; a new one past this evicts the oldest still-pending. |
| `MAX_TOKENS` | `lite_reactor.py` module constant | 60 | A one-line reply needs very little. |

## Log lines to grep

```text
[SENSE stage=react source=lite event=<n>] planned say=<...> vocalize=<...> gesture=<...> latency=<ms>ms
[SENSE stage=react source=lite event=<n>] dropped reason=lite-timeout
[SENSE stage=react source=lite event=<n>] dropped reason=lite-error: <detail>
[SENSE stage=react source=lite event=<n>] dropped reason=lite-malformed: reply=<...>
[SENSE stage=react source=lite event=<n>] dropped reason=lite-evicted cue=<...> (superseded by a newer cue before it was processed)
[SENSE stage=supervise source=nova event=component] component absent name=lite-reactor reason=<switch-off|...>
[SENSE stage=act source=nova event=lite-gesture] dropped reason=lite-gesture-failed name=<gesture> detail=<...>
```

`did Lite actually answer, or did we fall back to the template, and why` is
always answerable from the `event=<n>` line alone — every reaction, planned
or fallen back, costs exactly one.

## Failure modes

- **`NOVA_LITE_REACTIONS=0` or the reactor fails to construct**: every
  `react: lite` entry falls back to the plain template path, logged once as
  a named component absence — no cue is ever lost, only its variety.
- **Timeout** (`REASON_TIMEOUT`): the helper thread is abandoned in place;
  the template is delivered; the worker moves to the next queued cue
  immediately.
- **Bedrock/network error** (`REASON_ERROR`): same fallback, the exception
  named in the line.
- **Malformed reply** (`REASON_MALFORMED`): no line in the required format
  anywhere in the reply; same fallback, the raw reply named (truncated).
- **Queue full** (`REASON_EVICTED`): the oldest pending request is dropped
  with no delivery at all — not a fallback, since the newer cue already
  supersedes it.
- **A raising `deliver` or `on_gesture` callback**: caught and logged as a
  warning; never wedges the worker thread.
- **A raising `context_provider`**: degrades to an empty context (every
  part reads `(none)`), never fails the reaction.

## The measured facts behind the design

- A live probe of Nova 2 Lite from the robot (2026-09-06) for a 60-token
  reply returned a median of ~1.0 s and a max of 1.34 s — the basis for the
  2.0 s default timeout (roughly 1.5x the measured worst case).
- The same probe's replies sometimes carried a trailing "*Reasoning:*"
  paragraph despite the prompt instructing "reply with exactly one line and
  nothing else" — the reason `parse_plan` matches the first well-formed line
  and stops looking, rather than requiring the whole reply to be clean.
- Before this round, `llm_evaluate` on a rules.yaml entry was documented but
  never actually consulted ("no model is called here") — `react: lite` is
  the first per-entry field that actually routes a cue through a model
  before it becomes an inject.
