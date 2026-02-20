# Temporal Awareness

Nova experiences time the way a conscious being would — not as a clock, but as a felt sense of "how long ago" something happened. Events are tagged "just now" or "a moment ago", memories carry age labels like "yesterday" or "a few days ago", and a drifting sense of time-of-day keeps the conversation grounded. All underlying storage remains precise UTC for data alignment and training.

## Overview

The temporal awareness system adds three layers of time understanding:

1. **Current time anchor** — On startup, Nova receives both a precise timestamp (for data alignment) and a vague time-of-day phrase ("It's a Tuesday afternoon").
2. **Event timestamps** — Every injected event (face recognition, pats, vision, MQTT) is prefixed with a vague session-relative marker like `[just now]` or `[a little while into the conversation]`.
3. **Memory age labels** — Memories from MongoDB carry vague age labels ("a few hours ago", "yesterday") in both startup context and runtime queries.

**Files:**
- `reachy_nova/temporal.py` — pure-function utility module for vague human-like time formatting
- `reachy_nova/main.py` — startup anchor, event wrapping (4 sites), periodic time sense injection
- `reachy_nova/session_state.py` — vague restart context and conversation message timestamps
- `reachy_nova/nova_memory.py` — memory age labels in startup context and runtime queries

## How It Works

### Startup Sequence

```
App starts
  -> _inject_startup_context()
     -> [Current time: Tuesday, Feb 20, 2026, 3:42 PM UTC. It's Tuesday, in the afternoon.]
     -> [Session context — you were away for a little while. It's Tuesday, in the afternoon.]
     -> [Recent memories] Ori likes tea (a few hours ago) | The sensor is broken (yesterday)
```

The precise timestamp is the only absolute time reference Nova ever sees. Everything else uses vague, human-like language.

### Event Injection

Every `inject_text()` call that represents a sensory event is wrapped with `format_event()`, which prepends a session-relative temporal marker:

| Event | Before | After |
| :--- | :--- | :--- |
| Face recognized | `You notice Ori is here.` | `[a moment ago] You notice Ori is here.` |
| Pat level 1 | `You feel a gentle tap on your head.` | `[just now] You feel a gentle tap on your head.` |
| Pat level 2 | `Someone is scratching your head...` | `[just now] Someone is scratching your head...` |
| MQTT inject | `{text}` | `[a little while into the conversation] {text}` |

### Periodic Time Sense

Every 10 minutes (while awake), a vague time-of-day update is injected:

```
[Time sense: It's Tuesday, late in the evening.]
```

This gives Nova a drifting sense of time passing without being robotic.

### Restart Context

Session restart messages now use vague durations and include the current time-of-day:

| Restart type | Before | After |
| :--- | :--- | :--- |
| Crash recovery | `you just restarted due to a brief interruption.` | `you just restarted due to a brief interruption. It's Tuesday, in the afternoon.` |
| Short break | `you were offline for 5 minutes.` | `you were away for a little while. It's Tuesday, in the afternoon.` |
| Long absence | `you've been offline for about 2 hours.` | `you've been away for a few hours. It's Tuesday, in the afternoon.` |

Conversation history in restart context also carries vague timestamps:

```
Before:   USER: Hello Nova
After:    USER (a little while ago): Hello Nova
```

### Memory Age Labels

Both startup context and runtime queries now label memories with vague age:

```
Before: [Recent memories] Ori likes tea | The sensor is broken
After:  [Recent memories] Ori likes tea (a few hours ago) | The sensor is broken (yesterday)
```

This applies to keyword search results, vector search results, and the startup memory summary.

## temporal.py

A small pure-function module with no imports from the rest of the codebase. All functions use `datetime.now(timezone.utc)` internally.

### Functions

| Function | Purpose | Example output |
| :--- | :--- | :--- |
| `utc_now_vague()` | Vague time-of-day with weekday | `It's Tuesday, in the afternoon.` |
| `utc_now_precise()` | Precise UTC for startup anchor | `Tuesday, Feb 20, 2026, 3:42 PM UTC` |
| `relative_vague(timestamp)` | Vague age from a Unix timestamp | `a few hours ago`, `yesterday` |
| `session_vague(session_start)` | Vague session-elapsed marker | `a little while into the conversation` |
| `format_event(text, session_start)` | Wraps text with temporal prefix | `[a moment ago] You notice Ori is here.` |
| `format_elapsed_vague(elapsed)` | Vague duration from seconds | `a little while`, `a few hours` |

### Relative Time Bands

`relative_vague()` maps elapsed time to human expressions:

| Elapsed | Expression |
| :--- | :--- |
| < 30s | `just now` |
| < 5m | `a moment ago` |
| < 30m | `a little while ago` |
| < 2h | `a while ago` |
| < 6h | `a few hours ago` |
| < 24h | `earlier today` |
| < 48h | `yesterday` |
| < 7d | `a few days ago` |
| < 14d | `last week` |
| < 30d | `a couple of weeks ago` |
| >= 30d | `a long time ago` |

### Time-of-Day Bands

`utc_now_vague()` maps the UTC hour to phrases:

| Hour range | Expression |
| :--- | :--- |
| 0–4 | `late at night` |
| 5–8 | `early in the morning` |
| 9–11 | `in the morning` |
| 12–13 | `around midday` |
| 14–16 | `in the early afternoon` / `in the afternoon` |
| 17–19 | `in the evening` |
| 20–21 | `late in the evening` |
| 22–23 | `late at night` |

## Design Principles

- **Vague, not precise**: The model should feel time, not read a clock. "A while ago" is more natural than "47 minutes ago".
- **UTC everywhere**: All stored timestamps remain `time.time()` (UTC epoch). Vague formatting happens only at the model-facing boundary.
- **One precise anchor**: The startup context contains one absolute timestamp for data alignment and training correlation. Everything else is relative.
- **Training quality**: Temporal context in injected text improves the quality of RLHF training data, since the model's reasoning about recency becomes grounded.

## Thread Safety

`temporal.py` contains only pure functions operating on `datetime.now()` and numeric arguments. No shared state, no locking required. Safe to call from any thread.

## Related

- [Session Persistence](continous_experience.md) — restart classification and context injection (now uses vague time)
- [Nova Memory](nova_memory.md) — long-term memory retrieval (now carries age labels)
- [Nova Sonic](nova_sonic.md) — voice model that receives all temporal context via `inject_text()`
- [Growth](growth.md) — RLHF feedback pipeline (benefits from temporal context in training data)
- [Patting](patting.md) — touch events that now carry temporal prefixes
- [Face Recognition](face_recognition.md) — face events that now carry temporal prefixes
