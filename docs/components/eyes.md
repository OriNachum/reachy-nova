# Eyes Documentation

This documentation covers `Eyes`/`EyesComponent` — the harness's own watch
on whether the camera is actually delivering frames (task t11).

## Why it exists

Live finding (2026-09-06): the runtime published
`reachy/events/sense/snapshot` at 50 Hz with `frame_available: false` in
251/251 and 435/435 sampled messages, while the runtime's own availability
report (`reachy/state/senses`) said the camera senses were "available" the
whole time — that field reports *composition* (a camera pipeline is wired
in), not *liveness* (it is actually producing frames). Nothing in the
harness named the gap: the camera had been dark since shortly after boot,
and it took a manual, targeted probe to notice, two days in. Vision, face
recognition, and the face lock that item 2 of this round depends on all fail
*silently* in exactly this way without a line that says so.

**Files:** `reachy_nova/harness/eyes.py` (`Eyes`, the pure state machine;
`EyesComponent`, the supervisor-shaped MQTT reader),
`tests/test_harness_eyes.py`.

## What it watches

`EyesComponent` opens its **own** small `paho-mqtt` connection — deliberately
separate from `NovaBus` (`reachy/events/sense/snapshot` is off that bus's
default subscription by design; see `bus.py`'s module docstring on the sense
flood) — and subscribes to `SNAPSHOT_TOPIC =
"reachy/events/sense/snapshot"`. The topic itself arrives at ~50 Hz;
`EyesComponent` samples it down to about 1 Hz (`SAMPLE_INTERVAL_S = 1.0`,
tracked via an injectable clock) before feeding each decoded
`frame_available` boolean to `Eyes.note(...)`.

## The state machine

`Eyes` is pure — no threads, no network — and holds one of three states:
`"unknown"` (no note yet), `"live"`, or `"dead"`. It speaks **once per
transition**, never once per sample:

- Continuous `frame_available=False` for `dead_after_s` (default 60 s,
  `NOVA_EYES_DEAD_AFTER_S`, `DEFAULT_DEAD_AFTER_S = 60.0`) latches `"dead"`
  and logs exactly one line. Further `False` notes while already dead log
  nothing.
- The first `True` after `"dead"` latches `"live"` and logs exactly one
  line, with the actual downtime (since the continuous-`False` stretch
  began, not since the 60 s latch fired).
- The first `True` ever seen (`"unknown"` -> `"live"`) latches `"live"` and
  logs exactly one `live first_seen` line, so "frames arrived and kept
  arriving" is distinguishable from "frames never arrived at all" — which
  both used to look like silence. Later `True` samples log nothing: a healthy
  camera arrives at ~1 Hz and must not cost a line per second, so the
  transitions are the record.
- A later `False` stretch after a restoration latches again only after
  another full `dead_after_s` — the false-streak clock resets on every
  `True`.

`dead_after_s()` is parsed defensively: unset, blank, non-numeric or
non-positive values all fall back to `DEFAULT_DEAD_AFTER_S` (with a warning),
never raising and never silently disabling the latch.

## The three log lines

```text
[SENSE stage=vision source=runtime event=frames] live first_seen
[SENSE stage=vision source=runtime event=frames] dropped reason=no-frames after=60s
[SENSE stage=vision source=runtime event=frames] restored after=61.30s
```

`after=` in the `dropped` line is the configured threshold
(`_fmt_seconds(dead_after_s)`, trimmed of trailing `.0`); `after=` in the
`restored` line is the measured downtime.

## Status field

`EyesComponent.eyes_state` (a passthrough to `self.eyes.state`) is what
`supervisor.status()` discovers and reports as `eyes: "unknown" | "live" |
"dead"` — the same three values `Eyes.state` holds.

## The broker fallback

`EyesComponent.start()` never raises. Building the paho client, connecting,
or subscribing can each fail (no `paho` installed, an unreachable broker) —
every one of those degrades to one `component absent name=eyes
reason=broker-unreachable detail=...` line and the component simply never
feeds `Eyes.note`, leaving `eyes_state` at `"unknown"` rather than a guess.
`stop()` is idempotent and disconnects without raising.

`app.py` builds `EyesComponent` via `eyes_module.build_component()` inside
the same total `try/except` every other optional leg in `build_components`
uses — a construction failure there also degrades to one
`component absent name=eyes reason=<err>` line.

## Configuration

| Env | Default | Meaning |
| --- | --- | --- |
| `NOVA_EYES_DEAD_AFTER_S` | `60.0` | how long continuous `frame_available=False` must persist before the camera is declared dead |
