# Build Plan — harness round 2: alive senses + resilient start

slug: `harness-round-2-alive-senses-resilient-start` · status: `exported` · from frame: `harness-round-2-alive-senses-resilient-start`

> Wireless harness round 2: resilient auto-start (clock-jump-proof Sonic, no competing auto-run), working barge-in, pat + face + tracking senses routed into Nova, and Nova Act browsing as a tool

## Tasks

### t1 — Sonic resilience: clock-step detector + response-liveness watchdog in `nova_sonic.py`

- covers: c2, h1, c22, h16
- acceptance:
  - a simulated wall-clock step (>60s drift between time.time() and time.monotonic()) triggers exactly one session restart, in a unit test
  - a synthetic zombie (injects continue, response events muted) triggers exactly one forced restart within the watchdog window, in a unit test
  - forced restarts are clean (system prompt only, no recap) and never fire twice for one cause; existing 526-test suite stays green

### t2 — Gate policy split: hearing suppression becomes `NOVA_ECHO_GATE`=off|half-duplex (default off), playback windowing untouched

- covers: c4, c5, c6, h9
- acceptance:
  - with `NOVA_ECHO_GATE`=off (and unset), TeeHearing feeds mic chunks to Sonic while the gate is armed; with half-duplex it suppresses exactly as today, in unit tests
  - SonicSpeaker still serializes utterances one-at-a-time through the gate in both modes, in a unit test
  - gate.py/hearing.py docstrings drop the disproven 'no verified hardware AEC' premise and cite the 2026-08-10 AEC verification

### t3 — Units + install encoded in repo: unit.py renders exclusivity, install script carries journald persistence drop-in

- covers: c3, h2, c23, h17
- acceptance:
  - unit.py-rendered harness unit and a repo-carried reachy-runtime.service template pair Conflicts= with demo-mode; demo-mode template has no \[Install\]; rendering tests assert all three
  - install script/doc applies a journald drop-in (persistent storage, SystemMaxUse cap) and masks reachy-nova-autostart; idempotent on re-run

### t4 — browse tool: `TOOL_SPECS` entry + executor over the flag-gated NovaBrowser queue

- covers: c10, h5
- acceptance:
  - with `NOVA_ACT_ENABLED` unset, importing tools.py imports neither `nova_act` nor playwright (boundary test)
  - with the flag on (mocked NovaBrowser), the browse tool queues the instruction and returns a typed acknowledgment; progress callbacks inject back, in unit tests

### t5 — Cross-repo seams via reachy-mini-cli issues: enroll intent kind + unit-template exclusivity

- covers: c25, h19
- acceptance:
  - two issues filed on agentculture/reachy-mini-cli with full seam specs (enroll command kind through the intent registry; Conflicts=/manual-only demo in service unit templates), signed per convention
  - the enroll seam's wire contract (op, name, temp-face reference, result) is agreed in the issue thread before t7 builds against it

### t6 — Device bring-up: disk headroom, \[vision\] extra, journald drop-in, senses verified

- depends on: t3
- covers: c12, h6, c24, h18
- acceptance:
  - df recorded before/after; cleanup achieves >=10% root free BEFORE installs, else the step aborts loudly
  - after installing the \[vision\] extra: reachy/state/senses shows face available:true, reachy/state/clip carries a real path, control loop holds ~50Hz, and the discrete pat/face event topic names are recorded (resolves v3)

### t7 — Bus + rules routing for pat/face/vision events into Sonic injects

- acceptance:
  - `DEFAULT_SOURCES` gains the pat/face/vision event sources (exact names from the live topic check, v3) and rules.yaml routes them with priority/urgency; route tests cover fire and no-template paths
  - a pat event injected through `route_event` produces a Sonic inject whose text names the touch, in a unit test

### t8 — `enroll_face` tool riding the runtime's enroll seam

- depends on: t4, t5
- covers: c16, h7
- acceptance:
  - `enroll_face` appears in `TOOL_SPECS`; executor submits the agreed enroll intent and returns the engine's typed result; degraded/no-seam path returns a named refusal, in unit tests

### t9 — Vision leg component: clip-rider reader -> NovaOmni.understand -> inject

- acceptance:
  - a new harness/`vision_leg.py` consumes the retained clip state (path+availability), calls NovaOmni.understand with clip+context, and injects the answer; unavailable clip is a named drop, in unit tests
  - `vision_leg` imports no `reachy_mini` and no cv2 (boundary test)

### t10 — Composition root integration: wire gate policy, browse, enroll, vision leg into app.py

- depends on: t2, t4, t8, t9
- covers: c9, h10
- acceptance:
  - `build_app`() wires the new tools and vision leg behind their flags; component count and degraded paths asserted in tests
  - `test_harness_boundary.py` AST gate passes: no harness module imports `reachy_mini` or runs a tracking loop

### t11 — Live acceptance on the wireless: five signals + playback-tolerance watch, recorded in the scope doc

- depends on: t1, t6, t10
- covers: c1, h8, c17, h11, c18, h12, c19, h13, c20, h14, c21, h15, h3, h4
- acceptance:
  - signals 1-5 of c21 demonstrated live and recorded (overnight cold boot zero-SSH; mid-sentence interrupt; pat reaction; greet + learn a face by voice; spoken web search)
  - during playback, look-toward-sound does not orient to the robot's own speaker and no moving-floor poisoning recurs (c26 observed live); barge-in preempt visible in the journal, no echo loop

## Risks

- [unknown_nonblocking] cv2 + clip encode CPU load next to the 50Hz control loop on a Pi already at load ~2.3 — watch `compose_hz` during bring-up (task t6)
- [unknown_nonblocking] enroll seam is cross-repo: reachy-mini-cli must ship the intent kind before t8 can build; t8 has a named degraded path if it slips (task t8)
- [unknown_nonblocking] chromium RAM footprint next to runtime+harness (2.5G free) if AgentCore hosted browser doesn't pan out (v2) (task t4)
- [unknown_nonblocking] tools.py is shared by t4 and t8 — serialized by the t8->t4 dependency; app.py touched only by t10
