# Build Plan — Reachy Nova ships as an on-device AI harness over reachy-mini-cli's symbolic runtime on the Reachy Mini Wireless: Nova 2 Sonic live voice, Nova 2 Omni deep understanding, and Nova Act attach through the runtime's documented seams, and when the network is gone the robot keeps its full rules-and-actions autonomy

slug: `reachy-nova-ships-as-an-on-device-ai-harness-over` · status: `exported` · from frame: `reachy-nova-ships-as-an-on-device-ai-harness-over`

> Reachy Nova ships as an on-device AI harness over reachy-mini-cli's symbolic runtime on the Reachy Mini Wireless: Nova 2 Sonic live voice, Nova 2 Omni deep understanding, and Nova Act attach through the runtime's documented seams, and when the network is gone the robot keeps its full rules-and-actions autonomy

## Tasks

### t1 — Backend-free Nova core: extract all model IDs + region into `reachy_nova`/config.py, env-configurable with current defaults (incl. hidden Lite-2 in `nova_sonic.py`:121); record migration baseline (git rev-parse both checkouts + coupling snapshot)

- instruction: New `reachy_nova`/config.py with env-var reads (`NOVA_SONIC_MODEL_ID` etc) defaulting to current literals; update `nova_sonic`/`nova_vision`/`nova_memory`/`nova_browser` imports; append baseline revs to the scope doc
- covers: c3, h9
- acceptance:
  - grep finds no hardcoded model ID outside config.py and existing pytest stays green
  - baseline revs + coupling list recorded in the scope doc

### t2 — NovaOmni client (request-response clip+image+text) with Nova 2 Lite fallback in `reachy_nova`/`nova_omni.py`

- instruction: Model on `nova_vision.py` invoke pattern; boto3 `invoke_model` with video bytes; fallback wraps NovaVision call; tests mock boto3 client
- covers: c14, h3
- acceptance:
  - forced-Omni-failure test exercises the Lite-2 fallback with mocked bedrock, no network

### t3 — Packaging + publish pipeline: pyproject name reachy-nova with AWS base deps and reachy-nova-harness console script; .github/workflows/publish.yml mirroring reachy-mini-cli (pytest gate, TestPyPI devN on same-repo PRs, PyPI trusted publishing on main)

- instruction: Copy reachy-mini-cli publish.yml, drop the alias job, swap package paths; rename pyproject name field; stub `reachy_nova`/harness/`__main__.py` behind the console script
- acceptance:
  - uv build succeeds and the workflow matches the cited publish.yml pattern

### t4 — Harness boundary CI test: tests/`test_harness_boundary.py` AST-walks `reachy_nova`/harness forbidding `reachy_mini` imports and `set_target` references

- instruction: Follow reachy-mini-cli tests/`test_zero_llm_boundary.py` AST pattern; parametrize forbidden imports/attrs
- covers: c6, h12
- acceptance:
  - test fails on an injected violation and passes clean, wired into the pytest merge gate

### t5 — Behavior lock-in PRs upstream: port curious/pondering/boredom/nuzzle/purr/enjoy gestures, sleep breathing, mood antenna as LibraryEntry PRs to reachy-mini-cli (#162 Ask 2)

- instruction: One PR per behavior family against reachy-mini-cli; source motion curves from gestures.py, `sleep_orchestrator.py`, `antenna_animator.py`; pure fn(t, params, sense) form
- covers: c15, c5, h11
- acceptance:
  - PRs open following library.py conventions with their zero-LLM boundary and offline suites green in CI

### t6 — Bus subscriber + nervous-system on reachy/\*: paho subscriber modeled on `embody_bus_feed.py`; nervous-system as plain systemd --user process re-pointed to reachy/events/#; mosquitto bound to localhost

- instruction: Adapt scripts/`embody_bus_feed.py` paho pattern into `reachy_nova`/harness/bus.py; move docker/nervous-system logic to a module entry point; mosquitto conf listener 127.0.0.1
- acceptance:
  - unit test routes a fake reachy/events sense event through rules.yaml to an inject call; broker config binds 127.0.0.1

### t7 — Audio-tee client with echo-safe half-duplex gate: tee reader, resample to 16kHz, `feed_audio` suppressed while the speaker feed is active

- instruction: `reachy_nova`/harness/hearing.py; header-declared samplerate from tee, np.interp resample per `audio_pipeline.py` conventions; gate on speaker-feed active flag
- depends on: t1
- covers: c12, c19
- acceptance:
  - fake-socket test verifies resample correctness and that feed is suppressed during playback

### t8 — Speaker-feed writer + mouth-loss grace: Sonic `on_audio_output` streams to the speaker feed; preemption and write failure route to `on_interruption`

- instruction: `reachy_nova`/harness/speaking.py; write 24kHz PCM to the #162 feed; on preemption/write-fail call sonic.`on_interruption` path and clear pending audio
- depends on: t1
- covers: c21, h16
- acceptance:
  - preemption test leaves Sonic responsive with no stuck `_speaking` state

### t9 — Sonic tool registry over the intents spool: toolConfiguration built from `register_intent_tools` + `create_rule` wrapper

- instruction: `reachy_nova`/harness/tools.py; build toolConfiguration specs from `register_intent_tools` registry + `create_rule` + omni/memory/act executors; mirror skills.py `get_tool_specs` shape
- depends on: t1
- covers: c13, h2
- acceptance:
  - every tool returns the `await_result` (or degraded submitted-only) payload as its tool result against a fake spool

### t10 — Cognition feed emitter: thinking/message/emotion NDJSON per the export contract

- instruction: `reachy_nova`/harness/`cognition_feed.py` emitting the reachy-mini-cli `_export.py` block shapes (thinking/message/emotion) on stdout/export target
- covers: c17, h6
- acceptance:
  - the reterminal bridge script parses sample harness output unmodified

### t11 — Supervisor + systemd unit + exclusivity + observability: PID supervisor mirroring agent embody, --user unit After=reachy-runtime.service, refuses start while embody PID is live, SENSE-style journald lines and retained state topic

- instruction: `reachy_nova`/harness/supervisor.py mirroring reachy/embody/supervisor.py + reachy/procsup.py; render unit like service/units.py; check embody PID file before start
- covers: c4, c23, h18
- acceptance:
  - exclusivity and named-drop tests pass; rendered unit file orders after reachy-runtime

### t12 — Chaos/degradation suite: harness kill, wifi down/up, bad AWS creds, runtime-restart-under-live-harness; local fakes + on-robot checklist entries

- instruction: tests/chaos/ with fake tee/spool/broker; each case asserts a \[SENSE\] line and recovery; document on-robot equivalents in the checklist
- depends on: t7, t8, t11
- covers: c16, h5, c20, h15, h10
- acceptance:
  - all chaos cases pass locally with fakes, each producing a named SENSE drop and clean reconnect

### t13 — Security hardening P0: chmod 600 .env on-robot, IAM policy scoped to invoked model ARNs only, mosquitto localhost listener, documented residual LAN exposure

- instruction: scripts/harden-robot.sh (chmod, ss checks) + docs/security.md with the IAM policy JSON and the accepted-residual statement from the spec
- covers: c22, h17
- acceptance:
  - hardening script + doc exist; ls -l and ss verification steps scripted

### t14 — Memory leg v1: qq context injection routed through nervous-system prioritization

- instruction: Port `nova_memory` query path onto the bus inject route; keep qq env config; no direct sonic.`inject_text` bypass
- depends on: t6
- acceptance:
  - mocked qq test injects context via the rules path with the inject throttle preserved

### t15 — Nova Act headless behind a default-off flag

- instruction: Wrap `nova_browser` behind `NOVA_ACT_ENABLED`=0 default; lazy import; headless chromium args
- depends on: t1
- acceptance:
  - flag off means zero Playwright import; on-device smoke test deferred to the acceptance run

### t16 — Acceptance run + rollback drill + retirement: scripted 3-scenario checklist plus sub-5-minute rollback drill in the scope doc, executed on the robot; ReachyMiniApp removal PR gated on the pass; includes on-robot echo test, CM4 latency measurement, Omni live verification, and behavior visual acceptance

- instruction: Extend scope-doc verification section into a numbered checklist incl. echo test, clap-latency, rollback drill timing, behavior side-by-side; record results inline; open retirement PR referencing it
- depends on: t5, t12, t13
- covers: c18, h13, c1, h7, c2, h8, h1, h4, h14, c24, h19
- acceptance:
  - checklist committed to the scope doc verification section with recorded on-robot results; retirement PR references the passing run

## Risks

- [unknown_nonblocking] On-robot honesty conditions (h1, h4, h7, h10, h13, h14, h19) are unverifiable in CI - they gate at the t16 acceptance run, not at merge
- [unknown_nonblocking] Nova 2 Omni preview enablement, quotas, and clip-format acceptance unverified until console access - Lite-2 fallback is the mitigation
- [unknown_nonblocking] Speaker-feed wire format adapts to whatever reachy-mini-cli#162 ships (socket vs spool, sample format)
- [unknown_nonblocking] CM4 headroom for Act headless Chromium and XVF3800 AEC status remain open (parked frame-side v2/v6)
