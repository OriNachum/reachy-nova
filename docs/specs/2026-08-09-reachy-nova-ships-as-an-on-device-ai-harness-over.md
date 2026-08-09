# Reachy Nova ships as an on-device AI harness over reachy-mini-cli's symbolic runtime on the Reachy Mini Wireless: Nova 2 Sonic live voice, Nova 2 Omni deep understanding, and Nova Act attach through the runtime's documented seams, and when the network is gone the robot keeps its full rules-and-actions autonomy

> Reachy Nova ships as an on-device AI harness over reachy-mini-cli's symbolic runtime on the Reachy Mini Wireless: Nova 2 Sonic live voice, Nova 2 Omni deep understanding, and Nova Act attach through the runtime's documented seams, and when the network is gone the robot keeps its full rules-and-actions autonomy
> instruction: Demo script: start conversation, show object, ask for a browser task; verify via systemctl + /api/media ownership that only the runtime holds the session

## Audience

- People living with the wireless Reachy Mini (conversation, reactions, presence) plus the AgentCulture agents (nova, reachy-mini-cli) who maintain the two-layer system
  - instruction: Validate docs suffice: build the harness against reachy-mini-cli docs/export-schema.md + `intent_tools` docstrings only; file doc gaps as issues

## Before → After

- Before: `reachy_nova` is a monolithic ReachyMiniApp owning the SDK, the 50Hz motion loop, and all Nova cognition; the wireless robot runs a stale reachy-nova copy; no cloud means no behavior at all
  - instruction: Record baseline: git rev-parse on both checkouts; snapshot current main.py coupling list from docs/plans/2026-08-09-wireless-cli-harness-scope.md
- After: reachy-runtime.service (symbolic runtime) is main at boot and owns the single SDK/media session; the reachy-nova harness runs beside it as a peripheral systemd --user unit attaching only via MQTT events, audio tee, intents spool, rules overlay, and the speaker feed; killing wifi mid-conversation leaves breathing, pat reactions, and rule-driven speech intact
  - instruction: Install harness unit After=reachy-runtime.service; run the wifi-pull chaos test from the acceptance list

## Why it matters

- Cognition and body separate cleanly: the robot is never dead without the cloud, personality is locked into model-free rules that survive harness-off, and each layer evolves in its own repo against a documented wire contract
  - instruction: Run reachy-mini-cli `test_zero_llm_boundary` + pytest -m offline after each behavior-port PR

## Requirements

- Hear+speak loop: harness reads the audio tee socket, resamples to 16kHz for sonic.`feed_audio`, and streams Sonic's 24kHz reply PCM to the runtime's speaker feed (#162 Ask 1) for sub-second first-audio latency
  - instruction: Implement tee reader -> np.interp resample -> sonic.`feed_audio`; Sonic `on_audio_output` -> speaker-feed writer; measure first-audio latency with a timestamped clap test
  - honesty: Measured on the CM4: tee-to-Sonic-to-speaker-feed round trip yields first reply audio under 1s in a live conversation, and the tee's oldest-dropped queue never starves Sonic of contiguous audio (no ASR garbling from drops)
- Act loop: Sonic toolConfiguration tool calls map to the intents spool via `register_intent_tools` (`run_behavior`, `declare_goal`, `set_mode`, goto with `await_result`) and to `create_rule` for standing reactions that survive harness shutdown
  - instruction: Build the Sonic tool registry from `register_intent_tools` + `create_rule` wrapper; every executor returns the `await_result` payload (or degraded note) as the tool result string
  - honesty: Every Sonic tool schema maps to an intents-spool command whose `await_result` confirmation (or degraded submitted-only note) is returned to the model as the tool result - no tool call silently vanishes
- See loop: Omni consumes the `clip_rider` rolling clip path and face events off the bus for scene/identity understanding, falling back to Lite 2 when Omni is unavailable; results inject into Sonic through the nervous-system prioritization (inject throttle preserved)
  - instruction: NovaOmni client: request-response invoke with clip file + context; feature-flag model ID via env; forced-failure test exercises Lite-2 fallback; injects go through nervous-system rules
  - honesty: Omni accepts the `clip_rider` file format (duration/codec) as video input at our region/quota, and the Lite-2 fallback is exercised by a test that forces Omni unavailability
- Behavior lock-in: the shed vocabulary (curious, pondering, boredom, nuzzle, purr, enjoy gestures; sleep breathing cycle; mood antenna animation) lands upstream as reachy-mini-cli LibraryEntry behaviors + rules (#162 Ask 2, PRs authored by us) so a wifi-off robot expresses the full set
  - instruction: Port each gesture/sleep/antenna behavior as a LibraryEntry PR to reachy-mini-cli with params + StopClass per their library.py conventions; side-by-side visual acceptance with Ori
  - honesty: Each ported behavior passes reachy-mini-cli's gates (zero-LLM boundary suite, offline lane) and is visually accepted by Ori on the robot as matching the `reachy_nova` original
- Degradation: AWS/network loss surfaces as named drops and reconnect-with-backoff, never a crash; the runtime is provably unaffected by harness kill (reachy-mini-cli pytest -m offline green with harness installed but stopped)
  - instruction: Chaos suite: kill -9 harness, wifi down/up, bad AWS creds; assert named \[SENSE\] drops, backoff reconnect, runtime tick metrics flat
  - honesty: A chaos test kills the harness and pulls wifi mid-conversation: the runtime tick never overruns from it, and reachy-mini-cli's offline suite passes with the harness unit installed-but-stopped
- Cognition feed: the harness emits the documented thinking/message/emotion NDJSON contract so existing consumers (reterminal bridge) work unchanged
  - instruction: Emit thinking/message/emotion via the shared `build_export_hook` contract shape; smoke-test against the reterminal bridge script unmodified
  - honesty: The reterminal bridge renders harness output with zero changes to the bridge script
- Echo safety without SDK register access: the harness enforces a half-duplex speaking guard (suppress/attenuate `feed_audio` to Sonic while the speaker feed is playing) until hardware AEC on the wireless GStreamer/alsa path is verified - today main.py:75 feeds unconditionally and relied on XMOS pokes only the local sounddevice backend exposes
  - instruction: P1: implement the speaking-window gate in the tee reader; run the echo test; separately probe whether XVF3800 AEC is active on the alsa capture path and relax the gate if so
  - honesty: An on-robot echo test (speaker playing Sonic audio, silent room) produces no self-transcription in Sonic ASR output
- Runtime-restart resilience: when reachy-runtime restarts (Restart=on-failure) the harness re-attaches to the recreated audio tee socket, spools, and bus with backoff and named drops - the degradation story covers the body dying under the mind, not just AWS loss
  - instruction: Chaos suite adds a runtime-restart case beside the wifi-pull case
  - honesty: systemctl --user restart reachy-runtime mid-conversation: harness logs named drops, re-attaches within 10s, conversation resumable without harness restart
- Mouth-loss grace: losing speaker-feed arbitration mid-utterance (rule say or lobes realtime wins the mouth) surfaces to Sonic as an interruption (barge-in path), never a hang or infinite retry
  - instruction: Wire speaker-feed write failures/preemption to the existing `on_interruption` path in `nova_sonic`
  - honesty: Forcing a say rule to preempt Sonic mid-sentence leaves Sonic responsive to the next utterance (interruption fired, no stuck `_speaking` state)
- Credential + broker hardening: the on-robot .env is chmod 600 (probe found -rw-rw-r-- world-readable today), the IAM principal is scoped to exactly the models the harness invokes, and mosquitto binds localhost-only (config ships `allow_anonymous` true) on a device whose daemon and Zenoh are already LAN-open unauthenticated
  - instruction: P0 device work: chmod, dedicated IAM user/policy, mosquitto listener config; document accepted residual LAN exposure of daemon/Zenoh (upstream reality)
  - honesty: ls -l shows 600 on .env; the IAM policy lists only the invoked model ARNs; ss shows mosquitto bound to 127.0.0.1; a LAN scan finds no new ports opened by the harness
- Harness observability: health visible via systemctl --user status, greppable \[SENSE stage=... source=nova ...\] journald lines for every drop/reconnect, and a retained state topic - no silent cognition death
  - instruction: Reuse `sensory_log`.stage conventions for the harness; publish retained harness state on the bus
  - honesty: Killing AWS creds mid-run produces a journalctl-greppable named drop line and a state-topic transition within one reconnect cycle
- Reversible cutover: the presence switch is rollback-safe (service enable demo-mode restores prior behavior; harness uninstall leaves the runtime standalone), noting reachy-mini-cli service verbs destructively purge `RETIRED_UNITS` on every enable
  - instruction: Write the rollback drill into the acceptance checklist; back up ~/.config/systemd/user/reachy-\* before first service call
  - honesty: A timed rollback drill (service enable demo-mode + harness stop) restores the pre-migration robot in under 5 minutes with no motor re-calibration

## Honesty conditions

- A live demo on the wireless robot exercises voice (Sonic), understanding (Omni or Lite-2 fallback), and Act through the harness while the runtime remains the only SDK/media owner
- Household use needs no terminal (power-on to working robot), and each repo's agent can build against the seams from the wire-contract docs alone without reading the other repo
- The monolith description matches `reachy_nova` HEAD (main.py owns SDK + cognition) and the on-robot ~/git/reachy-nova checkout is confirmed stale relative to it
- systemctl --user on the robot shows the runtime presence unit plus the harness peripheral unit, and the wifi-pull test leaves tick rate, pat reactions, and rule speech unchanged
- No rule can require a harness code path to fire: reachy-mini-cli's zero-LLM boundary suite and offline lane stay green with the full locked-behavior set
- An import/AST audit of `reachy_nova`/harness finds no `reachy_mini` import and no pose math, enforced by a CI test so the boundary cannot regress silently
- The three scenarios exist as a scripted/checklisted acceptance run and pass on the physical robot before the ReachyMiniApp path is deleted

## Success signals

- On-robot acceptance: (1) boot wifi-off - rules fire, pat reaction, offline voice answers; (2) wifi on - Sonic conversation with barge-in, Omni describes a shown object, a spoken request creates a standing rule that still fires after harness stop; (3) wifi pulled mid-conversation - runtime continues, harness reconnects
  - instruction: Write the acceptance checklist into docs/plans/2026-08-09-wireless-cli-harness-scope.md verification section; run it on the robot and record results before P5 retirement

## Scope / boundaries

- The harness never opens a `reachy_mini` SDK client and contains zero motion code; all motion/reflex lives in reachy-mini-cli rules and library behaviors
  - instruction: Add tests/`test_harness_boundary.py` (AST walk over `reachy_nova`/harness forbidding `reachy_mini` imports and `set_target` references) to the merge gate

## Non-goals

- The USB Reachy Mini on spark stays on reachy-mini-cli directly - it is not a harness target; the direct-SDK ReachyMiniApp path is retired after wireless acceptance, not maintained in parallel
- Nova 2 Omni is not used for realtime voice (it is request-response, preview); live conversation stays on Nova 2 Sonic; reachy-mini-cli's lobes-cli /v1/realtime duplex lane is preserved untouched
- No reachy-nova\[openai\] extra: reachy-nova is for Nova by design and Nova is always AWS - AWS deps are base dependencies, and the OpenAI-compatible lane remains reachy-mini-cli's lobes/embody stack (already exclusive with the harness per c26)

## Assumptions

- reachy-mini-cli#162 (streaming speaker feed + upstreaming shed behaviors) and #163 are handled - design proceeds unblocked on both

## Scope exploration

- `s1` — `challenge pass / adjacent-systems lens: nova_mqtt.py + docker/nervous-system/nervous_system.py:302 vs reachy-mini-cli export/mqtt.py topic map`: namespace mismatch: nervous-system subscribes nova/events/# while the runtime publishes reachy/events/#; broker ownership on the CM4 undecided - seeded q1
- `s2` — `challenge pass / unstated-assumptions lens: Bedrock streaming SDK on CM4 (probe: robot venv pip list)`: clean - `aws_sdk_bedrock_runtime` 0.3.0 + boto3 already installed in the robot's reachy-nova venv (aarch64/py3.12): Sonic on this hardware is prior art, not a bet
- `s3` — `challenge pass / failure-modes lens: echo path (reachy_nova/main.py:75 unconditional feed_audio; main.py:143-180 XMOS pokes; SDK audio_gstreamer alsasrc on wireless)`: no software echo gate exists and the hardware AEC pokes cannot ride the wireless backend - seeded c19 + park v6
  - seeds: `c19`
- `s4` — `challenge pass / lifecycle lens: reachy-mini-cli service/units.py Restart=on-failure + audio_tee socket recreation`: runtime restart under a live harness was uncovered by the degradation requirement - seeded c20
  - seeds: `c20`
- `s5` — `challenge pass / security lens: robot .env perms probe (-rw-rw-r--), config/mosquitto/mosquitto.conf:2 allow_anonymous true, SDK daemon 0.0.0.0:8000 + Zenoh 7447 no-auth`: world-readable creds on a LAN-open device; hardening requirement c22 + posture question q4
  - seeds: `c22`
- `s6` — `challenge pass / concurrency lens: reachy-mini-cli control.py spools (atomic os.replace) + overlay rules.toml writers`: spool path clean by construction; overlay multi-writer atomicity unverified - parked v7
- `s7` — `challenge pass / observability+recovery lens: senselog conventions in both repos, harness health surface`: harness had no observability claim - seeded c23; mouth-loss recovery seeded c21
  - seeds: `c23`
- `s8` — `challenge pass / reversibility lens: service/manager.py exactly-one-presence + RETIRED_UNITS destructive purge + P5 retirement`: rollback drill was implicit only - seeded c24; ReachyMiniApp deletion already gated by h13
  - seeds: `c24`
- `s9` — `challenge pass / cheap-probes: CM4 memory (free -h: 3.7Gi total, 3.2Gi available with demo-mode running)`: headroom context for parked Act-headless unknown v2; no new claim

## Decisions

- Model lineup: Nova 2 Sonic = live voice; Nova 2 Omni = clip/scene understanding with Nova 2 Lite fallback; Lite 2 = fast judgments (barge-in, nervous-system); Nova Act = v1 on-device headless behind a flag; all model IDs and region env-configurable
- Harness runs on-device at reachy-mini.local beside the runtime, packaged as `reachy_nova`/harness with its own entry point, PID supervisor mirroring agent embody, and a peripheral systemd --user unit (not a reachy-mini-cli presence unit)
- Bus topology: one localhost-bound mosquitto on the robot; nervous-system as a plain systemd --user process subscribed to reachy/events/#; nova/\* topics fold away (resolves q1)
- Cognition exclusivity: Nova harness and agent embody never run simultaneously - the harness supervisor refuses to start while embody's PID file is live, and vice-versa guidance goes upstream (resolves q2)
- v1 aux scope: `nova_memory` in, slack/feedback deferred to v2, reachy-mini-cli forge is the one self-extension path (resolves q3)
- Security posture: LAN-open daemon/Zenoh accepted as documented residual risk; hardening scope is exactly c22 (resolves q4)
- Packaging: reachy-nova is ours on PyPI - the harness publishes as distribution reachy-nova with a deploy pipeline mirroring reachy-mini-cli publish.yml (pytest gate; TestPyPI .devN publish on same-repo PRs; PyPI via uv build + uv publish with OIDC trusted publishing on main push)

## Open parks

- [unknown_nonblocking] Nova 2 Omni exact model ID, region availability, and quotas (preview; console enablement pending) - env-configurable IDs + Lite-2 fallback make this survivable either way
- [unknown_nonblocking] CM4 4GB headroom for Nova Act headless Chromium alongside Sonic + tee (measure in P3; fallback = spark-side peripheral over MQTT)
- [unknown_nonblocking] Final wire format of the #162 speaker feed (socket vs spool, f32 vs int16 header) - harness adapts to whatever ships
- [unknown_nonblocking] Whether XVF3800 hardware AEC is active on the wireless GStreamer/alsa capture path (echo test + firmware inspection in P1; c19's software gate is the mitigation either way)
- [unknown_nonblocking] Concurrent writers to the overlay rules.toml (harness `create_rule`, agent embody, human edits) - spool writes are atomic os.replace but overlay TOML write atomicity is unverified
- [unknown_nonblocking] OTA updates can bump the on-robot daemon outside reachy-mini-cli's reachy-mini>=1.9,<1.10 lockstep pin - upgrade ownership and sequencing undecided
- [unknown_nonblocking] A dashboard-launched HF app contending with the runtime for the robot (outside liveness.py's reach) - mitigated if #163's app-slot integration lands
- [follow_up] Migration of on-robot ~/.`reachy_nova` state (faces, session, forged skills) into the harness layout
- [follow_up] Rules-parity review for the USB Reachy Mini (same locked behaviors, no harness)
