# Scope: Reachy Nova as a harness over reachy-mini-cli on Reachy Mini Wireless

Date: 2026-08-09
Status: SCOPED — awaiting implementation kickoff
Related: `docs/plans/nervous-system.md` (local-reflex vs remote-deliberation split),
`docs/plans/nova-2-sonic.md` (barge-in notes), reachy-mini-cli `docs/export-schema.md`
(the wire contract this harness builds against).

## Context

Today reachy_nova is a monolithic `ReachyMiniApp`: one process owns the robot SDK,
the 50Hz motion loop, tracking, sleep, *and* all AWS Nova cognition (Sonic voice,
Lite vision, Act browser). Meanwhile reachy-mini-cli has matured into exactly the
"body" half of that split: its **symbolic runtime** (`behavior engine run`) is a
50Hz, model-free, CI-enforced zero-LLM tick loop with declarative TOML rules and a
library of named behaviors — designed explicitly for an external AI harness to
attach on top, never to be replaced by one.

The goal: on the **Reachy Mini Wireless**, reachy-mini-cli becomes *main* — it runs
at boot and gives the robot full standalone autonomy (rules + actions, speech via
its offline voice). reachy_nova is redesigned as the *harness* — a peripheral
process on the same device that attaches through reachy-mini-cli's documented
seams and supplies AWS Nova cognition. When the network/AWS is unreachable, the
harness degrades away and the symbolic runtime keeps the robot alive unchanged.

The USB reachy mini connected to spark uses reachy-mini-cli directly (no
reachy_nova); reachy_nova's direct-SDK `ReachyMiniApp` path is retired once the
wireless harness is proven.

## Decisions (locked, 2026-08-09)

1. **reachy-mini-cli is main.** `reachy-runtime.service` (systemd --user,
   `behavior engine run`) enabled at boot on the wireless robot. It owns the single
   SDK/media session. The harness NEVER opens an SDK client.
2. **Harness runs on-device** (reachy-mini.local, beside the runtime), integrating
   reachy-mini-cli as a Python library dependency. "Wireless off" = no internet =
   Nova unreachable = harness degrades; runtime autonomy is untouched.
3. **Model lineup:**
   - **Nova 2 Sonic** (`amazon.nova-2-sonic-v1:0`) — live voice, unchanged role.
     (Nova 2 Omni is preview, request-response, speech-in/text-out — it cannot do
     real-time voice.)
   - **Nova 2 Omni** (preview; ID TBD, env-configurable) — deep multimodal
     understanding: rolling camera clips, scenes, long context. Replaces Nova 2
     Lite as the *vision/understanding* model; falls back to Lite 2 when Omni is
     unavailable.
   - **Nova 2 Lite** (`us.amazon.nova-2-lite-v1:0`) — fast/cheap judgments only:
     barge-in judge, nervous-system LLM verdicts.
   - **Nova Act** — in scope for v1, **on-device headless** (risk-flagged; fallback
     is deferring to a spark-side peripheral).
   - **Nova 2 Multimodal Embeddings** — unchanged (nova_memory).
4. **All model IDs and region become env-configurable** (today every ID is a
   hardcoded default — including the hidden Lite dependency inside
   `nova_sonic.py:121`).

## Architecture

```
Reachy Mini Wireless (CM4)
├── reachy-mini-daemon.service        (system; Zenoh 7447, HTTP 8000, WebRTC)
├── reachy-runtime.service (--user)   ← MAIN: reachy-mini-cli behavior engine run
│     rules.toml overlay · behavior library · pat/rms/face/doa senses
│     publishes MQTT reachy/events/# + reachy/state/# · audio tee socket
│     spools: intents/commands · reload · state.json
└── reachy-nova-harness (--user unit, ours)   ← THE HARNESS (this repo)
      hear:  audio_tee.sock ──► Nova 2 Sonic bidirectional stream
      speak: Sonic 24kHz audio ──► speaker path (see R1)
      see:   clip_rider clip path / face events ──► Nova 2 Omni (fallback Lite 2)
      act:   Sonic toolUse ──► intents spool (register_intent_tools),
             create_rule → rules overlay (standing reactions that outlive us),
             Nova Act (headless), memory, slack, vocalize
      sense: MQTT reachy/events/# ──► nervous-system rules ──► sonic.inject_text
      tell:  cognition feed (thinking/message/emotion NDJSON) for reterminal etc.
```

### Attach seams used (all documented in reachy-mini-cli `docs/export-schema.md`)

| Need | Seam | Notes |
|---|---|---|
| Events in | MQTT `reachy/events/#`, `reachy/state/#` | subscribe with paho, modeled on reachy-mini-cli `scripts/embody_bus_feed.py` (events-cli is publish-only) |
| Live mic | `<state_dir>/audio_tee.sock` (mono f32 + header) | reference reader: `reachy/embody/media.py`; feeds `sonic.feed_audio()` after resample to 16k |
| Actions | intents spool via `reachy.speech.intent_tools.register_intent_tools` | `run_behavior`, `declare_goal`, `set_mode`, `goto` + `await_result` |
| Standing reactions | `create_rule` → overlay `rules.toml` + `submit_reload()` | validated, hot-reloaded, survives harness shutdown |
| Speak | daemon HTTP `POST /api/media/sounds/upload` + `play_sound` (stdlib, no SDK contention) | see risk R1 for the streaming variant |
| Camera | `clip_rider` rolling clip path published on state.json/bus | Omni takes the clip file directly (video input) |

### What the harness sheds (moves to the symbolic runtime)

YOLO tracking → `orient-to-sound` / face_sense rules; pat detection → `pat_sense`;
wake word/Parakeet → Sonic hears the tee continuously + runtime engagement gate;
sleep breathing → runtime behaviors; gesture playback → behavior library via
intents. The harness keeps **zero** motion code. Shedding is not deletion:
the shed behaviors get **locked in** as reachy-mini-cli library entries +
rules so the personality survives with the harness off — the gap list
(`curious`/`pondering`/`boredom`/`nuzzle`/`purr`/`enjoy` gestures, the sleep
breathing cycle, mood antenna animation) is Ask 2 of
[reachy-mini-cli#162](https://github.com/agentculture/reachy-mini-cli/issues/162);
we author those PRs during P0–P1.

## Work plan

### P0 — Groundwork (both repos, no behavior change)

- reachy_nova: extract Nova clients (`nova_sonic`, `nova_vision`, `nova_memory`,
  `nova_browser`, `nova_slack`, emotions/state) from the app loop; make every
  model ID + region env-configurable; add `NovaOmni` client (request-response,
  video+image+text in) with Lite-2 fallback.
- Wireless robot audit — **done 2026-08-09** (device found via the `wireless`
  registry — `reachy wireless list` gives its `base_url` and `hardware_id`;
  mDNS `reachy-mini.local` does NOT resolve from the host — use the registry
  `base_url` or `sudo reachy wireless pin`): ReachyMiniOS v0.2.3, daemon
  **1.9.0** (inside reachy-mini-cli's `reachy-mini>=1.9,<1.10` lockstep window
  — no daemon upgrade needed), SSH access verified, linger enabled,
  reachy-mini-cli installed with
  the **demo-mode presence enabled and running** (not the runtime), `~/git/`
  holds `reachy-nova`, `reachy-mini-cli`, `reachy-claude`,
  `reachy_mini_testbench`. Remaining P0 device work: switch presence
  `demo-mode → runtime` (`reachy service enable runtime` — presence units are
  mutually exclusive), verify offline autonomy (wifi off → rules still fire).

### P0.5 — Packaging + deploy

Publish the harness to PyPI as **`reachy-nova`** (ours; AWS-only by design — no
`[openai]` extra, Nova is always AWS; the OpenAI-compatible lane stays
reachy-mini-cli's lobes/embody stack). Deploy pipeline mirrors reachy-mini-cli's
`.github/workflows/publish.yml`: pytest gate → TestPyPI `.devN` on same-repo
PRs → PyPI on main push, `uv build`/`uv publish` with OIDC trusted publishing,
SHA-pinned actions. On-robot install becomes `uv tool install reachy-nova`
instead of a git checkout.

### P1 — Harness core (`reachy_nova/harness/`)

New package with its own entry point (`reachy-nova-harness`), depending on
`reachy-mini-cli` as a library. Components: bus subscriber (paho), audio-tee
client → Sonic, Sonic → speaker path, tool registry mapping Sonic
`toolConfiguration` to the intents spool + `create_rule`, cognition-feed emitter,
PID-file supervisor (mirror `agent embody start|stop|status`) + our own
systemd --user unit (peripheral, not a reachy-mini-cli presence unit).
Reuse `nova_mqtt`'s nervous-system rules for event→inject prioritization
(inject throttle in `nova_sonic.py` stays).

### P2 — Omni vision + judgments + memory

Clip-based scene understanding on Omni (rolling clip path from the bus), face
events → identity context, Lite 2 barge-in/nervous-system verdicts, nova_memory
unchanged.

### P3 — Nova Act on-device headless

Chromium headless on CM4 behind a feature flag; measure memory. Fallback
decision point: defer to spark peripheral over MQTT.

### P4 — Startup + degradation hardening

Unit ordering after `reachy-runtime.service`; chaos tests: kill harness (runtime
unaffected), wifi off (Sonic reconnect loop with backoff, no crash), AWS creds
bad (named drops, offline voice still answers via runtime rules).

### P5 — Retirement

Remove the direct-SDK `ReachyMiniApp` path from reachy_nova once wireless
harness is accepted; USB robot remains on reachy-mini-cli directly.

## reachy-mini-cli changes needed (upstream, minimal)

Filed 2026-08-09 as
[reachy-mini-cli#162](https://github.com/agentculture/reachy-mini-cli/issues/162)
(speaker feed + upstreaming the shed behaviors into the library, explicitly
preserving their lobes-cli `/v1/realtime` duplex voice lane), with the
`reachy_mini_apps` dashboard-app idea sent separately as a non-blocking
suggestion in
[reachy-mini-cli#163](https://github.com/agentculture/reachy-mini-cli/issues/163).

1. **Speaker feed (R1):** a streaming audio-*out* counterpart to the audio tee
   (unix socket or spool → runtime's media session), so Sonic's 24kHz stream can
   play with conversational latency instead of per-utterance WAV upload+play.
   Follows audio_tee's bounded/named-drop patterns. Without it, v1 ships with
   per-utterance latency via the HTTP playback path.
2. Optional: export `transcript` on the feed (their issue #93) — moot for us
   (Sonic does its own ASR from the tee) but worth closing while we're there.

## Risks

- **R1 speak latency:** HTTP upload+play is per-utterance (~1–2s added). The
  speaker-feed upstream change is the fix; decide in P1.
- **R2 Omni preview:** model ID/region/quotas unverified until enabled in the
  console; everything env-configurable with Lite-2 fallback.
- **R3 CM4 headroom:** Sonic stream + tee + Act headless on 4GB; measure in P1/P3,
  Act is the first thing to offload.
- **R4 device addressing:** `reachy-mini.local` does not resolve from spark
  (avahi name-collision class of problem, documented in reachy-mini-cli
  `reachy/discover/hosts.py`); everything must key off the `wireless` registry
  (hardware_id → base_url) or a pinned `/etc/hosts` entry, never the mDNS name.
- **R5 inject ceiling:** all events still funnel through Sonic `inject_text`
  (3s throttle, speaking guard) — nervous-system prioritization remains the
  pressure valve.

## Verification

- Both test suites stay green: `uv run pytest` here; reachy-mini-cli's
  `pytest -m offline` (endpoints unreachable) must pass with the harness
  installed but stopped.
- On-robot acceptance: (1) boot with wifi off → rules fire, pat reaction, offline
  voice answers; (2) wifi on → harness attaches, Sonic conversation with barge-in,
  Omni describes a shown object, a spoken request creates a standing rule that
  still fires after `reachy-nova-harness stop`; (3) pull wifi mid-conversation →
  runtime continues breathing/reacting, harness reconnects when back.

### Device bring-up record (2026-08-10, executed over ssh pollen@192.168.1.162)

- **Migration baseline (t1)**: reachy_nova `5f5c3a9` (spec/plan commit, branch
  `wireless-harness` forked from it); reachy-mini-cli on-device was a stale
  **0.29.0 pip install** over a `30ef881` (v0.47.0) checkout — the venv, not
  the checkout, is what runs. On-device unpushed work preserved as branch
  `raspberry-pi-fixes` (`be59d7c`, pushed to origin from spark; the device has
  no GitHub auth).
- **CLI upgrade**: checkout pulled to `8238c1a` (v0.48.0), `ensurepip` +
  `pip install -e .` into its venv → `reachy-mini-cli 0.48.0`.
- **Broker**: `mosquitto` installed via apt, `/etc/mosquitto/conf.d/local-only.conf`
  = `listener 1883 127.0.0.1` + `allow_anonymous true`; enabled at boot;
  verified `ss -tlnp` shows `127.0.0.1:1883` only.
- **Engine probe** (before cutover): demo-mode stopped, foreground
  `behavior engine run --max-ticks 600` → 600 ticks clean, `feel-alive`
  owning head/antennas/body_yaw, live DoA on the bus, tee socket created,
  retained `reachy/state/*` tree + `reachy/state/online true` published.
- **Cutover**: `reachy-runtime.service` **hand-authored** (upstream
  `units.py` text minus `Requires=reachy-daemon.service` — ReachyMiniOS runs
  the SDK daemon as the system service `reachy-mini-daemon.service` on :8000,
  so the CLI's user daemon unit would double-start it; deviation to file
  upstream). `demo-mode` disabled, `runtime` enabled; heartbeat age 0.5s.
  Unit backups in `~/unit-backups/`. Rollback = disable runtime, enable
  demo-mode (unit file preserved).
- **Runtime restart drill**: `systemctl --user restart reachy-runtime` →
  active again with tee socket rebound and retained `online true` within 6s.
- **Hardening P0**: on-device `.env` now `-rw-------` (chmod 600).
- **Harness deps**: robot's reachy-nova venv is editable against
  `~/git/reachy-nova` (branch switch = deploy);
  `aws_sdk_bedrock_runtime`/paho/dotenv/yaml/numpy import clean.

### Live acceptance record (2026-08-10 evening, harness deployed on-robot)

- **Attach**: every component start/attach is one `[SENSE stage=…]` journald
  line: tee `header accepted (f32le rate=16000)`, bus `connect … rc=Success`,
  `runtime-online`, `engine live`, `harness up pid=… components=4`.
- **Hear (human)**: Ori spoke at the robot → `[SENSE stage=hear event=transcript]
  heard '…'` lines and an audible conversational reply. Mic RMS for close
  speech ≈0.09 vs ≈0.002 room floor.
- **Echo safety**: robot-speaker audio barely registers on the tee (XVF3800
  hardware AEC IS active on the wireless capture path — resolves parked v6)
  AND the software half-duplex gate arms per playback (`echo gate armed …
  suppressed 68 chunks (6800 ms)`); gate now arms *before* the HTTP post after
  a live echo-loop incident (Nova conversed with its own tail).
- **Speak**: Sonic replies play through the daemon HTTP route (`queued
  duration=6.44s` → `played`); playback audible in the room (user-confirmed).
- **Act**: a live Sonic tool call crossed the intents spool and the ENGINE
  answered: `set_inhibition refused reason=unknown behavior 'look-toward-sound'
  (…have: feel-alive, pet-reaction, orient-to-sound, …)` — typed refusal with
  the real catalog, returned to the model as the tool result.
- **Read**: `mosquitto_pub` of a `rule/fire` event → nervous-system verdict →
  `inject … priority=NORMAL urgency=NOW` → spoken reply about the reflex.
- **Robot-to-robot**: Nova transcribed the *local* (USB) unit's embody voice —
  cross-unit conversation observed unprompted.
- **Runtime restart under live harness (h15)**: `runtime-offline` →
  `tee-unavailable` → `runtime-online` → `reconnected … header accepted`,
  same harness process, within ~8s. PASS.
- **Reboot survival**: full power cycle → runtime + harness + mosquitto all
  return unattended (linger), motors re-enabled by the media-client patch,
  Sonic session re-established. PASS.
- **Stalled-generation guard**: live sessions showed Bedrock generations that
  stop streaming audio without a terminal `contentEnd` (duplicate
  speculative/final texts); `_speaking` pinned forever. A 4s speaking watchdog
  now clears the stuck state (`nova_sonic.py`), after which the buffered
  utterance flushes and plays.
- **KNOWN HARDWARE BLOCKER (outside harness scope)**: head/body motors do not
  move — antennas only. Verified below every layer of ours: engine-confirmed
  gotos, raw `reachy_mini` SDK (`goto_target`, runtime stopped), AND the
  daemon's own HTTP `/api/move/goto` all produce zero motion; SDK goto hangs
  on the daemon motion service. Daemon backend never reports alive
  (`ready: false`, `last_alive: null`) despite a 49Hz motor-controller loop,
  with serial errors on `body_rotation` (id 10) and `stewart_6` (id 16).
  Survives daemon restart AND full power cycle. 2026-08-11 refinement: the
  head/body motors are physically limp (no holding torque — Ori's
  observation) while the antennas hold and animate, and the unit DID move
  fully before (with the old app and with demo-mode) — so this is not
  shipped-broken hardware and not today's software: prime suspect is a loose
  head/body motor-chain connection (exactly the ids throwing comm errors).
  Being verified physically.
- **Upstream findings for reachy-mini-cli** (branch `wireless-motor-enable`
  pushed): (1) wireless daemon boots `motor_control_mode=disabled` — the CLI
  never calls `enable_motors()`; patched best-effort in `HeldMediaClient`
  (+ the sdk_transport media session). (2) `service enable runtime` would
  double-start the SDK daemon on ReachyMiniOS (the OS owns it as
  `reachy-mini-daemon.service`); the deployed `reachy-runtime.service` is
  hand-authored without `Requires=reachy-daemon.service`.
