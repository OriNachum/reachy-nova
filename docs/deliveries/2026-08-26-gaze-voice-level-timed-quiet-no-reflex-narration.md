# Delivery Summary — gaze, voice-level, timed quiet, no reflex narration

plan: `gaze-voice-level-timed-quiet-no-reflex-narration` · run: `complete` ·
date: `2026-08-26`
baseline: `devague summary skeleton`

## Intent

Deliver issues #13–#16 (gaze actions), #17 (stop narrating reflexes, duplicate
pat inject), #18 (raise/lower voice) and #19 (timed silence) across the two
repos — body-side behaviors/intents in reachy-mini-cli, mind-side tools, rules
and gates in the reachy_nova harness — and raise the low-confidence cold-boot
Kiro item (h12) with a repeatable on-robot drill. The plan (17 tasks, 8 waves,
`docs/plans/2026-08-26-gaze-voice-level-timed-quiet-no-reflex-narration.md`)
was executed by `/assign-to-workforce` with one agent per task in isolated
worktrees and a TDD gate on every merge, then accepted live on the robot.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — runtime: DoaPoller exposes last-good age; new one-shot 'look-at-sound'
  LibraryEntry (reachy/behavior/gaze.py + library.py + sense.py)
- `t2` — runtime: face position + selection reach Sense — capture `bbox_norm`
  before the store match, keep last-seen age, choose biggest face with near-tie
  preference for a recognised one (reachy/behavior/face_sense.py +
  reachy/vision/face.py + sense.py)
- `t3` — runtime: one-shot 'look-at-face' LibraryEntry aiming at
  Sense.`face_bbox` (reachy/behavior/gaze.py)
- `t4` — runtime: `lock_face` / `release_face` intent kinds + looping
  'face-lock' behavior with its own yaw/pitch clamp
  (reachy/behavior/`face_lock.py`, registered at composition like goto in
  cli/_commands/behavior.py)
- `t5` — runtime: lock lifecycle events + lock cannot outlive the mind —
  'face-lost' (after N s, lock persists), 'lock-released' registered in
  reachy/export/runtime.py raw-action tables; release on mind-offline
  (reachy/state/nova retained offline for a grace period) or max hold
- `t6` — harness: rules.yaml gains voice (silent|brief|free, default free) and
  sense fields; quiet templates for rule/fire, rule/fire:pat-acknowledge,
  rule/fire:nova-face-noticed, pat/*; bus renders the voice marker
  (config/nervous-system/rules.yaml + reachy_nova/harness/bus.py route_event +
  tests/test_rules_coverage.py + tests/test_harness_bus.py)
- `t7` — harness: sense-class dedupe in NovaBus between `route_event` and
  `_on_inject` (`reachy_nova`/harness/bus.py + tests/`test_harness_bus.py`)
- `t8` — harness: recent-sense ring buffer + `recall_senses` tool
  (`reachy_nova`/harness/`sense_history.py`, wired at NovaBus.`_on_inject`; tool
  builder in tools.py; EXPECTED_TOOLS)
- `t9` — harness: system-prompt line (react naturally to body cues, never
  describe your own mechanism unless asked) + `lock_face`/`release_face` tools +
  gaze tool descriptions for look-at-sound/look-at-face
  (reachy_nova/harness/app.py HARNESS_SYSTEM_PROMPT + tools.py
  builders/_BUILDERS/TOOL_SPECS + tests/test_harness_tools.py)
- `t10` — harness: shared daemon HTTP client +
  `raise_voice`/`lower_voice`/`set_voice_level` tools + persisted volume
  (`reachy_nova`/harness/`daemon_client.py` new; speaking.py uses it; tools.py
  builders; statedir.py nova-volume.json; tests/test_harness_daemon_client.py +
  test_harness_tools.py)
- `t11` — harness: timed quiet — QuietState (deadline, later-wins extension,
  persistence to `<state>`/nova-quiet.json) + SonicSpeaker gate at `_play_one`
  with drop-not-failure semantics and post-acknowledgement arming +
  supervisor.status() quiet_until (reachy_nova/harness/quiet.py new;
  speaking.py; supervisor.py; tests/test_harness_quiet.py +
  test_harness_speaking.py)
- `t12` — harness: `stay_silent` / `end_silence` tools (bounded 1-180 min) that
  also inhibit the runtime 'speak' behavior for the window and mark injects with
  the quiet marker (tools.py builders + bus.py marker +
  tests/test_harness_tools.py)
- `t13` — harness: lock awareness + new-event rules — heartbeat loss while
  locked logs 'lock released reason=engine-restart'; status() shows lock state
  read from state.json intents view; rules.yaml entries for intent/face-lost and
  intent/lock-released rendered as quiet context (reachy_nova/harness/app.py or
  supervisor.py; config/nervous-system/rules.yaml; tests)
- `t14` — cold-boot h12: chaos case 5 in tests/chaos/`ON_ROBOT.md` —
  `KIRO_CLI_BIN`=/nonexistent, restart reachy-nova-harness with
  `FORGE_WRITER`=kiro, assert 'started degraded (initial spawn failed' +
  'started name=kiro_session' and no 'start failed'; restore and assert
  'recovered' with NRestarts unchanged; run it on the robot and paste the
  journal evidence
- `t15` — docs + version: docs/architecture.md §5.5 tool list and §4 seams (lock
  intents, quiet, volume), new docs/components/gaze.md and quiet-mode.md,
  rules.yaml header for voice/sense, deploy order (runtime first) noted; bump
  pyproject version (PyPI publish on main requires it)
- `t16` — live acceptance on the robot + delivery record: deploy runtime then
  harness; run every success-signal line (pat -> one reaction, no mechanism
  words; speak up/quieter; stay silent 2 min zero playbacks incl. engine say;
  look at me / look at the sound / keep looking at me / you can look away;
  recall 'why did you move?'); write
  docs/deliveries/2026-08-26-gaze-voice-quiet.md linking each 2026-08-26 pain
  point to its fix, with unit-only claims labelled
- `t17` — runtime: mute/unmute intent kinds gating SpeechActuator.say + retained
  reachy/state/nova/online subscriber wired into FaceLockDriver.`mind_online`
  (reachy/behavior/speech_act.py, intents or a new mute driver, export/mqtt.py
  subscriber, cli/_commands/behavior.py wiring)

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `Sense.doa_age_s` + `DoaPoller.age_s`; `look-at-sound` one-shot (STOPPABLE, 2 s) with `no recent sound direction` refusal past 8 s — reachy-mini-cli #172 |
| `t2` | delivered | `Sense.face_bbox` / `face_age_s` captured before the store match; `select_face` biggest-wins, near-tie (15 % area) prefers recognised; `FaceEngine.detect_all` — #172 |
| `t3` | delivered | `look-at-face` one-shot + shared `_GAZE_PLANNERS` branch; `no face known` refusal — #172 |
| `t4` | delivered | `lock_face` / `release_face` kinds, clamped slew-limited `face-lock`, later-wins snapshot, named no-ops — #172; snapshot bug found live fixed in #173 (see drift) |
| `t5` | partial | `face-lost` / `lock-released` registered and reaching both feeds; max-hold (30 min) release live. Mind-offline release shipped as a `None` seam in #172 and wired in t17, but inactive on the robot: events-cli's client is publish-only (`MindPresence` logs `client-incompatible`) |
| `t6` | delivered | `voice` / `sense` fields, quiet templates, generic `rule/fire` → `(body cue: {rule})` silent; header docs — harness PR #21 |
| `t7` | delivered | NovaBus dedupe (`NOVA_SENSE_DEDUPE_S`, 10 s); review fix made the identity an explicit `dedupe:` key so pat levels escalate — #21 |
| `t8` | delivered | `SenseHistory` (20, newest-first) recorded post-dedupe; `recall_senses` tool — #21 |
| `t9` | delivered | prompt rewrite (never describe mechanism), `lock_face`/`release_face` tools with verbatim pass-through incl. unknown-kind skew, `DaemonClient` + `restore_volume` wiring — #21 |
| `t10` | delivered | `DaemonClient`, `raise_voice`/`lower_voice`/`set_voice_level` clamped [10, 100], serialized (review fix), persisted to `nova-volume.json` (review fix) — #21 |
| `t11` | delivered | `QuietState` (later-wins, persisted, ack-first arming), SonicSpeaker gate (drop ≠ failure, latched drop + resume summary), `status()` `quiet_until` — #21 |
| `t12` | delivered | `stay_silent` / `end_silence` (1–180 min) spooling `set_inhibition ∪ {speak}` and the runtime `mute`/`unmute`; quiet marker on every inject; expiry tick; ownership persisted across restart (review fix) — #21 |
| `t13` | delivered | `LockState` belief from tool results + `motion/lock-released`; engine-drop clear with a 5 s grace (live fix); rules for `motion/face-lost` / `lock-released`; `status()` `locked` — #21 |
| `t14` | delivered | chaos case 5 documented and **run live** (18:24–18:25): degraded start → 5 watchdog attempts → `recovered`, NRestarts 0→0. Fault knob differs from the plan text (d3) |
| `t15` | delivered | `docs/components/gaze.md`, `quiet-mode.md`, architecture §4/§5/§9 (+ mind-presence topic corrected to `nova/harness/state`), CLAUDE.md map, version 0.2.0 → 0.3.0 — #21 |
| `t16` | delivered | live acceptance 18:26–19:08 on the robot (three rounds, fixes between rounds); this record |
| `t17` | partial | `mute`/`unmute` kinds gating `SpeechActuator.say` — live-proven (`dropped reason=voice-muted`). `MindPresence` subscriber shipped but inactive (publish-only events-cli client) |

## Mid-work Decisions

- `d1` — Add runtime task t17: a mute/unmute intent kind gating
  SpeechActuator.say for the quiet window, plus a retained mind-presence
  subscriber wired into FaceLockDriver.mind_online — the runtime has no MQTT
  subscriber (mind_online shipped as a None seam; only max-hold binds) and rule
  `say` bypasses the speak behavior (rule_engine.py:493 → speech_act.py:468), so
  inhibiting 'speak' only mutes rules that run=speak. Approved.
- `d2` — Add `voice: none` to rules.yaml — the cue is recorded in SenseHistory
  but never injected; applied to `intent/applied`, `intent/blocked` and
  `rule/fire:look-toward-sound` — live 19:05–19:08 the silent cues were still
  narrated ("a standing intention called set_inhibition is now active", "I'm
  still following that sound") and triggered self-initiated glances. Approved.
- `d3` — Chaos case 5 injects the fault by moving `kiro-cli` out of
  `~/.local/bin` instead of `KIRO_CLI_BIN=/nonexistent` — `KIRO_CLI_BIN` is read
  from `os.environ`, populated once by `load_dotenv` at harness start, so the
  plan's knob cannot recover without a harness restart. **Proposed, awaiting
  approval** at the time of writing.
- Qodo on reachy-mini-cli #173 found that lock ownership frozen at
  acquisition breaks the lock invariant when a required name was already
  operator-held; ownership is now recomputed on every replacement
  (`0b199d9`): the live set is always `new_set ∪ LOCK_INHIBITS` while locked;
  a replacement re-listing *all* lock names is treated as a `state.json` echo
  (ownership survives), one keeping only *some* hands those to the caller —
  a heuristic, because the mind cannot tell lock-held names apart from
  operator-held ones. A principled fix (runtime exposes lock-owned names in
  `state.json`) is follow-up.
- The mind-presence topic is `nova/harness/state` (`harness/bus.py:144`), not
  `reachy/state/nova/*` as `docs/architecture.md` said; t17 subscribed to the
  real one and the doc was corrected — no record, captured here.
- `nova-face-noticed` moved from `brief` to `silent` after the second live
  round: on its 30 s cooldown it produced a glance + greeting loop ("Hello
  again!") — commit `6d7f666`. No record; a rules.yaml value change within c29's
  schema.
- Review round (Qodo, 7 findings) and SonarCloud round (6 findings) were fixed
  on the PR before acceptance — `af47c08`, `6f4d464`, `1d97674`, `9d9c3cc`.
- The device's runtime checkout is branch `wireless-motor-enable` (motor-enable
  \+ `enroll` #166, both absent from `main`); `origin/main` was merged into it on
  spark (enroll vs gaze conflicts in `face_sense.py`/`intents.py` were additive)
  — reachy-mini-cli `d2a41ff`, then `82cbd3c` with #173.
- The robot's SD card was 100 % full (pip cache) and the engine was in an ENOSPC
  crash loop (NRestarts 11) when deployment started; freed 1.5 G + 1.8 G,
  installed both packages `--no-deps` (the device never had `nemo_toolkit`; a
  full install pulls CUDA wheels).

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t5` (`d1`) | runtime has no MQTT subscriber (mind_online shipped as a None seam; only max-hold binds) and rule `say` bypasses the speak behavior, so inhibiting 'speak' only mutes rules that run=speak | acceptable |
| `t5` | after d1/t17 the subscriber exists but events-cli's client cannot subscribe (`client-incompatible`); mind-offline release is not live — max-hold (30 min) and the harness's heartbeat-drop belief clear are what bind | needs-follow-up |
| `t17` | same events-cli limitation; `mute`/`unmute` half is complete and live-proven | needs-follow-up |
| `t4` | live: a whole-set `set_inhibition` while locked turned the lock's own inhibitions into operator-held ones; release left the presence loop inhibited. Fixed in reachy-mini-cli #173 (v0.51.1, on the device) | acceptable |
| `t6` / `t7` (`d2`) | silent cues for intent/applied and the runtime's look-toward-sound rule were still narrated and triggered self-initiated glances; a marker cannot stop a model from reacting to a cue it receives every few seconds | acceptable |
| `t11` | the 2 s acknowledgement grace was shorter than Sonic's real latency (9 s live) — raised to 15 s, reservation spent only by an utterance (`8c9bbf5`) | acceptable |
| `t13` | engine heartbeat flaps every ~2 s on the loaded CM4 (pre-existing), which cleared the lock belief immediately — 5 s drop grace added (`8c9bbf5`) | acceptable |
| `t14` (`d3`) | `KIRO_CLI_BIN` is read from os.environ, populated once by load_dotenv at harness start; restoring it on disk cannot reach the running process, so the plan's literal fault knob needs a harness restart to recover | acceptable |
| `t16` | delivery record file is named after the plan slug (`…-gaze-voice-level-timed-quiet-no-reflex-narration.md`), not the plan's `…-gaze-voice-quiet.md` | acceptable |

## Evidence

- tests (harness, branch head `6a2bb59`): `uv run pytest` — 1387 passed
  (baseline on main: 1077)
- tests (runtime, device branch `82cbd3c`): `uv run pytest -n auto` — 5636
  passed, 8 pre-existing skips; #173 branch — 5628 passed
- lint: `workflow.sh lint` (agex) clean in both repos; SonarCloud on #21 —
  Quality Gate OK, 0 open issues, 0 hotspots (after `9d9c3cc`);
  black/isort/flake8 clean on the runtime
- reviews: Qodo on #21 — 7 findings, 7 threads resolved (`6f4d464`)
- commits (harness): `e85772e` (spec) … `6a2bb59` on
  `spec/gaze-voice-quiet-no-reflex-narration`; live-round fixes `8c9bbf5`,
  `6d7f666`, `6a2bb59`
- commits (runtime): #172 merge `9d1b39e` (v0.51.0); #173 `df37236` + `a0c0ebc`
  (v0.51.1); device branch `82cbd3c`
- PRs / issues: OriNachum/reachy-nova#21, #22 (face-focus follow-up);
  agentculture/reachy-mini-cli#172 (merged), #173 (CI green)
- on-robot journal, 2026-08-26 (BST), harness pid 5538/7614: chaos case 5
  18:24:19 `kiro session unit started degraded (initial spawn failed: [Errno 2]
  … 'kiro-cli') — retrying under watchdog`, `started name=kiro_session`, 4 ×
  `restart failed:`, 18:25:16 `kiro session unit recovered (session live after 5
  watchdog attempt(s))`, NRestarts 0→0
- on-robot journal, acceptance round 1 (18:26–18:31): `event=volume old=62
  new=72`, `72→82`; `run_behavior look-at-sound` applied; `lock_face … inhibited
  ['feel-alive','orient-to-sound']`; `motion/face-lost` cue; `stay_silent …
  body_muted:true`, `set_inhibition ['feel-alive','orient-to-sound','speak']`,
  `mute … muted:true`, `quiet-drop`, `quiet-resume count=4`; `end_silence …
  body_restored:true`; cues rendered as `(body cue: look-toward-sound)`, `(a
  familiar face is in view)`
- round 2 (18:35–18:39): `event=volume old=82 new=72`; `release_face` called by
  Sonic (lock:4); `look-at-face` applied ×3; pat: `pat-acknowledge` +
  `nova-pat-cheer` fired, one `(someone is petting you)` cue
- round 3 (19:05–19:08, after `8c9bbf5`): `recall_senses` called ×3;
  `stay_silent` armed 19:07:49, acknowledgement **played** 19:07:58; pat during
  quiet → `dedupe suppressed key=pat-touch`, `quiet-drop`, runtime `say utt1
  dropped reason=voice-muted`; `end_silence`, `quiet-resume count=1`, `voice
  restored, voice-muted count=1`
- spool self-test on the robot (no motion): `release_face` → `{"released":
  false, "note": "not locked"}`, `mute` → `muted: true`, `unmute` → `muted:
  false`

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| a pat yields at most one quiet cue and no mechanism words; the duplicate pat inject is gone | high | round 2/3 journal (`pat-touch` dedupe); tests `tests/test_harness_bus_dedupe.py`, `tests/test_rules_voice.py` |
| "why did you do that?" is answered from the actual sense history | high | round 3 `recall_senses` ×3 + spoken answer; `tests/test_harness_sense_history.py` |
| speak up / quieter move the daemon volume one step with a brief acknowledgement, clamped and persisted | high | journal 62→72→82→72; `tests/test_harness_tools.py` volume tests incl. persistence + concurrency |
| look at the sound / look at me produce a glance or a named refusal | high (aim approximate) | journal `look-at-sound`/`look-at-face` applied; #22 tracks aim quality |
| keep looking at me / look away lock and release by voice, with the presence loop restored after release | high | round 2 `release_face` by Sonic; #173 fix + `tests/test_behavior_face_lock.py` |
| stay silent N minutes: one heard acknowledgement, zero playbacks incl. the body's own `say`, resumes without announcement, early release works | high | round 3 journal (ack played, `quiet-drop`, `voice-muted`, `end_silence`) |
| the quiet deadline and volume survive a harness restart | medium | `tests/test_harness_quiet.py`, `tests/test_harness_tools.py` restart tests — not exercised live |
| a face-lock cannot outlive the mind | low | max-hold and heartbeat-drop belief clear are live; mind-offline release blocked by events-cli (`client-incompatible` in the runtime journal) |
| cold-boot Kiro degraded start recovers under the watchdog with no restart | high | chaos case 5 journal 18:24–18:25 (via the `~/.local/bin` knob, d3) |
| the boot-ordering race itself (network-online.target before wlan0) | unverified | needs a power-cycle with the AP off — deferred by decision c27 |
| Kiro-authored reflexes are quiet by default | high | generic `rule/fire` is `voice: silent`; round 1/3 journal shows `(body cue: …) (quiet: …)` never narrated after d2 |

## Remaining Work / Follow-up

- `t5` / `t17` — mind-offline lock release: events-cli's client needs a `subscribe` API (or the runtime needs its own subscriber client); until then `MindPresence` stays `client-incompatible` and only max-hold (30 min) bounds a lock whose mind died. Owner: reachy-mini-cli.
- #22 — face-focus aim quality (calibrated bbox→angle, head-pose compensation,
  closed loop while locked).
- Engine tick overruns (146 ms vs 20 ms budget) and the resulting 2 s heartbeat
  flap under load on the CM4 — pre-existing; the harness now tolerates it (5 s
  grace) but the runtime should be profiled (face detection cadence,
  `MindPresence`/lock on_tick cost).
- `nemo_toolkit[asr]` is a hard dependency of `reachy-nova` used only by the
  legacy `main.py` path; on the robot it must be installed `--no-deps` or it
  pulls CUDA wheels. Move it to an optional extra.
- reachy-mini-cli `version-bump` should run `uv lock`; a stale lock made CI
  re-resolve and hit the `pycairo 1.29.1` sdist (main's Tests run after #172 is
  red for the same reason; #173 refreshes the lock).
- Lock ownership vs the mind's whole-set `set_inhibition`: expose lock-owned names in `state.json` so the harness can merge without re-listing them (removes the echo heuristic). Owner: reachy-mini-cli.
- Merge order: #173 → `main` (runtime), then #21 → `main` (harness) and publish
  0.3.0; the device already runs both.
- Unprompted Sonic monologues about "sensitive video file details" appeared
  twice with no inject behind them — not chased; watch for recurrence.
