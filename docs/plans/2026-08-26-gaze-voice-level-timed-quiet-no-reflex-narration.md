# Build Plan — gaze, voice-level, timed quiet, no reflex narration

slug: `gaze-voice-level-timed-quiet-no-reflex-narration` · status: `exported` · from frame: `gaze-voice-level-timed-quiet-no-reflex-narration`

> Nova now performs asked-for gaze actions (look toward sound, look toward face, lock on a face until released, release), raises or lowers its voice on request, can stay quiet for a set number of minutes and resume on its own, and no longer narrates its own reflexes unless asked — and the cold-boot Kiro degraded-start path has been exercised live

## Tasks

### t2 — runtime: face position + selection reach Sense — capture `bbox_norm` before the store match, keep last-seen age, choose biggest face with near-tie preference for a recognised one (reachy/behavior/`face_sense.py` + reachy/vision/face.py + sense.py)

- covers: c11, h7, c12, h17
- acceptance:
  - Sense gains `face_bbox` (x,y,w,h normalised) and `face_age_s`; an unmatched detection still yields a bbox (name stays None) — test: single unknown face -> bbox present, name None
  - Selection test: larger unknown vs smaller known -> larger chosen; two within 15% area of each other, one known -> known chosen
  - No change under reachy/vision/`face_store.py`; existing `face_sense` tests green; uv run pytest -n auto green

### t4 — runtime: `lock_face` / `release_face` intent kinds + looping 'face-lock' behavior with its own yaw/pitch clamp (reachy/behavior/`face_lock.py`, registered at composition like goto in cli/`_commands`/behavior.py)

- depends on: t2
- covers: c37, h27, c15, h18, c34
- acceptance:
  - `lock_face` with no face -> {ok:false, error:'no face known'}; with a face -> {ok:true, op:'`lock_face`', locked:true} and the behavior follows `face_bbox` every tick, inhibiting feel-alive and orient-to-sound
  - Inhibition snapshot is later-wins: lock (set A) -> `set_inhibition` B -> release -> inhibitions == B; with no intervening call release restores A
  - `release_face` when not locked -> {ok:true, op:'`release_face`', released:false, note:'not locked'}; when locked -> released:true, previous inhibitions restored, one event
  - Clamp test: bbox at frame edge for 500 ticks -> commanded yaw/pitch never exceed the entry's clamp; tests/`test_behavior_face_lock.py`; uv run pytest -n auto green

### t5 — runtime: lock lifecycle events + lock cannot outlive the mind — 'face-lost' (after N s, lock persists), 'lock-released' registered in reachy/export/runtime.py raw-action tables; release on mind-offline (reachy/state/nova retained offline for a grace period) or max hold

- depends on: t4
- covers: c14, h8, c36, h26
- acceptance:
  - Events reach both the stdout feed and MQTT as reachy/events/intent/face-lost and intent/lock-released (`test_export_runtime` + `test_export_events` extended); an unregistered action still drops with `REASON_UNKNOWN_EVENT`
  - Test: locked, face absent for N s -> one face-lost event, still locked; mind offline for the grace period -> lock-released reason=mind-offline and inhibitions restored; max-hold elapsed -> lock-released reason=max-hold

### t6 — harness: rules.yaml gains voice (silent|brief|free, default free) and sense fields; quiet templates for rule/fire, rule/fire:pat-acknowledge, rule/fire:nova-face-noticed, pat/\*; bus renders the voice marker (config/nervous-system/rules.yaml + `reachy_nova`/harness/bus.py `route_event` + tests/`test_rules_coverage.py` + tests/`test_harness_bus.py`)

- covers: c2, h15, c3, h2, c4, h3, c40, h30
- acceptance:
  - Rendering test over every entry named above: inject text contains none of 'reflex', 'rule', 'reacted on its own', 'leaning', 'antenna'
  - Generic rule/fire is voice: silent and still produces an inject carrying the marker '(quiet: do not speak about this)'; brief renders '(react briefly if at all)'; free renders no marker; absent voice == free
  - `test_rules_coverage` accepts the new keys and asserts rule/fire is silent; pre-change text is recorded in the test docstring as the before-state (journal 2026-08-26)

### t7 — harness: sense-class dedupe in NovaBus between `route_event` and `_on_inject` (`reachy_nova`/harness/bus.py + tests/`test_harness_bus.py`)

- depends on: t6
- covers: c6, h4
- acceptance:
  - rule/fire:pat-acknowledge then rule/fire:nova-pat-cheer (both sense: pat) within the window -> exactly one inject; after the window a fresh inject
  - Two generic rule/fire events with different rule names and no sense field within the window -> both inject (dedupe by exact key/rule name); each suppressed duplicate logs one \[SENSE stage=inject event=dedupe\] line

### t8 — harness: recent-sense ring buffer + `recall_senses` tool (`reachy_nova`/harness/`sense_history.py`, wired at NovaBus.`_on_inject`; tool builder in tools.py; `EXPECTED_TOOLS`)

- depends on: t7
- covers: c7, h5
- acceptance:
  - Three routed senses then recall -> the three in order with monotonic timestamps and source/type; buffer is bounded (default 20)
  - `recall_senses` tool returns {ok:true, senses:\[...\]} and appears in `EXPECTED_TOOLS`; system prompt tells Nova to use it when asked why it moved/felt something

### t10 — harness: shared daemon HTTP client + `raise_voice`/`lower_voice`/`set_voice_level` tools + persisted volume (`reachy_nova`/harness/`daemon_client.py` new; speaking.py uses it; tools.py builders; statedir.py nova-volume.json; tests/`test_harness_daemon_client.py` + `test_harness_tools.py`)

- covers: c16, h9, c17, h19
- acceptance:
  - Stubbed client: raise from 95 -> 100 with {ok:true, volume:100, note:'at maximum'}; lower from 15 -> 10 with note 'at minimum'; step default 10; absolute form clamps to \[10,100\]; HTTP failure -> {ok:false, error}
  - Each change logs one \[SENSE stage=act source=nova event=volume\] old=N new=M line; last level persisted atomically to <state>/nova-volume.json and re-applied at start only if the daemon reports a different value (test with stub)
  - No file under reachy-mini-cli changes; `NOVA_VOCALIZE_GAIN` untouched

### t11 — harness: timed quiet — QuietState (deadline, later-wins extension, persistence to <state>/nova-quiet.json) + SonicSpeaker gate at `_play_one` with drop-not-failure semantics and post-acknowledgement arming + supervisor.status() `quiet_until` (`reachy_nova`/harness/quiet.py new; speaking.py; supervisor.py; tests/`test_harness_quiet.py` + `test_harness_speaking.py`)

- depends on: t10
- covers: c19, h10, c21, h11, c38, h28, c39, h29, c22, h20
- acceptance:
  - Deadline armed: three utterances enqueued -> poster never called, `on_playback_failure` never called, echo gate never armed, exactly one \[SENSE stage=speak event=quiet-drop\] line; on release one summary line with count=3; after expiry the next utterance plays; hearing.feed keeps being invoked throughout
  - Arming order: the first utterance after arm() is played then the gate closes; if no utterance arrives within the grace (2 s) the gate closes anyway
  - Restart with a future deadline on disk comes up silent; status() shows `quiet_until`; an expired file is ignored and removed; extension later-wins returns 'extended'

### t14 — cold-boot h12: chaos case 5 in tests/chaos/`ON_ROBOT.md` — `KIRO_CLI_BIN`=/nonexistent, restart reachy-nova-harness with `FORGE_WRITER`=kiro, assert 'started degraded (initial spawn failed' + 'started name=`kiro_session`' and no 'start failed'; restore and assert 'recovered' with NRestarts unchanged; run it on the robot and paste the journal evidence

- covers: c24, h12, c25, h21
- acceptance:
  - `ON_ROBOT.md` has case 5 with exact commands and expected lines; the delivery record cites the journal lines and states that the radio-off attempt was non-reproducing and the boot-ordering race remains unproven

### t1 — runtime: DoaPoller exposes last-good age; new one-shot 'look-at-sound' LibraryEntry (reachy/behavior/gaze.py + library.py + sense.py)

- depends on: t2
- covers: c10, h6
- acceptance:
  - DoaPoller.`age_s`() returns seconds since the last good DoA reading (None when never); a test drives the clock and asserts it
  - `run_behavior` look-at-sound with no reading or `age_s` > 8.0 returns {ok:false, error:'no recent sound direction'} and admits no behavior; with a fresh reading it admits a one-shot that aims yaw via orient's `doa_angle_to_yaw` within `max_yaw`, `default_duration` 2.0, StopClass chosen and documented in the entry docstring
  - behavior name 'look-at-sound' does not collide with the react rule id 'look-toward-sound' (`default_rules.toml`:149); tests in tests/`test_behavior_gaze.py`; uv run pytest -n auto green

### t3 — runtime: one-shot 'look-at-face' LibraryEntry aiming at Sense.`face_bbox` (reachy/behavior/gaze.py)

- depends on: t1, t2
- covers: h7
- acceptance:
  - `run_behavior` look-at-face with no `face_bbox` (or `face_age_s` > threshold) returns {ok:false, error:'no face known'}; with a bbox it admits a one-shot that maps bbox centre to yaw/pitch within the same clamps as look-at-sound
  - tests/`test_behavior_gaze.py` covers refusal + aim; uv run pytest -n auto green

### t9 — harness: system-prompt line (react naturally to body cues, never describe your own mechanism unless asked) + `lock_face`/`release_face` tools + gaze tool descriptions for look-at-sound/look-at-face (`reachy_nova`/harness/app.py `HARNESS_SYSTEM_PROMPT` + tools.py builders/`_BUILDERS`/`TOOL_SPECS` + tests/`test_harness_tools.py`)

- depends on: t8, t10
- covers: c34, h13, c8, h16, c41, h31
- acceptance:
  - `EXPECTED_TOOLS` includes `lock_face` and `release_face`; fake-engine round-trip returns the typed result verbatim; release of an inactive lock passes the runtime's named no-op through unchanged
  - An engine that lacks the kind returns {ok:false, error:"unknown kind '`lock_face`'"} passed through verbatim (skew test)
  - tests/`test_harness_boundary.py` green (no `reachy_mini` import, no `set_target`); `HARNESS_SYSTEM_PROMPT` no longer says 'mention what you did in a few words' for body cues

### t12 — harness: `stay_silent` / `end_silence` tools (bounded 1-180 min) that also inhibit the runtime 'speak' behavior for the window and mark injects with the quiet marker (tools.py builders + bus.py marker + tests/`test_harness_tools.py`)

- depends on: t11, t8, t9
- covers: c35, h14, c43, h32
- acceptance:
  - First call -> {ok:true, until:<ts>, note:'armed'} and one brief acknowledgement is allowed; second call with a longer duration -> 'extended'; `end_silence` with nothing armed -> {ok:true, note:'not silent'}; minutes outside 1-180 refuse without arming
  - Entering quiet spools `set_inhibition` including 'speak' (merged with the current set), leaving restores it; verified against the fake engine; a runtime whose say path does not go through 'speak' is recorded as a plan risk outcome in the task notes
  - While armed, every rendered inject carries '(quiet mode: do not speak)' regardless of the rule's voice level

### t13 — harness: lock awareness + new-event rules — heartbeat loss while locked logs 'lock released reason=engine-restart'; status() shows lock state read from state.json intents view; rules.yaml entries for intent/face-lost and intent/lock-released rendered as quiet context (`reachy_nova`/harness/app.py or supervisor.py; config/nervous-system/rules.yaml; tests)

- depends on: t12
- covers: c36, h26, c14, h8, c43, h32
- acceptance:
  - Test: engine heartbeat lost while the harness believes it is locked -> one named line and the next `lock_face` is not blocked by stale local state
  - `test_rules_coverage` green with the two new entries; their templates contain no mechanism words

### t15 — docs + version: docs/architecture.md §5.5 tool list and §4 seams (lock intents, quiet, volume), new docs/components/gaze.md and quiet-mode.md, rules.yaml header for voice/sense, deploy order (runtime first) noted; bump pyproject version (PyPI publish on main requires it)

- depends on: t13
- covers: c30, h22, c32, h24
- acceptance:
  - architecture.md tool vocabulary lists `lock_face`, `release_face`, `raise_voice`, `lower_voice`, `set_voice_level`, `stay_silent`, `end_silence`, `recall_senses`; markdownlint-cli2 clean on touched docs
  - pyproject.toml version bumped from 0.2.0; uv run pytest green

### t16 — live acceptance on the robot + delivery record: deploy runtime then harness; run every success-signal line (pat -> one reaction, no mechanism words; speak up/quieter; stay silent 2 min zero playbacks incl. engine say; look at me / look at the sound / keep looking at me / you can look away; recall 'why did you move?'); write docs/deliveries/2026-08-26-gaze-voice-quiet.md linking each 2026-08-26 pain point to its fix, with unit-only claims labelled

- depends on: t5, t14, t15
- covers: c1, h1, c31, h23, c33, h25
- acceptance:
  - Each success-signal line has journal evidence pasted in the delivery record; any line not run live is labelled unit-only with the reason
  - The delivery record links narrated reflexes, SSH volume, and no gaze/quiet control to the tasks that removed them and states the deploy order

## Risks

- [unknown_nonblocking] Whether runtime rule 'say' speech routes through the 'speak' library behavior (so `set_inhibition` \['speak'\] mutes it) — if not, quiet needs a runtime-side mute intent (new task in the runtime) (task t12)
- [unknown_nonblocking] StopClass arbitration for the one-shot gaze behaviors against an active orient-to-sound/feel-alive holding the head channel — chosen when the LibraryEntry is written; wrong choice shows as the one-shot never winning the head (task t1)
- [unknown_nonblocking] bbox freshness from the throttled FaceSenseDriver (500 ms, one-tick latch) may make face-lock step rather than track; measure live, tune cadence if needed (task t4)
- [unknown_nonblocking] Robot unreachable from spark at challenge time; live tasks (t14, t16) need LAN or tailnet reachability re-verified first (task t16)
- [follow_up] Daemon test sound on POST /api/volume/set is unsuppressible; a query flag belongs in `reachy_mini` (task t10)
- [unknown_nonblocking] Two repos, two PRs: runtime must merge and deploy before the harness or `lock_face` refuses with unknown-kind (typed, survivable) — merge order is an operator constraint (task t16)
