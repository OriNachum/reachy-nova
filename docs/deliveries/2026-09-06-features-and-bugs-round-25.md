# Delivery Summary — features-and-bugs-round-25

plan: `features-and-bugs-round-25` · run: `partial` · date: `2026-09-06`
baseline: `devague summary skeleton`

## Intent

Fix and improve four things from issue #25: browser results that the browser
found but Nova never spoke; a face hold while someone talks to her, released
when the conversation fades; answering to "Nova" as well as "Reachy"; and a
think posture while she browses. The scope pass root-caused the first as the
harness's own inject throttle, the think pass added a cold/warm attention
window, and the challenge pass added eyes on the camera, start/stop hygiene
and a gated tool surface. The run executed the 20-task plan
(`docs/plans/2026-09-06-features-and-bugs-round-25.md`, from spec
`docs/specs/2026-09-06-features-and-bugs-round-25.md`) through
`/assign-to-workforce`: ten waves, one agent per task in an isolated
worktree, every merge gated by the full suite before and after, the runtime
half deployed to the robot mid-run, the harness deployed at the end, a
review round on the PR folded back through the same gate, and a live half
with the operator at the desk after a device-level camera fix, which
produced three more harness fixes and one prerequisite for the runtime's
configurable-names work (#27).

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Must-deliver inject path in `nova_sonic.py`: `inject_text`(`must_deliver`=True, `sense_class`='browse') skips the 3 s throttle, still parks under its own DeferredCues class while speaking, and a small retry queue re-sends a dropped-inactive must-deliver text (with its age) once the next session is listening
- `t2` — `nova_browser.py`: progress narration collapses to one status cue per phase worded as status ('Status: your browser is working on X — no action needed'), rate-limited to one per 10 s, 'Done! Reading results...' becomes a log line; `queue_task` dedupes a normalised instruction against the running and queued tasks within 300 s and returns {ok, queued:false, duplicate:true}; `_act_on_agentcore` / `act_get` untouched
- `t3` — LockState gains an owner: `mark_locked`(owner='auto'|'model'), owner property, `mark_released` clears it; motion/lock-released and engine-drop clear owner with the belief
- `t4` — Persona names both names: config/persona/nova.md and harness/persona.py `DEFAULT_PERSONA` gain 'People call you Nova or Reachy; Reachy often reaches you as Richie or Reach — both are you', kept in the dry register with no character names
- `t5` — harness/attention.py: AttentionState — cold/warm with `NOVA_ATTENTION_WINDOW_S` (default 45 s, env, fail-open), a local restatement of the runtime's name match (nova, reachy, richie, reach, noah; difflib+length ratio, no network), `note_transcript`(text) opens on a name and renews when warm, `note_utterance`() and `note_inject`() renew, `conversation_live` (any transcript or utterance, for the gaze) vs warm (name-opened, for the voice), a `stay_silent` window reads as cold, one senselog line per open/close under stage=attention, rotation-agnostic
- `t6` — Voice gating and memory hygiene: speaking.py takes an AttentionState and, at the same single place the quiet gate lives, drops an utterance whose Sonic 'speaking' edge followed a nameless transcript while cold (an utterance within 3 s of an inject still plays) with one senselog line and no daemon post; ledger.py skips appends for a gate-dropped reply the way it skips quiet-window appends; the verdict is taken on the speaking edge, before the first chunk is enqueued, with a preempt fallback if audio precedes the transcript
- `t7` — tools.py: THINK alias tool (`run_behavior` name='thoughtful', side 'left'|'right' -> yaw sign, alternating by default, duration 3); `_browse` passes NovaBrowser's typed duplicate result through; `lock_face`/`release_face` mark LockState owner 'model'; IntentTools.execute refuses effectful tools with ToolRefused('not addressed') while AttentionState is cold and the last transcript was nameless (browse, forge, `use_skill`, `author_rule`, goto, `run_behavior`, `declare_goal`, `set_mode`, `set_inhibition`, `create_rule`, `enroll_face`, `lock_face`), leaving `recall_senses`, `stay_silent`, `end_silence`, voice-level tools and `release_face` allowed
- `t8` — harness/`gaze_stack.py` core: a single-writer posture layer — producers (transcript, speaking edge, speaker idle, browser state) set flags under a lock, one worker thread computes the top layer (wander < browsing < conversation) and issues only transition intents serially through IntentTools, waiting out each await; browsing = `declare_goal` gaze-hold {pitch:10, yaw:±15, alternating} on busy and `declare_goal` None on idle/error; a `clear_for_result`() hook the app calls before a browse result inject; every transition logged under stage=gaze
- `t9` — `gaze_stack` conversation layer: on `conversation_live` rising submit `look_at_sound` then `lock_face` (owner 'auto'), retry a refused lock at 3, 6, 12, 24, 30, 30... s while live, treat ok:null as unknown and keep retrying, log one 'no face known' summary per conversation, hold through replies and gaps, submit `release_face` on conversation end only when owner is 'auto' (a model lock is left standing), leave a standing browsing goal in place under the lock and let it resume on release; on start and on `stop_event` submit `release_face` + `declare_goal` None once (idempotent)
- `t10` — Head reflexes kept off a held head: the browsing layer submits `set_inhibition` merged with the current inhibited set to add orient-to-sound and nod (restored on exit, later-wins respected), and `rules_overlay` gains `retire_rule`(id) writing an enabled=false tombstone inside the nova-managed block; app start retires nova-face-noticed once
- `t11` — harness/eyes.py: samples reachy/events/sense/snapshot at about 1 Hz on its own small MQTT subscription, latches one '\[SENSE stage=vision source=runtime event=frames\] dropped reason=no-frames' line after 60 s of `frame_available`=false and one 'restored' line when frames return; supervisor.status() gains eyes: live|dead|unknown
- `t12` — Integration in app.py + switches: `NOVA_FACE_HOLD`, `NOVA_THINK_POSTURE`, `NOVA_ATTENTION_GATE` in harness/switches.py (default on, fail-open, in the start-up line) and .env.sample; `build_app` wires AttentionState into `_on_transcript` / the inject wrapper / speaker, the gaze stack into transcript, speaking edge, speaker idle and NovaBrowser.`on_state_change`, eyes into the supervisor, the browse result through `clear_for_result`() then the must-deliver inject, and retires nova-face-noticed at start; each leg degrades to a named absent line when its switch is off
- `t13` — Docs and version: docs/components/attention.md (new), gaze.md (gaze stack, layers, think alias, owner), `nova_browser.md` (status cues, dedupe, must-deliver), architecture.md and CLAUDE.md module map entries, .env.sample knobs; pyproject.toml version 0.5.0
- `t14` — Boundary audit: uv run pytest -n auto green; tests/`test_harness_boundary.py` green; grep for gestures, `wake_word`, `face_recognition` imports under `reachy_nova`/harness/ empty; git diff of `nova_browser.py` confirms the act path untouched; runtime diff confirms only `select_face` and names changed
- `t15` — reachy-mini-cli names: add 'nova' to `DEFAULT_NAMES` in reachy/speech/engagement.py and reachy/behavior/`transcript_sense.py`, to `is_name_match` defaults in reachy/speech/`name_match.py`, and accept 'hey nova' as the sleep wake phrase default; extend tests/`test_name_match.py`'s collision table with now, no, know, nah, novel, November, nowhere and a four-letter floor for the n-family
- `t16` — reachy-mini-cli `select_face`: recognised-first, biggest among equals (issue #175); no other constant in `face_lock.py`, gaze.py or `face_sense.py` changes
- `t17` — Runtime deploy: merge origin/main into wireless-motor-enable on spark with t15 + t16, tests green, push; on the robot ff-pull, detached pip install with disk headroom checked, restart reachy-runtime; open the reachy-mini-cli PR to main
- `t18` — Harness deploy: first switch the robot's ~/git/reachy-nova from spec/fast-witty-remembering-nova to main (v0.4.0) and restart, then check out the round's branch, restart, and read the switches line and the persona line from the journal
- `t19` — Camera gate before item-2 acceptance: sample reachy/events/sense/snapshot for 10 s and record `frame_available` counts; if false, follow reachy-mini-cli #176 (daemon/media recovery) and record every item-2 live check as blocked; if true, run `look_at_face` and `lock_face` once with Ori in view and record bbox/name presence
- `t20` — Live acceptance and delivery doc: run the c23 script on the robot (browse result spoken once; duplicate request runs one session; head turns to the voice then locks and releases on fade with lock lines in both journals; five cold 'Nova, ...' and five 'Reachy, ...' answered outside any quiet window; browsing posture with antennas alive, conversation takes the head; no-frames line present within 90 s of start while the camera is dead), with Ori and a guest; write docs/deliveries/2026-09-06-features-and-bugs-round-25.md quoting journal lines per check and marking unverified or blocked honestly

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `inject_text(must_deliver=True)` — throttle-exempt, own deferred class, bounded restart queue; `1faa060` → merged `686f1b7`; hardened in the review round (`89982ff` → `7f49f30`): failed or stale sends re-queue, five attempts then `must-deliver-exhausted` |
| `t2` | delivered | status-worded progress cues (one per phase, 10 s), "Done" as a log line, `queue_task` dedupe with a typed result (#8); `06c5a14` → `a2fb4f9`; review: dedupe under one lock, the dashboard API returns the typed result (`7c35fa9` → `e958bb1`) |
| `t3` | delivered | `LockState` owner auto/model; `0acda0d` → `9345dae` |
| `t4` | delivered | persona names Nova, Reachy, Richie, Reach in file and embedded default; live fix `e3441fc`/`ed198f5` (`d3`): never speak tool names or write parentheses (1796 chars on the device) |
| `t5` | delivered | `harness/attention.py`, 59 tests; `a09cb02` → `ba625db`; live fix `7dfdba6` → `61bde6f` (`d5`): `note_utterance` renews but never opens `conversation_live` |
| `t6` | delivered | speaker verdict on the speaking edge, late-transcript preempt, `Ledger.append(dropped=)`; `dfbcddd` → `04d82b5` |
| `t7` | delivered | `think` alias, typed duplicate passthrough, model-owned locks, cold refusal of effectful tools; `49e2596` → `49979fb` |
| `t8` | delivered | `harness/gaze_stack.py` core; `cb75099` → `5d92414`; review: start hygiene moved onto the worker (`0d6c577` → `6619e82`) |
| `t9` | delivered | conversation layer with backoff, owner, fade release, start/stop hygiene; `cf04c47` → `6a9d652`; live fix `61bde6f` (`d5`): the layer opens only on someone else speaking, and `antenna-sway` (amp 10, period 5, re-issued) keeps the antennas alive under the automatic lock (`lock_liveness`) |
| `t10` | delivered | browsing inhibits merged with the live set, `retire_rule` tombstone; `253c843` → `20699b6`; review: stop hygiene also restores the inhibits; live: `motion/face-lost` demoted to `voice: none` (`d3`) |
| `t11` | delivered | `harness/eyes.py` + `eyes` in status; `f286b63` → `47ee560`; review: first-live line, fail-open broker parsing; live: the no-frames / restored pair logged across the camera fix and across a runtime restart |
| `t12` | delivered | three switches, all wiring; `f21cbaa` → `8f6d724`; main-agent follow-up `8827b56` (install the face-nod rule only when the hold is off); review: `conversation_enabled=switches.face_hold` |
| `t13` | delivered | docs (attention, gaze, eyes, browser, sonic, architecture, CLAUDE.md), version 0.5.0; `0334397` → `d5f044b`; plus `67d5d75` (names-table paragraph in attention.md, architecture 6.3) |
| `t14` | delivered | audit at `8827b56` and re-run at `6619e82`: boundary 19, no legacy imports, act path unchanged, daemon paths unchanged; suite 2096 at `67d5d75` |
| `t15` | partial | "nova" reached the runtime's name lists in the first form of reachy-mini-cli PR #178 (`a8535a2` → `b3fd068`), then the operator ruled on #175 that the runtime hardcodes no peer's name: the commit was dropped from #178 and reverted on the device branch. The runtime agent now builds a configurable `names` overlay table plus a `name_mentioned` sense field in reachy-mini-cli #177; its prerequisite on this repo (#27) landed as `67d5d75` — the overlay validator accepts both (`d6`). The harness's own `names = ["nova"]` write waits for #177 on the device |
| `t16` | delivered | `select_face` recognised-first (#175); `70e1905` → `c8e2575`; merged upstream as reachy-mini-cli 0.52.0 (#178) |
| `t17` | delivered | v0.52.0 (`ca64951`), reachy-mini-cli PR #178 (merged to main `f2515e0`), device branch `74629d3` → `cb1ab7c` (0.52.1 with #180's detection knobs), robot restarted — without the pip reinstall (`d1`); the device-branch merge and #180 predate the every-agent-owns-its-repo rule and are the runtime agent's to keep or drop |
| `t18` | delivered | robot moved to main (`19dde0c`, 08:25Z), then the round branch (`d5f044b` 08:31Z → `6619e82` → `ed198f5` → `61bde6f` 10:18Z → `67d5d75` 10:43Z); switches and persona lines read from the journal at every step |
| `t19` | delivered | camera sampled at 08:31:38Z: 0 of 501 snapshots carried a frame; reachy-mini-cli #176 filed. Then root-caused and fixed on the device (see Mid-work Decisions): every snapshot has carried a frame since 09:48Z (69/69, 67/67, 230/230, 191/191), eyes watch logged `restored after=251s` |
| `t20` | partial | item 2 proven live with Ori in view: 10:48–10:53 local lock 3.3 s after entry, held 42 s, released on fade, re-locked in 0.3 s; 11:30:33 local "noah look at me" → window opened by name, `look_at_sound`, lock confirmed on the first attempt 0.2 s later, antennas swaying under the lock, released on fade; cold nameless speech ("oh", "hello hello") dropped with no audio. Operator verdicts: "look at me works, but very slow, and not my face on center", "It feels rigid now. No liveness." (fixed, `d5`), "It follows me in a delay", "not focusing on me on center of the camera" — both remaining verdicts are the runtime's aim (reachy-mini-cli #181) and detection cadence (#179/#180). Still not run: the spoken browse (once, then deduped), the five-by-five name count, the browsing posture, a guest, the tone verdict |

## Mid-work Decisions

All seven deviation records are **proposed** (`--origin llm`) and await Ori's
confirmation; quoted as recorded.

- `d1` — t17 deployed the runtime without the pip reinstall the acceptance
  criterion names — the device runs an editable install, the root disk is
  91 % full with 1.3 GB free, pip on the CM4 takes minutes; only the
  installed metadata version lags
- `d2` — the PR #26 review round: nine Qodo findings fixed in three worktrees
  plus five Sonar new-code fixes, one pushback (the must-deliver queue is
  Sonic's own bounded parking, the same shape as `DeferredCues`)
- `d3` — t20's live acceptance surfaced two harness fixes applied after the
  review round: replies opened with "(look-at-face)" / "release_face" spoken
  as words (the persona now forbids spoken tool names and self-written
  parentheses, both copies, test-pinned), and the runtime's face-lost notice
  fired every 20–40 s at the ~7 Hz tick, each one a spoken "they wandered
  off" while the person sat there (`motion/face-lost` demoted to
  `voice: none` until reachy-mini-cli #179)
- `d4` — recorded when the runtime's names commit was found reverted
  upstream; its "not delivered" reading is superseded by `d6` (the change
  moved, it was not dropped)
- `d5` — Ori: "It feels rigid now. No liveness." Root cause: Nova's own
  opening line counted as a conversation, so the automatic lock (which
  inhibits the runtime's feel-alive) was held almost always. The conversation
  layer now opens only on someone speaking (Nova's own voice renews but
  never opens), and the antennas sway (amp 10, period 5, re-issued every
  50 s) while the automatic lock is held — commit `7dfdba6` → `61bde6f`
- `d6` — t15's runtime name-list change (c16) moved to a configurable path:
  reachy-mini-cli #177 (the runtime agent) adds a top-level `names` overlay
  table and a `name_mentioned` sense field; the harness mirrors both in
  `rules_overlay.py` (issue #27, commit `67d5d75`) and will write
  `names = ["nova"]` into the operator head once #177 is on the device.
  Reason: operator decision on reachy-mini-cli #175 (the runtime hardcodes no
  peer's name), and "every agent manages their own repo" — the interface
  between the two repos is issues
- `d7` — the face-detection load fix on the device is a systemd drop-in over
  reachy-mini-cli #180's knobs (`REACHY_FACE_DETECT_INTERVAL=1.0`,
  `REACHY_FACE_DETECT_MAX_WIDTH=640`): 0.5 s on the 640-px frame measured a
  9.9 Hz tick, 41 overruns/min and engine-heartbeat flaps every ~10 s;
  1.0 s measures 27.7 Hz, 11 overruns/min, no flaps. #180 and the device-branch
  merge `cb1ab7c` predate the every-agent-owns-its-repo rule and are the
  runtime agent's to keep or drop
- not covered by a record — the dead camera had two stacked causes, both
  outside this repo and both fixed on the device (`s27`, `s28`,
  reachy-mini-cli #176): WirePlumber's libcamera/v4l2 monitors held the whole
  imx708 pipeline since boot (disabled via
  `~/.config/wireplumber/wireplumber.conf.d/99-reachy-no-camera.conf`), and
  an operator-local `reachy agent embody` unit on the workstation,
  crash-looping with no USB robot attached, reconnected to the wireless
  daemon every ~6 s and made it release its media each time (stopped and
  disabled on the workstation)
- not covered by a record — the two remaining live verdicts are runtime
  behaviour, filed rather than patched: the face-lock's aim is open-loop
  (an absolute target of 20° per half-frame of box offset against an 87°
  field of view, so the head settles at about 0.31 of the face angle and
  chases one detection cycle at a time) — reachy-mini-cli #181 with the
  maths; and the 1.5 s box TTL no longer spans two detections at a 1.0 s
  interval, so a single missed detection blanks the face (`b8`)
- not covered by a record: wave 2's harness tasks started as soon as their
  own dependencies (t2, t3, t5) had merged, before the runtime tasks of
  wave 1 finished — the dependency graph was respected, the wave boundary
  was not waited for
- not covered by a record (filed as delta `b1`): t12 installed the face-nod
  rule and immediately tombstoned it on every boot; `8827b56` makes the two
  steps exclusive on `NOVA_FACE_HOLD`
- not covered by a record (deltas `b2`–`b5`): Nova's own opening line at a
  session start counted as conversation (since fixed by `d5`); an
  unconfirmed `release_face` clears the stack's `lock_held` without touching
  `LockState`; the "Navigating to" and "cloud session" progress phases are
  log lines only; an out-of-process `status` reports `eyes: unknown`
- the runtime's `CHANGELOG` bump script left an empty 0.52.0 section above
  t15's Unreleased entry; folded by hand in `ca64951`
- the Sonar finding S3415 (assertion argument order in
  `tests/test_harness_speaking_attention.py`) is left open: every equality
  in that file is actual-first, so it reads as a false positive for the
  operator to accept in the Sonar UI

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t17` (`d1`) | the device runs an editable install so the checkout's code is what runs; the root disk is 91 % full with 1.3 GB free and pip on the CM4 takes minutes; the only effect is the installed metadata version reading 0.51.1 while the code is 0.52.x | acceptable |
| `t14` (`d2`) | human gate 3 produced ten valid Qodo findings the plan's tests had not exercised; each fix is small, covered by new tests (suite 2041 → 2070) and merged through the same TDD gate | acceptable |
| `t12` | the plan's t12 retired the face-nod rule but kept the existing install step, so every boot wrote and tombstoned the same rule; fixed on the round branch after t12 merged | acceptable |
| `t19` | the camera gate found no frames (two device-level causes, fixed on the device during the round, reachy-mini-cli #176); item 2 was then proven live | acceptable |
| `t20` (`d3`) | two behaviours only visible with the camera live and a person at the desk: spoken tool names, and a face-lost notice every 20–40 s | acceptable |
| `t9` (`d5`) | the plan's c7 counted Nova speaking as conversation, so the automatic lock held almost always and inhibited the runtime's feel-alive ("rigid, no liveness"); the layer now opens only on someone else speaking, and the antennas sway under the automatic lock | acceptable |
| `t15` (`d6`) | the operator ruled the runtime hardcodes no peer's name; the change moved to a configurable names table the runtime agent builds in reachy-mini-cli #177, with this repo's overlay validator as the prerequisite (#27, `67d5d75`); the `names = ["nova"]` write waits for #177 on the device | needs-follow-up |
| `t20` (`d7`) | the CM4 tick fell to ~7 Hz under full-frame detection every 0.5 s; the fix is a device drop-in over reachy-mini-cli #180's knobs, settled at 1.0 s / 640 px after measuring both | acceptable |
| `t20` | the spoken browse, the five-by-five name count, the browsing posture, a guest and the tone verdict are still unmeasured; the head's aim (settles a third of the way, #181) and tracking cadence (#179/#180, box TTL) are runtime-side | needs-follow-up |

## Evidence

- tests: `uv run pytest -n auto` at `67d5d75` — 2096 passed, 1 skipped (baseline before the round: 1810; after the plan's waves at `d5f044b`: 2041; after the review round at `6619e82`: 2070; after the live fixes at `61bde6f`: 2083)
- tests: `tests/test_harness_boundary.py` — 19 passed at `8827b56` and `6619e82`
- tests: per-obligation node ids filed as plan evidence `e1`–`e44` against obligations `o1`–`o28` (`devague evidence --list`), all proposed
- runtime tests: reachy-mini-cli `uv run pytest -n auto` at `b3fd068` — 5668 passed, 7 skipped, 1 failed (`tests/test_vision_scene_integration.py::test_integration_scene_default_model_resolves_via_senses_role`, a live-gateway test that fails identically on `origin/main`)
- audit: `scratchpad/audit-t14.txt` (session) — no legacy imports under `reachy_nova/harness/`, `nova_browser.py` act path byte-unchanged, daemon paths `/api/media/{play_sound,sounds,sounds/upload,stop_sound}` and `/api/volume/{current,set}` only
- lint: `markdownlint-cli2` on the touched docs — 0 new findings (pre-existing table-row MD013 in `gaze.md`/`attention.md`, list-style findings in `nova_browser.md`)
- commits: `a144528..67d5d75` on `spec/features-and-bugs-round-25` (spec+plan, 20 task merges, 1 follow-up, 5 Sonar fixes, 3 review-fix merges, 3 live-fix commits, 1 liveness merge, 1 prerequisite for #177)
- runtime commits: `c8e2575..ca64951` on `feat/reachy-nova-25-names-select-face` (PR #178, merged as 0.52.0); device branch `wireless-motor-enable` @ `cb1ab7c` (0.52.1, #180)
- PRs / issues: OriNachum/reachy-nova#26 (this round), #25 (closed by it), #8 (closed by it), #27 (filed by the runtime agent, landed `67d5d75`); agentculture/reachy-mini-cli#178 (runtime PR, merged), #175 (enrolled-face preference), #176 (camera pipeline failure, two causes), #177 (configurable names, the runtime agent's), #179 (detect only in the still period), #180 (detection knobs), #181 (open-loop aim)
- robot journal, harness pid 15925 (round branch `d5f044b`, 09:31 local): `switches ... face_hold=on think_posture=on attention_gate=on`; `persona ... chars=1615`; `face-nod retired id=nova-face-noticed changed=True verdict=reload confirmed`; `start-hygiene release ok=true` / `start-hygiene clear goal ok=true`; `layer wander -> conversation` at 09:31:17, `look at sound ok=true`, `lock face attempt=1..4 ... error=no face known`, `layer conversation -> wander` at 09:32:03 with `lock never held: refusals=4`; `[SENSE stage=vision source=runtime event=frames] dropped reason=no-frames after=60s` at 09:32:15
- robot journal, runtime pid 12726 (v0.52.0): `release_face ... 'released': False`, `declare_goal ... 'goal': None`, `run_behavior ... look-at-sound`, `lock_face ... dropped reason=no face known` at 09:31:17/21/27/39
- robot journals 10:48:57–10:53:26 local (harness pid 17745, runtime): `look_at_face` confirmed; `layer wander -> conversation`; `lock_face` refused then `confirmed {"ok":true,"locked":true,"id":"face-lock:lock:1"}` (`locked after=3.3s attempts=2`); `release_face ... reason=requested` at fade; `locked after=0.3s attempts=1` on the next utterance; seven `motion/face-lost` injects; `heard 'nova, look at me.'` → `event=window] opened by=nova`; two `dropped reason=not-addressed`
- robot journals 11:26–11:35 local (harness pid 24999 @ `61bde6f`, runtime 0.52.1 with detect 1.0 s / 640 px): `locked after=45.9s attempts=5` then `liveness sway ok=true` at 11:26:41 (runtime: `antenna-sway` applied on the antennas channel under `face-lock:lock:2`, `inhibited: ['feel-alive', 'orient-to-sound']`); `heard 'noah look at me'` → `event=window] opened by=noah` → `look at sound ok=true` → `lock face attempt=1 reason=enter ok=true` → `locked after=0.2s attempts=1` → `liveness sway ok=true`, all at 11:30:33; `motion/face-lost` `absent_s=3.019` and `absent_s=3.026` at 11:31; `heard 'oh'` at 11:34:26 → `dropped reason=not-addressed` (`utterance dropped count=5`); no `event=window] opened` attributed to Nova's own voice since `61bde6f` (`e40`, `e41`)
- robot journal 11:43:30 local (harness @ `67d5d75`): switches line unchanged, `persona ... chars=1796`, `face-nod retired ... changed=False`, start hygiene ok (`e44`)
- MQTT samples on the robot: 08:31:38Z 501 snapshots, 0 with a frame; after the camera fix 69/69 (09:48Z), 67/67, 230/230 (10:27Z, 23.0 Hz, `face` named in 1), 191/191 (10:29Z); tick 9.9 Hz at detect 0.5 s with 41 overruns/min and heartbeat flaps every ~10 s (10:34Z); 27.7 Hz at 1.0 s with 11 overruns/min and no flaps (10:36Z) (`e42`)
- camera intrinsics from the daemon (`GET /api/camera/specs`): K fx ≈ 2002, cx ≈ 1906 on the 3840-wide sensor → ≈ 87° horizontal field of view; against the face-lock's 20° yaw gain per half-frame the aim settles at ≈ 0.31 of the face angle (#181)
- device overlay `~/.local/state/reachy/behavior/rules.toml`: `id = "nova-face-noticed"` / `enabled = false` in the nova-managed block; device drop-in `~/.config/systemd/user/reachy-runtime.service.d/override.conf`: `REACHY_FACE_DETECT_INTERVAL=1.0`, `REACHY_FACE_DETECT_MAX_WIDTH=640`
- deviations: `devague deviate --list` — `d1`–`d7`, all proposed; deltas `b1`–`b10`, obligations `o1`–`o28`, evidence `e1`–`e44`, all proposed
- PR #26 review: 10 Qodo threads, 10 answered, 10 resolved; Sonar quality gate passed, 1 open finding (S3415) left for the operator; CI green at `61bde6f`, running at `67d5d75`

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| a browse result is never dropped by the inject throttle, and survives a failed or stale send | high | `tests/test_sonic_inject_must_deliver.py` (33 tests) · commits `1faa060`, `89982ff` |
| a browse result is spoken once, within about 5 s of the browser finishing, on the robot | unverified | no spoken browse has been run since the deploy — not claimed done |
| a repeated browse request runs one hosted session | medium | `tests/test_nova_browser_dedupe.py` incl. the 20-thread race · not exercised live |
| the persona answers to Nova and Reachy, and never speaks its tool names | medium | `tests/test_harness_persona.py` · journal `persona ... chars=1796` · "nova, look at me" and "noah look at me" both opened the window · the spoken check is the operator's |
| cold, nameless speech gets no reply; a name opens a 45 s window | high | journal 10:44–10:56 and 11:30–11:34: `dropped reason=not-addressed` ×5, `opened by=nova`, `opened by=noah` · `e39`, `e41` · the unit suites |
| effectful tools are refused while cold and nameless | high | `tests/test_harness_tools.py` cold-refusal tests |
| a conversation turns the head toward the voice and retries the face lock with backoff, one summary per conversation | high | journal 09:31:17–09:32:03 (four refusals at +0/+3/+9/+22 s, fade at +46 s) · `tests/test_harness_gaze_stack_conversation.py` |
| the face lock is held through a conversation and released on fade | high | journal 10:48:58–10:49:45: lock 3.3 s after entry, held 42 s, released on fade, re-locked in 0.3 s · 11:30:33: locked on the first attempt 0.2 s after the name · `e36`, `e40` |
| Nova's own voice never opens a conversation, and the antennas stay alive under the automatic lock | high | `61bde6f` tests · journal `liveness sway ok=true` at 11:26:41 and 11:30:33 with `antenna-sway` applied by the runtime under the lock · no window opened by Nova's own line since 10:18Z · `e40` |
| the head lands on the face quickly and centred | unverified | Ori: "It follows me in a delay", "not focusing on me on center of the camera" — the runtime's open-loop aim (settles at ≈ 0.31 of the face angle, reachy-mini-cli #181) and detection cadence (#179/#180) — not claimed done |
| the head looks up and aside while browsing, antennas alive, and yields to a conversation | medium | `tests/test_harness_gaze_stack.py`, `tests/test_harness_gaze_stack_inhibit.py` · no browse has run live since the deploy |
| the face-nod rule is retired on the device | high | overlay read over ssh · journal `face-nod retired ... reload confirmed` |
| the harness names a dead camera and its return | high | journal `dropped reason=no-frames after=60s` at 09:32:15, `restored after=251.41s` at 10:47:54, and the same pair across the runtime restart at 11:13:39 / 11:16:46 · `tests/test_harness_eyes.py` |
| the harness releases what it holds on start and stop | high | journal `start-hygiene` lines applied by the runtime at every restart · stop tests |
| every new behaviour has a fail-open switch named at start | high | journal switches line at `67d5d75` (`e44`) · `tests/test_harness_switches.py` |
| the harness still opens no SDK client and touches no new daemon path | high | `tests/test_harness_boundary.py` · audit at `6619e82` |
| the runtime prefers an enrolled face | high | reachy-mini-cli face-selection suite · PR #178 merged as 0.52.0 · robot on `cb1ab7c` |
| the runtime answers to "nova" | unverified | the change moved to reachy-mini-cli #177 (in build by the runtime agent); the current runtime does not — not claimed done |
| the overlay validator accepts the runtime's names table and a `name_mentioned` predicate, and refuses a malformed table untouched | high | `tests/test_harness_rules_overlay.py` (13 tests) at `67d5d75` · `e43` · bounds pinned to the runtime's `rules.py` on its #177 branch |
| version 0.5.0 with a green suite | high | `pyproject.toml` · 2096 passed at `67d5d75` |
| it feels right to talk to (tone and behaviour) | unverified | the operator's verdict is pending — not claimed done |

## Remaining Work / Follow-up

- `t20` — the operator's live acceptance on `67d5d75`: five cold nameless sentences (expect silence and `dropped reason=not-addressed` lines), "Nova, how are you" then a nameless follow-up (both answered), the same with "Reachy", a spoken browse (posture up, result spoken once) repeated within a minute (one session), a guest by name, and the tone verdict; then update the unverified rows above. Owner: Ori + operator.
- item 2's quality — the hold works and lands on the first attempt; the head is slow to arrive and settles off-centre: reachy-mini-cli #181 (closed-loop aim through the camera FOV), #179 (detect only in the still grace period), #180 (detection interval / frame width; the 1.5 s box TTL should track the interval, `b8`). Runtime work, the runtime agent's
- the runtime's "nova" name — reachy-mini-cli #177 (the runtime agent); once it is on the device branch, this harness writes `names = ["nova"]` into the overlay's operator head and submits a reload at start (same mechanics as the face-nod tombstone), behind `NOVA_ATTENTION_GATE`; c16 is delivered when that round-trips on the robot
- the camera's two causes need durable fixes outside this repo: the WirePlumber config belongs in the OS image or the runtime's `service` setup, and the daemon should not let a second SDK client release media held by the runtime (both on reachy-mini-cli #176)
- confirm or reject deviations `d1`–`d7`, obligations `o1`–`o28`, evidence `e1`–`e44`, deltas `b1`–`b10` (all proposed)
- Sonar S3415 on `tests/test_harness_speaking_attention.py:150` — accept as a false positive in the Sonar UI, or ask for the two `in` assertions to be reshaped
- PR #26 is open with all ten review threads resolved; the operator wants everything right on the robot before it merges; merging publishes 0.5.0 to PyPI, after which the robot's checkout goes back to `main`
- follow-ups noted, not built: a trailing-name grace for fragmented transcripts (park `v4`); multi-angle enrolment (park `v5`, #127); a wandering "thinking" runtime library entry instead of the static gaze-hold (decision `c19`); the runtime's lock still inhibits feel-alive wholesale (the antenna sway is the harness's workaround, `d5`); observations: the cognition feed logs every assistant line twice (pre-existing), one ASR-hallucinated "refusal" transcript (10:49:59), memory replay carrying odd topics
