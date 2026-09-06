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
half deployed to the robot mid-run, the harness deployed at the end, and a
review round on the PR folded back through the same gate.

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
| `t4` | delivered | persona names Nova, Reachy, Richie, Reach in file and embedded default (1615 chars); `6177dcb` → `c870399` |
| `t5` | delivered | `harness/attention.py`, 59 tests; `a09cb02` → `ba625db` |
| `t6` | delivered | speaker verdict on the speaking edge, late-transcript preempt, `Ledger.append(dropped=)`; `dfbcddd` → `04d82b5` |
| `t7` | delivered | `think` alias, typed duplicate passthrough, model-owned locks, cold refusal of effectful tools; `49e2596` → `49979fb` |
| `t8` | delivered | `harness/gaze_stack.py` core; `cb75099` → `5d92414`; review: start hygiene moved onto the worker (`0d6c577` → `6619e82`) |
| `t9` | delivered | conversation layer with backoff, owner, fade release, start/stop hygiene; `cf04c47` → `6a9d652` |
| `t10` | delivered | browsing inhibits merged with the live set, `retire_rule` tombstone; `253c843` → `20699b6`; review: stop hygiene also restores the inhibits |
| `t11` | delivered | `harness/eyes.py` + `eyes` in status; `f286b63` → `47ee560`; review: first-live line, fail-open broker parsing |
| `t12` | delivered | three switches, all wiring; `f21cbaa` → `8f6d724`; main-agent follow-up `8827b56` (install the face-nod rule only when the hold is off); review: `conversation_enabled=switches.face_hold` |
| `t13` | delivered | docs (attention, gaze, eyes, browser, sonic, architecture, CLAUDE.md), version 0.5.0; `0334397` → `d5f044b` |
| `t14` | delivered | audit at `8827b56` and re-run at `6619e82`: suite 2070 passed, boundary 19, no legacy imports, act path unchanged, daemon paths unchanged |
| `t15` | delivered | reachy-mini-cli names gain "nova" with n-family guards, wake phrase accepts both; `a8535a2` → `b3fd068` |
| `t16` | delivered | `select_face` recognised-first (#175); `70e1905` → `c8e2575` |
| `t17` | delivered | v0.52.0 (`ca64951`), reachy-mini-cli PR #178, device branch `74629d3`, robot restarted 07:49:56Z — without the pip reinstall (`d1`) |
| `t18` | delivered | robot moved to main (`19dde0c`, 08:25Z), then the round branch (`d5f044b` 08:31Z, `6619e82` 08:56Z); switches and persona lines read from the journal |
| `t19` | delivered | camera sampled at 08:31:38Z: 0 of 501 snapshots carried a frame; reachy-mini-cli #176 filed; item-2 live checks recorded as blocked |
| `t20` | partial | live evidence collected from the journals for start-up, the gaze layer transitions and lock retries, the tombstone, the hygiene and the eyes latch; the operator's spoken acceptance (browse spoken once and deduped, "Nova" from cold, cold nameless silence, browsing posture, a guest, the tone verdict) has not happened yet; this document is the partial record |

## Mid-work Decisions

Both deviation records are **proposed** (`--origin llm`) and await Ori's
confirmation; quoted as recorded.

- `d1` — t17 deployed the runtime without the pip reinstall the acceptance
  criterion names — the device runs an editable install, the root disk is
  91 % full with 1.3 GB free, pip on the CM4 takes minutes; only the
  installed metadata version lags (0.51.1 vs code 0.52.0)
- `d2` — the PR #26 review round: nine Qodo findings fixed in three worktrees
  plus five Sonar new-code fixes, one pushback (the must-deliver queue is
  Sonic's own bounded parking, the same shape as `DeferredCues`)
- not covered by a record: wave 2's harness tasks started as soon as their
  own dependencies (t2, t3, t5) had merged, before the runtime tasks of
  wave 1 finished — the dependency graph was respected, the wave boundary
  was not waited for
- not covered by a record (filed as delta `b1`): t12 installed the face-nod
  rule and immediately tombstoned it on every boot; `8827b56` makes the two
  steps exclusive on `NOVA_FACE_HOLD`
- not covered by a record (deltas `b2`–`b5`): Nova's own opening line at a
  session start counts as conversation, so the gaze stack glances toward
  the last sound and tries the lock at every boot; an unconfirmed
  `release_face` clears the stack's `lock_held` without touching
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
| `t17` (`d1`) | the device runs an editable install so the checkout's code is what runs; the root disk is 91 % full with 1.3 GB free and pip on the CM4 takes minutes; the only effect is the installed metadata version reading 0.51.1 while the code is 0.52.0 | acceptable |
| `t14` (`d2`) | human gate 3 produced ten valid Qodo findings the plan's tests had not exercised; each fix is small, covered by new tests (suite 2041 → 2070) and merged through the same TDD gate | acceptable |
| `t12` | the plan's t12 retired the face-nod rule but kept the existing install step, so every boot wrote and tombstoned the same rule; fixed on the round branch after t12 merged | acceptable |
| `t19` / `t20` | the robot's camera receive pipeline fails at media acquisition (GStreamer "state change failed"), so no face lock can be held live; the hold-and-release half of item 2 is verified by unit tests only until reachy-mini-cli #176 is fixed | needs-follow-up |
| `t20` | the operator's spoken acceptance has not been run; the tone verdict and the spoken-browse, cold-silence and "Nova"-from-cold checks are unmeasured | needs-follow-up |

## Evidence

- tests: `uv run pytest -n auto` at `6619e82` — 2070 passed, 1 skipped (baseline before the round: 1810; after the plan's waves at `d5f044b`: 2041)
- tests: `tests/test_harness_boundary.py` — 19 passed at `8827b56` and `6619e82`
- tests: per-obligation node ids filed as plan evidence `e1`–`e35` against obligations `o1`–`o27` (`devague evidence --list`), all proposed
- runtime tests: reachy-mini-cli `uv run pytest -n auto` at `b3fd068` — 5668 passed, 7 skipped, 1 failed (`tests/test_vision_scene_integration.py::test_integration_scene_default_model_resolves_via_senses_role`, a live-gateway test that fails identically on `origin/main`)
- audit: `scratchpad/audit-t14.txt` (session) — no legacy imports under `reachy_nova/harness/`, `nova_browser.py` act path byte-unchanged, daemon paths `/api/media/{play_sound,sounds,sounds/upload,stop_sound}` and `/api/volume/{current,set}` only
- lint: `markdownlint-cli2` on the touched docs — 0 new findings (pre-existing table-row MD013 in `gaze.md`/`attention.md`, list-style findings in `nova_browser.md`)
- commits: `a144528..6619e82` on `spec/features-and-bugs-round-25` (spec+plan, 20 task merges, 1 follow-up, 5 Sonar fixes, 3 review-fix merges)
- runtime commits: `c8e2575..ca64951` on `feat/reachy-nova-25-names-select-face`; device branch `wireless-motor-enable` @ `74629d3`
- PRs / issues: OriNachum/reachy-nova#26 (this round), #25 (closed by it), #8 (closed by it); agentculture/reachy-mini-cli#178 (runtime PR), #175 (enrolled-face preference), #176 (camera pipeline failure)
- robot journal, harness pid 15925 (round branch `d5f044b`, 09:31 local): `switches ... face_hold=on think_posture=on attention_gate=on`; `persona ... chars=1615`; `face-nod retired id=nova-face-noticed changed=True verdict=reload confirmed`; `start-hygiene release ok=true` / `start-hygiene clear goal ok=true`; `layer wander -> conversation` at 09:31:17, `look at sound ok=true`, `lock face attempt=1..4 ... error=no face known`, `layer conversation -> wander` at 09:32:03 with `lock never held: refusals=4`; `[SENSE stage=vision source=runtime event=frames] dropped reason=no-frames after=60s` at 09:32:15
- robot journal, runtime pid 12726 (v0.52.0): `release_face ... 'released': False`, `declare_goal ... 'goal': None`, `run_behavior ... look-at-sound`, `lock_face ... dropped reason=no face known` at 09:31:17/21/27/39
- robot journal, harness pid 17745 (`6619e82`, 09:56 local): `started name=gaze` then `engine live` then `start-hygiene release ok=true` — the hygiene no longer runs on the supervisor thread
- MQTT sample on the robot at 08:31:38Z: 501 `sense/snapshot` messages, `frame_available` true in 0, `face_bbox` in 0
- device overlay `~/.local/state/reachy/behavior/rules.toml`: `id = "nova-face-noticed"` / `enabled = false` in the nova-managed block
- deviations: `devague deviate --list` — `d1`, `d2`, both proposed
- PR #26 review: 10 Qodo threads, 10 answered, 10 resolved; Sonar quality gate passed, 1 open finding (S3415) left for the operator

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| a browse result is never dropped by the inject throttle, and survives a failed or stale send | high | `tests/test_sonic_inject_must_deliver.py` (33 tests) · commits `1faa060`, `89982ff` |
| a browse result is spoken once, within about 5 s of the browser finishing, on the robot | unverified | no spoken browse has been run since the deploy — not claimed done |
| a repeated browse request runs one hosted session | medium | `tests/test_nova_browser_dedupe.py` incl. the 20-thread race · not exercised live |
| the persona answers to Nova and Reachy | medium | `tests/test_harness_persona.py` · journal `persona ... chars=1615` · the spoken check is the operator's |
| cold, nameless speech gets no reply; a name opens a 45 s window | medium | `tests/test_harness_attention.py` (59), `tests/test_harness_speaking_attention.py` (16) · not exercised live |
| effectful tools are refused while cold and nameless | high | `tests/test_harness_tools.py` cold-refusal tests |
| a conversation turns the head toward the voice and retries the face lock with backoff, one summary per conversation | high | journal 09:31:17–09:32:03 (four refusals at +0/+3/+9/+22 s, fade at +46 s) · `tests/test_harness_gaze_stack_conversation.py` |
| the face lock is held through a conversation and released on fade | unverified | the camera delivers no frames (#176) — proven by unit tests only, not claimed done live |
| the head looks up and aside while browsing, antennas alive, and yields to a conversation | medium | `tests/test_harness_gaze_stack.py`, `tests/test_harness_gaze_stack_inhibit.py` · no browse has run live since the deploy |
| the face-nod rule is retired on the device | high | overlay read over ssh · journal `face-nod retired ... reload confirmed` |
| the harness names a dead camera | high | journal `dropped reason=no-frames after=60s` at 09:32:15 · `tests/test_harness_eyes.py` |
| the harness releases what it holds on start and stop | high | journal `start-hygiene` lines applied by the runtime · stop tests |
| every new behaviour has a fail-open switch named at start | high | journal switches line · `tests/test_harness_switches.py` |
| the harness still opens no SDK client and touches no new daemon path | high | `tests/test_harness_boundary.py` · audit at `6619e82` |
| the runtime answers to "nova" and prefers an enrolled face | high | reachy-mini-cli tests (96 name-match, face-selection suite) · PR #178 · robot on `74629d3` |
| version 0.5.0 with a green suite | high | `pyproject.toml` · 2070 passed at `6619e82` |
| it feels right to talk to (tone and behaviour) | unverified | the operator's verdict is pending — not claimed done |

## Remaining Work / Follow-up

- `t20` — the operator's live acceptance on `6619e82`: five cold nameless sentences (expect silence and `dropped reason=not-addressed` lines), "Nova, how are you" then a nameless follow-up (both answered), the same with "Reachy", a spoken browse (posture up, result spoken once) repeated within a minute (one session), a guest by name, and the tone verdict; then update the unverified rows above. Owner: Ori + operator.
- item 2's live half — blocked on agentculture/reachy-mini-cli#176 (camera receive pipeline fails at media acquisition; a runtime restart does not recover it); once frames return, run `look_at_face`/`lock_face` with Ori in view and re-check the hold and the fade release live
- confirm or reject deviations `d1`, `d2`, obligations `o1`–`o27`, evidence `e1`–`e35`, deltas `b1`–`b5` (all proposed)
- Sonar S3415 on `tests/test_harness_speaking_attention.py:150` — accept as a false positive in the Sonar UI, or ask for the two `in` assertions to be reshaped
- PR #26 is open with CI green and all ten review threads resolved; merging publishes 0.5.0 to PyPI, after which the robot's checkout goes back to `main`; reachy-mini-cli PR #178 merges the runtime half to its main
- follow-ups noted, not built: a trailing-name grace for fragmented transcripts (park `v4`); multi-angle enrolment (park `v5`, #127); Nova's opening line opening a conversation at every boot (delta `b2`); a wandering "thinking" runtime library entry instead of the static gaze-hold (decision `c19`)
