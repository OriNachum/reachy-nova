# Build Plan — features-and-bugs-round-25

slug: `features-and-bugs-round-25` · status: `exported` · from frame: `features-and-bugs-round-25`

> Nova now tells you what her browser found instead of losing it, holds a face in view while she talks and settles back to her idle wander after, answers to 'Nova' as readily as to 'Reachy', and looks up in thought while she is off browsing.

## Tasks

### t1 — Must-deliver inject path in `nova_sonic.py`: `inject_text`(`must_deliver`=True, `sense_class`='browse') skips the 3 s throttle, still parks under its own DeferredCues class while speaking, and a small retry queue re-sends a dropped-inactive must-deliver text (with its age) once the next session is listening

- covers: c2, h1, c39, h31
- acceptance:
  - tests/`test_sonic_inject_must_deliver.py`: `on_result`-shaped inject 0 ms after a throttled progress inject is sent (status 'sent'), a plain inject in the same window is still 'dropped-throttled'
  - a must-deliver inject while `_speaking` is parked under class 'browse' and drained after the utterance without being overwritten by an unclassed cue
  - with `inject_text` answering dropped-inactive then a new session, the queued must-deliver text is sent exactly once with 'age' in the text; a plain inject is not queued

### t2 — `nova_browser.py`: progress narration collapses to one status cue per phase worded as status ('Status: your browser is working on X — no action needed'), rate-limited to one per 10 s, 'Done! Reading results...' becomes a log line; `queue_task` dedupes a normalised instruction against the running and queued tasks within 300 s and returns {ok, queued:false, duplicate:true}; `_act_on_agentcore` / `act_get` untouched

- covers: c3, h2, c5, h3, c24, h15
- acceptance:
  - tests/`test_nova_browser_progress.py`: four progress messages inside one second produce at most one `on_progress` call per phase, each starting with 'Status:'; a stub session fed that text alone never calls `queue_task`
  - tests/`test_nova_browser_dedupe.py`: same instruction twice within 300 s runs one task and the second `queue_task` returns {'ok': True, 'queued': False, 'duplicate': True} with one senselog line; after 300 s it runs again
  - git diff `nova_browser.py` shows no change inside `_act_on_agentcore` or the `act_get` call

### t3 — LockState gains an owner: `mark_locked`(owner='auto'|'model'), owner property, `mark_released` clears it; motion/lock-released and engine-drop clear owner with the belief

- covers: c9, h6
- acceptance:
  - tests/`test_harness_lock_state.py`: owner is None when unlocked, 'auto' or 'model' as marked, None after `mark_released`, after a motion/lock-released event and after the engine-drop grace expires

### t4 — Persona names both names: config/persona/nova.md and harness/persona.py `DEFAULT_PERSONA` gain 'People call you Nova or Reachy; Reachy often reaches you as Richie or Reach — both are you', kept in the dry register with no character names

- covers: c14, h8
- acceptance:
  - tests/`test_harness_persona.py`: file and embedded default both contain 'Nova', 'Reachy' and 'Richie'; the existing no-character-name test still passes; chars stay under 2000

### t5 — harness/attention.py: AttentionState — cold/warm with `NOVA_ATTENTION_WINDOW_S` (default 45 s, env, fail-open), a local restatement of the runtime's name match (nova, reachy, richie, reach, noah; difflib+length ratio, no network), `note_transcript`(text) opens on a name and renews when warm, `note_utterance`() and `note_inject`() renew, `conversation_live` (any transcript or utterance, for the gaze) vs warm (name-opened, for the voice), a `stay_silent` window reads as cold, one senselog line per open/close under stage=attention, rotation-agnostic

- covers: c28, h23
- acceptance:
  - tests/`test_harness_attention.py`: nameless transcript from cold leaves it cold; 'nova, how are you' / 'hello richie' / 'reach, come here' open it; a nameless transcript while warm renews; 45 s of nothing closes it; a quiet window forces cold; 'now', 'no', 'know', 'nah' do not open it
  - `conversation_live` is true after any transcript or utterance and false after the window, independent of the name; a session-rotation callback changes nothing
  - exactly one log line per open and per close, none per renewal

### t6 — Voice gating and memory hygiene: speaking.py takes an AttentionState and, at the same single place the quiet gate lives, drops an utterance whose Sonic 'speaking' edge followed a nameless transcript while cold (an utterance within 3 s of an inject still plays) with one senselog line and no daemon post; ledger.py skips appends for a gate-dropped reply the way it skips quiet-window appends; the verdict is taken on the speaking edge, before the first chunk is enqueued, with a preempt fallback if audio precedes the transcript

- depends on: t5
- covers: c28, h23, c32, h24
- acceptance:
  - tests/`test_harness_speaking_attention.py`: cold + nameless transcript -> speaking edge -> chunks never reach the daemon poster, one 'dropped reason=not-addressed' line, `attention_drops` == 1; warm -> played; cold + inject 1 s before the edge -> played
  - a scripted event order textOutput USER, textOutput ASSISTANT, audioOutput yields the verdict before the first enqueue; audio before transcript triggers preempt exactly once
  - tests/`test_harness_ledger.py`: an ASSISTANT transcript flagged gate-dropped is not appended and is counted; the compactor input for that period holds no such line

### t7 — tools.py: THINK alias tool (`run_behavior` name='thoughtful', side 'left'|'right' -> yaw sign, alternating by default, duration 3); `_browse` passes NovaBrowser's typed duplicate result through; `lock_face`/`release_face` mark LockState owner 'model'; IntentTools.execute refuses effectful tools with ToolRefused('not addressed') while AttentionState is cold and the last transcript was nameless (browse, forge, `use_skill`, `author_rule`, goto, `run_behavior`, `declare_goal`, `set_mode`, `set_inhibition`, `create_rule`, `enroll_face`, `lock_face`), leaving `recall_senses`, `stay_silent`, `end_silence`, voice-level tools and `release_face` allowed

- depends on: t2, t3, t5
- covers: c17, h10, c24, c9, c40, h32
- acceptance:
  - tests/`test_harness_tools.py`: think builds {'op':'`run_behavior`','name':'thoughtful','params':{'yaw':±10},'lifetime':{'duration':3}} and alternates sign across calls; the spec is published in the tool list
  - a duplicate browse returns the typed duplicate dict unchanged; a confirmed `lock_face` marks owner 'model'
  - cold + nameless: browse raises ToolRefused('not addressed') and `queue_task` is never called; `recall_senses` still executes; warm: browse queues

### t8 — harness/`gaze_stack.py` core: a single-writer posture layer — producers (transcript, speaking edge, speaker idle, browser state) set flags under a lock, one worker thread computes the top layer (wander < browsing < conversation) and issues only transition intents serially through IntentTools, waiting out each await; browsing = `declare_goal` gaze-hold {pitch:10, yaw:±15, alternating} on busy and `declare_goal` None on idle/error; a `clear_for_result`() hook the app calls before a browse result inject; every transition logged under stage=gaze

- depends on: t5
- covers: c25, h16, c18, h11, c20, h12, c37, h29
- acceptance:
  - tests/`test_harness_gaze_stack.py`: busy -> `declare_goal` gaze-hold; idle -> `declare_goal` None; error -> `declare_goal` None; the fake spool sees exactly those ops in order
  - events fired from three threads for 2 s produce a strictly serial, causally ordered op list on the fake spool (no interleaved pairs), asserted by a per-op sequence number
  - `clear_for_result`() submits `declare_goal` None before returning; a browse result injected after it never precedes the clear in the spool

### t9 — `gaze_stack` conversation layer: on `conversation_live` rising submit `look_at_sound` then `lock_face` (owner 'auto'), retry a refused lock at 3, 6, 12, 24, 30, 30... s while live, treat ok:null as unknown and keep retrying, log one 'no face known' summary per conversation, hold through replies and gaps, submit `release_face` on conversation end only when owner is 'auto' (a model lock is left standing), leave a standing browsing goal in place under the lock and let it resume on release; on start and on `stop_event` submit `release_face` + `declare_goal` None once (idempotent)

- depends on: t8, t3
- covers: c7, h4, c8, h5, c35, h27, c36, h28
- acceptance:
  - tests/`test_harness_gaze_stack_conversation.py`: transcript -> `look_at_sound`, `lock_face` within one worker tick; refusals produce retries at 3, 6, 12, 24, 30, 30 s (fake clock) and exactly one summary line; ok:null keeps the belief unknown and retrying
  - lock held across a 10 s gap; `release_face` lands after the window closes; with owner 'model' no release is submitted; a browsing goal set before the conversation is still standing after release
  - start() and stop() each submit `release_face` and `declare_goal` None exactly once; no face-presence state exists in the module (grep)

### t10 — Head reflexes kept off a held head: the browsing layer submits `set_inhibition` merged with the current inhibited set to add orient-to-sound and nod (restored on exit, later-wins respected), and `rules_overlay` gains `retire_rule`(id) writing an enabled=false tombstone inside the nova-managed block; app start retires nova-face-noticed once

- depends on: t9
- covers: c34, h26
- acceptance:
  - tests/`test_harness_rules_overlay.py`: `retire_rule`('nova-face-noticed') leaves operator text byte-identical, writes id + enabled=false in the managed block, re-validates, and a second call is a no-op
  - tests/`test_harness_gaze_stack_inhibit.py`: entering browsing submits `set_inhibition` with the prior set plus orient-to-sound and nod; leaving restores the prior set; a lock in between does not drop them

### t11 — harness/eyes.py: samples reachy/events/sense/snapshot at about 1 Hz on its own small MQTT subscription, latches one '\[SENSE stage=vision source=runtime event=frames\] dropped reason=no-frames' line after 60 s of `frame_available`=false and one 'restored' line when frames return; supervisor.status() gains eyes: live|dead|unknown

- covers: c38, h30
- acceptance:
  - tests/`test_harness_eyes.py`: 60 s of false snapshots -> exactly one dropped line, then true -> one restored line, then false again 60 s -> one more; status reports unknown before any snapshot, dead, then live
  - the subscriber is optional: broker unreachable degrades to eyes: unknown with one named line and never raises

### t12 — Integration in app.py + switches: `NOVA_FACE_HOLD`, `NOVA_THINK_POSTURE`, `NOVA_ATTENTION_GATE` in harness/switches.py (default on, fail-open, in the start-up line) and .env.sample; `build_app` wires AttentionState into `_on_transcript` / the inject wrapper / speaker, the gaze stack into transcript, speaking edge, speaker idle and NovaBrowser.`on_state_change`, eyes into the supervisor, the browse result through `clear_for_result`() then the must-deliver inject, and retires nova-face-noticed at start; each leg degrades to a named absent line when its switch is off

- depends on: t1, t2, t4, t6, t7, t10, t11
- covers: c22, h14, c20, h12
- acceptance:
  - tests/`test_harness_app.py`: with all switches on, `build_app` returns components including gaze stack and eyes and the browser's `on_state_change` is the gaze stack's; with `NOVA_FACE_HOLD`=0 no lock op is ever submitted and a 'component absent' line names it; same for the other two switches
  - tests/`test_harness_switches.py`: the three new switches parse like the existing ones and appear in the start-up line
  - a browse result reaching `_on_browse_result` produces `declare_goal` None on the fake spool before `inject_text`(`must_deliver`=True) is called

### t13 — Docs and version: docs/components/attention.md (new), gaze.md (gaze stack, layers, think alias, owner), `nova_browser.md` (status cues, dedupe, must-deliver), architecture.md and CLAUDE.md module map entries, .env.sample knobs; pyproject.toml version 0.5.0

- depends on: t12
- covers: c21, h13
- acceptance:
  - markdownlint-cli2 passes on the touched docs; pyproject.toml carries version = "0.5.0"; CLAUDE.md names harness/attention.py, harness/`gaze_stack.py` and harness/eyes.py

### t14 — Boundary audit: uv run pytest -n auto green; tests/`test_harness_boundary.py` green; grep for gestures, `wake_word`, `face_recognition` imports under `reachy_nova`/harness/ empty; git diff of `nova_browser.py` confirms the act path untouched; runtime diff confirms only `select_face` and names changed

- depends on: t13
- covers: c22, h14, c5, h3, c11, h7
- acceptance:
  - the audit's four commands and their output are pasted into the PR body; any failure blocks the PR

### t15 — reachy-mini-cli names: add 'nova' to `DEFAULT_NAMES` in reachy/speech/engagement.py and reachy/behavior/`transcript_sense.py`, to `is_name_match` defaults in reachy/speech/`name_match.py`, and accept 'hey nova' as the sleep wake phrase default; extend tests/`test_name_match.py`'s collision table with now, no, know, nah, novel, November, nowhere and a four-letter floor for the n-family

- covers: c16, h9
- acceptance:
  - tests/`test_name_match.py`: 'Nova, come here', 'hey nova', 'nova what time is it' match; 'now', 'no', 'know', 'nah', 'novel', 'November', 'nowhere', 'not now' do not; the r-family table still passes
  - uv run pytest -n auto green in reachy-mini-cli; CHANGELOG entry; version bump per the repo's rule with uv.lock re-locked

### t16 — reachy-mini-cli `select_face`: recognised-first, biggest among equals (issue #175); no other constant in `face_lock.py`, gaze.py or `face_sense.py` changes

- covers: c26, h17, c11, h7
- acceptance:
  - tests/`test_behavior_face_selection.py`: two faces, the smaller one recognised -> the recognised one; equal status -> the biggest; a single unknown face still yields a bbox
  - git diff limited to `select_face` and its tests

### t17 — Runtime deploy: merge origin/main into wireless-motor-enable on spark with t15 + t16, tests green, push; on the robot ff-pull, detached pip install with disk headroom checked, restart reachy-runtime; open the reachy-mini-cli PR to main

- depends on: t15, t16
- covers: c21, h13
- acceptance:
  - robot runtime checkout is wireless-motor-enable at the merged commit; 'engine live' in the journal; df -h / shows headroom before the install; PR open with CI green

### t18 — Harness deploy: first switch the robot's ~/git/reachy-nova from spec/fast-witty-remembering-nova to main (v0.4.0) and restart, then check out the round's branch, restart, and read the switches line and the persona line from the journal

- depends on: t14
- covers: c21, h13
- acceptance:
  - journal shows the switches line naming `face_hold`, `think_posture` and `attention_gate`, the persona source line, and engine live; no traceback in the first two minutes

### t19 — Camera gate before item-2 acceptance: sample reachy/events/sense/snapshot for 10 s and record `frame_available` counts; if false, follow reachy-mini-cli #176 (daemon/media recovery) and record every item-2 live check as blocked; if true, run `look_at_face` and `lock_face` once with Ori in view and record bbox/name presence

- depends on: t17, t18
- covers: c41, h33
- acceptance:
  - the delivery doc carries the sample counts with a timestamp and the #176 status; item-2 checks are marked blocked or run, never assumed

### t20 — Live acceptance and delivery doc: run the c23 script on the robot (browse result spoken once; duplicate request runs one session; head turns to the voice then locks and releases on fade with lock lines in both journals; five cold 'Nova, ...' and five 'Reachy, ...' answered outside any quiet window; browsing posture with antennas alive, conversation takes the head; no-frames line present within 90 s of start while the camera is dead), with Ori and a guest; write docs/deliveries/2026-09-06-features-and-bugs-round-25.md quoting journal lines per check and marking unverified or blocked honestly

- depends on: t19
- covers: c1, h18, c23, h19, c29, h20, c30, h21, c31, h22, h1, h4, h8, h23, h30, h11, h26
- acceptance:
  - every c23 check has a quoted journal line or an explicit blocked/unverified mark; the before-state quotes the three throttled results and the two-day face silence; Ori's tone and behaviour verdict is recorded verbatim

## Risks

- [unknown_nonblocking] The robot's camera delivers no frames (GStreamer 'state change failed' at media acquisition; reachy-mini-cli #176); item-2 live evidence is blocked until the daemon/media path recovers — the harness ships regardless behind `NOVA_FACE_HOLD` (task t19)
- [unknown_nonblocking] Fuzzy name match on 'nova' can engage on 'now'/'no'/'know' (difflib 0.86 for 'now'); the collision table and the n-family floor are the guard, measured live in t20 (task t15)
- [unknown_nonblocking] Sonic may emit reply audio before the USER transcript on some turns (assumption c33); the preempt fallback clips one chunk — measured live in t20 (task t6)
- [follow_up] Fragmented transcripts with the name arriving after the request (park v4) cannot be revived by the gate; a trailing-name grace is a follow-up if live use shows it (task t20)
- [unknown_nonblocking] `set_inhibition` is later-wins and the runtime re-asserts the lock's own inhibitions on every replacement; the browsing layer's merge must read the live set from state.json each time or it can drop an operator-held name (task t10)
- [follow_up] Runtime deploy on the CM4: pip takes minutes and the SD card sits near 90 %; run pip detached and check df first (device ops notes) (task t17)
- [follow_up] Recognition may need multi-angle enrolment (park v5, #127); only testable once frames return (task t19)
