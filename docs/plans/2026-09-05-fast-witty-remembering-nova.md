# Build Plan — fast-witty-remembering-nova

slug: `fast-witty-remembering-nova` · status: `exported` · from frame: `fast-witty-remembering-nova`

> Reachy Nova now answers within about a second of what you say instead of after a long silence, speaks like a friend with a dry wit rather than an eager helper, varies how it reacts to the same touch or face, hands the deeper reactions to Nova 2 Lite, and remembers who you are and what you talked about across restarts and session rotations.

## Tasks

### t1 — Daemon sounds API: DaemonClient gains `list_sounds`() and `delete_sound`(filename) over GET /api/media/sounds and DELETE /api/media/sounds/{filename}

- instruction: Files: `reachy_nova`/harness/`daemon_client.py`, tests/`test_harness_daemon_client.py` only. Mirror `get_volume`/`set_volume`; endpoints verified in the robot daemon's OpenAPI 2026-09-05.
- covers: c27
- acceptance:
  - DaemonClient.`delete_sound`('x.wav') issues DELETE {base}/api/media/sounds/x.wav through the injectable transport and returns the parsed JSON; `list_sounds`() returns the daemon's list
  - an HTTP failure on delete raises (the caller names the drop); tests in tests/`test_harness_daemon_client.py` cover both calls with the fake transport

### t2 — Sonic session config and watchdogs: endpointingSensitivity on sessionStart, contentEnd type/role/stopReason logged at INFO, liveness watchdog counts only speech-energy input

- instruction: Files: `reachy_nova`/`nova_sonic.py` (`_start_session`, `_process_responses` contentEnd branch, `feed_audio`/`_note_input_sent`), tests/`test_sonic_resilience.py` + a new tests/`test_sonic_session_config.py`. Keep the clock-step watchdog untouched.
- covers: c5, h3, c26, h18
- acceptance:
  - sessionStart payload carries turnDetectionConfiguration.endpointingSensitivity = `NOVA_SONIC_ENDPOINTING`, default HIGH; an unrecognised value logs a warning and sends HIGH; one INFO line per session start names the value (test asserts the payload and the log)
  - every contentEnd event logs type, role and stopReason at INFO (test with a fake event stream)
  - `feed_audio` marks input as flowing only when chunk RMS exceeds `NOVA_SONIC_SPEECH_FLOOR` (default 0.01) — a test feeds 200 s of silent chunks past the liveness window and the watchdog does not fire; the same test with one speech-level chunk then silence fires it; an inject or tool result still counts as input

### t3 — Persona file and loader: config/persona/nova.md in the Wit-inspired register with two one-shot exchanges, harness/persona.py resolving `NOVA_PERSONA_PATH` with an embedded default

- instruction: Files: config/persona/nova.md (new), `reachy_nova`/harness/persona.py (new), tests/`test_harness_persona.py` (new). Register per decision c18: quick wordplay, deflates pretension, teases the person it likes, sudden sincerity, never cruel to the vulnerable, never a helper; do NOT name or quote the character. Show tone with 2 one-shot User/Nova exchanges; avoid phrase lists (Sonic 2 over-uses them). Do not touch app.py here.
- covers: c8, c34, h26, c35, h27
- acceptance:
  - persona.load() returns the file text when `NOVA_PERSONA_PATH` (or the repo default path) exists, else the embedded `DEFAULT_PERSONA` plus one senselog line naming the missing path
  - the persona text contains no offer of help, no 'how can I help', names no character and quotes no book: a test greps both the file and the embedded default for the character names and asserts the register keywords (teasing, dry, sincere, never cruel) are present
  - a wheel built with uv build and installed into a clean venv imports `reachy_nova`.harness.persona and load() returns the embedded default (test spawns a subprocess with the config dir absent)

### t4 — Conversation ledger: harness/ledger.py with locked NDJSON append of transcripts and delivered senses, quiet-window exclusion, 24 h truncation, atomic writes and latched drop lines

- instruction: Files: `reachy_nova`/harness/ledger.py (new), tests/`test_harness_ledger.py` (new). Path: statedir.`ledger_path`() -> <state>/nova-conversation.jsonl (add the helper to harness/statedir.py). Reuse quiet.py's atomic-write helper if it is importable; otherwise copy the temp+replace pattern.
- covers: c30, h22
- acceptance:
  - Ledger.append(`role_or_source`, text, ts) from two threads concurrently yields well-formed NDJSON with no interleaved lines (test with 2 writer threads x 500 lines)
  - while quiet.active() is true, append() is a no-op that counts a skipped line; truncate(now) drops lines older than 24 h and rewrites the file via temp+os.replace
  - with the state dir read-only, append() raises nothing, emits exactly one senselog drop line for the whole run, and a later successful append emits one recovery line

### t5 — Feature switches: harness/switches.py resolving `NOVA_CHUNKED_PLAYBACK`, `NOVA_LITE_REACTIONS`, `NOVA_MEMORY` (default on) and `NOVA_PERSONA_PATH` once, fail-open with a warning, logged in one line

- instruction: Files: `reachy_nova`/harness/switches.py (new), tests/`test_harness_switches.py` (new). Same shape as gate.`resolve_policy`. Wiring into app.py happens in the integration task, not here.
- covers: c33, h25
- acceptance:
  - resolve() returns a frozen dataclass; '0'/'false'/'off' turn a switch off, unset means on, anything else means on plus a named warning line
  - describe() renders one senselog line listing every switch's resolved value; tests cover each switch off and on

### t6 — Mood: harness/mood.py — a small decaying state fed by pats, recognised faces, conversation turns and silence, rendered as one short context sentence

- instruction: Files: `reachy_nova`/harness/mood.py (new), tests/`test_harness_mood.py` (new). Port the idea of config/emotions.yaml (baseline + decay), not the module. Consumers: the Lite reactor context and the history replay block; nothing on the cognition feed (assumption c9).
- acceptance:
  - Mood.note(event) adjusts a bounded scalar per event class from a small table; Mood.render(now) returns one sentence ('you are in a playful mood', 'nobody has spoken to you for ten minutes') that changes across at least three states in tests as events and time pass
  - levels decay toward baseline with elapsed time (injectable clock); a fresh Mood renders the neutral sentence

### t7 — Device probes on the robot: measure back-to-back `play_sound` chunk gap/click and Nova 2 Lite round-trip latency; record the numbers as plan evidence

- instruction: Scratch scripts only, run over ssh as pollen; nothing committed to the repo except the numbers in the plan/delivery notes. Resolves parks v2 and v3; informs t8's chunk size and t11's timeout.
- acceptance:
  - two 1 s WAVs posted back-to-back through the daemon with the second posted at the first's expected end: the audible gap or overlap is measured (timestamps + a human ear) and recorded in the plan risk notes
  - five timed `invoke_model` calls to Nova 2 Lite from the robot's network with a 60-token prompt: min/median/max recorded; the Lite tier's default timeout is set from the median x 2 (cap 2 s)

### t8 — Chunked speaker: speaking.py flushes ~1 s chunks at low-energy boundaries or after 300 ms of audio inactivity, per-chunk filenames with delete-after-window, gate-serialised, preempt purges the rest

- instruction: Files: `reachy_nova`/harness/speaking.py, tests/`test_harness_speaking.py`. Split point: the lowest-RMS 50 ms window in the last 200 ms before the target; inactivity timer on a small thread or checked by the worker loop. Keep the quiet gate and `_mouth_loss` paths. Chunk size may be tuned from t7's numbers.
- depends on: t1
- covers: c2, h2, c27, h19
- acceptance:
  - feeding 5 s of synthetic audio at real-time pace produces the first queued chunk within 1.2 s of the first sample and no chunk longer than 1.5 s; a 0.5 s reply followed by silence is queued within 0.4 s of its last sample without any state change (inactivity flush)
  - chunks of one utterance post in order, each waiting for the previous gate window; a preempt() between chunk 2 and 3 stops the sound, purges chunks 3..n and none of them ever posts (epoch check)
  - each chunk uploads as nova-<utt>-<seq>.wav and is deleted after its window via the injectable deleter; at most 8 files are ever outstanding; a failed delete is one named senselog line and playback continues
  - `NOVA_CHUNKED_PLAYBACK`=0 (passed as a constructor flag) restores today's whole-utterance behaviour and tests/`test_harness_speaking.py`'s existing tests still pass under it

### t9 — Deferred cues: a cue arriving while Sonic generates is parked (latest-wins per sense class, TTL 5 s) and drained through `inject_text` with its age in the text when the utterance ends

- instruction: Files: `reachy_nova`/harness/`deferred_cues.py` (new), `reachy_nova`/`nova_sonic.py` (`inject_text` speaking-guard branch + `_set_state` hook), tests/`test_harness_deferred_cues.py` (new). Sense class comes from the rules entry's sense field; `inject_text` gains an optional `sense_class` kwarg the bus passes.
- depends on: t2
- covers: c6, h4
- acceptance:
  - `inject_text` during `_speaking` stores the cue in DeferredCues instead of dropping it, logging 'deferred' not 'dropped reason=speaking'; a second cue of the same class replaces the first; classes are independent
  - on the transition to listening the slot drains in arrival order through the normal inject path with the text prefixed by its age ('a few seconds ago, while you were talking'); a cue older than the TTL is dropped with reason=deferred-expired
  - the 3 s throttle and the bus dedupe are unchanged (tests/`test_harness_bus_dedupe.py` green); the drain happens off the response thread's critical path (no await inside `_process_responses` beyond scheduling)

### t10 — Memory compactor: harness/`memory_compactor.py` — a background thread that asks Nova 2 Lite to distil the ledger into topics and important items with timestamps, 24 h expiry, atomic write, and a history() view for session replay

- instruction: Files: `reachy_nova`/harness/`memory_compactor.py` (new), tests/`test_harness_memory_compactor.py` (new). Lite call via boto3 bedrock-runtime like `nova_omni`.`_invoke` (inject the client). Prompt asks for topics + important items (requests, preferences, running jokes, things Nova was told to stop). Expiry uses timestamps written at compaction; skip expiry while the wall clock is older than the newest entry (stale boot clock, no RTC).
- depends on: t4
- covers: c11, h7, c28
- acceptance:
  - compact() runs on its own thread, never on the caller's; with a fake Lite returning JSON it writes <state>/nova-memory.json atomically with {topics:\[{text,ts}\], items:\[{text,ts,kind}\]} and truncates the ledger to 24 h
  - entries older than 24 h are absent after the next compact(); a Lite failure leaves the previous file intact and emits one named drop line; a compaction runs at most every N minutes (default 5) and once at shutdown
  - history(`max_chars`=2000) returns a list of {role, text} blocks: one USER-role context block summarising topics and items ('earlier today we talked about ...') followed by the last two or three exchanges verbatim, within the cap

### t11 — Lite reactor: harness/`lite_reactor.py` — a single worker with a bounded latest-wins queue that turns an opted-in cue plus context (sense history, day memory, mood, last exchanges) into a one-line reaction plan within a timeout, falling back to the template

- instruction: Files: `reachy_nova`/harness/`lite_reactor.py` (new), tests/`test_harness_lite_reactor.py` (new). Model id from config.`lite_model_id`(); reuse `nova_omni`'s message schema; never import paho or `nova_sonic` here.
- depends on: t6
- covers: c10, h6, c28, h20
- acceptance:
  - react(cue, context) enqueues and returns immediately; the worker calls the injected Lite client with a prompt carrying all four context parts and delivers plan.text (and plan.gesture, if any, through the injected intents callable) via the delivery callback
  - a Lite client that hangs 30 s does not block react() or the delivery of other cues; the plan times out at the configured deadline (default from t7, cap 2 s), the template text is delivered instead and one named drop line is emitted
  - the plan text is returned raw — markers are applied by the bus (t13); an empty or malformed Lite reply falls back to the template

### t12 — Session rotation with history replay: NovaSonic gains a `history_provider` hook sent between the system prompt and the audio contentStart, a ~7 min rotation timer that waits for idle with a hard deadline, zero delay for a healthy rotation, and one journal line per rotation

- instruction: Files: `reachy_nova`/`nova_sonic.py` (`_start_session`, `_run_loop` wait loop, `request_immediate_restart` path), tests/`test_sonic_rotation.py` (new). Depends on t9 only because both edit `nova_sonic.py`. The provider is wired in t14.
- depends on: t9
- covers: c12, h8
- acceptance:
  - `_start_session` sends each history block from `history_provider`() as a USER/ASSISTANT TEXT content (interactive false) after the system prompt and before the audio contentStart — asserted by the event order in the fake send log
  - a rotation timer (`NOVA_SONIC_ROTATE_S`, default 420) requests a restart only when state is listening, no tool use is in flight and an injected `speaker_idle`() returns True, with a hard deadline at 470 s; the restart uses delay 0 and logs 'rotation delay=0 replay=<n>'
  - liveness, clock-step and network restarts replay the same history; a fresh session sends no assistant-initiating text (Nova stays silent until spoken to) — asserted by the send log

### t13 — Bus routes Lite-tier cues: rules.yaml entries gain react: lite; the bus reserves dedupe, hands the rendered cue to the reactor, and applies the voice/quiet markers and SenseHistory record to the plan text; voice: none never reaches Lite

- instruction: Files: `reachy_nova`/harness/bus.py (`_deliver` + a reactor kwarg), config/nervous-system/rules.yaml (react: lite on rule/fire:pat-acknowledge, rule/fire:nova-pat-cheer, rule/fire:nova-face-noticed, pat/\*, face/recognized), tests/`test_harness_bus.py` + tests/`test_rules_voice.py`. SenseHistory records the delivered plan text.
- depends on: t11
- covers: c10, h6
- acceptance:
  - an entry with react: lite and voice: brief yields an inject ending in the brief marker built from the reactor's plan text; the second same-key event inside the dedupe window produces neither a Lite call nor an inject; a voice: none entry produces no Lite call
  - entries without react: lite render byte-identically to today (tests/`test_rules_coverage.py` and tests/`test_rules_voice.py` unchanged and green); the pat and face entries in rules.yaml opt in
  - with `NOVA_LITE_REACTIONS` off (reactor=None) every entry renders as today

### t14 — Integration in app.py: persona + amy voice, switches, ledger, compactor, reactor, deferred cues and the history provider wired into `build_app`, with every leg degrading to a named absent line

- instruction: Files: `reachy_nova`/harness/app.py, tests/`test_harness_app.py`. Only this task edits app.py. Keep every component optional-degraded with the standard 'component absent name=... reason=...' line.
- depends on: t3, t5, t8, t9, t10, t11, t12, t13
- covers: c8, h5, c33, h25, c13, h11, c34
- acceptance:
  - `build_app`() constructs NovaSonic with `system_prompt` = persona text + a short tool paragraph and `voice_id`='amy'; `TOOL_SPECS` are unchanged (test compares against the pre-change list)
  - with `NOVA_MEMORY`=0 no ledger or compactor is built; with `NOVA_LITE_REACTIONS`=0 the bus gets reactor=None; with `NOVA_CHUNKED_PLAYBACK`=0 the speaker is constructed in whole-utterance mode; the first senselog lines name every switch (tests/`test_harness_app.py`)
  - `on_transcript` appends USER/ASSISTANT lines to the ledger and the bus appends delivered senses; sonic.`history_provider` = compactor.history; the compactor thread starts as a supervisor component and stops cleanly

### t15 — Docs and version: architecture.md, CLAUDE.md module map, docs/components/{persona,memory,lite-reactor}.md, .env.sample entries for the new switches, pyproject version 0.4.0

- instruction: Files: docs/architecture.md, CLAUDE.md, docs/components/\*.md (new pages), .env.sample, pyproject.toml. No IP addresses anywhere.
- depends on: t14
- covers: c16, h13
- acceptance:
  - pyproject.toml version is 0.4.0 and tests/`test_packaging.py` is green; .env.sample lists `NOVA_SONIC_ENDPOINTING`, `NOVA_CHUNKED_PLAYBACK`, `NOVA_LITE_REACTIONS`, `NOVA_MEMORY`, `NOVA_PERSONA_PATH`, `NOVA_SONIC_ROTATE_S`, `NOVA_SONIC_SPEECH_FLOOR` with one-line meanings
  - docs/architecture.md section 5.4 describes chunked playback and the rotation, section 6 adds the Lite reaction tier and the memory compactor; CLAUDE.md's module map lists the new modules; markdownlint-cli2 passes on the edited docs (line-length aside where generated)

### t16 — Boundary audit: run the boundary and packaging tests, grep the harness for legacy modules and daemon endpoints, diff hearing.py, and check every after/before-state clause maps to a requirement

- instruction: No code changes; this task produces the PR body's audit section and fails the round if any check fails.
- depends on: t14
- covers: c4, h9, c13, h11, c15, h12, c23, h15, c24, h16
- acceptance:
  - uv run pytest -n auto is green; tests/`test_harness_boundary.py` passes; grep of `reachy_nova`/harness for `session_state`|`nova_feedback`|emotions finds nothing; git diff main -- `reachy_nova`/harness/hearing.py is empty
  - a short audit note in the PR body maps each after-state clause (c23) to its requirement+honesty ids and confirms the before-state numbers (c24) against the 2026-09-05 journal quotes

### t17 — Deploy to the robot: check disk headroom, pull the merged branch into ~/git/reachy-nova, set .env (`NOVA_SONIC_ENDPOINTING` unset=HIGH, switches on), restart reachy-nova-harness, confirm the switch line and persona in the journal

- instruction: Per memory: editable install, uv absent on device, run any pip detached, never pkill patterns that match your own ssh command, no IPs in docs.
- depends on: t15, t16
- covers: c16, h13
- acceptance:
  - df -h / shows at least 1 GB free before the pull; the harness restarts with 'engine live', the switch summary line, the persona source line and the endpointing line in the journal within 60 s
  - no pip install unless pyproject or uv.lock changed; the robot runs the exact merged commit (git rev-parse HEAD matches)

### t18 — Live acceptance and delivery doc: a 10-minute conversation on the robot with two rotations, journal evidence for every timing honesty condition, Ori's verdict on tone, docs/deliveries/<date>-fast-witty-remembering-nova.md

- instruction: Write docs/deliveries/2026-09-XX-fast-witty-remembering-nova.md following the 2026-08-26 delivery docs' shape. Ori runs the conversation; the operator collects the journal.
- depends on: t17
- covers: c1, h1, c7, h10, c22, h14, c25, h17, h2, h4, h8, h19, h18, c23
- acceptance:
  - journal: every 'played' within 1.5 s of its 'Utterance audio started'; median heard-to-first-audio at most 2.5 s; a pat during a reply produces a 'deferred' line and a reaction within 1 s of the utterance end; two 'rotation delay=0 replay=' lines with no greeting and Nova answering 'what were we talking about' afterwards; no liveness restart during a 10-minute quiet stretch; GET /api/media/sounds shows no stale chunk files
  - a guest (not Ori) talks to Nova and gets the same speed and register; Ori's one-line verdict on tone is recorded; the delivery doc quotes the journal lines and lists what was left out, if anything

## Risks

- [unknown_nonblocking] Chunk boundary gap or click on the daemon is unmeasured (park v3); t7 measures it and t8 tunes chunk size and a pre-roll if needed. If the daemon cannot play back-to-back files seamlessly, the fallback is larger chunks (2-3 s) rather than whole utterances. (task t8)
- [unknown_nonblocking] Nova 2 Lite round-trip from the robot is unmeasured (park v2); t7 measures it and sets t11's timeout. If the median exceeds ~2 s the Lite tier stays opt-in for the slow cues only (face), not pats. (task t11)
- [unknown_nonblocking] Why Sonic's contentEnd never ends the speaking state on this stream is unknown (hard question on c2); t2 logs type/role/stopReason and the answer may simplify t8's inactivity flush or reveal a second bug. (task t2)
- [unknown_nonblocking] Sonic may speak unprompted on a fresh session with replayed history (a re-greeting every 7 min); c31 forbids it and t12 asserts the send log, but the model's behaviour is only provable live in t18. (task t12)
- [unknown_nonblocking] `reachy_nova`/`nova_sonic.py` is edited by t2, t9 and t12 — serialised by dependencies so no two land in the same wave; a rebase conflict there is expected and must be resolved by hand, never by taking one side. (task t12)
- [unknown_nonblocking] The robot's root disk is at 90 % (1.4 GB free); t17 refuses to deploy under 1 GB and the sounds cleanup in t8 is what keeps chunked playback from filling it. (task t17)
- [follow_up] Upstream follow-up: reachy-mini-cli#162 streaming speaker-feed seam — the gap-free sub-second path once this round's chunked playback is proven.
- [follow_up] Upstream follow-up: per-person memory needs the runtime to publish the recognised name (face cue today names the rule, not the person).
- [follow_up] Pre-existing: a wheel install lacks config/nervous-system/rules.yaml; the persona now embeds a default (t3) but rules do not — package config under `reachy_nova`/ in a later round.
