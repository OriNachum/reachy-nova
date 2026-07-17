# Build Plan — event-based senses + seamless reactions

slug: `event-based-senses-seamless-reactions` · status: `exported` · from frame: `event-based-senses-seamless-reactions`

> Reachy Nova's senses become one verifiable event pipeline: speech is captured with an always-on ring buffer + VAD that backtracks 1-2s to the utterance start, and every sense — speech, audio direction, touch, image, face — reaches the agent as a logged, event-based signal it can answer with speech, harmonic vocalizations, movement, or any combination, seamlessly

## Tasks

### t1 — Test scaffold: pytest wiring for reachy_nova

- instruction: Add tests/ with conftest.py stubbing hardware-heavy imports (reachy_mini, nemo, cv2, boto3, paho are heavy or absent on dev machines — module-level sys.modules stubs or importorskip keeps unit tests import-light). Add pytest to a [dependency-groups] dev group in pyproject.toml. No CI exists in this repo — the gate is local 'uv run pytest'.
- acceptance:
  - uv run pytest exits 0 on a fresh checkout with a passing placeholder test under tests/
  - pytest lands in a dev dependency group without touching runtime dependencies

### t2 — Speech-capture lane: ring buffer + parakeet VAD with measured 1-2s backtrack

- instruction: New file reachy_nova/speech_events.py: SpeechEventDetector with a rolling float32 ring buffer (pre_roll_seconds default 2.0 at 16kHz) fed from the main loop; periodic background transcription mirroring wake_word.py's single-worker pattern, sharing the SAME loaded parakeet instance (constructor arg — wake_word owns it asleep, this lane awake). Onset = transcript becomes non-empty; emit on_speech(dict) with the clip backtracked pre_roll seconds before estimated onset. This resolves frame park v2: payload = clip path + transcript + duration + onset ts. Do NOT touch main.py — integration is a later task.
- depends on: t1
- covers: c2, h2
- acceptance:
  - A synthetic test feeds silence then speech starting at a known t0: the emitted speech_detected clip's audio begins at or before t0 (backtrack measured, not assumed)
  - The detector reuses an injected ASR handle (no second parakeet load); with no handle it falls back to the XMOS speech_detected flag as the onset trigger
  - Event payload carries local clip path, transcript, duration_seconds and onset timestamp; clips stay under ~/.reachy_nova/, never uploaded

### t3 — Per-stage sensory logging + loud inject drops

- instruction: New file reachy_nova/sensory_log.py: stdlib-only helper (logger 'nova.sensory', line shape like '[SENSE stage=vad source=speech event=<id>] detail'). In nova_sonic.py inject_text, raise the two silent drop paths (3s throttle; skipped-while-speaking) from DEBUG to INFO through the helper, including the reason. Files owned: sensory_log.py, nova_sonic.py, their tests.
- depends on: t1
- covers: c3, h3
- acceptance:
  - sensory_log.stage() emits one parseable INFO line per stage carrying stage name, source, event id and detail; a unit test asserts the format
  - A throttled or skipped-while-speaking inject_text logs at INFO with a reason and the dropped text's first 60 chars (tests cover both drop paths)

### t4 — Vocalize: harmonic non-speech voice as a skill

- instruction: New file reachy_nova/vocalize.py: additive harmonic synthesis (fundamental + 2-4 harmonics, pitch envelopes for rising/falling/trill shapes, 0.3-1.5s, amplitude envelope against clicks). Executor in skill_executors.py enqueues via the existing handle_audio_output path so barge-in clearing and the speech_enabled gate apply; because output rides the speaker reference like Sonic speech, hardware AEC already treats it as far-end (the live no-self-hearing check is the live-proof task's job). Files owned: vocalize.py, skills/vocalize/SKILL.md, skill_executors.py, tests.
- depends on: t1
- covers: c6, h6
- acceptance:
  - vocalize.synthesize(kind, ...) returns float32 mono in [-1,1] at a requested sample rate for at least 3 expressive kinds (chirp_up, trill, purr_tone); pure numpy, no new deps; tests assert shape/range/duration and click-free envelopes (first/last samples near zero)
  - skills/vocalize/SKILL.md + executor registered in skill_executors.py: the executor pushes synthesized audio through the same speaker buffer the Sonic output rides and returns a short result string

### t5 — DoA becomes rate-limited audio_direction events

- instruction: Only file: tracking.py (+ its test). Extend TrackingManager.update_doa: after computing doa_yaw_target, fire 'audio_direction' with {bearing_deg, label, speech_active} through self._fire_event (main.py already forwards tracking events to MQTT). Keep head-tracking behavior unchanged — events are purely additive.
- depends on: t1
- covers: c4, h4
- acceptance:
  - update_doa fires an audio_direction event (bearing_deg + left/front/right label per the XMOS 0=left, pi/2=front mapping) via the existing _fire_event path, rate-limited to one event per 2s window unless the bearing moves 15+ degrees
  - Unit tests cover the bearing-to-label mapping and the rate limit (continuous speech from one bearing yields exactly one event)

### t6 — rules.yaml explicit coverage for every event source

- instruction: Files: config/nervous-system/rules.yaml + one test. Suggested: face_recognized NORMAL/NOW llm_evaluate false, warm template; pat_level1/2 llm false (reflex already handled in-process); audio_direction LOW/BACKGROUND llm false; speech/speech_detected NORMAL/NOW llm false; forge/* LOW/BACKGROUND except forge/rejected NORMAL/NOW. The test can regex-scan for publish_event( and _fire_event( call sites to build the source/type inventory.
- depends on: t1
- covers: c5, h5
- acceptance:
  - rules.yaml gains explicit entries for face/face_recognized, tracking/pat_level1, tracking/pat_level2, tracking/audio_direction, speech/speech_detected, and forge/staged|activated|rejected with priority/urgency/inject templates
  - A test enumerates every publish_event source/type pair in the codebase and fails if any lacks an explicit rules.yaml entry — no sense rides the default rule

### t7 — Skill-forge client: dispatch to qwen3, stage results, forge events

- instruction: New file reachy_nova/skill_forge.py. The prompt template instructs the coder model to output exactly two fenced files (SKILL.md with name/description frontmatter; executor.py defining execute(params, ctx)); parse fences defensively. Call the validator (separate task's module) before handing to activation; on validation failure move the folder to skills-forged/.rejected/<name>/ and emit forge/rejected with the reasons. requests or urllib both fine. Files owned: skill_forge.py + tests.
- depends on: t1
- covers: c13, c16, h14, h11
- acceptance:
  - SkillForge.dispatch(goal, context, improve=None) sends an OpenAI-compatible chat request to FORGE_BASE_URL / FORGE_MODEL (env-configured, Bearer via FORGE_API_KEY) from a worker thread and writes the returned skill as a STAGED folder (SKILL.md + executor.py) under ~/.reachy_nova/skills-forged/<name>/ — never directly into the live skills dir
  - Every transition emits an event through a provided publish callback: forge/staged, forge/activated, forge/rejected with reason; an unreachable endpoint or unparseable reply yields forge/rejected + a WARNING log, never an exception to the caller and never a blocked 50Hz loop (tests mock the HTTP layer for success, timeout, and garbage-reply paths)

### t8 — Forge validator: sanctioned-primitives allow-list, AST-only, never imports

- instruction: New file reachy_nova/forge_validator.py, stdlib-only (ast + pathlib). Walk the AST: Import/ImportFrom against ALLOWED_IMPORTS; attribute/call roots against the ctx surface; forbid getattr on ctx (allow-list bypass); cap executor size (~200 lines). Return (ok, reasons). This is the security-critical module — the negative tests are the deliverable as much as the code. Files owned: forge_validator.py + tests.
- depends on: t1
- covers: c15, h13
- acceptance:
  - validate(skill_dir) parses executor.py with ast (never imports or executes it) and rejects: imports outside an allow-list (numpy, math, time, typing, dataclasses), any os/subprocess/socket/urllib/requests/open usage, exec/eval/compile/__import__, and calls outside the injected ctx primitive surface (ctx.gesture, ctx.vocalize, ctx.say/inject, ctx.state_get/update, ctx.emotion)
  - Negative tests exist and pass: skills importing subprocess, opening a socket, writing via Path.write_text, and shelling via os.system are each rejected with a reason; a well-formed vocalize-composing skill passes; validation runs BEFORE any generated code is imported anywhere in the codebase

### t9 — Main-loop integration: speech lane + DoA + logs wired, Sonic feed untouched

- instruction: The ONLY task allowed to edit main.py — keep the diff tight. Feed the lane right after preprocess_mic_audio, parallel to sonic.feed_audio, never replacing it. Pause the lane while sleep_orch.state != 'awake' (wake_word owns the parakeet model asleep; hand ownership back on wake). Inject phrasing short and sensory: 'You hear someone speaking from your left.' Files owned: main.py + integration tests.
- depends on: t2, t3, t5
- covers: c1, c8, h8, c9, c10, h10
- acceptance:
  - main.py instantiates the speech lane sharing the wake-word parakeet handle and feeds it the SAME preprocessed chunks Sonic gets; on_speech publishes speech/speech_detected via mqtt.publish_event and injects a concise notice, including direction when an audio_direction bearing arrived within the last 3s
  - A test proves the Sonic feed is byte-identical with the lane enabled vs disabled (spy on sonic.feed_audio: identical call sequence and arrays)
  - Per-stage sensory_log lines appear for capture, vad, event and inject in a simulated pass; all new publishes go only through NovaMQTT (broker-down fallback still works, no direct sockets); no reachy_mini SDK or firmware files are modified; no cloud upload is added (clips stay local)

### t10 — Forged-skill hot registration + activation announce

- instruction: Files: skills.py (discover_runtime reusing _parse_skill_md; import forged executor.py ONLY after validation passed, inside try — failure emits forge/rejected and skips), skills/forge/SKILL.md + the forge executor in skill_executors.py so Nova can call forge(goal=...) as a tool, and the small activation wiring. Auto-activate per frame decision q2 — no admin gate.
- depends on: t7, t8, t9
- covers: c14, h12
- acceptance:
  - SkillManager.discover_runtime(dir) loads validated forged skills from the runtime area; after activation the NEW Sonic session's toolConfiguration contains the forged toolSpec (asserted via get_tool_specs()) and activation is announced to the agent via inject + a forge/activated event
  - Activation path: validated skill moves from staged to the active runtime dir, then sonic.restart() runs immediately when voice_state is idle or is deferred to the next natural restart when a conversation is live; a forged skill whose execute() raises returns an error string through SkillManager.execute, never crashing the loop

### t11 — Live proof on the robot (operator-assisted): trace, benchmark, vocalize, forge round-trip

- instruction: Run on the robot with the qwen3 rig reachable. Script the evidence capture (grep the [SENSE ...] lines; mosquitto_sub -t 'nova/events/#' recording) so the artifact is saved, not remembered. If any leg fails, record it honestly per the honesty conditions — a failed leg blocks claiming its targets, never gets smoothed over.
- depends on: t4, t6, t9, t10
- covers: c7, h1, h7, h2, h6, h9, h11, c17, h15, c19, h17, c20, h18
- acceptance:
  - One real end-to-end session log shows a spoken sentence's trace capture -> vad (backtrack visible) -> event -> inject -> reaction, and all five senses (speech, direction, touch, image, face) each produce a real MQTT event id observed on the bus; the trace is saved as an artifact
  - Parakeet-as-VAD benchmark recorded on-device: transcribe latency over the rolling window + main-loop timing under load; PASS keeps parakeet, FAIL flips the lane's onset trigger to the XMOS speech_detected flag — either way the numbers and the decision are recorded
  - Vocalize audibly plays on the robot speaker and its chirps are NOT transcribed as user speech (AEC check); the forge round-trip is proven live once (dispatched goal -> generated skill validates, stages, activates, and is called by the model); a diff review confirms zero vendored/patched reachy_mini or XMOS files; the operator is present and confirms the experience

### t12 — Component docs + CLAUDE.md refresh with cited baseline

- instruction: Follow docs/components/patting.md's voice. The CLAUDE.md drift fix is in-scope here: main-loop description + module list. Benchmark numbers arrive from the live-proof task — if docs land first, mark the number as pending and update in the same PR.
- depends on: t10
- covers: c18, h16
- acceptance:
  - docs/components/ gains speech-events.md, vocalize.md and skill-forge.md (overview, event/payload shapes, config env vars, honest limits including the benchmark outcome and the auto-activate policy); CLAUDE.md's architecture section is updated to the real tree (it currently predates audio_pipeline, wake_word, face_recognition, sleep_orchestrator)
  - Docs state the before-state baseline with file citations (continuous Sonic feed, DEBUG-level drops, DoA private to head-tracking) and the local-only clip policy; markdownlint clean

## Risks

- [unknown_nonblocking] The qwen3 rig is a separate machine: gateway auth (Bearer), model reloads and downtime make the forge seam flaky — mirrors what lobes-cli#91/#95 did to colleague; backoff + clear forge/rejected reasons required (task t7)
- [unknown_nonblocking] Parakeet transcribe cadence on the robot's compute is unmeasured — if the on-device benchmark fails, the XMOS speech_detected flag becomes the primary onset trigger (fallback designed into the speech lane) (task t11)
- [follow_up] reachy_nova has no CI — the TDD gate is local pytest only; adding CI is a follow-up outside this plan
