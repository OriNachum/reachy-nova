# event-based senses + seamless reactions

> Reachy Nova's senses become one verifiable event pipeline: speech is captured with an always-on ring buffer + VAD that backtracks 1-2s to the utterance start, and every sense — speech, audio direction, touch, image, face — reaches the agent as a logged, event-based signal it can answer with speech, harmonic vocalizations, movement, or any combination, seamlessly

## Audience

- Ori (operator and admin of the robot) and anyone sharing a room with Nova; secondarily the nervous-system/dashboard consumers reading the event stream, and future contributors extending senses or reaction skills

## Before → After

- Before: Today the mic feeds Sonic continuously with no speech events, no pre-roll, and near-silent logging (RMS every 500 chunks, injects dropped at DEBUG); DoA is a private head-tracking signal; touch/image/face events exist but with rules gaps; reactions are speech or movement only — no non-speech voice, and the skill set is fixed at build time
- After: Every sense is a logged MQTT event the agent actually receives: speech onsets arrive with their first word intact (1-2s backtrack), direction/touch/image/face all reach the agent through one observable fabric, the agent answers with any mix of speech, harmonic vocalization, and movement, and Nova can forge new reaction skills live via qwen3 — validated, staged, auto-activated, and visible as forge/* events

## Why it matters

- Being heard correctly is the robot's front door: a clipped first word or a silently dropped sensory inject reads as deafness or indifference. One verifiable event pipeline makes the experience seamless AND debuggable, and the live skill-forge turns feedback into growth instead of a backlog

## Requirements

- A speech-capture lane: the mic already records continuously (main.py feeds every chunk straight to Sonic), but nothing detects speech — add a rolling pre-roll ring buffer + VAD so a speech onset emits a speech_detected event whose audio starts 1-2s BEFORE the onset, so the first word is never clipped
  - honesty: Honest only if a test plays speech starting at t0 and the captured clip's audio begins at or before t0 — the backtrack is measured, not assumed; detection delay up to the transcribe cadence is covered by the ring buffer, not by luck
- Per-stage structured logs across the whole sensory path — capture, AEC/preprocess, VAD gate, event publish, inject, reaction — so 'speech was captured and heard correctly' is verifiable from the log alone; today main.py logs RMS once per 500 chunks and nova_sonic silently drops throttled injects at DEBUG level
  - honesty: Honest only if a dropped or throttled inject appears in the log at INFO+ with a reason — 'seamless' must never be achieved by silently discarding sensory input
- Audio direction becomes an event: XMOS DoA already returns (angle, speech_detected) but is only polled at 5Hz for head tracking inside tracking.py — publish audio_direction events to MQTT and make direction available to the agent (e.g. 'the voice came from your left')
  - honesty: Honest only if audio_direction events carry a usable bearing (left/right/front semantics verified against the XMOS 0=left, pi/2=front mapping) and are rate-limited so continuous speech does not flood the bus
- Close the event-coverage gaps for the already-event-based senses: touch (pat_level1/2), image (vision_description) and face (face_recognized) all publish MQTT + inject today, but rules.yaml has NO explicit entries for face/* or tracking/pat_* — every sensory source gets an explicit nervous-system rule, including the new speech and audio_direction sources
  - honesty: Honest only if rules.yaml has an explicit entry for EVERY source that publishes events (grep-verifiable) — no sense rides the default rule
- A harmonics-vocal reaction seam: the agent can currently speak (Sonic) or move (gesture skill) but has NO non-speech voice — add a vocalize skill that synthesizes harmonic chirps/tones through the existing speaker path (resample_output + push_audio_sample), so reactions compose freely: speak, vocalize, move, or all at once
  - honesty: Honest only if vocalize produces audible non-speech sound on the robot speaker in a live run AND respects the AEC reference path (the robot must not hear its own chirps as user speech)
- A skill-forge seam: Nova can dispatch a code task (goal + sensory context + optionally an existing skill to improve, e.g. one that earned negative feedback) to a configurable qwen3 coder endpoint (OpenAI-compatible, env-configured — the robot and the rig are different machines); the result lands as a new/updated skill folder (SKILL.md + executor) in a runtime skills area, distinct from the shipped built-ins
  - honesty: Honest only if the forge round-trip is proven live against the real qwen3 endpoint at least once (request -> generated skill on disk), and an unreachable endpoint degrades to a logged forge/rejected event — never a hang or crash in the 50Hz loop
- Hot skill registration: SkillManager can discover and load a forged skill at runtime — noting Sonic's toolConfiguration is FIXED at session start (main.py passes get_tool_specs() to the NovaSonic constructor), so activation rides the existing nova_sonic.restart() path or waits for the next natural session restart; the activation moment is logged and announced to the agent
  - honesty: Honest only if a forged skill is actually callable by the model after activation (toolSpec present in the NEW Sonic session) — activation that only writes files but never reaches Sonic's tool list does not count

## Honesty conditions

- Honest only if every one of the five senses demonstrably reaches the agent as an event in a live run — a sense that only reaches head-tracking or a log line does not count as 'sent to the agent'
- Honest only if the trace comes from one real end-to-end run on the robot (not stitched from unit tests), with the five senses each showing a real event id
- Honest only if a diff of the Sonic feed path shows zero behavioral change when the VAD lane is disabled — byte-identical audio reaches Sonic with the lane on or off
- Honest only if no vendored/patched SDK or firmware files appear in the diff; new capability comes from consuming existing APIs only
- Honest only if new senses publish through NovaMQTT.publish_event and nothing else — no direct sockets, no second broker, and the direct-callback fallback still works when MQTT is down
- Honest only if the validator provably rejects a skill that imports os/subprocess/socket or calls outside the primitive allow-list (negative tests exist), and validation runs BEFORE any generated code is imported or executed
- Honest only if every forge lifecycle transition observed in a live run has a matching forge/* MQTT event — staging with no event is a violation, not an optimization
- Honest only if the delivered UX is validated with the operator physically present in a live session — audience claims are not satisfied by CI alone
- Honest only if each stated gap is verifiable in the pre-change tree at a cited location (main.py audio feed, nova_sonic inject throttle, rules.yaml coverage) — no strawman baseline
- Honest only if every after-state capability maps to a confirmed requirement claim with its own honesty condition met — the after-state is the sum of proven parts, not a vision statement
- Honest only if the seamlessness claims trace to measurable signals (backtrack test, drop logging, live trace) — 'feels alive' language never substitutes for the measured pipeline

## Success signals

- A single log trace shows one spoken sentence traveling capture -> VAD (with the 1-2s backtrack visible) -> event -> inject -> reaction, and each of the five senses (speech, direction, touch, image, face) produces a visible MQTT event that reaches the agent

## Scope / boundaries

- The continuous mic->Sonic feed stays untouched: Sonic does its own turn detection and barge-in (nova_sonic.py), so the VAD lane runs IN PARALLEL for eventing/logging/clip capture — never as a gate in front of Sonic's audio input, which would change conversation behavior
- reachy_mini SDK and XMOS firmware are consumed as-is: DoA via media.audio.get_DoA(), AEC channel 0 via preprocess_mic_audio — no SDK or firmware changes
- The MQTT nervous system stays the single event fabric: new senses join nova/events/<source>/<type> + rules.yaml + the existing LLM interrupt evaluator — no second bus, no new transport
- Forged skills are constrained, not arbitrary code: executors may only compose sanctioned reaction primitives (gesture engine, vocalize, inject_text, state reads, emotion events) — no network, filesystem, or subprocess access from generated code; a generation that fails validation (syntax, primitive allow-list) is rejected and loudly logged, never activated, and never crashes the robot
- Forged skills are staged before activation and every forge/activate/reject transition is a nervous-system event (forge/*) — the pipeline stays observable end-to-end like every other sense

## Non-goals

- No raw audio/frame firehose to the cloud: events carry metadata + short summaries; speech clips stay local (the existing feedback store's local/S3 policy is unchanged); this is perception plumbing, not a data-collection feature

## Scope exploration

- `s1` — `reachy_nova/main.py (50Hz loop, audio feed)`: mic audio is preprocessed (AEC ch0, resample to 16k) and fed to Sonic EVERY chunk with no VAD, no pre-roll, no speech event; the only audio log is RMS once per 500 chunks; DoA is polled at 5Hz solely for head tracking
  - seeds: `c2`, `c3`, `c7`
- `s2` — `reachy_nova/wake_word.py`: a rolling 4s buffer + parakeet-TDT ASR already exists, but only for the wake phrase during sleep mode — proof the always-buffered-audio pattern runs on this hardware
  - seeds: `c2`
- `s3` — `reachy_nova/tracking.py (update_doa, detect_snap, PatDetector)`: XMOS DoA returns (angle_radians, speech_detected) — a hardware VAD flag is already available; snap and pat fire events via _fire_event, but DoA itself never becomes an event
  - seeds: `c2`, `c4`
- `s4` — `nova_mqtt.py + docker/nervous-system/nervous_system.py + config/nervous-system/rules.yaml`: the event fabric exists end-to-end: publish_event -> nova/events/<source>/<type> -> rules (priority/urgency) -> optional Nova-2-Lite interrupt evaluator -> nova/inject -> sonic.inject_text; rules.yaml covers tracking/vision/slack/browser/memory/emotions but has NO face/* or pat entries
  - seeds: `c5`, `c10`
- `s5` — `reachy_nova/face_recognition.py + docs/components/patting.md + main.py callbacks`: face (YuNet+SFace, on_match with 30s cooldown), touch (two-tier pat reflex+deliberate) and image (vision_description) are already event-based with MQTT publish + inject — the user's 'basic face recognition' already exists; the work is coverage + logging, not new detectors
  - seeds: `c5`
- `s6` — `reachy_nova/skills.py + skill_executors.py`: skills ARE the reaction seams: SKILL.md folders + register_executor become Sonic toolSpecs the model calls freely — a new reaction (e.g. vocalize) is a new skill folder + executor, exactly the shape a code-writing model can produce
  - seeds: `c6`
- `s7` — `reachy_nova/nova_sonic.py (feed_audio, inject_text)`: inject_text has a 3s throttle and a skipped-while-speaking path that drop sensory notifications at DEBUG level — a 'seamless' pipeline must at least LOG these drops loudly; Sonic owns turn detection and barge-in, so VAD must not gate its input
  - seeds: `c3`, `c8`
- `s8` — `pyproject.toml + repo-wide grep for VAD libs`: no VAD dependency exists (no silero/webrtcvad hits); nemo_toolkit[asr] + parakeet are already installed, and the XMOS speech flag is free — the VAD engine choice is open but well-supplied
  - seeds: `c2`
- `s9` — `reachy_nova/gestures.py + audio output path (audio_pipeline.resample_output, media.push_audio_sample)`: movement gestures exist (nuzzle/purr/enjoy) and speech exists (Sonic), but NO sound-synthesis module anywhere — harmonics-vocal is green-field; the speaker path it would ride is already there
  - seeds: `c6`
- `s10` — `parakeet-as-VAD feasibility (wake_word.py timings)`: wake_word.py already runs parakeet over a 4s rolling buffer every 2s in a single background worker without blocking the sleep animation — evidence the same model can serve as the awake-mode speech detector; detection delay up to the transcribe interval is covered by the ring-buffer backtrack
  - seeds: `c12`
- `s11` — `skills discovery + Sonic tool lifecycle (skills.py, main.py, nova_sonic.py)`: skills are discovered once at startup and Sonic tools are fixed at session start; nova_sonic.restart() exists and is the plausible activation path for a newly forged skill
  - seeds: `c13`, `c14`
- `s12` — `face_manager.is_admin + FaceRecognition.is_admin_authenticated`: camera-based admin authentication already exists (voice claims explicitly rejected) — a ready-made primitive for gating forged-skill activation on a physically present admin
  - seeds: `c15`

## Decisions

- VAD engine = parakeet (resolves park v1): reuse the already-loaded parakeet-TDT model as the speech detector — periodic transcription over the rolling ring buffer, speech onset = transcript appears — more precise than an energy/XMOS flag and no second model in memory (the wake-word instance is reused, not duplicated). CONDITIONAL on being fast enough on-device
  - instruction: Benchmark on the robot before committing: parakeet transcription latency over a 2-4s rolling window must keep speech-event detection delay within the pre-roll budget (backtrack still captures the utterance start) and must not starve the 50Hz main loop or the sleep-mode wake-word cadence. If the latency check fails, fall back to the XMOS speech_detected flag as onset trigger with parakeet confirming asynchronously.

## Hard questions

- risk: The robot and the qwen3 rig are separate machines: rig downtime, model reloads, or lobes-gateway auth (Bearer) can make the forge seam flaky in exactly the way lobes-cli#91/#95 already bit colleague — the endpoint config and failure path need the same care as the senses
