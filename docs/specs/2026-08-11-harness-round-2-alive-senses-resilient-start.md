# harness round 2: alive senses + resilient start

> Wireless harness round 2: resilient auto-start (clock-jump-proof Sonic, no competing auto-run), working barge-in, pat + face + tracking senses routed into Nova, and Nova Act browsing as a tool
> instruction: verify by the five live acceptance signals in c21 on the deployed wireless unit, after a real overnight power-off

## Audience

- Ori and anyone sharing a room with the two Reachy Mini units — plus the harness itself as the consumer of the runtime's senses

## Before → After

- Before: today a cold boot after a long power-off leaves a zombie Sonic stream until someone restarts the harness; the robot is deaf while it speaks (no barge-in); pats and faces are sensed by the runtime but never reach Nova; the camera and face senses are dead for a missing cv2; browsing exists as dormant flag-gated code
- After: the robot comes up alive from any cold boot with no manual restarts and nothing else able to seize it; you can talk over it and it stops to listen; it feels pats, recognizes and learns faces, sees through the camera clip, and can browse the web on request

## Why it matters

- a companion robot that must be manually revived, talks over you, and ignores touch is a demo, not a companion — this round closes the gap between senses the runtime already has and the mind that should feel them

## Requirements

- `nova_sonic.py` detects a wall-clock step (drift between time.time() and time.monotonic() beyond ~60s) and forces a Sonic session restart — the 2026-08-11 boot showed an NTP +14h step leaving a zombie Bedrock stream: injections flowed, zero responses, no error, until a manual harness restart
  - honesty: a forced clock step on the device (or a simulated one in tests) triggers exactly one Sonic session restart and the next spoken utterance gets a reply
- boot-time exclusivity is encoded in the repo, not just hand-applied on the device: harness/unit.py's rendered units (and an install doc/script for the hand-authored reachy-runtime.service) carry Conflicts=reachy-demo-mode.service, demo-mode stays manual-only (no \[Install\]), and the old ReachyMiniApp reachy-nova-autostart.service stays masked
  - honesty: a fresh render/install from the repo reproduces tonight's exclusivity: demo-mode cannot be enabled by preset, and starting it stops the runtime (Conflicts both ways)
- barge-in works by fixing gate policy, not adding barge-in code: `nova_sonic.py` already has `_handle_barge_in` (USER transcript while `_speaking` -> Nova Lite decision -> `on_interruption` -> speaker.preempt, wired in harness/app.py); it never fires because hearing.py drops every mic chunk while the EchoGate is armed, so Sonic is deaf during its own speech
  - honesty: with the gate policy off, speaking over the robot mid-utterance reaches Sonic as a USER transcript and `_handle_barge_in` fires — observable as a preempt in the journal
- the echo gate becomes a policy (env-selectable, e.g. `NOVA_ECHO_GATE`=off|half-duplex, default off on the wireless): XVF3800 hardware AEC was verified active live on 2026-08-10 — gate.py's docstring premise 'no verified hardware AEC (spec c19)' is now disproven for this device
  - honesty: with hearing suppression off on the wireless, the robot does not converse with its own echo (the 2026-08-10 echo-loop regression stays fixed, now by hardware AEC alone)
- Nova Act browsing surfaces as a Sonic tool: `nova_browser.py` already exists flag-gated (`NOVA_ACT_ENABLED`, default off, lazy imports — `nova_act`/playwright never imported when off); the harness adds a browse tool to `TOOL_SPECS` wired to the NovaBrowser queue in app.py, with progress emitted back via inject
  - honesty: with `NOVA_ACT_ENABLED` unset nothing imports `nova_act`/playwright (boundary test), and with it set a spoken browse request produces a queued task and a spoken result
- the device CLI venv gets the \[vision\] extra installed: live reachy/state/senses on 2026-08-11 shows face and clip both unavailable with reason=vision-extra-absent (cv2 missing), while pat is available:true and `frame_available`:true — one install unblocks both the face sense and the clip rider
  - honesty: after installing the \[vision\] extra on the device, reachy/state/senses reports face available:true and reachy/state/clip carries a real clip path
- conversational face learning rides FaceStore's existing temporary-face seam (`remember_temporary` -> `temp_id` -> enroll(name, embedding) in reachy/vision/`face_store.py`): the harness gets an `enroll_face` tool Sonic can call when someone introduces themselves; if the runtime exposes no enrollment intent, that seam is requested via a reachy-mini-cli issue rather than the harness writing `face_store` files directly
  - honesty: saying 'I'm <name>' to an unknown face results in a `face_store` identity the runtime greets by name on the next sighting, via a sanctioned runtime seam (intent or CLI), never by the harness writing `face_store` files

## Honesty conditions

- all five c21 acceptance signals pass live on the wireless unit
- with hearing suppression off, utterances still play one at a time (no overlapping playback), and setting `NOVA_ECHO_GATE`=half-duplex restores the old suppression behavior
- `test_harness_boundary.py` still passes after the round: no harness module imports `reachy_mini` or runs a tracking loop
- acceptance is judged by people in the room interacting naturally (speech, touch, presence) — no SSH, dashboard, or debug tooling involved in the happy path
- an overnight unplugged robot answers the first spoken sentence after power-on with no human intervention beyond the power switch
- each before-state fact is journal-cited: the 2026-08-11 zombie stream, gate suppression in hearing.py, vision-extra-absent in reachy/state/senses, `NOVA_ACT_ENABLED` default off
- after the round, a first-time visitor can hold a natural interrupted conversation and get a touch reaction without being told any special rules
- each signal is demonstrated live and recorded in the scope doc's acceptance section, not simulated

## Success signals

- live acceptance on the wireless: (1) power-off overnight, power-on, robot answers speech with zero SSH; (2) interrupting it mid-sentence makes it stop and respond; (3) a pat gets a spoken/gesture reaction; (4) it greets an enrolled face by name and learns a new name by voice; (5) 'search the web for X' returns a spoken answer

## Scope / boundaries

- EchoGate stays: speaking.py also uses it for one-speaker-at-a-time playback windowing (0.48.0 has no speaker arbitration), and half-duplex must remain selectable for devices without hardware AEC — hearing suppression and playback serialization get decoupled, not deleted
- tracking behavior (orienting to sound/faces, feel-alive motion) stays in the runtime (orient.py, `feel_alive.py`); the harness never runs its own tracking loop — it receives events and may command goto via the existing intent tools. The old-app tracking.py and `face_recognition.py` are NOT ported; `test_harness_boundary.py`'s AST gate (harness never imports `reachy_mini`) holds

## Assumptions

- patting is a wiring job, not a port: reachy-mini-cli already ships `pat_sense.py` / `pet_reaction.py` / motion/pat\*.py with availability verdicts (`sense_availability.py` carries pat + `pat_event` vocabularies); the harness adds the pat source to `NOVA_BUS_SOURCES` and rules.yaml routes so Sonic hears 'you are being petted'
- face recognition is runtime-owned too: `face_sense.py` + vision/face.py + `face_store.py` exist in reachy-mini-cli; issue #120 says the face sense needs the \[vision\] extra installed in the device venv, with availability published to reachy/state — the harness leg is bus routing + injection, plus verifying the extra is installed on-device
- the vision leg reads the runtime's camera clip rider: `clip_rider.py` overwrite-in-place clip path is published on retained reachy/state/clip, and `nova_omni.py`'s understand() already takes a rolling clip + still + context — no new camera ownership anywhere

## Scope exploration

- `s1` — `reachy_nova/nova_sonic.py (session lifecycle)`: restart machinery exists (session generation counter, restart(), stream-died 3s backoff, 4s speaking watchdog at line ~449) but nothing detects a wall-clock step; the 2026-08-11 journal shows the NTP +14h jump at 21:10 left a zombie stream from 06:59 with injections but zero responses
  - seeds: `c2`
- `s2` — `device systemd units (pollen@192.168.1.162, live journals 2026-08-11)`: reachy-nova-autostart.service (old ReachyMiniApp, preset-enabled) retired to ~/unit-backups and masked; reachy-demo-mode.service rewritten manual-only with Conflicts=reachy-runtime.service; runtime unit given Conflicts=reachy-demo-mode.service — applied live but not yet encoded in harness/unit.py or an install doc
  - seeds: `c3`
- `s3` — `reachy_nova/harness/hearing.py + gate.py + app.py`: TeeHearing.`_drain` drops every chunk while gate.active (hearing.py:406-408); app.py already wires sonic.`on_interruption` = speaker.preempt; `nova_sonic.py`:323 fires `_handle_barge_in` on USER-transcript-while-speaking — the gate is the sole reason barge-in never triggers
  - seeds: `c4`, `c5`
- `s4` — `XVF3800 capture path (live test 2026-08-10)`: hardware AEC verified active: robot's own speaker barely registers on the mic (human speech RMS ~0.09 vs ~0.002 floor during playback), contradicting gate.py's 'no verified hardware AEC (spec c19)' premise
  - seeds: `c5`, `c6`
- `s5` — `reachy_nova/harness/speaking.py`: SonicSpeaker uses the same EchoGate for one-speaker-at-a-time playback windowing (waits for the prior window before posting the next utterance) — gate removal would break playback serialization; policy split needed
  - seeds: `c6`
- `s6` — `reachy-mini-cli reachy/behavior/pat_sense.py + sense_availability.py + motion/pat*.py`: pat detection, warmup/idle publication and pat/`pat_event` availability vocabularies already exist runtime-side; the old app's tracking.py pat levels (level1/level2, scratch/`side_pat`) need no port
  - seeds: `c7`, `c9`
- `s7` — `reachy-mini-cli reachy/behavior/face_sense.py + reachy/vision/ (face.py, face_store.py, producer.py)`: face recognition with identity store is runtime-owned; issue #120 documents the \[vision\]-extra-absent failure mode with availability verdicts published to reachy/state — device venv extra status unverified
  - seeds: `c8`
- `s8` — `reachy_nova/harness/bus.py`: `DEFAULT_SOURCES`='rule,intent,motion' with `NOVA_BUS_SOURCES` override (bus.py:113-115); pat/face/vision sources are not subscribed today and rules.yaml has no entries for them
  - seeds: `c7`, `c8`
- `s9` — `reachy_nova/nova_browser.py`: NovaBrowser is flag-gated by `NOVA_ACT_ENABLED` (default off) with lazy `nova_act`/playwright imports and a queue-based worker; nothing in harness/tools.py `TOOL_SPECS` or app.py references it yet
  - seeds: `c10`
- `s10` — `reachy-mini-cli reachy/behavior/clip_rider.py + reachy_nova/nova_omni.py`: clip rider writes an overwrite-in-place clip under `behavior_dir` and the path reaches retained reachy/state/clip via the engine's state writer; `nova_omni`.understand() already accepts clip+still+context — the vision leg composes from existing parts
  - seeds: `c11`
- `s11` — `live MQTT bus (mosquitto_sub on-device, 2026-08-11)`: retained reachy/state/senses: pat available:true, `frame_available`:true, face+clip unavailable reason=vision-extra-absent; cv2 import fails in the CLI venv; 10s event sample shows only reachy/events/sense/snapshot (~27/s) — discrete pat/face event topic names still to be confirmed on a first live pat
  - seeds: `c12`

## Decisions

- browser execution: on-device Nova Act confirmed; AgentCore-hosted browser to be verified as the preferred surface (user decision, q1)
- face learning is BOTH ways (user amendment to q2): one-time CLI enrollment into the runtime's `face_store`, AND Nova learns names from conversation — an unknown face seen now can be named by voice ('I'm Ori') and becomes a stored identity

## Open parks

- [unknown_nonblocking] whether nova-act supports Bedrock AgentCore Browser as a remote execution surface from the harness — verify during implementation; local chromium is the fallback
- [unknown_nonblocking] exact discrete MQTT topic names for pat/face events (reachy/events/<source>/<type>) — confirm on first live pat/face after the vision extra lands

## Resolved vagueness

- [unknown_blocking] whether the deployed runtime venv has the \[vision\] extra and whether pat/face events actually appear on reachy/events/\* MQTT topics (vs only internal cues) — verifiable on-device before planning — resolved: verified live 2026-08-11: pat sense available:true on the deployed runtime; face + clip rider unavailable solely for vision-extra-absent (no cv2 in the CLI venv); only residual unknown is the exact discrete pat/face event topic names, confirmable on first live pat — nonblocking
