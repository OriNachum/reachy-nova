# Delivery Summary — fast-witty-remembering-nova

plan: `fast-witty-remembering-nova` · run: `partial` · date: `2026-09-06`
baseline: `devague summary skeleton`

## Intent

Make the Reachy Mini's mind answer within about a second instead of after a
long silence, speak like a dry, teasing friend rather than an eager helper,
vary its reactions, hand deeper reactions to Nova 2 Lite, and remember the
day's topics across restarts and session rotations. The run executed the
18-task plan exported from the converged frame (spec
`docs/specs/2026-09-05-fast-witty-remembering-nova.md`, plan
`docs/plans/2026-09-05-fast-witty-remembering-nova.md`) through
`/assign-to-workforce`: seven waves, one agent per task in an isolated
worktree, each merge gated by the full suite before and after, then deployed
to the robot and exercised live. Five defects found only on the robot were
fixed on top of the plan and are recorded as deviations.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Daemon sounds API: DaemonClient gains `list_sounds`() and `delete_sound`(filename) over GET /api/media/sounds and DELETE /api/media/sounds/{filename}
- `t2` — Sonic session config and watchdogs: endpointingSensitivity on sessionStart, contentEnd type/role/stopReason logged at INFO, liveness watchdog counts only speech-energy input
- `t3` — Persona file and loader: config/persona/nova.md in the Wit-inspired register with two one-shot exchanges, harness/persona.py resolving `NOVA_PERSONA_PATH` with an embedded default
- `t4` — Conversation ledger: harness/ledger.py with locked NDJSON append of transcripts and delivered senses, quiet-window exclusion, 24 h truncation, atomic writes and latched drop lines
- `t5` — Feature switches: harness/switches.py resolving `NOVA_CHUNKED_PLAYBACK`, `NOVA_LITE_REACTIONS`, `NOVA_MEMORY` (default on) and `NOVA_PERSONA_PATH` once, fail-open with a warning, logged in one line
- `t6` — Mood: harness/mood.py — a small decaying state fed by pats, recognised faces, conversation turns and silence, rendered as one short context sentence
- `t7` — Device probes on the robot: measure back-to-back `play_sound` chunk gap/click and Nova 2 Lite round-trip latency; record the numbers as plan evidence
- `t8` — Chunked speaker: speaking.py flushes ~1 s chunks at low-energy boundaries or after 300 ms of audio inactivity, per-chunk filenames with delete-after-window, gate-serialised, preempt purges the rest
- `t9` — Deferred cues: a cue arriving while Sonic generates is parked (latest-wins per sense class, TTL 5 s) and drained through `inject_text` with its age in the text when the utterance ends
- `t10` — Memory compactor: harness/`memory_compactor.py` — a background thread that asks Nova 2 Lite to distil the ledger into topics and important items with timestamps, 24 h expiry, atomic write, and a history() view for session replay
- `t11` — Lite reactor: harness/`lite_reactor.py` — a single worker with a bounded latest-wins queue that turns an opted-in cue plus context (sense history, day memory, mood, last exchanges) into a one-line reaction plan within a timeout, falling back to the template
- `t12` — Session rotation with history replay: NovaSonic gains a `history_provider` hook sent between the system prompt and the audio contentStart, a ~7 min rotation timer that waits for idle with a hard deadline, zero delay for a healthy rotation, and one journal line per rotation
- `t13` — Bus routes Lite-tier cues: rules.yaml entries gain react: lite; the bus reserves dedupe, hands the rendered cue to the reactor, and applies the voice/quiet markers and SenseHistory record to the plan text; voice: none never reaches Lite
- `t14` — Integration in app.py: persona + amy voice, switches, ledger, compactor, reactor, deferred cues and the history provider wired into `build_app`, with every leg degrading to a named absent line
- `t15` — Docs and version: architecture.md, CLAUDE.md module map, docs/components/{persona,memory,lite-reactor}.md, .env.sample entries for the new switches, pyproject version 0.4.0
- `t16` — Boundary audit: run the boundary and packaging tests, grep the harness for legacy modules and daemon endpoints, diff hearing.py, and check every after/before-state clause maps to a requirement
- `t17` — Deploy to the robot: check disk headroom, pull the merged branch into ~/git/reachy-nova, set .env (`NOVA_SONIC_ENDPOINTING` unset=HIGH, switches on), restart reachy-nova-harness, confirm the switch line and persona in the journal
- `t18` — Live acceptance and delivery doc: a 10-minute conversation on the robot with two rotations, journal evidence for every timing honesty condition, Ori's verdict on tone, docs/deliveries/\<date\>-fast-witty-remembering-nova.md

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `DaemonClient.list_sounds`/`delete_sound` with an injectable DELETE transport; commit `6507802`, merged `5cc051e` |
| `t2` | delivered | endpointing on sessionStart, contentEnd logged, speech-gated liveness; commit `c0e885b`, merged `0d8dd2d`; the gate was reworked twice after live evidence (`d7`, `d9`) |
| `t3` | delivered | `config/persona/nova.md` (1,464 chars, dry/teasing register, two one-shot exchanges) + `harness/persona.py` with embedded default; commit `c25f366`, merged `0acbd12` |
| `t4` | delivered | `harness/ledger.py`, `statedir.ledger_path()`; commit `eb1e615`, merged `2402488` |
| `t5` | delivered | `harness/switches.py`; commit `327fb28`, merged `0b3b6c8` |
| `t6` | delivered | `harness/mood.py`; commit `86618df`, merged `406a137` |
| `t7` | delivered | probes run over ssh: Lite 0.97 / 1.06 median / 1.34 s; gate-exact chunk boundary seamless, 100 ms pre-roll audibly early (plan risks `r1`, `r2` resolved) |
| `t8` | delivered | chunked speaker (size / inactivity / state-change flush, per-chunk files with cleanup); commit `b461555`, merged `0fd79df`; queue bound raised after live drops (`d10`) |
| `t9` | delivered | `harness/deferred_cues.py` + `inject_text(..., sense_class=)`; commit `fee4b8c`, merged `6c5f375` |
| `t10` | delivered | `harness/memory_compactor.py`, `statedir.memory_path()`; commit `f6e1c55`, merged `05ec5ce`; history shape fixed after the live replay loop (`d8`) |
| `t11` | delivered | `harness/lite_reactor.py`; commit `062ae17`, merged `c966e61`; recent-line feedback added after live repeats (`d9`) |
| `t12` | delivered | history replay hook + idle-gated rotation with hard deadline; commit `93f2b7c`, merged `dca2896`; replay normalised + circuit breaker added (`d8`) |
| `t13` | delivered | `react: lite` in rules.yaml (7 entries), bus routing with markers on delivery; commit `0b7d885`, merged `0c4af78` |
| `t14` | delivered | `build_app` wires persona + amy, switches, ledger, compactor, reactor, mood, history provider; commit `c890eb8`, merged `bb20aae`; vision cue wrapper added after live monologues (`d11`) |
| `t15` | delivered | docs (architecture §5–§9, CLAUDE.md, three component pages, nova_sonic.md), `.env.sample`, version 0.4.0; commit `f6244d1`, merged `dbeb7c1` |
| `t16` | delivered | audit run at `bb20aae`: suite green, boundary test green, no legacy imports, hearing.py zero diff, daemon paths enumerated; note in the PR body |
| `t17` | delivered | robot pulled the branch (editable install, no pip — `d5`), restarted, journal shows switches/persona/endpointing/engine-live; redeployed after each live fix, last at `f4ee46c` |
| `t18` | partial | live evidence collected for timings, one rotation, quiet-room survival and reaction variety (see Evidence); NOT yet evidenced: the heard-to-audio median (no USER transcript this evening), a pat during speech, Nova answering "what were we talking about" after a rotation, a guest speaking, and Ori's verdict on tone |

## Mid-work Decisions

All eleven deviation records are **proposed** (`--origin llm`) and await
Ori's confirmation; they are quoted as recorded, not as approved.

- `d1` — fixed a pre-existing race in `tests/chaos/test_chaos_aws_loss.py` (the flap test healed the poster on a counter bumped before the purge) — t11's added test load made it fail about one run in three, blocking the TDD gate; the reactor was not at fault
- `d2` — task branches are `agent/fwrn-<task>` rather than `agent/<task>` — `agent/t3..t8` already existed from the kiro-writer round
- `d3` — the chunked speaker waits out the previous chunk's audio, not the echo gate's padded window — `EchoGate()` carries a 1 s ear-side margin that would have put a second of silence between every chunk
- `d4` — mood reaches Sonic only through the Lite reactor's context, not as direct text — injecting a mood sentence on every cue would add noise; left for Ori's call
- `d5` — deployed without a pip reinstall although `pyproject.toml` changed — only the version line changed, the device runs an editable install, pip on the CM4 takes minutes on a 91 %-full card
- `d6` — Sonic's `contentEnd` carries no role field on the wire; speaking now also ends on an AUDIO `END_TURN` — the role check alone never matched and every turn ended on the 4 s watchdog
- `d7` — liveness input became a sustained speech burst and injects/tool results stopped counting — a single loud chunk latched "input flowing" and a quiet body cue legitimately gets no reply
- `d8` — the first memory replay opened with one of Nova's own lines; Bedrock refused every restart for ten minutes — history is now USER-first and alternating, with a replay circuit breaker
- `d9` — liveness window 900 s and floor 0.05 by default; the Lite reactor feeds back its last five lines — a person at the desk still tripped the 180 s window, and three pats got the identical line
- `d10` — playback queue bound 90 chunks (was 8); a 35 s reply dropped 12 chunks — plus a process slip: that commit shipped with one red test for a few minutes, fixed in `03c73d1`
- `d11` — scene descriptions reach Sonic as a brief, capped body cue — the raw 400–850 char Omni text became a 30 s monologue at every harness start
- `d12` — the PR #24 review round: eight fixes in `d7fbcab` (failed chunk deletes retried then abandoned by name; at most two Lite calls in flight plus botocore timeouts; Lite vocalizations synthesised and played through the speaker; the ledger records delivered senses only; persisted memory entries validated; the vision cue through the ledger wrapper; the speech-burst window reset per session; the ledger's quiet check under its lock) and one pushback (the reactor's internal bounded queue is the repo's existing never-block-the-caller pattern; CLAUDE.md's pattern line now names it)
- `d13` — Bedrock cuts a stream with no interactive content for 295 s (silent audio does not count; three deaths at exactly 296 s of quiet); a quiet idle session now rotates cleanly after 270 s without interactive content, the cutoff restarts with no backoff if it still hits, and the Lite reactor refuses an exact repeat of a recent line (`fc2178b`)
- not covered by any record: `t14` kept the bare `sonic.inject_text` for the browse and qq-memory legs (only the bus and now the vision leg go through the wrapper), per the task's stated allowance

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t11` (`d1`) | t11's tests added enough scheduler load that a pre-existing chaos-test race failed ~1 in 3 parallel runs, blocking the TDD gate; the reactor itself was not at fault | acceptable |
| `t1` (`d2`) | avoid reusing or deleting another round's branches | acceptable |
| `t8` (`d3`) | app.py builds EchoGate() with margin_s=1.0 for the ear; paying it between chunks of one sentence would insert 1 s of silence per chunk | acceptable |
| `t14` (`d4`) | injecting a mood sentence on every cue would add noise to the live voice model; the Lite plans already carry it | needs-follow-up |
| `t17` (`d5`) | a pip install on the CM4 takes many minutes and the root disk is at 91 % with 1.3 GB free; the code that runs is the checkout's | acceptable |
| `t2` (`d6`) | with the role check alone every turn ended on the 4 s speaking watchdog, which delayed deferred cues by 4 s | acceptable |
| `t2` (`d7`) | deployed 0.4.0 still restarted a quiet room; one loud chunk latched "input flowing"; a quiet body cue inject legitimately gets no reply | risky |
| `t12` (`d8`) | c11/c12 specified the replay order but not Bedrock's role constraints; the live wire shape supplied both | needs-follow-up |
| `t2` (`d9`) | h18 promised no liveness restart in a 10-minute quiet stretch; two attempts at a mic-energy gate could not tell a person moving about from a person talking, and the rotation makes a long window safe | acceptable |
| `t8` (`d10`) | the whole-utterance design never queued more than one item per utterance, so the bound of 8 was never exercised | acceptable |
| `t14` (`d11`) | a bare scene report is not a cue the persona knows how to treat briefly; invisible in tests because the vision leg is Omni-model-gated on the dev box | acceptable |
| `t8`, `t11`, `t14` (`d12`) | Human gate 3 (the PR) produced valid reliability and correctness findings the plan's tests had not exercised; each is small, covered by a new test, and deployed to the robot | acceptable |
| `t12`, `t11` (`d13`) | the spec's rotation targeted the 8-minute connection limit; the 295 s interactive-content rule is a second, undocumented Bedrock limit that only a quiet room exposes | acceptable |
| `t18` | the 10-minute spoken session with Ori has not happened; only pats and the robot's own replies are in the journal, so the tone verdict and the heard-to-audio median are unmeasured | needs-follow-up |

## Evidence

- tests: `uv run pytest -n auto` at `f4ee46c` — 1795 passed (baseline before the round: 1387)
- tests: `tests/test_harness_boundary.py` — 19 passed
- tests: per-honesty-condition node ids filed as plan evidence `e1`–`e22` against obligations `o1`–`o16` (`devague evidence --list`)
- lint: `markdownlint-cli2 --config ~/.markdownlint-cli2.yaml` on the round's docs — 0 errors
- commits: `4583c44..fc2178b` on `spec/fast-witty-remembering-nova` (42 commits; 15 task commits, 15 TDD-gated merges, 1 test fix, 5 post-deploy fixes, the delivery summary and devague state, 1 review-fix commit)
- tests: `uv run pytest -n auto` at `fc2178b` — 1810 passed, 1 skipped
- PR #24: CI test + SonarCloud gate green; Qodo review of 9 threads answered in `d7fbcab` (8 fixes, 1 pushback)
- robot journal (harness on `9bb4d90`/`f4ee46c`, 2026-09-06 00:42–00:53 BST, read over ssh): `rotation delay=0 replay=2 age=420s` at 00:49:43.76 with the new session listening 0.5 s later; first chunk played 0.80 / 0.37 / 0.59 / 0.56 / 0.50 s after `Utterance audio started` on five replies; 0 liveness restarts, 0 stream deaths, 0 `queue-full` drops in the window; four consecutive pats answered "Thank you!", "Mmm, that feels nice!", "That tickles!", "Yup, that's nice!" — filed as `e23`–`e26`
- robot journal (before the round, harness `f936f11`, 2026-09-05): 9.9 s of silence before a 12.52 s reply, 6.0 s before a 4.20 s reply; short replies queued 4.3–4.6 s after first audio on the 4 s speaking watchdog; six liveness restarts 180 s apart in a quiet room
- device probes (t7, 2026-09-06): Nova 2 Lite from the robot 0.97 / 1.06 median / 1.34 s; two 1 s tones posted gate-exact heard seamless, 100 ms pre-roll heard early (plan risks `r1`, `r2`)
- deviations: `devague deviate --list` — `d1`–`d11`, all proposed
- PRs / issues: none opened yet (final PR is the next step); upstream follow-ups reachy-mini-cli#162 (speaker feed) and the clip rider not producing new clips

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| first audio lands within about a second of Sonic's first chunk, regardless of reply length | high | journal 00:43–00:51: 0.37–0.80 s on five replies · `tests/test_harness_speaking.py::test_a_short_reply_is_flushed_by_inactivity_alone` · commit `b461555` |
| long replies play every chunk in order without drops | high | 0 `queue-full` lines after `9bb4d90` (12 before) · `tests/test_harness_speaking.py::test_a_long_reply_generated_faster_than_real_time_never_drops_chunks` |
| the spoken turn ends on Sonic's own end-of-turn event, not the 4 s watchdog | high | journal: `contentEnd type=AUDIO role=? stopReason=END_TURN` followed by a `state-change` flush at the same millisecond · commit `2912fef` |
| a session rotates before the 8-minute limit with history replayed and no audible hole | high | journal `rotation delay=0 replay=2 age=420s`, next session 0.5 s later · `tests/test_sonic_rotation.py` |
| Nova still knows the topic after a rotation | unverified | no one asked it after the rotation — not claimed done |
| a quiet room no longer restarts the stream every 3 minutes | medium | 0 liveness restarts in the 00:42–00:53 window (three in the same window on `f936f11`); a full 10-minute silent stretch with nobody at the desk has not been observed |
| the persona is on disk, in the dry/teasing register, with the amy voice and no character names | high | file `config/persona/nova.md` · `tests/test_harness_persona.py::test_persona_names_no_character_and_no_book` · journal `persona source=file:…/config/persona/nova.md chars=1464` |
| the same cue gets different reactions | medium | four different pat lines in a row after `f104731`; only pats were exercised live |
| a pat during a reply is reacted to after the reply ends, with its age | medium | `tests/test_harness_deferred_cues.py` (49 tests); not exercised live — no pat landed during speech this evening |
| the day's topics and important items survive restarts and rotations | medium | `history replayed blocks=2` on every start after `1309c69` · `tests/test_harness_memory_compactor.py` (27 tests); memory file still held no distilled topics (no conversation to distil) |
| Nova 2 Lite plans opted-in reactions within 2 s with template fallback | high | journal `stage=react … latency=6xx–1210ms` · `tests/test_harness_lite_reactor.py::test_hung_lite_call_times_out_without_blocking_other_cues` |
| every new behaviour has an env kill switch and the journal names them at start | high | journal `switches chunked_playback=on lite_reactions=on memory=on persona=default` · `tests/test_harness_switches.py` |
| the harness still opens no SDK client and touches only the daemon's upload/play/stop/sounds/volume paths | high | `tests/test_harness_boundary.py` (19) · audit grep at `bb20aae` |
| the robot runs the delivered code | high | device checkout `f4ee46c`, `engine live` 4 s after restart at 00:55:54 |
| version 0.4.0 with a green suite | high | `pyproject.toml` · `uv run pytest -n auto` 1795 passed |
| it feels like talking to a friend with a dry wit rather than a helper | unverified | Ori's verdict pending — not claimed done |

## Remaining Work / Follow-up

- `t18` — run the spoken 10-minute session with Ori on `f4ee46c`: talk (to measure heard-to-audio), pat mid-sentence (deferred cue live), stay quiet 10 minutes with nobody at the desk (h18 in full), ask "what were we talking about" after a rotation (h8 second half), have a guest speak (h14), record the tone verdict (h1/h17); then update this document's claims. Owner: Ori + operator.
- confirm or reject deviations `d1`–`d13` (`devague deviate --confirm dN`); `d4` (mood only via Lite), `d7` and `d8` are the ones with substance
- PR #24 is open with CI green and the first review round answered; merging publishes 0.4.0 to PyPI, after which the robot's checkout goes back to `main`
- `d4` follow-up: decide whether the compactor's replay context block should carry the mood sentence so Sonic sees it directly
- upstream reachy-mini-cli: the clip rider has produced no new clip since boot (same `ts` all evening) — the vision leg only ever sees the boot clip; and #162 (streaming speaker feed) remains the gap-free sub-second path
- pre-existing, untouched: an awscrt `InvalidStateError` traceback on every Sonic stream close; a wheel install lacks `config/nervous-system/rules.yaml` (plan risk `r9`); the installed metadata version on the device stays 0.3.0 until the next pip reinstall
- per-person memory needs the runtime to publish the recognised name (plan risk, frame park `v5`)
