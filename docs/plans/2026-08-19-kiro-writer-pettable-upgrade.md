# Build Plan — kiro writer + pettable upgrade

slug: `kiro-writer-pettable-upgrade` · status: `exported` · from frame: `kiro-writer-pettable-upgrade`

> `reachy_nova` runs on the pettable reachy-mini-cli and gains Kiro CLI over ACP as an optional on-device writer engine — code and rules authored by a Kiro agent holding cognition-tier device control at the Nova 2 Lite boundary

## Tasks

### t1 — Device upgrade: align ~/git/reachy-mini-cli on the Wireless to the released v0.49.0 pat-gate fix (#168), restart reachy-runtime, and measure the gate

- instruction: On-device (pollen@192.168.1.162): fetch/checkout the released v0.49.0 tag (or its merge into the deployed branch) in ~/git/reachy-mini-cli, reinstall if editable-install requires it, restart reachy-runtime, then sample reachy/events/sense/snapshot for ~32s and tally `pat_state`.availability values; compare against the archived 19/19-blocked measurement in #168
- covers: c2, c3, h9
- acceptance:
  - `pat_state`.availability shows a nonzero open fraction sampled over a window comparable to the original 19/19-blocked measurement (~32s)
  - `reachy_nova`'s diff is empty for dependencies: reachy-mini-cli absent from pyproject.toml and uv.lock before and after

### t2 — Live pat acceptance: a human pat produces the full reaction chain; record t11 signal 3

- instruction: With t1's gate measurably open: pet the head (level1-style scratches), grep both service journals for 'rule/fire' + 'pat', verify pat-acknowledge -> sensory inject + pet-reaction; append signal 3 with the journal evidence to the round-2 acceptance section and close OriNachum/reachy-nova#7 citing it
- depends on: t1
- covers: h2
- acceptance:
  - journal evidence on both services: pat cue -> pat-acknowledge rule/fire -> harness sensory inject ('You feel someone petting you...') + pet-reaction motion
  - signal 3 recorded in docs/plans/2026-08-11-harness-round-2-alive-senses-resilient-start.md acceptance section and OriNachum/reachy-nova#7 closed

### t3 — Stdlib-only ACP client for kiro-cli: new module `reachy_nova`/`kiro_acp.py` (cited from cultureagent's ACP runtime, cite-don't-import) speaking initialize -> session/new -> prompt over stdio JSON-RPC to 'kiro-cli acp'

- instruction: New file `reachy_nova`/`kiro_acp.py` + tests/`test_kiro_acp.py`. Cite (don't import) cultureagent's clients/acp runtime transport for the stdio JSON-RPC framing. Public surface: a KiroAcpSession class with start/initialize/`new_session`/prompt/close; env: `KIRO_MODEL` (default minimax-m2.5), `KIRO_AGENT_ENGINE` (default v2, v3 allowed), `KIRO_CLI_BIN` (default kiro-cli). No new pyproject deps
- covers: c5
- acceptance:
  - unit tests drive a fake ACP subprocess through the full round-trip and assert request/response framing
  - engine flag defaults to v2 with v3 accepted as pass-through (`KIRO_AGENT_ENGINE`); model from `KIRO_MODEL` default minimax-m2.5; a test asserts the module imports stdlib only

### t4 — Standing session unit: `reachy_nova`/harness/`kiro_session.py` keeps one warm ACP session under the supervisor — watchdog, restart with backoff, history compaction/recycle

- instruction: New file `reachy_nova`/harness/`kiro_session.py` + tests/`test_kiro_session.py`, following the harness supervisor/unit pattern (see harness/supervisor.py, harness/unit.py) and the Sonic watchdog shape. Must pass the harness boundary test (no `reachy_mini` import, no `set_target`). Liveness = ACP ping or last-activity age; restart with capped exponential backoff; compaction: probe for a kiro-native ACP method, else recycle (close + respawn with a fresh session) past `KIRO_HISTORY_MAX` prompts
- depends on: t3
- covers: c13, h8
- acceptance:
  - unit tests with a fake process prove: spawn on start, dead/hung session detected via liveness check and restarted with backoff, `stop_event` shuts it down cleanly
  - compaction policy: kiro-native if the ACP surface exposes it, else session recycle at an env-tunable history threshold (`KIRO_HISTORY_MAX` or similar) — tested at the threshold boundary

### t5 — Kiro agent config, trust rationale, and design-revision docs: repo-provisioned agent config granting the full tool surface (read/write/shell) as pollen; docs/security.md documents the trust decision; docs/components/skill-forge.md's separate-machine note explicitly amended to record the on-device Kiro writer

- instruction: Ship config/kiro/nova-writer.json (agent config: full tool set read/write/shell/aws, model field, name nova-writer); extend scripts/install-device-units.sh or docs/setup.md to provision it to ~/.kiro/agents; add the trust rationale to docs/security.md (full shell as pollen, why, blast radius); amend the 'different machines by design' paragraph in docs/components/skill-forge.md to record the on-device Kiro writer as a revision with date and rationale
- covers: c12, h7, c9, h10
- acceptance:
  - an agent config JSON ships in the repo (config/kiro/ or install script provisioned to ~/.kiro/agents) with the full tool set and model field; a test validates its shape against kiro's `agent_config` schema
  - docs/security.md gains the full-shell-as-pollen trust rationale; skill-forge.md's 'different machines by design' sentence is amended in the same PR, not deleted

### t6 — Kiro writer backend in `skill_forge.py`: writer selection (`FORGE_WRITER`=http|kiro) routes dispatch through the standing ACP session; same two-fenced-file parse -> stage -> validate -> auto-activate pipeline and forge/\* events

- instruction: Modify `reachy_nova`/`skill_forge.py` only (plus tests): introduce a writer seam — `FORGE_WRITER` env (http default, kiro) — where the kiro path submits the same two-fenced-file authoring prompt through the standing session handle passed in at construction; parametrize existing forge tests over both writers; add failure-mode tests (dead session, timeout, garbage output) asserting forge/rejected. Do NOT touch `forge_validator.py`, skills.py activation, or ForgedSkillContext
- depends on: t3
- covers: c4, h3
- acceptance:
  - with `FORGE_WRITER`=kiro, a mocked session returning SKILL.md+executor.py flows through stage->validate->activate identically to the HTTP path (existing tests parametrized over both writers)
  - dead session, timeout, unparseable output each resolve to forge/rejected with a reason — no exception on the caller's thread (tests); `forge_validator.py` and ForgedSkillContext have an empty diff

### t7 — Rule authoring through the Kiro writer: a goal-shaped rule request goes to the standing session and the resulting rule lands ONLY via harness `rules_overlay`'s nova-managed block

- instruction: New module (e.g. `reachy_nova`/harness/`kiro_rules.py`) + tests — do NOT edit `skill_forge.py` (t6 owns it) or `rules_overlay.py`'s write logic; call `rules_overlay`'s existing API with rules parsed from the Kiro session's output. Validate the rule dict against the overlay's schema before landing; byte-compare operator regions in tests; surface the reload verdict in the return value and senselog
- depends on: t3
- covers: c7, h5
- acceptance:
  - byte-compare test: every operator byte outside the sentinel block is verbatim-identical after a Kiro-authored rule lands; rule merges by id inside the block
  - the engine reload verdict is awaited and reported to the caller — a REJECTED verdict surfaces in the result and the senselog, never swallowed (test)

### t8 — Packaging: first \[project.optional-dependencies\] table with a \[kiro\] extra (empty-or-tiny), zero mandatory new deps, writer inert unless configured

- instruction: pyproject.toml: add \[project.optional-dependencies\] with kiro = \[\] (or the genuinely-needed minimum if t3/t4 surface one); extend tests/`test_packaging.py` to assert the mandatory dependency list is unchanged and that no kiro-cli process spawns on import/startup without `FORGE_WRITER`=kiro
- covers: c16, h13
- acceptance:
  - bare 'uv sync' installs and runs unchanged: `test_packaging` asserts no new mandatory dependencies and that importing `reachy_nova` without kiro config never spawns kiro-cli
  - the Kiro writer activates only when `FORGE_WRITER`=kiro (or equivalent) is set; unset config leaves the HTTP path and all existing behavior untouched (test)

### t9 — CM4 live proof: standing Kiro session on the Wireless completes a real ACP round-trip, one real forge dispatch (activated, callable skill) and one real rule-authoring pass, within resource bounds

- instruction: On-device, after waves 0-1 deploy: start the harness with `FORGE_WRITER`=kiro, confirm the standing session in the supervisor's unit list; run one voice-or-API forge(goal=...) producing an activated skill Sonic can call; author one rule via the kiro path and see it fire; record top/free/df samples and ~/.kiro/sessions size across the run; verify 50Hz-equivalent responsiveness held
- depends on: t4, t5, t6, t7, t8
- covers: c11, h4, h6
- acceptance:
  - live round-trip against kiro-cli 2.18.1 on-device with --agent-engine v2 and --model minimax-m2.5 produces an artifact back over ACP
  - resource bounds hold with the session warm plus one active dispatch: harness responsiveness (50Hz-equivalent) holds, no OOM, disk stays within the ~2.4G headroom and ~/.kiro/sessions stays bounded by the compaction policy

### t10 — Acceptance and delivery record: all three success signals recorded with evidence; before/after states verified; spec honesty conditions checked off

- instruction: Write docs/deliveries/2026-08-19-kiro-writer-pettable-upgrade.md following the round-2 delivery doc's shape: each of the three signals with its evidence (journal lines, snapshot tallies, activated-skill name) or an explicit 'pending'; a before/after section citing s2/s3 provenance; the operability check (no new infrastructure beyond existing device/install/deploy paths); update the wireless-harness memory afterwards
- depends on: t2, t9
- covers: c1, h1, c14, h11, c15, h12, c17, h14
- acceptance:
  - a delivery doc (docs/deliveries/) records each of the three signals — pat reaction, forged-skill activation, Kiro-authored rule firing — with journal/snapshot/skill-name evidence, or an explicit 'pending', never claimed early
  - the record verifies the before-state description matched reality (s2/s3 provenance) and the after-state: operable by Ori with existing device/install/deploy paths, no new infrastructure

## Risks

- [unknown_nonblocking] kiro-native history compaction may not exist on the ACP surface of kiro-cli 2.18.1 — t4's fallback (session recycle at an env-tunable threshold) is the committed path until probed (task t4)
- [unknown_nonblocking] minimax-m2.5's fitness for the two-fenced-file forge protocol (SKILL.md + executor.py) is unmeasured — prompt/agent-config iteration may be needed before a dispatch validates; failures land as forge/rejected, so the loop stays safe while tuning (task t6)
- [unknown_nonblocking] agent-engine v3 behavior differences vs v2 are unmeasured — v3 stays opt-in pass-through, never the default (frame park v2) (task t3)
