# kiro writer + pettable upgrade

> `reachy_nova` runs on the pettable reachy-mini-cli and gains Kiro CLI over ACP as an optional on-device writer engine — code and rules authored by a Kiro agent holding cognition-tier device control at the Nova 2 Lite boundary
> instruction: Acceptance is on-robot, not in unit tests: run the issue #7 checklist for the pat leg, then drive one real forge dispatch and one real rule-authoring pass through the standing Kiro session; record both in the delivery/acceptance doc

## Audience

- Ori (operator of the Wireless) and the `reachy_nova` harness itself — the runtime self-extension loop that consumes the writer's output; secondarily anyone running `reachy_nova` on a Reachy Mini Wireless CM4

## Before → After

- Before: Pat sense structurally dead on the Wireless (19/19 'blocked', reachy-mini-cli#168); the only writer engine is a remote OpenAI-compatible qwen3 endpoint (`FORGE_BASE_URL`) on a separate machine — no on-device writer, no ACP client anywhere in `reachy_nova`, no optional-dependencies table in pyproject
- After: The robot feels pats again (t11 signal 3 recorded in the round-2 acceptance record), and a standing Kiro agent on the CM4 — full shell as pollen, ACP agent-engine v2 with v3 opt-in, model configurable — authors skill code through the forge pipeline and rules through the rules-overlay seam; the feature installs with zero mandatory new dependencies (a \[kiro\] extra exists only if something genuinely needs packaging)

## Requirements

- Upgrade leg is an on-device deployment, not a code change here: agentculture/reachy-mini-cli#168 closed 2026-08-19 by the v0.49.0 cadence-invariant deg/s stillness gate; the device checkout ~/git/reachy-mini-cli (branch wireless-motor-enable) already carries merge ab565cf 'deploy: merge spec/pettable-wireless-168'. Remaining work is OriNachum/reachy-nova#7's resume checklist: align to the released fix, restart reachy-runtime, re-measure `pat_state`.availability leaving 'blocked', live pat test (pat-acknowledge -> rule/fire -> sensory inject + pet-reaction), record t11 signal 3 in the round-2 acceptance record
  - honesty: `pat_state`.availability measurably leaves 'blocked' on the deployed v0.49.0 gate (nonzero open fraction over a sampling window comparable to the 19/19 measurement), AND a human pat produces the full chain: pat cue -> pat-acknowledge rule/fire -> harness sensory inject + pet-reaction motion, visible in both journals
- Kiro plugs in at the existing forge writer seam: SkillForge.dispatch today POSTs to `FORGE_BASE_URL`/chat/completions (qwen3), parses two fenced files (SKILL.md + executor.py), stages, AST-validates, auto-activates. A Kiro/ACP writer is an alternative dispatch backend behind the SAME stage -> validate -> activate pipeline and the same forge/\* event vocabulary (staged/activated/rejected)
  - honesty: A real dispatch through the Kiro writer produces two fenced files that parse, validate, stage and auto-activate exactly like the `FORGE_BASE_URL` path — and every Kiro-side failure (dead session, timeout, unparseable output) resolves to forge/rejected with a reason, never an exception on the caller's thread or a blocked 50Hz loop
- ACP is the operation channel and 'v2/v3' means the agent engine, not the CLI major: kiro-cli 2.18.1 on the Wireless ships 'kiro-cli acp' (Agent Client Protocol over stdio) with --agent, --model, --effort, --trust-all-tools/--trust-tools, and --agent-engine v1|v2(default)|v3 — v2 is the supported default, v3 a pass-through option, exactly matching 'v2/v3 (v3 optional)'
  - honesty: A full ACP round-trip (initialize -> session/new -> prompt -> artifact back) completes against kiro-cli 2.18.1 on the device with --agent-engine v2 and --model minimax-m2.5; v3 is accepted purely as a pass-through flag value and the harness never defaults to it
- Kiro-authored RULES flow through the existing `rules_overlay` seam, never raw file writes: the nova-managed sentinel block inside the operator's <state>/behavior/rules.toml, schema-validated before touching the file, temp-file + os.replace atomic write, reload verdict awaited and REJECTED reported — the same three-stage fail-closed path `create_rule` uses today
  - honesty: After a Kiro-authored rule lands: every operator byte outside the nova-managed block is verbatim-identical, the rule merged by id inside the block, and the engine's reload verdict was awaited and reported — a REJECTED verdict reaches the caller, never silently swallowed
- Runs on the CM4 of the Reachy Mini Wireless: kiro-cli 2.18.1 is already installed and has run sessions there (arm64, user pollen), and the harness this feature extends is the thing deployed on that device — no cross-compilation or new runtime is introduced
  - honesty: The standing Kiro session plus one active dispatch run on the CM4 without starving the harness: 50Hz-equivalent responsiveness holds, no OOM, and disk stays within the ~2.4G headroom (session history compaction keeps ~/.kiro/sessions bounded)
- Kiro agent tool surface = full shell as pollen on the CM4 (--trust-all-tools; resident dev agent). 'Nova 2 Lite boundary' is its architectural tier — cognition-level harness citizen, never a second `reachy_mini` SDK owner — NOT a tool allow-list. This amends c10's 'acts only through sanctioned seams' reading: the seams govern how it participates in the harness, not what its shell may touch (per resolved q1)
  - instruction: Ship a kiro agent config (JSON under ~/.kiro/agents or repo-provisioned) granting the full tool set; start the ACP session with it; add the trust rationale to docs/security.md
  - honesty: The standing agent genuinely runs with the full kiro tool surface as pollen (--trust-all-tools or an agent config allow-listing read/write/shell) and this trust decision is documented in the repo (security doc note), not implicit
- Kiro runs as one standing kiro-cli acp session the harness keeps warm and watchdogs (Sonic-watchdog pattern), with auto-compact of session history to preserve tokens — kiro-native compaction if the ACP surface offers it, otherwise recycle the session at a history threshold (per resolved q2)
  - instruction: Implement as a harness unit: spawn kiro-cli acp under the supervisor, health-check via ACP ping/session liveness, restart on failure with backoff, and add a compaction/recycle policy with its threshold env-tunable
  - honesty: The session survives the harness lifecycle: supervisor starts it, a watchdog detects a dead/hung session and restarts it (Sonic-watchdog pattern), and history is auto-compacted — kiro-native if the ACP surface exposes it, else session recycle at a measured threshold — so tokens and disk stay bounded across days

## Honesty conditions

- Holds only when BOTH legs are live-proven on the Wireless: a real pat gets a reaction (not just an open gate in telemetry), and a Kiro-authored artifact (skill or rule) is produced over ACP on-device and reaches its live surface (activated skill / firing rule)
- The delivered diff shows no change to \[project.dependencies\]; reachy-mini-cli still appears nowhere in pyproject.toml or uv.lock after the upgrade leg lands
- The skill-forge doc's separate-machine design note is explicitly amended (not deleted silently) to name the on-device Kiro writer as a recorded revision, in the same PR that ships the writer
- Holds if the shipped feature is operable by Ori without new infrastructure (uses the existing device, kiro-cli install, and harness deploy path) and the forge/rules surface remains callable by the harness itself
- Accurately describes today: `FORGE_BASE_URL` is the only writer path in `skill_forge.py`, no ACP code exists in the repo, and pyproject has no optional-dependencies table — verified in the scope exploration (s2, s3)
- After delivery: 'uv sync' with no extras installs and runs unchanged (zero mandatory new deps), the Kiro writer activates only when configured, and both authoring paths (skill + rule) work from the standing session on the CM4
- Each of the three signals is observed live on the Wireless and recorded in an acceptance/delivery doc with its evidence (journal lines / snapshot samples / activated-skill name) — a signal not yet observed is recorded as pending, never claimed

## Success signals

- Three live signals on the Wireless: (1) a pat produces a spoken/gesture reaction (pat-acknowledge -> rule/fire -> sensory inject + pet-reaction), recorded as t11 signal 3; (2) a forge(goal=...) dispatch through the Kiro writer yields a skill that validates, stages, auto-activates and is callable by Sonic; (3) a Kiro-authored rule lands in the nova-managed block, survives the reload verdict, and fires from the behavior engine

## Scope / boundaries

- pyproject.toml is untouched by the upgrade leg: reachy-nova depends on PyPI reachy-mini 1.2.10 (uv.lock), NOT on reachy-mini-cli — the CLI is the runtime deployed as a git checkout on the Wireless, outside this package's dependency graph
- The forge doc records 'the robot and the coder rig are different machines by design' — Kiro-on-device deliberately revises that recorded decision and the revision must be recorded, not slipped in. Practical footing: kiro-cli is a thin client (inference is cloud-side; the on-device session already ran modelId minimax-m2.5), it is already installed (no new disk on the ~2.4G-free CM4), so the on-device cost is one CLI process per dispatch

## Non-goals

- Forge-artifact sandboxing is unchanged: Kiro-generated skill code still passes `forge_validator`'s AST allow-list gate and runs against ForgedSkillContext's seven-method surface. 'Full control' belongs to the Kiro AGENT, never to the executor.py artifacts it writes — per the user's explicit 'not the code it generates within forge'

## Assumptions

- A zero-dependency ACP client is feasible, so the \[kiro\] extra can be empty-or-tiny: cultureagent's ACP runtime (clients/acp/runtime/transport.py et al.) drives ACP agents — KiroCLI named explicitly in its backend capabilities sheet — over stdio JSON-RPC using stdlib-only imports; per the cite-don't-import policy `reachy_nova` cites that implementation rather than depending on cultureagent
- 'The Nova 2 Lite boundary' is the harness cognition tier: today Lite (via `NOVA_OMNI_MODEL_ID`) understands the runtime's clip and acts ONLY through sanctioned seams — `inject_text` into the conversation, and by extension MQTT/HTTP/intents/rules-overlay. The harness boundary is machine-enforced (tests/`test_harness_boundary.py`: no `reachy_mini` import, no `set_target`, AST-checked) and the Kiro agent, as a new harness citizen, sits inside that same enforced boundary — a cognition-level actor, not a second SDK owner

## Scope exploration

- `s1` — `OriNachum/reachy-nova#7 + agentculture/reachy-mini-cli#168 + device ~/git/reachy-mini-cli`: \#168 closed today by v0.49.0 'The robot can be petted again: a cadence-invariant deg/s stillness gate'; issue #7 body carries the 4-step resume checklist; device checkout already has the spec/pettable-wireless-168 merge (ab565cf) staged for t6 live acceptance
  - seeds: `c2`
- `s2` — `pyproject.toml + uv.lock`: dependency is 'reachy-mini' from PyPI (1.2.10 in uv.lock); reachy-mini-cli appears nowhere in the dependency graph; pyproject also has NO \[project.optional-dependencies\] table today — a \[kiro\] extra would be the first one
  - seeds: `c3`
- `s3` — `reachy_nova/skill_forge.py + docs/components/skill-forge.md`: the writer seam is already a clean boundary: dispatch runs on a daemon thread, every failure path resolves to forge/rejected, and the validator gate is writer-agnostic — an ACP-backed dispatcher slots in without touching the activation pipeline
  - seeds: `c4`
- `s4` — `device pollen@192.168.1.162: kiro-cli 2.18.1 (--help, acp --help, agent --help)`: kiro-cli 2.18.1 installed at ~/.local/bin (kiro-cli, kiro-cli-chat, kiro-cli-term); 'kiro-cli acp' subcommand exists with per-session model/effort/trust flags and --agent-engine {v1,v2,v3}; agent configs are JSON under ~/.kiro/agents with a tools allow-list (read/write/shell/aws/mcp...) and a model field
  - seeds: `c5`
- `s5` — `cultureagent clients/acp runtime (site-packages via culture repo) + agent_experience/backends/capabilities/acp.yaml`: transport/harness/wiring modules import stdlib only (no agent-client-protocol PyPI package); the capabilities sheet says 'ACP-speaking agents (KiroCLI, OpenCode) do not currently expose hooks; mcp: true, skills: false' — a proven, citable stdlib ACP client that already speaks to Kiro
  - seeds: `c6`
- `s6` — `reachy_nova/harness/rules_overlay.py`: rules are durable operator-owned config; the module owns only the sentinel-delimited nova block, merges by id, and never imports the peer engine — any new writer (Kiro included) must land rules through this module to keep the operator's bytes verbatim
  - seeds: `c7`
- `s7` — `reachy_nova/forge_validator.py + ForgedSkillContext (skills.py) per docs/components/skill-forge.md`: the validator is the sole activation gate (no admin gate, a recorded frame decision) and ForgedSkillContext caps the runtime surface at 7 ctx methods; swapping the writer does not touch either
  - seeds: `c8`
- `s8` — `docs/components/skill-forge.md (separate-machine design note) + device disk/session state`: separate-machine is an explicit design sentence in the doc; kiro sessions on-device prove cloud-side inference (minimax-m2.5 in ~/.kiro/sessions); disk headroom ~2.4G per the round-2 live-state record
  - seeds: `c9`
- `s9` — `tests/test_harness_boundary.py + reachy_nova/harness/vision_leg.py + docs/plans/2026-08-09-wireless-cli-harness-scope.md`: the boundary is already a machine-checked gate over `reachy_nova`/harness/ (forbidden: `reachy_mini` imports, `set_target` refs); the Lite/Omni vision leg consumes clip state and injects text only — it opens no camera and steers no head; single-owner constraint: reachy-mini-cli's runtime owns the SDK
  - seeds: `c10`
- `s10` — `device probe: kiro-cli binaries + ~/.kiro/{sessions,logs} on 192.168.1.162`: binaries present in ~/.local/bin, version 2.18.1 runs, session/log artifacts from 2026-08-18 exist — the CLI demonstrably works on this CM4 today
  - seeds: `c11`
- `s11` — `kiro model evidence (~/.kiro/sessions on device)`: one real session used modelId minimax-m2.5 — non-Anthropic frontier models are selectable on this account; kimi-2.1/glm-5 presence not yet observed, needs a live ACP model-list probe

## Decisions

- Model roster resolved by Ori (2026-08-19): minimax-m2.5 (or m2.1) is CONFIRMED on the account, GLM-5 is available, kimi is NOT. Default the Kiro writer to minimax-m2.5; GLM-5 stays a selectable alternative; model id remains env-configurable (`KIRO_MODEL` or similar), never hardcoded

## Open parks

- [unknown_nonblocking] kiro-cli agent-engine v3 behavior differences vs v2 are unknown (v3 ships behind --agent-engine v3 but is undocumented here); treated as an opt-in flag value passed through, never the default, until measured

## Resolved vagueness

- [unknown_nonblocking] Model roster on this Kiro account is unverified: kimi-2.1 and glm-5 were requested, but only modelId minimax-m2.5 is proven (a real on-device session). The actual selectable list needs an on-device probe — kiro-cli acp initialize/session-new model listing or the chat model picker — before any model id is hardcoded; model must stay configurable regardless — resolved: Ori verified the roster live: minimax-m2.5/m2.1 confirmed, GLM-5 available, no kimi. Default = minimax-m2.5, env-configurable (see c18)
