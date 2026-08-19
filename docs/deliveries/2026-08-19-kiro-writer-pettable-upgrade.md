# Delivery Summary — kiro writer + pettable upgrade

plan: `kiro-writer-pettable-upgrade` · run: `complete` · date: `2026-08-19`
baseline: `devague summary skeleton`

## Intent

Execute the converged plan seeded from the frame "`reachy_nova` runs on the
pettable reachy-mini-cli and gains Kiro CLI over ACP as an optional on-device
writer engine — code and rules authored by a Kiro agent holding cognition-tier
device control at the Nova 2 Lite boundary". Two legs: resume the parked pat
sense on the Reachy Mini Wireless (upstream reachy-mini-cli#168 fix, issue
OriNachum/reachy-nova#7 checklist, t11 signal 3), and ship Kiro CLI 2.18.1 as
an optional on-device writer engine over ACP (standing watchdogged session,
agent-engine v2 with v3 pass-through, minimax-m2.5 default, zero mandatory
dependencies). Run via /assign-to-workforce: waves [t1,t3,t5,t8] →
[t2,t4,t6,t7] → t9 → t10, TDD-gated merges. All live evidence produced on the
CM4 (`pollen@192.168.1.162`) on 2026-08-19.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Device upgrade: align ~/git/reachy-mini-cli on the Wireless to the released v0.49.0 pat-gate fix (#168), restart reachy-runtime, and measure the gate
- `t2` — Live pat acceptance: a human pat produces the full reaction chain; record t11 signal 3
- `t3` — Stdlib-only ACP client for kiro-cli: new module `reachy_nova`/`kiro_acp.py` (cited from cultureagent's ACP runtime, cite-don't-import) speaking initialize -> session/new -> prompt over stdio JSON-RPC to 'kiro-cli acp'
- `t4` — Standing session unit: `reachy_nova`/harness/`kiro_session.py` keeps one warm ACP session under the supervisor — watchdog, restart with backoff, history compaction/recycle
- `t5` — Kiro agent config, trust rationale, and design-revision docs: repo-provisioned agent config granting the full tool surface (read/write/shell) as pollen; docs/security.md documents the trust decision; docs/components/skill-forge.md's separate-machine note explicitly amended to record the on-device Kiro writer
- `t6` — Kiro writer backend in `skill_forge.py`: writer selection (`FORGE_WRITER`=http|kiro) routes dispatch through the standing ACP session; same two-fenced-file parse -> stage -> validate -> auto-activate pipeline and forge/\* events
- `t7` — Rule authoring through the Kiro writer: a goal-shaped rule request goes to the standing session and the resulting rule lands ONLY via harness `rules_overlay`'s nova-managed block
- `t8` — Packaging: first \[project.optional-dependencies\] table with a \[kiro\] extra (empty-or-tiny), zero mandatory new deps, writer inert unless configured
- `t9` — CM4 live proof: standing Kiro session on the Wireless completes a real ACP round-trip, one real forge dispatch (activated, callable skill) and one real rule-authoring pass, within resource bounds
- `t10` — Acceptance and delivery record: all three success signals recorded with evidence; before/after states verified; spec honesty conditions checked off

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | v0.49.0 merged into device `wireless-motor-enable` (merge `d458fd5`; conflicts resolved toward the released gate, local motor-enable/face work kept); 493 on-device pat tests pass; gate measured **9/68 available (~13 %)** vs the archived 19/19 (0 %); `blocked_reason` live. No reachy_nova dependency change (pyproject/uv.lock untouched by this leg). |
| `t2` | delivered | Live pat 10:59:04 BST: `Pat level1! type=side_pat` → `pat-acknowledge`/`pet-reaction` (Ori saw the antennas react) → kiro-authored `nova-pat-cheer` spoke → harness inject. Recorded as t11 signal 3 in the round-2 acceptance section; issue #7 closed with evidence. Needed `REACHY_PAT_STILL_EPS_DEG_S=8.0` (see decisions). |
| `t3` | delivered | `reachy_nova/kiro_acp.py` (stdlib-only, AST-asserted) + 27 tests: full JSON-RPC framing round-trip, engine v1/v2/v3 pass-through (invalid rejected), env seams `KIRO_CLI_BIN`/`KIRO_MODEL`/`KIRO_AGENT_ENGINE` (+`KIRO_AGENT`, PATH fallback — added post-merge, see decisions). Merged `b1f766b`. |
| `t4` | delivered | `reachy_nova/harness/kiro_session.py` (`KiroSessionUnit`) + 20 tests: spawn-on-start, dead/hung detection, capped exponential backoff with healthy-period reset, recycle at `KIRO_HISTORY_MAX` (default 50), clean stop; boundary gate passes. Merged `795bc78`. |
| `t5` | delivered | `config/kiro/nova-writer.json` (full tool set) + 11 config tests, guarded provisioning step in `scripts/install-device-units.sh`, trust rationale in `docs/security.md`, dated design-revision amendment in `docs/components/skill-forge.md`. Merged `9c567dc`. |
| `t6` | delivered | `FORGE_WRITER=http|kiro` seam in `skill_forge.py` (+14 tests): shared `_finish_from_content` tail so both writers hit the identical stage→validate→activate pipeline and `forge/*` vocabulary; every kiro failure mode lands as `forge/rejected`; `forge_validator.py`/`skills.py` empty diff. Merged `697878e`. |
| `t7` | delivered | `reachy_nova/harness/kiro_rules.py` (`author_rule`) + 22 tests: schema-restating prompt, fence parse, land ONLY via `rules_overlay.upsert_rule`, operator bytes byte-compare-verbatim, REJECTED verdict surfaced. Merged `1ed378a`. |
| `t8` | delivered | First `[project.optional-dependencies]` table, `kiro = []` deliberately empty; packaging tests pin the 15 mandatory deps and assert no kiro import at package-import time. Merged `e37a395`. |
| `t9` | delivered | Live on the CM4: real ACP round-trip 6.1 s (engine v2, minimax-m2.5, nova-writer agent); forge dispatch → `greet-cheer` authored, validated, **activated in 8 s**, ran via `use_skill`, survives service restarts; `author_rule` → `nova-pat-cheer` reload **CONFIRMED**, later fired on the live pat; `kiro_session` in the supervisor's unit list under the real service. Memory stable, `~/.kiro/sessions` ~48 K. Needed the d1 wiring plus 4 live-found fixes (below). |
| `t10` | delivered | This artifact (restructured into the delivery-summary shape after the live pat landed); round-2 acceptance record updated; issue #7 closed. |

## Mid-work Decisions

- `d1` — (recorded via /deviate, **proposed — awaiting Ori's confirm**) t9
  presumes the harness runs the kiro writer, but no wave-2 task ships the
  integration wiring: nothing constructs KiroSessionUnit in build_app(),
  IntentTools has no forge tool, and SkillForge is not instantiated anywhere
  in the harness path. Main agent added the wiring leg directly:
  `reachy_nova/harness/forge_leg.py` (SkillForge(kiro) + runtime-only
  SkillManager + restricted ForgedSkillContext, activation with no Sonic
  restart via the generic `use_skill` tool), `forge`/`use_skill` on the
  published tool surface (refused with a named reason when unwired),
  KiroSessionUnit under the supervisor gated on `FORGE_WRITER=kiro`
  (`abf67eb`, +10 tests).
- Four bugs found by live testing, fixed and redeployed the same hour — none
  were in the plan: `KIRO_AGENT` env → `--agent` flag (t5 shipped the config,
  nothing selected it; `244d626`); the forge prompt never enumerated the
  sanctioned ctx surface, so kiro's first skill used `ctx.speak` and was
  correctly AST-rejected (`9f9d241`); the rules prompt didn't state the
  `duration_s` requirement for looping runs, so the engine rejected an
  unbounded `speak` rule (`8cc7e66`); systemd's PATH lacks `~/.local/bin`, so
  the standing session died at spawn — bare-name fallback added (`b18a8a0`).
- Device-side operational fixes (not repo code): stale `nova-typing` rule
  (`run="look"`, unknown to the current engine) removed from the nova-managed
  block — it poisoned every full-file reload; `.env` gained
  `FORGE_WRITER=kiro` + `KIRO_AGENT=nova-writer`;
  `REACHY_PAT_STILL_EPS_DEG_S=8.0` set via systemd drop-in after measuring
  the gate only ~3 % open while face-tracking (the fix's own env seam, used
  as designed).
- t2's dependency on human presence was handled by running the code waves
  first and folding the physical pat into the same event that proved the
  kiro rule fires.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t9` (`d1`) | plan gap: component tasks (t3,t4,t6,t7) each built their module but no task covered composing them into build_app()/IntentTools — t9's acceptance was unreachable without the wiring leg | acceptable |
| `t2` | the confirmed task assumed the deployed gate alone would make the robot pettable; live measurement showed ~3 % open while face-tracking, so an env tune (`REACHY_PAT_STILL_EPS_DEG_S=8.0`) was required beyond the plan's steps; repeat-pat detection remains intermittent | needs-follow-up |
| `t5` | the shipped agent config was inert until the post-merge `--agent` seam (`244d626`) — the task's own acceptance ("start the ACP session with it") was only met after that fix | acceptable |

## Evidence

- tests: full suite `uv run pytest` — **809 passed** (last full run at `b18a8a0`; baseline before the run: 716, one xdist-only flake in `tests/chaos/test_chaos_aws_loss.py::test_repeated_cloud_flaps_never_require_a_restart`, passes serially)
- tests (on-device): reachy-mini-cli `-k pat` — 493 passed post-merge
- commits: `a5c1247..58491f1` on `wireless-harness` (task merges `b1f766b`, `9c567dc`, `e37a395`, `697878e`, `1ed378a`, `795bc78`; wiring `abf67eb`; fixes `244d626`, `9f9d241`, `8cc7e66`, `b18a8a0`)
- PRs / issues: PR #6 (branch), issue #7 (closed with evidence), agentculture/reachy-mini-cli#168 (upstream fix, closed by their PR #169 / v0.49.0)
- journals (device, 2026-08-19): pat chain at 10:59:04 BST on both services; `component started name=kiro_session` + "one warm session, watchdog armed" at 10:47:15; `Hot-registered forged skill: greet-cheer` on service restart
- live measurements (device): pat gate 9/68 available idle, 3/109 while tracking (pre-tune); ACP round-trip 6.1 s; forge-to-activation 8 s; mem ~1.1–1.2 G used / no OOM; `~/.kiro/sessions` 48 K; disk 561 M free

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| The robot feels pats again and reacts (t11 signal 3) | high | journal 10:59:04 chain + Ori's independent confirmation; recorded in `docs/plans/2026-08-11-harness-round-2-alive-senses-resilient-start.md`; issue #7 closed |
| A kiro-authored skill was forged, validated, activated and called live on the CM4 | high | journal `greet-cheer` activation + `use_skill` output; survives restarts (`Hot-registered forged skill` line) |
| A kiro-authored rule landed via rules_overlay and fired live | high | `author_rule` result `ok=True, reload confirmed (react: 5)`; `nova-pat-cheer fired` in the 10:59:04 journal |
| The AST validator rejects unsafe generated code fail-closed, live | high | first dispatch rejected: `ctx.speak is outside the sanctioned primitive surface (line 2)` |
| The standing session runs under the supervisor with watchdog | high | journal `component started name=kiro_session`; `kiro-cli acp` child process under the service |
| Zero mandatory new dependencies; writer inert unless configured | high | `tests/test_packaging.py` (10 tests) · `tests/test_harness_forge_leg.py::test_tools_refuse_forge_without_a_wired_leg` |
| Session history stays bounded across days of runtime | low | recycle logic unit-tested (`tests/test_kiro_session.py`) and `~/.kiro/sessions` 48 K during the run — multi-day soak not yet observed |
| Repeat pats are reliably detected while the robot tracks a face | unverified | one clean detection; a follow-up pat ~2 min later did not re-trigger — not claimed done |

## Remaining Work / Follow-up

- **Ori: confirm (or reject) deviation `d1`** — recorded proposed; the wiring
  it covers is merged and live-proven.
- **Repeat-pat reliability while tracking** — intermittent even at
  eps=8.0 deg/s; upstream receptive-window discussion continues on
  reachy-mini-cli (their #168 thread's options 2/3). Consider a receptive
  still pose during engagement.
- **Upstream runtime tick overruns** — `duration_ms≈100–185 budget_ms=20`
  and matching harness `engine-heartbeat-lost` flapping (known "23 Hz"
  upstream issue); harness degrades to named drops; also contributes
  `clock-gap` blocked time to the pat gate.
- **Disk: 561 M free (96 %)** on the device — below the ~2.4 G the plan
  assumed; needs a cleanup pass before anything new lands.
- **`rules_overlay` has no removal API** — a stale rule inside the nova block
  poisons every reload until hand-removed (bit us live via `nova-typing`);
  consider `remove_rule(id)`.
- **Forged executors may return `None`** — `greet-cheer` worked but returned
  no status string; the authoring prompt could ask for a return value.
- **Multi-day soak of the standing session** — restarts/backoff/recycle are
  unit-tested but not yet observed across days (the `low`-confidence claim
  above).
