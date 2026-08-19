# Delivery — kiro writer + pettable upgrade (2026-08-19)

Spec: `docs/specs/2026-08-19-kiro-writer-pettable-upgrade.md` · Plan:
`docs/plans/2026-08-19-kiro-writer-pettable-upgrade.md` (frame/plan slug
`kiro-writer-pettable-upgrade`). Built via workforce fan-out: waves
[t1,t3,t5,t8] → [t2,t4,t6,t7] → t9 → t10, TDD-gated merges, **809 tests
green** at delivery. All live evidence below was produced on the Reachy Mini
Wireless (CM4, `pollen@192.168.1.162`) on 2026-08-19.

## The three success signals (spec c17/h14)

### Signal 1 — a pat produces a spoken/gesture reaction: **PROVEN (10:59:04 BST)**

Ori patted the head; the journal shows the full chain in one tick —
`Pat level1! type=side_pat (2 presses)` → `pat-acknowledge fired
run=pet-reaction` (Ori saw the antennas react) → **`nova-pat-cheer fired
run=speak say='I love getting pats on my head! This feels wonderful!'`**
(the kiro-authored rule) → harness inject: "I feel someone petting me and my
body is already leaning into it — that feels nice!". Recorded in the round-2
acceptance section (t11 signal 3); closes OriNachum/reachy-nova#7.

Tuning that made it land: while face-tracking the default gate was only
~3 % open (vs ~13 % idle), so `REACHY_PAT_STILL_EPS_DEG_S=8.0` was set via a
systemd drop-in (`~/.config/systemd/user/reachy-runtime.service.d/pat-eps.conf`).
Honest caveat: a follow-up pat ~2 min later did not re-trigger — detection
while actively tracking remains intermittent; the upstream receptive-window
discussion continues.

The supporting evidence chain:

- Upstream fix deployed (t1): reachy-mini-cli#168 closed by the v0.49.0
  cadence-invariant deg/s stillness gate (PR #169); merged into the device's
  `wireless-motor-enable` checkout (merge `d458fd5`, version 0.49.0, local
  motor-enable/face work preserved; conflicts resolved toward the released
  gate). 493 on-device pat-related tests pass post-merge.
- Gate measurably open: sampling `reachy/events/sense/snapshot` for ~35 s
  after the runtime restart: **68 samples — 59 blocked / 9 available
  (~13 % open fraction)**, vs the archived 19/19 (0 %) measurement in #168 —
  inside the #82 design's expected 10–15 % band. `pat_state.blocked_reason`
  is live (`stillness` 38, `clock-gap` 21).
- Two rules now react to a pat: the runtime's own `pat-acknowledge` →
  `pet-reaction`, and the kiro-authored `nova-pat-cheer` (below).

All of the above landed together in the live pat at 10:59:04 — see the
PROVEN header. Also recorded in
`docs/plans/2026-08-11-harness-round-2-alive-senses-resilient-start.md`
(t11 signal 3).

### Signal 2 — a forge dispatch yields an activated, callable skill: **PROVEN**

Live on the CM4, through the real standing session (`kiro-cli acp
--trust-all-tools --model minimax-m2.5 --agent-engine v2 --agent
nova-writer`):

- First dispatch: kiro authored a skill using `ctx.speak` → the AST
  validator **rejected it fail-closed**
  (`ctx.speak is outside the sanctioned primitive surface (line 2)`) and the
  rejection was announced to the conversation — the safety gate proven on
  real generated code. Root cause fixed (see "bugs fixed live").
- Second dispatch (after the prompt fix): goal → authored → validated →
  staged → **auto-activated in 8 s** ("I just learned a new skill:
  greet-cheer."), then `use_skill("greet-cheer")` ran it and the robot's
  conversation received "Hey there! Great to see you!" via `ctx.say`.
- The forged skill **survives service restarts**: on the next harness boot
  the journal shows `Hot-registered forged skill: greet-cheer` /
  `re-registered 1 forged skill(s)`.

### Signal 3 — a kiro-authored rule lands, survives reload, is live: **PROVEN**

- `author_rule(goal, session)` end-to-end result:
  `{'ok': True, 'rule_id': 'nova-pat-cheer', 'verdict': 'reload confirmed:
  {"ok":true,...,"react":5,...}'}` — goal → kiro → fenced rule → schema
  validation → sentinel-block merge (operator bytes verbatim) → engine
  reload **CONFIRMED**, 5 react rules live.
- The intermediate failures were themselves the fail-closed contract working
  live: an unbounded `speak` rule (no `duration_s`) came back
  `reload rejected: ... would let it hold its channel forever` — surfaced
  verbatim, never swallowed. Both prompts were then fixed (below).
- The rule *fires* on a pat — folded into Signal 1's pending live pat.

## Service-level state (t9)

`reachy-nova-harness.service` runs with `FORGE_WRITER=kiro`,
`KIRO_AGENT=nova-writer` (device `.env`): the supervisor lists
`kiro_session` ("one warm session, watchdog armed"), the real
`kiro-cli acp` child runs under it, and `forge`/`use_skill` joined the
published tool surface (refused with a named reason when the writer is not
configured).

Resource bounds during and after live runs: memory stable (~1.1–1.2 G used,
2.3 G cache, no OOM), `~/.kiro/sessions` bounded at ~48 K, session-recycle
threshold env-tunable (`KIRO_HISTORY_MAX`, default 50). **Disk is the tight
one: 561 M free (96 % full)** — below the ~2.4 G the plan assumed; flagged
as follow-up.

## Bugs found and fixed live (all pushed on `wireless-harness`)

1. `fix(kiro): KIRO_AGENT env -> --agent flag` — t5 shipped the nova-writer
   agent config but no seam selected it (`244d626`).
2. `fix(forge): the authoring prompt now enumerates the sanctioned ctx
   surface` — PROMPT_TEMPLATE said "only the primitives available on ctx"
   without listing them; affects both writers (`9f9d241`).
3. `fix(kiro-rules): teach the authoring prompt the duration_s requirement`
   — the engine rejects looping runs (`speak`) without `duration_s`; the
   same trap the round-2 face rule hit (`8cc7e66`).
4. `fix(kiro): fall back to ~/.local/bin for a bare kiro-cli PATH miss` —
   under systemd the service PATH lacks `~/.local/bin`, so the standing
   session died at spawn (`b18a8a0`).

Device-side operational fixes (not repo code): stale `nova-typing` rule
(`run = "look"`, unknown to the current engine) removed from the
nova-managed block — it poisoned every full-file reload; `.env` gained
`FORGE_WRITER=kiro` + `KIRO_AGENT=nova-writer`.

## Deviations

- **d1 (recorded on the plan, proposed→for confirmation):** no wave-2 task
  composed the kiro components into the harness. Added directly:
  `reachy_nova/harness/forge_leg.py` (SkillForge(kiro) + runtime-only
  SkillManager + restricted ForgedSkillContext, activation with no Sonic
  restart via the generic `use_skill` tool), `forge`/`use_skill` in
  `harness/tools.py`, `KiroSessionUnit` wired in `harness/app.py` gated on
  `FORGE_WRITER=kiro` (`abf67eb`).

## Before/after (spec c15/c16, h12/h13)

Before matched the scope exploration's record (s2/s3): FORGE_BASE_URL was
the only writer, no ACP code in the repo, no optional-dependencies table.
After: bare `uv sync` unchanged (packaging tests pin the mandatory list;
`[kiro]` extra is deliberately empty), the writer inert unless
`FORGE_WRITER=kiro`, and the whole loop operable with the existing device,
install and deploy paths — no new infrastructure.

## Known warts / follow-ups

- **Upstream: runtime tick overruns** — `[SENSE stage=rule source=tick
  event=overrun] duration_ms≈100–185 budget_ms=20` and the harness's
  matching `engine-heartbeat-lost` flapping. Known upstream issue
  (reachy-mini-cli "behavior engine achieves 23 Hz, not 50", cross-linked
  on #168); harness degrades to named drops, pat gate still opens.
- **Disk: 561 M free (96 %)** — needs a cleanup pass before anything new
  lands on the device.
- **rules_overlay has no removal API** — a stale rule inside the nova block
  poisons every reload until hand-removed; consider `remove_rule(id)`.
- **`use_skill` result of `greet-cheer` was `None`** — the forged executor
  returned no status string; harmless, but the authoring prompt could ask
  for a return value.
