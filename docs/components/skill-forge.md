# Skill Forge Documentation

This documentation covers the skill-forge loop — Nova's runtime
self-extension seam: dispatch a goal to a coder model, stage what comes
back, statically validate it, and auto-activate it into the live skill set.

## Overview

Before this arc, Nova's skill set was fixed at build time — whatever
`SkillManager.discover()` found under `reachy_nova/skills/` at startup, and
nothing else. The skill forge closes that gap: Nova can call a `forge` tool
with a goal (optionally: an existing skill to improve, e.g. one that earned
negative feedback), which dispatches to a configurable OpenAI-compatible
coder-model endpoint on a **separate machine** — the robot and the coder rig
are different machines by design. The reply is parsed into two fenced files
(`SKILL.md` + `executor.py`), staged to disk, and run through a static
validator before it is ever eligible to run. Only a validated skill gets
**auto-activated** — hot-registered into the live runtime and, once Sonic
next allows it, callable by the model.

The runtime self-extension loop, end to end:

```text
forge(goal=...) tool call
    -> SkillForge.dispatch()            (reachy_nova/skill_forge.py, daemon thread)
       -> POST to FORGE_BASE_URL/chat/completions (qwen3 endpoint)
       -> parse two fenced files: SKILL.md, executor.py
       -> write to ~/.reachy_nova/skills-forged/<name>/         [forge/staged fires here, ONLY if validation passed]
    -> forge_validator.validate(skill_dir)   (reachy_nova/forge_validator.py, AST-only)
       -> ok=False  -> move to skills-forged/.rejected/<name>/, forge/rejected fires
       -> ok=True   -> forge/staged fires
    -> _on_forge_event (reachy_nova/main.py) sees forge/staged
       -> activate_forged()             (reachy_nova/skills.py)
          -> move staged/<name> -> skills-active/<name>/
          -> SkillManager.discover_runtime(active_root)   (imports executor.py ONLY now, post-validation)
          -> restart Sonic now (voice_state idle) or defer to next natural restart
          -> forge/activated fires, Nova is told "I just learned a new skill: <name>."
```

**Files:** `reachy_nova/skill_forge.py` (`SkillForge`, the dispatch client),
`reachy_nova/forge_validator.py` (the AST-only static validator),
`reachy_nova/skills.py` (`discover_runtime`, `activate_forged`,
`ForgedSkillContext`), `reachy_nova/skill_executors.py`'s `_forge_executor`
(the tool-callable entry point), wired into the live loop in
`reachy_nova/main.py`.

## Auto-Activate Policy: No Admin Gate

**A validated forged skill activates automatically — there is no admin
approval gate in front of activation.** This is a deliberate, recorded frame
decision (q2 in the build plan for this arc), not an oversight: the only
gate between "the coder model generated something" and "it's live and
callable" is the static validator described below. `main.py`'s
`_on_forge_event` fires `_auto_activate_forged` on every `forge/staged`
event, unconditionally:

```python
# reachy_nova/main.py
def _on_forge_event(event_type: str, payload: dict):
    _forge_publish(event_type, payload)
    if event_type == "forge/staged":
        threading.Thread(
            target=_auto_activate_forged,
            args=(str(payload.get("name", "")),),
            daemon=True,
        ).start()
```

State this plainly because it's easy to assume otherwise: passing static
validation is **sufficient** for a forged skill to go live, not merely
sufficient for a human to be asked whether it should.

## Env Vars

| Var | Default | Purpose |
| :--- | :--- | :--- |
| `FORGE_BASE_URL` | *(none — required)* | Base URL of the OpenAI-compatible coder endpoint. Unset → every dispatch immediately rejects with `"endpoint not configured"`. |
| `FORGE_MODEL` | `qwen3` | Model name sent in the chat-completions request. |
| `FORGE_API_KEY` | *(none)* | If set, sent as `Authorization: Bearer <key>`. |

`SkillForge` also takes a constructor `timeout` (default `120.0`s) for the
HTTP round-trip.

## `forge/*` Events

Every lifecycle transition publishes through the caller's `publish`
callback (`main.py` republishes as MQTT `forge/<transition>`, e.g.
`forge/staged`), and `config/nervous-system/rules.yaml` carries an explicit
rule for each — no forge event rides the default rule:

| Event | Fired when | Payload | rules.yaml priority/urgency |
| :--- | :--- | :--- | :--- |
| `forge/staged` | A generated skill passed validation and is staged | `{name, path}` | LOW / BACKGROUND |
| `forge/activated` | A staged skill was hot-registered into the live skill set | `{name}` | LOW / BACKGROUND |
| `forge/rejected` | Endpoint unreachable/timeout/unparseable, or validation failed | `{reason, reasons, name?, path?}` | NORMAL / NOW, template `"A new skill idea didn't pass its safety check: {reason}"` |

`forge/rejected` is deliberately surfaced louder (`NORMAL`/`NOW`) than the
other two (`LOW`/`BACKGROUND`) — a failed forge attempt is never silent.

## The Validator's Allow-List Contract

`forge_validator.validate(skill_dir)` (`reachy_nova/forge_validator.py`) is
**AST-only** — it parses `executor.py` with the stdlib `ast` module and
never imports, compiles-to-exec, or otherwise runs the generated code. It
returns `(ok, reasons)`; rejection is fail-closed — anything that cannot be
positively verified is rejected with reasons, never waved through.

Checks:

- **Line cap**: `executor.py` over `MAX_EXECUTOR_LINES = 200` lines is
  rejected outright as "too large to trust", without even parsing further.
- **Import allow-list** (`ALLOWED_IMPORTS`): `numpy`, `math`, `time`,
  `typing`, `dataclasses`. Any other top-level import — direct or `from` —
  is rejected by name and line number.
- **Forbidden names** (`FORBIDDEN_NAMES`): mere appearance of `exec`,
  `eval`, `compile`, `__import__`, `open`, `input`, `getattr`, `setattr`,
  `delattr`, `globals`, `locals`, `vars`, `breakpoint`, `exit`, `quit`, or
  the roots `os`, `sys`, `subprocess`, `socket`, `shutil`, `pathlib`,
  `urllib`, `requests`, `http`, `importlib`, `ctypes`, `pickle`, `marshal`
  is a rejection — this closes off `getattr`-based allow-list bypasses on
  the `ctx` surface too.
- **The sanctioned `ctx` primitive surface** (`ALLOWED_CTX_ATTRS`): a
  forged skill's `execute(params, ctx)` may only call `ctx.gesture`,
  `ctx.vocalize`, `ctx.say`, `ctx.inject`, `ctx.state_get`,
  `ctx.state_update`, `ctx.emotion`. Any other `ctx.<attr>` access is
  rejected by name and line number.
- **Dunder attribute access** (`.__anything__`) is rejected outright,
  everywhere — not just on `ctx`.
- **Call-target allow-list**: every call must resolve to a safe builtin
  (`SAFE_BUILTIN_CALLS` — `abs`, `bool`, `dict`, `enumerate`, `float`,
  `format`, `int`, `isinstance`, `len`, `list`, `max`, `min`, `print`,
  `range`, `repr`, `reversed`, `round`, `set`, `sorted`, `str`, `sum`,
  `tuple`, `zip`), a locally-defined function, or a name imported from the
  allow-list above.
- **Shape check**: the module must define a top-level `execute(params, ctx)`
  taking exactly two positional arguments, no `*args`/`**kwargs`.

The `ctx` a forged skill actually receives at runtime is never the full
`NovaContext` — it's `ForgedSkillContext` (`reachy_nova/skills.py`), a
purpose-built object exposing exactly the seven `ALLOWED_CTX_ATTRS` methods
as thin, defensive delegations to the real subsystems (gesture engine,
skill manager, sonic, state, emotional state). Even if the validator's
allow-list check were somehow wrong, there is nothing else reachable on the
object a forged skill is handed.

## Fail-Closed Behaviors

- **No validator available → nothing activates.** Both `SkillForge` and
  `SkillManager.discover_runtime` lazy-import `forge_validator.validate` on
  first use; if that import fails (e.g. a checkout missing the module),
  they fail closed: `SkillForge` rejects with `"validator unavailable"`
  rather than ever staging an un-vetted skill as activatable, and
  `discover_runtime` refuses to register anything from the runtime
  directory at all.
- **Every dispatch failure path resolves to `forge/rejected` + a
  `logging.warning`, never an exception on the caller's thread and never a
  blocked 50Hz loop.** The whole network round-trip runs on a daemon worker
  thread (`SkillForge.dispatch` returns the thread immediately); an
  unreachable endpoint, a timeout, a non-200, an unparseable reply, a
  rejecting validator, or even an unexpected internal bug in `SkillForge`
  itself is caught and turned into `forge/rejected` rather than propagating.
- **A validator that raises is treated as a rejection**, not as "no
  problems found" — `SkillForge._run_inner` and `discover_runtime` both wrap
  the validator call in `try/except` and reject on any exception.
- **Generated code is imported exactly once, and only after validation
  passed.** `_import_forged_execute` (`reachy_nova/skills.py`) is the one
  place a forged `executor.py` is ever imported (`importlib.util`, not
  registered in `sys.modules`, so each forged skill's code stays isolated
  from the next) — `discover_runtime`'s calling code runs the validator
  first and only reaches the import on `ok=True`.
- **A buggy forged skill's `execute()` can't crash the loop.** The executor
  wrapper built in `discover_runtime` catches every exception and returns
  `f"[Skill error: {e}]"`, matching `SkillManager.execute`'s own contract
  for built-in skills.
- **Path traversal is closed by sanitization, not by trust.** The staged
  folder name comes from the `name:` field in the generated `SKILL.md`,
  sanitized to `[a-z0-9-]` (`_extract_and_sanitize_name`) — a name like
  `../../etc/passwd` cannot survive with a `/` or `.` in it, so the staged
  folder can never escape `staging_root`.

## Honest Limits

- **Live forge round-trip: PENDING.** The spec's honesty condition for this
  seam is explicit: it's only proven once a real dispatch to the real qwen3
  endpoint produces a generated skill on disk that validates, stages,
  activates, and is actually called by the model — live, on the robot. That
  proof is part of the live-proof task (t11 in the build plan) and has not
  run as of this doc landing. Everything described above (dispatch parsing,
  validator allow-list, activation wiring, fail-closed behaviors) is
  unit-tested against a mocked HTTP transport, not the live endpoint.
- **The validator is a static gate, not a sandbox — say so honestly.**
  `forge_validator.validate` never runs the generated code; it only proves
  that the *source text* doesn't contain a disallowed import, name, or call
  by AST inspection. It does not bound CPU time, memory, or recursion; it
  does not catch a logic bug that stays entirely within the allowed
  surface (e.g. an infinite loop built only from `range`/`while` and
  allowed calls); and it cannot reason about what an allowed `ctx` call
  itself does at runtime (e.g. repeatedly calling `ctx.say` in a tight loop
  passes validation and would only be caught, if at all, by whatever
  throttling exists downstream in `NovaSonic.inject_text`). It is a real,
  meaningful gate against the dangerous surface (filesystem, network,
  process, `ctx` escape) — it is not a substitute for a real execution
  sandbox, and none is claimed here.
