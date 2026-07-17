"""Skills system for Reachy Nova.

Discovers, loads, and manages skills following the Agent Skills pattern.
Skills are folders with SKILL.md files containing YAML frontmatter metadata
and markdown body instructions.
"""

import importlib.util
import json
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default skills directory is reachy_nova/skills/
DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"


@dataclass
class Skill:
    """A discovered skill with metadata and optional executor."""

    name: str
    description: str
    body: str  # full markdown body from SKILL.md
    metadata: dict = field(default_factory=dict)
    input_schema: dict = field(default_factory=dict)
    executor: Callable[[dict], str] | None = None


def _parse_skill_md(path: Path) -> dict:
    """Parse a SKILL.md file, extracting YAML frontmatter and body."""
    text = path.read_text()

    # Extract YAML frontmatter between --- markers
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not fm_match:
        return {"name": path.parent.name, "description": "", "body": text}

    frontmatter_str = fm_match.group(1)
    body = fm_match.group(2).strip()

    # Simple YAML parsing (avoids PyYAML dependency for basic frontmatter)
    meta = {}
    current_key = None
    current_value_lines = []

    for line in frontmatter_str.split("\n"):
        # Check for key: value
        kv_match = re.match(r"^(\w[\w-]*):\s*(.*)", line)
        if kv_match:
            # Save previous key
            if current_key:
                meta[current_key] = "\n".join(current_value_lines).strip()
            current_key = kv_match.group(1)
            current_value_lines = [kv_match.group(2).lstrip(">").strip()]
        elif current_key and line.startswith("  "):
            current_value_lines.append(line.strip())

    if current_key:
        meta[current_key] = "\n".join(current_value_lines).strip()

    return {
        "name": meta.get("name", path.parent.name),
        "description": meta.get("description", ""),
        "body": body,
        "metadata": {k: v for k, v in meta.items() if k not in ("name", "description")},
    }


class ForgedSkillContext:
    """The restricted ``ctx`` surface handed to a forged skill's execute().

    A forged skill is runtime-generated code that only ever passed static
    analysis (``reachy_nova.forge_validator``), never a human review — so it
    NEVER gets the full ``NovaContext`` (mqtt, browser, memory, face_manager,
    slack_bot, ...). It gets exactly the sanctioned reaction primitives the
    validator allow-lists (``forge_validator.ALLOWED_CTX_ATTRS``): gesture,
    vocalize, say, inject, state_get, state_update, emotion.

    Each method is a thin, defensive delegation to the real subsystem handed
    in at construction (either as individual kwargs, or bundled in a
    ``subsystems`` mapping — whichever is convenient for the caller; a
    per-kwarg value always wins over the mapping). When the underlying
    subsystem is absent or the call fails, the method logs a warning and
    returns a bracketed status string instead of raising — a forged skill's
    execute() should never crash just because e.g. `sonic` isn't wired up in
    a given test/host.
    """

    def __init__(
        self,
        subsystems: dict | None = None,
        *,
        gesture_engine=None,
        skill_manager=None,
        sonic=None,
        state=None,
        emotional_state=None,
    ):
        subsystems = subsystems or {}
        self._gesture_engine = gesture_engine if gesture_engine is not None else subsystems.get("gesture_engine")
        self._skill_manager = skill_manager if skill_manager is not None else subsystems.get("skill_manager")
        self._sonic = sonic if sonic is not None else subsystems.get("sonic")
        self._state = state if state is not None else subsystems.get("state")
        self._emotional_state = (
            emotional_state if emotional_state is not None else subsystems.get("emotional_state")
        )

    def gesture(self, name: str) -> str:
        """Run a named gesture animation via the real GestureEngine."""
        engine = self._gesture_engine
        if engine is None or not hasattr(engine, "execute"):
            logger.warning("ForgedSkillContext.gesture: no gesture_engine available")
            return "[gesture unavailable]"
        try:
            return engine.execute(name)
        except Exception as e:
            logger.warning(f"ForgedSkillContext.gesture failed: {e}")
            return f"[gesture error: {e}]"

    def vocalize(self, kind: str, **kw) -> str:
        """Play an expressive non-speech sound via the built-in vocalize skill."""
        manager = self._skill_manager
        if manager is None or not hasattr(manager, "execute"):
            logger.warning("ForgedSkillContext.vocalize: no skill_manager available")
            return "[vocalize unavailable]"
        try:
            return manager.execute("vocalize", {"kind": kind, **kw})
        except Exception as e:
            logger.warning(f"ForgedSkillContext.vocalize failed: {e}")
            return f"[vocalize error: {e}]"

    def say(self, text: str) -> str:
        """Speak immediately, bypassing the speaking guard (force=True)."""
        sonic = self._sonic
        inject_text = getattr(sonic, "inject_text", None) if sonic is not None else None
        if not callable(inject_text):
            logger.warning("ForgedSkillContext.say: no sonic available")
            return "[say unavailable]"
        try:
            inject_text(text, force=True)
            return "[said]"
        except Exception as e:
            logger.warning(f"ForgedSkillContext.say failed: {e}")
            return f"[say error: {e}]"

    def inject(self, text: str) -> str:
        """Inject text into the live conversation (subject to the speaking guard)."""
        sonic = self._sonic
        inject_text = getattr(sonic, "inject_text", None) if sonic is not None else None
        if not callable(inject_text):
            logger.warning("ForgedSkillContext.inject: no sonic available")
            return "[inject unavailable]"
        try:
            inject_text(text)
            return "[injected]"
        except Exception as e:
            logger.warning(f"ForgedSkillContext.inject failed: {e}")
            return f"[inject error: {e}]"

    def state_get(self, key: str):
        """Read a single field from the shared State."""
        state = self._state
        if state is None or not hasattr(state, "get"):
            logger.warning("ForgedSkillContext.state_get: no state available")
            return None
        try:
            return state.get(key)
        except Exception as e:
            logger.warning(f"ForgedSkillContext.state_get failed: {e}")
            return None

    def state_update(self, **kw) -> str:
        """Update one or more fields on the shared State."""
        state = self._state
        if state is None or not hasattr(state, "update"):
            logger.warning("ForgedSkillContext.state_update: no state available")
            return "[state_update unavailable]"
        try:
            state.update(**kw)
            return "[state updated]"
        except Exception as e:
            logger.warning(f"ForgedSkillContext.state_update failed: {e}")
            return f"[state_update error: {e}]"

    def emotion(self, event: str) -> str:
        """Apply a named emotion event via the real EmotionalState."""
        emotional_state = self._emotional_state
        if emotional_state is None or not hasattr(emotional_state, "apply_event"):
            logger.warning("ForgedSkillContext.emotion: no emotional_state available")
            return "[emotion unavailable]"
        try:
            emotional_state.apply_event(event)
            return f"[emotion '{event}' applied]"
        except Exception as e:
            logger.warning(f"ForgedSkillContext.emotion failed: {e}")
            return f"[emotion error: {e}]"


def _lazy_import_forge_validator():
    """Lazy-import reachy_nova.forge_validator.validate; None if unavailable.

    Mirrors SkillForge._resolve_validator's fail-closed lazy-import pattern:
    the validator is a sibling module that may not exist in every checkout,
    so discover_runtime never imports it at module load time.
    """
    try:
        from .forge_validator import validate as _validate
    except ImportError:
        return None
    return _validate


def _import_forged_execute(executor_path: Path, name: str):
    """Import a forged executor.py and return its execute(params, ctx) function.

    SECURITY CONTRACT: callers must invoke this ONLY after a validator has
    already returned ok=True for the containing folder — see the top-of-file
    note on forge_validator.validate. Uses importlib.util.spec_from_file_location
    (not a plain `import`) so the module is never registered in sys.modules —
    each forged skill's code stays isolated from the next.

    Returns None (never raises for a missing/non-callable `execute`) so the
    caller can treat "no usable execute()" the same as any other discovery
    failure — skip the folder, log a warning, keep going.
    """
    module_name = f"_forged_skill_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, executor_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # may raise — caller wraps in try/except
    execute_fn = getattr(module, "execute", None)
    if not callable(execute_fn):
        return None
    return execute_fn


class SkillManager:
    """Discovers, registers, and executes skills."""

    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir = skills_dir or DEFAULT_SKILLS_DIR
        self.skills: dict[str, Skill] = {}

    def discover(self) -> None:
        """Scan skills_dir for folders containing SKILL.md files."""
        if not self.skills_dir.is_dir():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return

        for skill_md in self.skills_dir.glob("*/SKILL.md"):
            try:
                parsed = _parse_skill_md(skill_md)
                name = parsed["name"]
                self.skills[name] = Skill(
                    name=name,
                    description=parsed["description"],
                    body=parsed["body"],
                    metadata=parsed.get("metadata", {}),
                )
                logger.info(f"Discovered skill: {name}")
            except Exception as e:
                logger.error(f"Error loading skill from {skill_md}: {e}")

    def discover_runtime(self, runtime_dir, ctx=None, validator: Callable | None = None) -> list[str]:
        """Hot-load validated forged skills from a runtime directory.

        For each immediate subfolder of `runtime_dir` that contains both a
        SKILL.md and an executor.py:

          1. Run `validator` (default: a lazy import of
             `forge_validator.validate`) against the folder. If no validator
             is resolvable at all, nothing in `runtime_dir` is registered —
             fail closed, same as SkillForge's own contract.
          2. Only on a passing validation, parse SKILL.md via
             `_parse_skill_md` for name/description/body/metadata.
          3. Only on a passing validation, import executor.py (see
             `_import_forged_execute` — this is the ONE place forged code
             ever gets imported, and only after step 1 said ok).
          4. Wrap the imported `execute(params, ctx)` in a closure over the
             `ctx` argument given here (typically a `ForgedSkillContext`) so
             it matches `Skill.executor`'s `Callable[[dict], str]` shape, and
             catch ALL exceptions inside that wrapper — matching
             `SkillManager.execute`'s own `"[Skill error: {e}]"` contract, so
             a buggy forged skill can never crash the loop.
          5. Register via `register_executor`.

        Any failure along the way (validation rejection, a missing
        SKILL.md/executor.py, a parse error, an import error, or a missing/
        non-callable `execute`) skips just that folder — logged as a
        warning, discovery continues with the rest.

        Returns the list of skill names actually registered.
        """
        runtime_dir = Path(runtime_dir)
        registered: list[str] = []

        if not runtime_dir.is_dir():
            logger.warning(f"Runtime skills directory not found: {runtime_dir}")
            return registered

        resolved_validator = validator if validator is not None else _lazy_import_forge_validator()
        if resolved_validator is None:
            logger.warning("discover_runtime: no validator available — refusing to register anything (fail closed)")
            return registered

        for skill_dir in sorted(p for p in runtime_dir.iterdir() if p.is_dir()):
            folder_name = skill_dir.name
            skill_md_path = skill_dir / "SKILL.md"
            executor_path = skill_dir / "executor.py"
            if not skill_md_path.is_file() or not executor_path.is_file():
                continue  # not a forged-skill folder — nothing to discover here

            try:
                ok, reasons = resolved_validator(skill_dir)
            except Exception as e:
                logger.warning(f"discover_runtime: validator raised for {folder_name}: {e}")
                continue
            if not ok:
                logger.warning(f"discover_runtime: {folder_name} failed validation: {reasons}")
                continue

            try:
                parsed = _parse_skill_md(skill_md_path)
            except Exception as e:
                logger.warning(f"discover_runtime: failed parsing SKILL.md for {folder_name}: {e}")
                continue

            # SECURITY: executor.py is imported ONLY after validate() passed.
            try:
                execute_fn = _import_forged_execute(executor_path, folder_name)
            except Exception as e:
                logger.warning(f"discover_runtime: failed importing executor.py for {folder_name}: {e}")
                continue
            if execute_fn is None:
                logger.warning(f"discover_runtime: {folder_name}'s executor.py has no usable execute(params, ctx)")
                continue

            skill_name = parsed.get("name") or folder_name

            def _make_executor(fn, bound_name):
                # `fn` and `bound_name` are bound as default-free closures over
                # this call's arguments (not the loop variable) — avoids the
                # classic late-binding bug where every executor would end up
                # referencing whatever `skill_name`/`execute_fn` the loop last
                # landed on.
                def _executor(params: dict) -> str:
                    try:
                        return fn(params, ctx)
                    except Exception as e:
                        logger.error(f"Forged skill '{bound_name}' execution error: {e}")
                        return f"[Skill error: {e}]"

                return _executor

            self.skills[skill_name] = Skill(
                name=skill_name,
                description=parsed.get("description", ""),
                body=parsed.get("body", ""),
                metadata=parsed.get("metadata", {}),
            )
            self.register_executor(skill_name, _make_executor(execute_fn, skill_name))
            registered.append(skill_name)
            logger.info(f"Hot-registered forged skill: {skill_name}")

        return registered

    def register_executor(
        self,
        name: str,
        executor: Callable[[dict], str],
        input_schema: dict | None = None,
    ) -> None:
        """Register an executor callable for a skill.

        If the skill was already discovered, attaches the executor.
        Otherwise creates a minimal skill entry.
        """
        if name in self.skills:
            self.skills[name].executor = executor
            if input_schema:
                self.skills[name].input_schema = input_schema
        else:
            self.skills[name] = Skill(
                name=name,
                description="",
                body="",
                executor=executor,
                input_schema=input_schema or {},
            )
        logger.info(f"Registered executor for skill: {name}")

    def get_tool_specs(self) -> list[dict]:
        """Return tool specs formatted for Nova Sonic's toolConfiguration."""
        tools = []
        for skill in self.skills.values():
            if skill.executor is None:
                continue

            schema = skill.input_schema or {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to ask or look for",
                    }
                },
                "required": ["query"],
            }

            tools.append({
                "toolSpec": {
                    "name": skill.name,
                    "description": skill.description,
                    "inputSchema": {"json": json.dumps(schema)},
                }
            })
        return tools

    def execute(self, tool_name: str, params: dict) -> str:
        """Execute a skill by name, return result string."""
        skill = self.skills.get(tool_name)
        if not skill:
            return f"[Unknown skill: {tool_name}]"
        if not skill.executor:
            return f"[Skill '{tool_name}' has no executor]"
        try:
            return skill.executor(params)
        except Exception as e:
            logger.error(f"Skill '{tool_name}' execution error: {e}")
            return f"[Skill error: {e}]"

    def get_system_context(self) -> str:
        """Return available skills description for the system prompt."""
        if not self.skills:
            return ""

        lines = ["You can also do these things:"]
        for skill in self.skills.values():
            if skill.executor:
                lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)


def activate_forged(
    staging_root: Path | str,
    active_root: Path | str,
    name: str,
    skill_manager: SkillManager,
    forged_ctx,
    sonic,
    state,
    publish: Callable[[str, dict], None],
    validator: Callable | None = None,
    restart: Callable[[], None] | None = None,
) -> str:
    """Move a validated staged forged skill into the live runtime, then announce it.

    Sequencing:

      1. `shutil.move` the staged folder (`staging_root/<name>`) into
         `active_root/<name>`. A missing staged folder, or a filesystem
         failure, returns a `"[Activation failed: ...]"` string — never
         raises.
      2. Re-run `SkillManager.discover_runtime` over `active_root` (not just
         the moved folder — `discover_runtime` itself only ever scans
         *immediate* subfolders of the directory it's given, so the parent
         is the right unit; this also keeps every already-active forged
         skill re-registered idempotently). `forged_ctx` is closed over as
         every forged skill's execution context — never the full
         NovaContext.
      3. Restart timing: `NovaSonic.restart()` takes a `stop_event` this
         function has no business owning, so the actual restart is a
         zero-arg `restart` callable the caller binds (e.g.
         `functools.partial(sonic.restart, stop_event)`):
           - `state.get("voice_state")` in (None, "idle"): call `restart()`
             immediately — a new Sonic session picks up the forged
             toolSpec via `get_tool_specs()`.
           - any other voice_state (a conversation is live): do NOT
             restart — defer to the next natural restart, and say so in the
             returned string.
      4. Announce either way (whether restarted now or deferred): via
         `sonic.inject_text` (getattr-defensive, so a missing/broken sonic
         never crashes activation — NOT a separate injected `inject`
         callable, since `sonic` is already a required parameter and this
         keeps the announce seam singular), and via
         `publish("forge/activated", {"name": name})` — the ONE event seam
         used here. This is deliberately the same `PublishFn` contract
         `SkillForge` itself is constructed with, called directly, rather
         than requiring a `SkillForge` instance and routing through its
         `mark_activated()` (which just calls the same publish callback
         under the hood) — `activate_forged` has no other need for a
         `SkillForge`, so this keeps its dependency surface minimal.

    Never raises: every subsystem call is wrapped defensively, mirroring
    `SkillManager.execute`'s own "never crash the loop" contract.
    """
    staging_root = Path(staging_root)
    active_root = Path(active_root)
    staged_dir = staging_root / name

    if not staged_dir.is_dir():
        return f"[Activation failed: no staged skill named '{name}']"

    active_dir = active_root / name
    try:
        active_root.mkdir(parents=True, exist_ok=True)
        if active_dir.exists():
            shutil.rmtree(active_dir)
        shutil.move(str(staged_dir), str(active_dir))
    except OSError as e:
        logger.warning(f"activate_forged: failed moving '{name}' into active root: {e}")
        return f"[Activation failed: could not move '{name}' into place: {e}]"

    registered = skill_manager.discover_runtime(active_root, forged_ctx, validator=validator)
    if name not in registered:
        logger.warning(f"activate_forged: '{name}' moved but did not pass discovery/validation afterward")
        return f"[Activation failed: '{name}' did not pass discovery after move]"

    voice_state = None
    try:
        if state is not None:
            voice_state = state.get("voice_state")
    except Exception as e:
        logger.warning(f"activate_forged: failed reading voice_state: {e}")

    if voice_state in (None, "idle"):
        restarted = False
        if restart is not None:
            try:
                restart()
                restarted = True
            except Exception as e:
                logger.warning(f"activate_forged: restart() raised for '{name}': {e}")
        status = (
            f"[Skill '{name}' activated and Sonic restarted]"
            if restarted
            else f"[Skill '{name}' activated — no restart callable provided]"
        )
    else:
        status = f"[Skill '{name}' activated — restart deferred (voice_state={voice_state!r})]"

    inject_text = getattr(sonic, "inject_text", None) if sonic is not None else None
    if callable(inject_text):
        try:
            inject_text(f"I just learned a new skill: {name}.")
        except Exception as e:
            logger.warning(f"activate_forged: inject_text failed for '{name}': {e}")
    else:
        logger.warning(f"activate_forged: sonic has no inject_text — cannot announce '{name}'")

    try:
        publish("forge/activated", {"name": name})
    except Exception as e:
        logger.warning(f"activate_forged: publish failed for '{name}': {e}")

    return status
