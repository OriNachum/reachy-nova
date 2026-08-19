"""Forge leg — the harness's kiro-writer wiring (plan deviation d1).

The kiro-writer arc shipped its components separately: ``kiro_acp`` (the
stdlib ACP client, t3), ``kiro_session`` (the standing watchdogged session
unit, t4), the ``FORGE_WRITER=kiro`` seam inside :class:`SkillForge` (t6) and
``kiro_rules`` (t7). None of those tasks composed them into the running
harness — that composition is this module, recorded as deviation d1 on the
build plan rather than slipped in silently.

What it wires, and what it deliberately does not:

* ONE :class:`ForgeLeg` object owns the forge pipeline for the harness —
  a :class:`~reachy_nova.skill_forge.SkillForge` constructed with the
  standing kiro session, a runtime-only
  :class:`~reachy_nova.skills.SkillManager` (it never calls ``discover()``;
  built-in skills belong to the direct-SDK app, not the harness), and the
  same ``activate_forged`` machinery the direct-SDK path uses — staged →
  validated → moved to the active root → registered.
* Forged skills become callable through the generic ``use_skill`` tool
  (``harness/tools.py``) rather than per-skill Sonic toolSpecs — so
  activation needs NO Sonic session restart: ``activate_forged`` is called
  with ``restart=None`` and an always-idle state shim, and the model learns
  the skill exists from the activation announce it already injects.
* The forged execution context is the restricted
  :class:`~reachy_nova.skills.ForgedSkillContext` with only ``sonic`` and
  the skill manager wired — gesture/emotion/state degrade to bracketed
  status strings by that class's own contract. The harness stays inside its
  boundary: nothing here imports ``reachy_mini`` or steers the head.

Every lifecycle transition emits one ``[SENSE stage=forge ...]`` line and a
staged skill is announced to the conversation exactly the way the direct-SDK
path announces it (``activate_forged`` injects "I just learned a new
skill: ...").
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from reachy_nova.sensory_log import stage as _stage
from reachy_nova.skill_forge import DEFAULT_STAGING_ROOT, SkillForge
from reachy_nova.skills import ForgedSkillContext, SkillManager, activate_forged

logger = logging.getLogger(__name__)

#: Sibling of the staging root, matching the direct-SDK app's layout
#: (``main.py``'s FORGED_ACTIVE_DIR) so a robot that ran either path sees
#: one shared set of activated skills.
DEFAULT_ACTIVE_ROOT = DEFAULT_STAGING_ROOT.parent / "skills-active"


class ForgeLeg:
    """The harness's forge pipeline: dispatch, auto-activate, use.

    ``session_unit`` is anything with ``prompt(text, timeout=...) -> str`` —
    in production the standing :class:`KiroSessionUnit`; in tests a fake.
    ``sonic`` only needs ``inject_text`` (getattr-defensive downstream).
    """

    def __init__(
        self,
        sonic,
        session_unit,
        *,
        staging_root: Path | str | None = None,
        active_root: Path | str | None = None,
    ) -> None:
        self._sonic = sonic
        self._staging_root = Path(staging_root) if staging_root is not None else DEFAULT_STAGING_ROOT
        self._active_root = Path(active_root) if active_root is not None else DEFAULT_ACTIVE_ROOT

        self._skill_manager = SkillManager(skills_dir=self._active_root)
        self._forged_ctx = ForgedSkillContext(skill_manager=self._skill_manager, sonic=sonic)
        self._forge = SkillForge(
            publish=self._on_forge_event,
            staging_root=self._staging_root,
            kiro_session=session_unit,
        )

        # Re-register skills forged in earlier runs; a missing/empty active
        # root is the ordinary first-boot state, not an error.
        try:
            if self._active_root.is_dir():
                reloaded = self._skill_manager.discover_runtime(self._active_root, ctx=self._forged_ctx)
                if reloaded:
                    _stage("forge", "nova", "reload", f"re-registered {len(reloaded)} forged skill(s)")
        except Exception as err:  # noqa: BLE001 - a bad prior skill must not kill the leg
            logger.warning(f"forge_leg: startup discover_runtime failed: {err}")

    # -- tool-facing surface -------------------------------------------------

    def forge(self, goal: str, improve: str | None = None) -> dict:
        """Queue a forge dispatch; the result arrives as forge/* events."""
        goal = (goal or "").strip()
        if not goal:
            return {"ok": False, "error": "forge needs a non-empty goal"}
        self._forge.dispatch(goal, improve=improve or None)
        _stage("forge", "nova", "dispatch", f"queued goal={goal[:80]!r}")
        return {
            "ok": True,
            "queued": goal,
            "note": "forging in the background — you will be told when it is staged or rejected",
        }

    def use_skill(self, name: str, params: dict | None = None) -> dict:
        """Execute an activated forged skill by name."""
        name = (name or "").strip()
        known = sorted(self._skill_manager.skills)
        if name not in self._skill_manager.skills:
            return {
                "ok": False,
                "error": f"no activated skill named {name!r}",
                "available": known,
            }
        result = self._skill_manager.execute(name, dict(params or {}))
        return {"ok": True, "skill": name, "result": result}

    def known_skills(self) -> list[str]:
        return sorted(self._skill_manager.skills)

    # -- forge event plumbing ------------------------------------------------

    def _on_forge_event(self, event_type: str, payload: dict) -> None:
        _stage("forge", "nova", event_type, f"payload={payload!r}"[:200])
        if event_type == "forge/staged":
            name = str(payload.get("name", ""))
            threading.Thread(
                target=self._activate,
                args=(name,),
                name=f"nova-forge-activate-{name}",
                daemon=True,
            ).start()
        elif event_type == "forge/rejected":
            reason = str(payload.get("reason", "unknown"))
            self._inject(f"A new skill idea didn't pass its safety check: {reason}")

    def _activate(self, name: str) -> None:
        # The always-idle state shim + restart=None: forged skills are reached
        # through the generic use_skill tool, so no Sonic restart is needed.
        result = activate_forged(
            self._staging_root,
            self._active_root,
            name,
            self._skill_manager,
            self._forged_ctx,
            self._sonic,
            {"voice_state": "idle"},
            self._on_forge_event,
            restart=None,
        )
        _stage("forge", "nova", "activate", f"name={name} result={result[:120]!r}")

    def _inject(self, text: str) -> None:
        inject = getattr(self._sonic, "inject_text", None)
        if callable(inject):
            try:
                inject(text)
            except Exception as err:  # noqa: BLE001 - announce failure must not crash the leg
                logger.warning(f"forge_leg: inject failed: {err}")
