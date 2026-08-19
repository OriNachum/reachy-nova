"""Skill-forge client for Reachy Nova.

Dispatches a natural-language goal (plus sensory context, and optionally an
existing skill to improve) to a configurable OpenAI-compatible coder-model
endpoint (``FORGE_BASE_URL`` / ``FORGE_MODEL``, env-configured — the robot and
the coder rig are different machines), parses the two fenced files the prompt
asks for (``SKILL.md`` + ``executor.py``), stages them under
``~/.reachy_nova/skills-forged/<name>/`` and runs them through a validator
before they are ever eligible for activation.

Every lifecycle transition is reported through a caller-supplied ``publish``
callback as a ``forge/*`` event (``forge/staged``, ``forge/activated``,
``forge/rejected``) so the nervous system stays observable end-to-end, the
same way every other sense is. The whole network round-trip runs on a daemon
worker thread — ``dispatch()`` returns immediately, and every failure path
(unreachable endpoint, timeout, non-200, unparseable reply, a rejecting
validator, or even an unexpected internal bug) resolves to ``forge/rejected``
plus a ``logging.warning`` — never an exception on the caller's thread, and
never a blocked 50Hz loop.

The validator (``reachy_nova/forge_validator.py``) is built by a sibling task
and may not exist in every checkout. ``SkillForge`` accepts a ``validator``
callable; when none is given it lazy-imports ``forge_validator.validate`` on
first use and, if that import fails, fails CLOSED — treats the skill as
rejected with reason ``"validator unavailable"`` rather than ever staging an
un-vetted skill as activatable.

``FORGE_WRITER`` selects *where* the authoring dispatch runs, without
touching anything downstream of it: ``"http"`` (the default) is the
behavior described above — a POST to ``FORGE_BASE_URL``. ``"kiro"`` sends
the exact same authoring instructions through an injected "kiro prompt
callable" (any object exposing ``.prompt(text, timeout=...) -> str``, e.g.
``reachy_nova.kiro_acp.KiroAcpSession``) instead — a standing on-device
coding-agent session rather than a separate-machine HTTP endpoint. Both
writers feed the identical parse -> stage -> validate -> ``forge/*`` event
pipeline; only the transport differs. An unconfigured or dead kiro session,
a prompt() that times out or raises, unparseable kiro output, and an
unrecognized ``FORGE_WRITER`` value all fail closed to ``forge/rejected``
with a clear reason, exactly like every HTTP failure mode.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Where staged (and rejected) forged skills live by default. Never the live
# skills dir (reachy_nova/skills/) — activation is a separate, later task.
DEFAULT_STAGING_ROOT = Path.home() / ".reachy_nova" / "skills-forged"

DEFAULT_FORGE_MODEL = "qwen3"
DEFAULT_TIMEOUT = 120.0

# FORGE_WRITER selects the authoring transport. "http" (default) is the
# original FORGE_BASE_URL POST path; "kiro" routes through an injected
# kiro-prompt-callable instead. Anything else fails closed.
DEFAULT_FORGE_WRITER = "http"

PROMPT_TEMPLATE = (
    "You are the skill-forge for Reachy Nova, a physical robot. Given a goal "
    "and sensory context, respond with EXACTLY two fenced code blocks and "
    "nothing else outside them:\n\n"
    "1. A block fenced as ```SKILL.md``` containing YAML frontmatter with a "
    "`name` field (lowercase, hyphenated, e.g. `wave-hello`) and a "
    "`description` field, followed by a short markdown body describing when "
    "to use the skill.\n\n"
    "2. A block fenced as ```executor.py``` containing a single Python "
    "function `def execute(params, ctx):` implementing the skill's "
    "behavior using only the primitives available on `ctx`.\n\n"
    "Output nothing else: no prose before, between, or after the two fenced "
    "blocks."
)

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)\n?```", re.DOTALL)
_NAME_RE = re.compile(r"^name:\s*(.+)$", re.MULTILINE)
_SANITIZE_RE = re.compile(r"[^a-z0-9-]+")
_DASH_COLLAPSE_RE = re.compile(r"-+")

# publish(event_type, payload) — event_type is already "forge/<transition>".
PublishFn = Callable[[str, dict], None]
# validate(skill_dir) -> (ok, reasons) — the sibling forge_validator contract.
ValidatorFn = Callable[[Path], tuple[bool, list[str]]]
# transport(url, payload, headers, timeout) -> parsed JSON response dict.
TransportFn = Callable[[str, dict, dict[str, str], float], dict]
# Anything with .prompt(text, timeout=...) -> str, e.g. kiro_acp.KiroAcpSession.
# Duck-typed deliberately — SkillForge never imports kiro_acp itself.
KiroSession = object


def _default_transport(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    """Real HTTP transport: POST JSON via urllib, return the parsed JSON body.

    Raises on any transport failure (connection error, timeout, non-2xx
    status, malformed JSON) — the caller is responsible for catching those
    and turning them into a forge/rejected event.
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _build_messages(goal: str, context: dict, improve: str | None) -> list[dict]:
    """Build the OpenAI-compatible chat messages for a dispatch."""
    user_lines = [f"Goal: {goal}"]
    if context:
        user_lines.append("Sensory context:")
        user_lines.append(json.dumps(context, indent=2, default=str))
    if improve:
        user_lines.append("Improve this existing skill (address feedback, keep what works):")
        user_lines.append(improve)
    return [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {"role": "user", "content": "\n\n".join(user_lines)},
    ]


def _build_kiro_prompt(goal: str, context: dict, improve: str | None) -> str:
    """Same authoring instructions as `_build_messages`, collapsed to one string.

    The kiro writer speaks one prompt string per turn (ACP has no separate
    system/user roles at this call site) — collapsing system+user here keeps
    the two-fenced-file contract identical across both writers.
    """
    messages = _build_messages(goal, context, improve)
    return "\n\n".join(m["content"] for m in messages)


def _extract_fences(content: str) -> dict[str, str]:
    """Defensively pull the SKILL.md and executor.py fenced blocks out of a reply.

    Matches fences labeled with the filename directly (```SKILL.md``` /
    ```executor.py```). Any fence whose label doesn't look filename-shaped is
    classified by content-sniffing (frontmatter ``---`` for SKILL.md,
    ``def execute(`` for executor.py) so a slightly-off label from the coder
    model doesn't sink an otherwise-good reply.
    """
    found: dict[str, str] = {}
    unlabeled: list[str] = []

    for label, body in _FENCE_RE.findall(content):
        label_lower = label.strip().lower()
        if "skill.md" in label_lower:
            found.setdefault("SKILL.md", body)
        elif "executor.py" in label_lower:
            found.setdefault("executor.py", body)
        else:
            unlabeled.append(body)

    for body in unlabeled:
        stripped = body.strip()
        if "SKILL.md" not in found and stripped.startswith("---"):
            found["SKILL.md"] = body
        elif "executor.py" not in found and "def execute(" in body:
            found["executor.py"] = body

    return found


def _extract_and_sanitize_name(skill_md: str) -> str | None:
    """Pull `name:` out of a SKILL.md's frontmatter and sanitize to [a-z0-9-].

    Sanitizing to that charset also closes off path traversal — a name like
    ``../../etc/passwd`` cannot survive with a '/' or '.' in it, so the
    staged folder can never escape ``staging_root``.
    """
    match = _NAME_RE.search(skill_md)
    if not match:
        return None
    raw = match.group(1).strip().strip("\"'").lower()
    raw = raw.replace("_", "-").replace(" ", "-")
    sanitized = _SANITIZE_RE.sub("", raw)
    sanitized = _DASH_COLLAPSE_RE.sub("-", sanitized).strip("-")
    if not sanitized or sanitized in (".", ".."):
        return None
    return sanitized


class SkillForge:
    """Dispatches skill-generation goals to a coder model and stages results.

    Every transition emits an event through `publish` (forge/staged,
    forge/activated, forge/rejected). The network round-trip runs on a
    daemon worker thread so a slow or unreachable coder rig never blocks the
    caller (in practice, the 50Hz main loop).
    """

    def __init__(
        self,
        publish: PublishFn,
        validator: ValidatorFn | None = None,
        staging_root: Path | str | None = None,
        transport: TransportFn | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        kiro_session: KiroSession | None = None,
    ):
        self._publish = publish
        self._validator = validator
        self._staging_root = Path(staging_root) if staging_root is not None else DEFAULT_STAGING_ROOT
        self._transport = transport or _default_transport
        self._timeout = timeout
        # Any object exposing .prompt(text, timeout=...) -> str. Only consulted
        # when FORGE_WRITER=kiro; ignored (not even required) on the http path.
        self._kiro_session = kiro_session

        # Lazy-import cache for the sibling forge_validator module, tried
        # (at most) once per instance the first time no validator is given.
        self._lazy_checked = False
        self._lazy_validator: ValidatorFn | None = None

    # -- public API ---------------------------------------------------

    def dispatch(self, goal: str, context: dict | None = None, improve: str | None = None) -> threading.Thread:
        """Kick off a forge round-trip on a daemon thread; return immediately.

        Returns the (already-started) worker thread so tests/callers can
        `.join()` it deterministically instead of guessing at timing.
        """
        thread = threading.Thread(
            target=self._run,
            args=(goal, context or {}, improve),
            daemon=True,
            name="skill-forge-dispatch",
        )
        thread.start()
        return thread

    def mark_activated(self, name: str) -> None:
        """Announce that `name` was activated. Emits forge/activated only.

        This does NOT move files or touch the live skills dir — activation
        itself is a later task's responsibility; this is purely the event.
        """
        self._emit("forge/activated", {"name": name})

    # -- worker thread body --------------------------------------------

    def _run(self, goal: str, context: dict, improve: str | None) -> None:
        try:
            self._run_inner(goal, context, improve)
        except Exception as e:  # noqa: BLE001 - last-resort safety net
            logger.warning("Skill-forge dispatch crashed unexpectedly: %s", e)
            self._reject(None, [f"internal error: {e}"])

    def _run_inner(self, goal: str, context: dict, improve: str | None) -> None:
        writer = os.environ.get("FORGE_WRITER", DEFAULT_FORGE_WRITER)
        if writer == "http":
            self._run_http(goal, context, improve)
        elif writer == "kiro":
            self._run_kiro(goal, context, improve)
        else:
            logger.warning("Skill-forge got unknown FORGE_WRITER=%r — failing closed", writer)
            self._reject(None, [f"unknown FORGE_WRITER: {writer!r}"])

    def _run_http(self, goal: str, context: dict, improve: str | None) -> None:
        base_url = os.environ.get("FORGE_BASE_URL")
        if not base_url:
            self._reject(None, ["endpoint not configured"])
            return

        model = os.environ.get("FORGE_MODEL", DEFAULT_FORGE_MODEL)
        api_key = os.environ.get("FORGE_API_KEY")

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "messages": _build_messages(goal, context, improve)}

        try:
            response = self._transport(url, payload, headers, self._timeout)
        except TimeoutError as e:
            logger.warning("Skill-forge request to %s timed out: %s", url, e)
            self._reject(None, [f"request timed out: {e}"])
            return
        except Exception as e:  # noqa: BLE001 - any transport failure is a rejection
            logger.warning("Skill-forge endpoint %s unreachable: %s", url, e)
            self._reject(None, [f"endpoint unreachable: {e}"])
            return

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("Skill-forge got an unparseable reply shape from %s: %s", url, e)
            self._reject(None, ["unparseable reply"])
            return

        self._finish_from_content(content, source=url)

    def _run_kiro(self, goal: str, context: dict, improve: str | None) -> None:
        if self._kiro_session is None:
            logger.warning("Skill-forge FORGE_WRITER=kiro but no kiro_session was configured — failing closed")
            self._reject(None, ["kiro writer not configured"])
            return

        prompt_text = _build_kiro_prompt(goal, context, improve)
        try:
            content = self._kiro_session.prompt(prompt_text, timeout=self._timeout)
        except Exception as e:  # noqa: BLE001 - dead session, timeout, or any other kiro failure
            logger.warning("Skill-forge kiro session prompt failed: %s", e)
            self._reject(None, [f"kiro session failed: {e}"])
            return

        self._finish_from_content(content, source="kiro")

    def _finish_from_content(self, content: str, source: str) -> None:
        """Shared tail of both writers: parse fences -> stage -> validate -> emit.

        Identical for the http and kiro paths — only how `content` (the raw
        assistant reply text) was obtained differs upstream.
        """
        fences = _extract_fences(content)
        skill_md = fences.get("SKILL.md")
        executor_py = fences.get("executor.py")

        if not skill_md or not skill_md.strip():
            logger.warning("Skill-forge reply from %s missing/empty SKILL.md fence", source)
            self._reject(None, ["missing or empty SKILL.md fence"])
            return
        if not executor_py or not executor_py.strip():
            logger.warning("Skill-forge reply from %s missing/empty executor.py fence", source)
            self._reject(None, ["missing or empty executor.py fence"])
            return

        name = _extract_and_sanitize_name(skill_md)
        if not name:
            logger.warning("Skill-forge reply from %s had no usable skill name", source)
            self._reject(None, ["invalid or missing skill name"])
            return

        skill_dir = self._staging_root / name
        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(skill_md)
            (skill_dir / "executor.py").write_text(executor_py)
        except OSError as e:
            logger.warning("Skill-forge failed writing staged folder for %s: %s", name, e)
            self._reject(name, [f"failed to stage: {e}"])
            return

        validator = self._resolve_validator()
        if validator is None:
            logger.warning("Skill-forge has no validator available for %s — failing closed", name)
            self._reject(name, ["validator unavailable"], skill_dir)
            return

        try:
            ok, reasons = validator(skill_dir)
        except Exception as e:  # noqa: BLE001 - a buggy validator must not activate anything
            logger.warning("Skill-forge validator raised for %s: %s", name, e)
            self._reject(name, [f"validator error: {e}"], skill_dir)
            return

        if not ok:
            logger.warning("Skill-forge rejected %s: %s", name, reasons)
            self._reject(name, reasons or ["validation failed"], skill_dir)
            return

        self._emit("forge/staged", {"name": name, "path": str(skill_dir)})

    # -- helpers ---------------------------------------------------------

    def _resolve_validator(self) -> ValidatorFn | None:
        if self._validator is not None:
            return self._validator
        if not self._lazy_checked:
            self._lazy_checked = True
            try:
                from .forge_validator import validate as _validate
            except ImportError:
                self._lazy_validator = None
            else:
                self._lazy_validator = _validate
        return self._lazy_validator

    def _reject(self, name: str | None, reasons: list[str], skill_dir: Path | None = None) -> None:
        """Emit forge/rejected, moving a staged folder into .rejected/<name>/ if one exists."""
        final_dir = skill_dir
        if skill_dir is not None and name is not None:
            rejected_dir = self._staging_root / ".rejected" / name
            try:
                rejected_dir.parent.mkdir(parents=True, exist_ok=True)
                if rejected_dir.exists():
                    shutil.rmtree(rejected_dir)
                shutil.move(str(skill_dir), str(rejected_dir))
                final_dir = rejected_dir
            except OSError as e:
                logger.warning("Skill-forge failed moving rejected folder for %s: %s", name, e)

        payload: dict = {"reason": "; ".join(reasons), "reasons": list(reasons)}
        if name:
            payload["name"] = name
        if final_dir is not None:
            payload["path"] = str(final_dir)
        self._emit("forge/rejected", payload)

    def _emit(self, event_type: str, payload: dict) -> None:
        try:
            self._publish(event_type, payload)
        except Exception as e:  # noqa: BLE001 - a broken publish callback must not crash us
            logger.warning("Skill-forge publish callback raised for %s: %s", event_type, e)
