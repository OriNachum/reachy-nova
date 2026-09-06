"""Nova Act - Browser automation triggered by voice commands.

Gated behind ``NOVA_ACT_ENABLED`` (default off, see :func:`act_enabled`).
When the flag is off, no code path in this module imports ``nova_act`` or
``playwright`` — those imports are function-local (inside
``_ensure_workflow``/``_execute_task``, on the enabled path only) precisely
so that flag-off means zero import of either package.
"""

import base64
import logging
import os
import threading
import queue
import time
from collections.abc import Callable

from . import sensory_log

logger = logging.getLogger(__name__)

WORKFLOW_MODEL_ID = "nova-act-latest"

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


#: Where the browser itself runs (``NOVA_ACT_BROWSER``): ``local`` launches
#: playwright chromium on this machine; ``agentcore`` connects NovaAct over CDP
#: to a Bedrock AgentCore hosted browser session (no chromium on the robot —
#: the user decision recorded as deviation d5). Anything unrecognized is local.
SURFACE_ENV = "NOVA_ACT_BROWSER"
SURFACE_LOCAL = "local"
SURFACE_AGENTCORE = "agentcore"


def browser_surface() -> str:
    """The configured execution surface, resolved at call time."""
    value = os.environ.get(SURFACE_ENV, SURFACE_LOCAL).strip().lower()
    return SURFACE_AGENTCORE if value == SURFACE_AGENTCORE else SURFACE_LOCAL


#: Per-act browser actuation budget (``NOVA_ACT_MAX_STEPS``, default the SDK's
#: own 30). The old hard-coded 15 failed a plain Google weather search live
#: ("Exceeded max steps 15 without return", 2026-08-12 00:06).
MAX_STEPS_ENV = "NOVA_ACT_MAX_STEPS"
DEFAULT_MAX_STEPS = 30


def act_max_steps() -> int:
    """The per-act step budget, resolved at call time; bad values fall back."""
    raw = os.environ.get(MAX_STEPS_ENV, "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_STEPS
    return value if value > 0 else DEFAULT_MAX_STEPS


def act_enabled() -> bool:
    """Whether Nova Act browser automation is enabled.

    Reads the ``NOVA_ACT_ENABLED`` environment variable, default-off
    (``"0"``). Truthy values (case-insensitive): ``1``, ``true``, ``yes``,
    ``on``. Anything else — including an absent variable, ``"0"``,
    ``"false"``, ``"no"``, ``"off"``, or an empty string — is off.
    """
    value = os.environ.get("NOVA_ACT_ENABLED", "0").strip().lower()
    return value in _TRUTHY_VALUES


class NovaBrowser:
    """Manages browser automation tasks via Amazon Nova Act."""

    #: Progress narration collapses to at most these two phases (issue #8):
    #: a fresh Sonic session reading a mid-task "Working on: X..." line as a
    #: request would call the browse tool again, so every cue is worded as a
    #: status, never a request.
    PROGRESS_PHASE_START = "start"
    PROGRESS_PHASE_WORKING = "working"

    #: At most one on_progress callback per this many seconds, per task
    #: (monotonic clock) — most Sonic conversation injects inside this
    #: window get dropped by the voice model's own throttle anyway, and a
    #: burst read back-to-back risks looking like a fresh request.
    PROGRESS_RATE_LIMIT_S = 10.0

    #: Duplicate-instruction window for queue_task dedupe (issue #8): a
    #: fresh Sonic session restarted mid-flight can re-issue the same browse
    #: request; within this window (and always while the same instruction is
    #: still the running task) the repeat is dropped rather than re-queued.
    DEDUPE_WINDOW_S = 300

    def __init__(
        self,
        on_result: Callable[[str], None] | None = None,
        on_screenshot: Callable[[str], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
        on_progress: Callable[[str], None] | None = None,
        headless: bool = True,
        chrome_channel: str = "chromium",
        clock: Callable[[], float] | None = None,
    ):
        self.on_result = on_result
        self.on_screenshot = on_screenshot
        self.on_state_change = on_state_change
        self.on_progress = on_progress
        self.headless = headless
        self.chrome_channel = chrome_channel
        self._clock = clock or time.monotonic

        self._task_queue: queue.Queue[dict] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._nova = None
        self._workflow = None

        self.state = "idle"  # idle, busy, error
        self.last_result = ""
        self.last_screenshot_b64 = ""
        self.current_task = ""
        self.history: list[dict] = []

        self._last_progress_emit_ts: float | None = None
        # (normalised instruction, enqueue timestamp) for queue_task dedupe.
        self._recent_instructions: list[tuple[str, float]] = []
        # Guards queue_task's prune/compare/insert/enqueue as one step
        # (PR #26 review, finding 6) — see queue_task's docstring.
        self._dedupe_lock = threading.Lock()

    def _begin_task(self, instruction: str) -> None:
        """Record a task as the currently-running one (PR #26 review, finding 6).

        ``current_task``/``state`` are written under ``self._dedupe_lock``
        so :meth:`queue_task`'s running-task comparison — read under the
        same lock — never observes a torn state. The ``on_state_change``
        callback fires outside the lock, since it can run arbitrary
        (dashboard-notifying) code and must not be part of the critical
        section.
        """
        with self._dedupe_lock:
            self.current_task = instruction
            self.state = "busy"
        if self.on_state_change:
            try:
                self.on_state_change("busy")
            except Exception:
                pass

    def _set_state(self, state: str) -> None:
        self.state = state
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception:
                pass

    def _emit_progress(self, message: str) -> None:
        """Emit a progress update for narration.

        The single emission point for browser progress narration. Always
        logs (grep-able even when the callback is rate-limited), but the
        ``on_progress`` callback itself fires at most once per
        ``PROGRESS_RATE_LIMIT_S`` seconds per task — a burst of phase
        messages landing inside the same second must not multiply into a
        burst of conversation injects.
        """
        logger.info(f"[Browser progress] {message}")
        now = self._clock()
        if (
            self._last_progress_emit_ts is not None
            and now - self._last_progress_emit_ts < self.PROGRESS_RATE_LIMIT_S
        ):
            return
        self._last_progress_emit_ts = now
        if self.on_progress:
            try:
                self.on_progress(message)
            except Exception:
                pass

    def _phase_message(self, phase: str, instruction: str) -> str:
        """Build the status-shaped narration for one progress phase.

        Worded as a status report, never a request — a fresh Sonic session
        that hears "Working on: X..." mid-flight can read it as an
        instruction and call the browse tool again (issue #8).
        """
        if phase == self.PROGRESS_PHASE_START:
            return f"Status: your browser is starting on '{instruction}' — no action needed."
        if phase == self.PROGRESS_PHASE_WORKING:
            return f"Status: your browser is working on '{instruction}' — no action needed."
        raise ValueError(f"unknown browser progress phase: {phase!r}")

    @staticmethod
    def _normalize_instruction(instruction: str) -> str:
        return " ".join((instruction or "").strip().lower().split())

    def queue_task(self, instruction: str, url: str | None = None) -> dict:
        """Queue a browser automation task (fire-and-forget).

        No-op when ``NOVA_ACT_ENABLED`` is off (the default) — nothing
        consumes the queue in that case since :meth:`start` never spins up
        the worker thread.

        Deduplicates (issue #8): a normalised instruction equal to the
        currently running task's (while ``state == "busy"``), or to any
        queued/recently-enqueued task's within ``DEDUPE_WINDOW_S`` seconds,
        is dropped instead of enqueued — a fresh Sonic session restarted
        mid-flight otherwise re-issues the same browse request.

        Synchronized (PR #26 review, finding 6): pruning
        ``_recent_instructions``, the running-task comparison, the
        recent-task comparison, the recent-list insertion, and the queue
        insertion all happen inside one ``self._dedupe_lock`` critical
        section. There are two independent callers — the voice tool (via
        ``skill_executors.py``) and the dashboard API
        (``api_routes.py``'s ``submit_browser_task``) — and without a
        shared lock, two concurrent calls with the same instruction could
        both pass the duplicate check before either recorded it, enqueuing
        (and later executing) the same browse twice. The critical section
        stays short: it never holds the lock across the act itself, only
        across the bookkeeping that decides whether to enqueue.

        Args:
            instruction: Natural language instruction for what to do.
            url: Optional URL to navigate to first.

        Returns:
            A dict describing what happened: ``{"ok": False, "queued":
            False, "reason": "nova-act-disabled"}`` when disabled;
            ``{"ok": True, "queued": False, "duplicate": True,
            "instruction": ..., "url": ...}`` when deduped; otherwise
            ``{"ok": True, "queued": True, "instruction": ..., "url": ...}``.
        """
        if not act_enabled():
            sensory_log.stage(
                "act", "browser", "queue_task", "dropped reason=nova-act-disabled"
            )
            return {"ok": False, "queued": False, "reason": "nova-act-disabled"}

        normalized = self._normalize_instruction(instruction)
        with self._dedupe_lock:
            now = self._clock()
            self._recent_instructions = [
                (norm, ts)
                for norm, ts in self._recent_instructions
                if now - ts < self.DEDUPE_WINDOW_S
            ]

            is_running_duplicate = (
                self.state == "busy"
                and self._normalize_instruction(self.current_task) == normalized
            )
            is_recent_duplicate = any(
                norm == normalized for norm, _ts in self._recent_instructions
            )
            if is_running_duplicate or is_recent_duplicate:
                sensory_log.stage(
                    "act",
                    "browser",
                    "queue_task",
                    f"dropped reason=duplicate window=300s instruction={normalized}",
                )
                return {
                    "ok": True,
                    "queued": False,
                    "duplicate": True,
                    "instruction": instruction,
                    "url": url,
                }

            self._recent_instructions.append((normalized, now))
            self._task_queue.put({"instruction": instruction, "url": url})
        return {"ok": True, "queued": True, "instruction": instruction, "url": url}

    def execute(self, instruction: str, url: str | None = None) -> str:
        """Execute a browser task synchronously, blocking until complete.

        Used by the skill executor so Sonic can wait for the result.

        Args:
            instruction: Natural language instruction for what to do.
            url: Optional URL to navigate to first.

        Returns:
            The result string from the browser task.
        """
        if not act_enabled():
            sensory_log.stage(
                "act", "browser", "execute", "dropped reason=nova-act-disabled"
            )
            return "Browser automation is disabled (set NOVA_ACT_ENABLED=1 to enable)."

        done_event = threading.Event()
        result_holder: list[str] = []

        self._task_queue.put({
            "instruction": instruction,
            "url": url,
            "_done_event": done_event,
            "_result_holder": result_holder,
        })

        done_event.wait()
        return result_holder[0] if result_holder else "Browser task completed with no result."

    def _capture_screenshot(self) -> str:
        """Capture a screenshot of the current browser page as base64."""
        if not self._nova:
            return ""
        try:
            screenshot_bytes = self._nova.page.screenshot()
            b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            self.last_screenshot_b64 = b64
            if self.on_screenshot:
                self.on_screenshot(b64)
            return b64
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return ""

    def _ensure_workflow(self) -> None:
        """Ensure the Workflow context is active."""
        from nova_act.types.workflow import Workflow

        if self._workflow is None:
            name = os.environ.get("AWS_NOVA_ACT_WORKFLOW_NAME", "Reachy-Nova")
            self._workflow = Workflow(
                model_id=WORKFLOW_MODEL_ID,
                workflow_definition_name=name,
            )
            self._workflow.__enter__()
            logger.info("Nova Act Workflow started: %s", name)

    def _cleanup_workflow(self) -> None:
        """Clean up the Workflow context."""
        if self._workflow is not None:
            try:
                self._workflow.__exit__(None, None, None)
            except Exception:
                pass
            self._workflow = None

    def _act_on_agentcore(self, instruction: str, url: str | None):
        """Run one act() against a Bedrock AgentCore hosted browser (per-task session).

        The browser runs in AWS, NovaAct connects over CDP
        (``browser_session(region) -> generate_ws_headers()``, per the ACBT
        quickstart) — no chromium on this machine. Sessions are per-task on
        purpose: hosted sessions time out, and a fresh one per voice request
        beats holding a zombie across the robot's idle hours.
        """
        from bedrock_agentcore.tools.browser_client import browser_session
        from nova_act import NovaAct

        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._ensure_workflow()
        logger.info("Starting a cloud browser session...")
        with browser_session(region) as client:
            ws_url, headers = client.generate_ws_headers()
            with NovaAct(
                cdp_endpoint_url=ws_url,
                cdp_headers=headers,
                workflow=self._workflow,
                starting_page=url or "https://www.google.com",
                tty=False,
            ) as nova:
                self._emit_progress(
                    self._phase_message(self.PROGRESS_PHASE_WORKING, instruction)
                )
                # act_get, not act: a voice browse request is almost always a
                # question, and in this SDK the model's answer only comes back
                # through act_get's parsed_response (ActResult carries none).
                return nova.act_get(instruction, max_steps=act_max_steps())

    def _execute_task(self, task: dict) -> str:
        """Execute a single browser task."""
        instruction = task["instruction"]
        url = task.get("url", "https://www.google.com")
        done_event = task.get("_done_event")
        result_holder = task.get("_result_holder")
        self._begin_task(instruction)
        self._last_progress_emit_ts = None

        try:
            self._emit_progress(
                self._phase_message(self.PROGRESS_PHASE_START, instruction)
            )
            if browser_surface() == SURFACE_AGENTCORE:
                result = self._act_on_agentcore(instruction, url)
            else:
                from nova_act import NovaAct

                self._ensure_workflow()

                if self._nova is None:
                    self._nova = NovaAct(
                        starting_page=url or "https://www.google.com",
                        headless=self.headless,
                        chrome_channel=self.chrome_channel,
                        workflow=self._workflow,
                        tty=False,
                    )
                    self._nova.start()
                    logger.info("Navigating to %s", url or "Google")
                elif url:
                    logger.info("Navigating to %s", url)
                    self._nova.go_to_url(url)

                self._emit_progress(
                    self._phase_message(self.PROGRESS_PHASE_WORKING, instruction)
                )
                result = self._nova.act(instruction, max_steps=act_max_steps())

            logger.info("[Browser progress] Done! Reading results...")
            self._capture_screenshot()

            result_text = f"Done: {instruction}"
            if result and hasattr(result, "success"):
                if result.success:
                    result_text = f"Completed: {instruction}"
                else:
                    result_text = f"Could not complete: {instruction}"
            # The agent's actual answer beats a status line: a voice request
            # like "search the web for X" is only served if what it FOUND is
            # what gets spoken. act_get carries it as parsed_response; older
            # SDK results carried response.
            for attr in ("parsed_response", "response"):
                value = getattr(result, attr, None)
                if value is not None and str(value).strip():
                    result_text = str(value).strip()
                    break

            self.last_result = result_text
            self.history.append({
                "instruction": instruction,
                "result": result_text,
                "time": time.time(),
                "success": result.success if result and hasattr(result, "success") else True,
            })

            if self.on_result:
                self.on_result(result_text)

            self._set_state("idle")
            if result_holder is not None:
                result_holder.append(result_text)
            if done_event:
                done_event.set()
            return result_text

        except Exception as e:
            error_msg = f"Browser error: {e}"
            logger.error(error_msg, exc_info=True)
            self.last_result = error_msg
            self._set_state("error")
            # Report error back so voice can inform user
            if self.on_result:
                self.on_result(error_msg)
            # Reset the browser instance on error
            try:
                if self._nova:
                    self._nova.stop()
            except Exception:
                pass
            self._nova = None
            self._cleanup_workflow()
            if result_holder is not None:
                result_holder.append(error_msg)
            if done_event:
                done_event.set()
            return error_msg

    def _run_loop(self, stop_event: threading.Event) -> None:
        logger.info("Nova Browser loop started")
        while not stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self._execute_task(task)

        # Cleanup
        if self._nova:
            try:
                self._nova.stop()
            except Exception:
                pass
            self._nova = None
        self._cleanup_workflow()

    def start(self, stop_event: threading.Event) -> None:
        """Start the browser automation worker thread.

        No-op when ``NOVA_ACT_ENABLED`` is off (the default): no worker
        thread is spun up, and — since ``_run_loop``/``_execute_task`` are
        never reached — neither ``nova_act`` nor ``playwright`` is ever
        imported.
        """
        if not act_enabled():
            sensory_log.stage(
                "act", "browser", "start", "dropped reason=nova-act-disabled"
            )
            logger.info("Nova Act disabled (NOVA_ACT_ENABLED=0); browser thread not started")
            return
        self._thread = threading.Thread(
            target=self._run_loop, args=(stop_event,), name="nova-browser", daemon=True
        )
        self._thread.start()
        logger.info("Nova Browser thread started")
