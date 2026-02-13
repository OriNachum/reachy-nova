"""Nova Act - Browser automation triggered by voice commands."""

import base64
import logging
import os
import threading
import queue
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

WORKFLOW_MODEL_ID = "nova-act-latest"


class NovaBrowser:
    """Manages browser automation tasks via Amazon Nova Act."""

    def __init__(
        self,
        on_result: Callable[[str], None] | None = None,
        on_screenshot: Callable[[str], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
        on_progress: Callable[[str], None] | None = None,
        headless: bool = False,
        chrome_channel: str = "chromium",
    ):
        self.on_result = on_result
        self.on_screenshot = on_screenshot
        self.on_state_change = on_state_change
        self.on_progress = on_progress
        self.headless = headless
        self.chrome_channel = chrome_channel

        self._task_queue: queue.Queue[dict] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._nova = None
        self._workflow = None

        self.state = "idle"  # idle, busy, error
        self.last_result = ""
        self.last_screenshot_b64 = ""
        self.current_task = ""
        self.history: list[dict] = []

    def _set_state(self, state: str) -> None:
        self.state = state
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception:
                pass

    def _emit_progress(self, message: str) -> None:
        """Emit a progress update for narration."""
        logger.info(f"[Browser progress] {message}")
        if self.on_progress:
            try:
                self.on_progress(message)
            except Exception:
                pass

    def queue_task(self, instruction: str, url: str | None = None) -> None:
        """Queue a browser automation task (fire-and-forget).

        Args:
            instruction: Natural language instruction for what to do.
            url: Optional URL to navigate to first.
        """
        self._task_queue.put({"instruction": instruction, "url": url})

    def execute(self, instruction: str, url: str | None = None) -> str:
        """Execute a browser task synchronously, blocking until complete.

        Used by the skill executor so Sonic can wait for the result.

        Args:
            instruction: Natural language instruction for what to do.
            url: Optional URL to navigate to first.

        Returns:
            The result string from the browser task.
        """
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

    def _execute_task(self, task: dict) -> str:
        """Execute a single browser task."""
        from nova_act import NovaAct

        instruction = task["instruction"]
        url = task.get("url", "https://www.google.com")
        done_event = task.get("_done_event")
        result_holder = task.get("_result_holder")
        self.current_task = instruction
        self._set_state("busy")

        try:
            self._emit_progress("Opening browser...")
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
                self._emit_progress(f"Navigating to {url or 'Google'}...")
            elif url:
                self._emit_progress(f"Navigating to {url}...")
                self._nova.go_to_url(url)

            self._emit_progress(f"Working on: {instruction}...")
            result = self._nova.act(instruction, max_steps=15)

            self._emit_progress("Done! Reading results...")
            self._capture_screenshot()

            result_text = f"Done: {instruction}"
            if result and hasattr(result, "success"):
                if result.success:
                    result_text = f"Completed: {instruction}"
                else:
                    result_text = f"Could not complete: {instruction}"

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
        """Start the browser automation worker thread."""
        self._thread = threading.Thread(
            target=self._run_loop, args=(stop_event,), name="nova-browser", daemon=True
        )
        self._thread.start()
        logger.info("Nova Browser thread started")
