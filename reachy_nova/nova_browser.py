"""Nova Act - Browser automation triggered by voice commands."""

import base64
import logging
import threading
import queue
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class NovaBrowser:
    """Manages browser automation tasks via Amazon Nova Act."""

    def __init__(
        self,
        on_result: Callable[[str], None] | None = None,
        on_screenshot: Callable[[str], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
        headless: bool = False,
    ):
        self.on_result = on_result
        self.on_screenshot = on_screenshot
        self.on_state_change = on_state_change
        self.headless = headless

        self._task_queue: queue.Queue[dict] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._nova = None

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

    def queue_task(self, instruction: str, url: str | None = None) -> None:
        """Queue a browser automation task.

        Args:
            instruction: Natural language instruction for what to do.
            url: Optional URL to navigate to first.
        """
        self._task_queue.put({"instruction": instruction, "url": url})

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

    def _execute_task(self, task: dict) -> str:
        """Execute a single browser task."""
        from nova_act import NovaAct

        instruction = task["instruction"]
        url = task.get("url", "https://www.google.com")
        self.current_task = instruction
        self._set_state("busy")

        try:
            if self._nova is None:
                self._nova = NovaAct(
                    starting_page=url or "https://www.google.com",
                    headless=self.headless,
                )
                self._nova.start()
            elif url:
                self._nova.go_to_url(url)

            result = self._nova.act(instruction, max_steps=15)

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
            return result_text

        except Exception as e:
            error_msg = f"Browser error: {e}"
            logger.error(error_msg)
            self.last_result = error_msg
            self._set_state("error")
            # Reset the browser instance on error
            try:
                if self._nova:
                    self._nova.stop()
            except Exception:
                pass
            self._nova = None
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

    def start(self, stop_event: threading.Event) -> None:
        """Start the browser automation worker thread."""
        self._thread = threading.Thread(
            target=self._run_loop, args=(stop_event,), name="nova-browser", daemon=True
        )
        self._thread.start()
        logger.info("Nova Browser thread started")
