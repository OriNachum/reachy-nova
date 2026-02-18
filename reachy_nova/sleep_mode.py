"""Sleep mode state machine for Reachy Nova.

Pure state machine — no pose generation.  The SDK's goto_sleep() and
wake_up() handle all physical transitions.  The main loop uses the
state to decide what to do:

  awake          → normal operation
  falling_asleep → SDK owns robot, main loop only reads audio
  sleeping       → main loop sends breathing (antennas + body only)
  waking_up      → SDK owns robot, main loop only reads audio
"""

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class SleepManager:
    """Sleep/wake state machine.

    States: awake → falling_asleep → sleeping → waking_up → awake
    """

    def __init__(self, on_state_change: Callable[[str], None] | None = None):
        self.state = "awake"
        self._on_state_change = on_state_change
        self._sleep_start_time = 0.0

    def _set_state(self, state: str) -> None:
        self.state = state
        logger.info(f"[Sleep] State → {state}")
        if self._on_state_change:
            try:
                self._on_state_change(state)
            except Exception:
                pass

    def trigger_sleep(self) -> None:
        """Enter falling_asleep — the SDK's goto_sleep() will own the robot."""
        if self.state != "awake":
            return
        self._set_state("falling_asleep")

    def enter_sleeping(self) -> None:
        """Called by the SDK thread when goto_sleep() finishes."""
        if self.state != "falling_asleep":
            return
        self._sleep_start_time = time.time()
        self._set_state("sleeping")

    def trigger_wake(self) -> None:
        """Enter waking_up — the SDK's wake_up() will own the robot."""
        if self.state != "sleeping":
            return
        self._set_state("waking_up")

    def enter_awake(self) -> None:
        """Called by the SDK thread when wake_up() finishes."""
        if self.state != "waking_up":
            return
        self._set_state("awake")
