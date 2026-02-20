"""Multi-subsystem sleep/wake lifecycle coordination for Reachy Nova.

Orchestrates SDK animation, Sonic restart, vision pause, gesture cancellation,
and state transitions across the full sleep cycle. Composes the SleepManager
(pure FSM) as an internal component.
"""

import logging
import threading
import time

import numpy as np
from reachy_mini.reachy_mini import SLEEP_HEAD_JOINT_POSITIONS, SLEEP_ANTENNAS_JOINT_POSITIONS

from .sleep_mode import SleepManager
from .temporal import utc_now_precise, utc_now_vague

logger = logging.getLogger(__name__)


class SleepOrchestrator:
    """Coordinates sleep/wake across all subsystems."""

    def __init__(self, state, sonic, vision, reachy_mini,
                 gesture_cancel_event: threading.Event,
                 session, memory, feedback, mqtt, stop_event: threading.Event,
                 restart_type: str, restart_elapsed: float, previous_session: dict | None,
                 t0: float):
        self._state = state
        self._sonic = sonic
        self._vision = vision
        self._reachy_mini = reachy_mini
        self._gesture_cancel_event = gesture_cancel_event
        self._session = session
        self._memory = memory
        self._feedback = feedback
        self._mqtt = mqtt
        self._stop_event = stop_event
        self._restart_type = restart_type
        self._restart_elapsed = restart_elapsed
        self._previous_session = previous_session
        self._t0 = t0

        self.sleep_manager = SleepManager(
            on_state_change=lambda s: self._state.update(sleep_mode=s),
        )
        self.high_boredom_start = 0.0
        self._startup_context_injected = False

    @property
    def state(self) -> str:
        return self.sleep_manager.state

    def initiate_sleep(self) -> None:
        """Stop LLM subsystems, run SDK sleep animation, then enter breathing loop."""
        if self.sleep_manager.state != "awake":
            return
        logger.info("[Sleep] Initiating sleep — cancelling movements, stopping subsystems")

        # 1. Cancel any running gesture
        self._gesture_cancel_event.set()

        # 2. Clear head override and disable tracking
        self._state.update(
            vision_enabled=False,
            tracking_enabled=False,
            tracking_mode="idle",
            head_override=None,
            gesture_active=False,
            gesture_name="",
        )

        # Brief pause for gesture thread to see the cancel
        time.sleep(0.05)
        self._gesture_cancel_event.clear()

        # 3. Stop Sonic
        self._sonic.stop()

        # 4. Clear pending vision analysis
        self._vision._force_analyze.clear()

        self.high_boredom_start = 0.0

        # 5. Mark as falling_asleep
        self.sleep_manager.trigger_sleep()

        # 6. Run SDK sleep animation in background
        def _sdk_sleep():
            try:
                self._reachy_mini.goto_sleep()
            except Exception as e:
                logger.warning(f"[Sleep] SDK goto_sleep failed: {e}")
            self.sleep_manager.enter_sleeping()

        threading.Thread(target=_sdk_sleep, daemon=True, name="sdk-sleep").start()

    def initiate_wake(self) -> None:
        """Run SDK wake animation and restart LLM subsystems."""
        if self.sleep_manager.state != "sleeping":
            return
        logger.info("[Sleep] Initiating wake — restarting Sonic, re-enabling vision/tracking")
        self.sleep_manager.trigger_wake()

        def _sdk_wake():
            try:
                self._reachy_mini.wake_up()
            except Exception as e:
                logger.warning(f"[Sleep] SDK wake_up failed: {e}")
            self.sleep_manager.enter_awake()
            self._sonic.restart(self._stop_event)
            self._state.update(vision_enabled=True, tracking_enabled=True)
            # Deferred startup context injection on first wake
            if not self._startup_context_injected:
                self._startup_context_injected = True
                threading.Thread(
                    target=self._inject_startup_context, daemon=True,
                    name="memory-startup",
                ).start()

        threading.Thread(target=_sdk_wake, daemon=True, name="sdk-wake").start()

    def startup_sleep(self) -> None:
        """Enter sleep mode at startup — skip Sonic, check if already in position."""
        self._state.update(
            vision_enabled=False,
            tracking_enabled=False,
            tracking_mode="idle",
            head_override=None,
        )
        if self._is_in_sleep_position():
            logger.info("[Sleep] Already in sleep position — entering sleeping directly")
            self.sleep_manager.enter_sleeping_direct()
        else:
            logger.info("[Sleep] Not in sleep position — running goto_sleep transition")
            self.sleep_manager.trigger_sleep()

            def _sdk_startup_sleep():
                try:
                    self._reachy_mini.goto_sleep()
                except Exception as e:
                    logger.warning(f"[Sleep] SDK goto_sleep failed: {e}")
                self.sleep_manager.enter_sleeping()

            threading.Thread(
                target=_sdk_startup_sleep, daemon=True, name="sdk-startup-sleep",
            ).start()

    def _is_in_sleep_position(self) -> bool:
        """Check if robot is already physically in the sleep position."""
        try:
            head_pos, ant_pos = self._reachy_mini.get_current_joint_positions()
            head_dist = np.linalg.norm(
                np.array(head_pos) - np.array(SLEEP_HEAD_JOINT_POSITIONS)
            )
            ant_diff = np.max(np.abs(
                np.array(ant_pos) - np.array(SLEEP_ANTENNAS_JOINT_POSITIONS)
            ))
            return head_dist < 0.3 and ant_diff < 0.5
        except Exception as e:
            logger.warning(f"[Sleep] Could not read joint positions: {e}")
            return False

    def _inject_startup_context(self) -> None:
        """Inject session + memory context into Sonic after first wake."""
        time.sleep(2)  # Wait for Sonic to be ready
        try:
            parts = []

            time_anchor = f"[Current time: {utc_now_precise()}. {utc_now_vague()}]"
            parts.append(time_anchor)

            session_ctx = self._session.get_restart_context(
                self._restart_type, self._restart_elapsed, self._previous_session,
            )
            if session_ctx:
                parts.append(session_ctx)

            memory_ctx = self._memory.get_startup_context()
            if memory_ctx:
                parts.append(f"Things you remember:\n{memory_ctx}")

            if parts:
                combined = "\n\n".join(parts)
                self._sonic.inject_text(combined)
                self._mqtt.publish_event("memory", "startup_context", {"context": combined})
                logger.info(
                    f"Injected startup context ({len(combined)} chars, "
                    f"restart={self._restart_type})"
                )
        except Exception as e:
            logger.warning(f"Startup context injection failed: {e}")
