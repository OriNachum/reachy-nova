"""Thread-safe reactive state container for Reachy Nova."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class AppState:
    """Typed schema for all robot state fields."""

    voice_state: str = "idle"
    vision_enabled: bool = True
    vision_description: str = ""
    vision_analyzing: bool = False
    browser_state: str = "idle"
    browser_task: str = ""
    browser_result: str = ""
    browser_screenshot: str = ""
    last_user_text: str = ""
    last_assistant_text: str = ""
    antenna_mode: str = "auto"
    mood: str = "happy"
    tracking_enabled: bool = True
    tracking_mode: str = "idle"
    memory_state: str = "idle"
    memory_last_context: str = ""
    slack_state: str = "idle"
    slack_last_event: str = ""
    head_override: dict | None = None
    head_override_time: float = 0.0
    speech_enabled: bool = True
    movement_enabled: bool = True
    face_recognized: str = ""
    face_count: int = 0
    emotion_levels: dict = field(default_factory=dict)
    emotion_boredom: float = 0.0
    emotion_wounds: list = field(default_factory=list)
    gesture_active: bool = False
    gesture_name: str = ""
    sleep_mode: str = "awake"
    pat_antenna_time: float = 0.0

    # Head smoothing state — previously unprotected nonlocal vars
    smooth_yaw: float = 0.0
    smooth_pitch: float = 0.0
    body_yaw: float = 0.0


class State:
    """Thread-safe wrapper around AppState with change notifications."""

    def __init__(self, on_change: Callable[[dict], None] | None = None):
        self._state = AppState()
        self._lock = threading.Lock()
        self._on_change = on_change

    def update(self, **kwargs) -> None:
        """Set fields under lock, then fire on_change outside the lock."""
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)
        if self._on_change and kwargs:
            try:
                self._on_change(kwargs)
            except Exception:
                pass

    def get(self, field_name: str):
        """Read a single field under lock."""
        with self._lock:
            return getattr(self._state, field_name)

    def get_many(self, *fields: str) -> tuple:
        """Read multiple fields atomically under a single lock acquisition."""
        with self._lock:
            return tuple(getattr(self._state, f) for f in fields)

    def snapshot(self) -> dict:
        """Full copy for API responses."""
        with self._lock:
            return {
                k: v for k, v in self._state.__dict__.items()
                if not k.startswith("_")
            }
