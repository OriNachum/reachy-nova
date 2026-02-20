"""Dependency injection container for Reachy Nova.

Bundles all subsystem references into a single typed object for passing
into extracted modules, replacing implicit closure capture.
"""

import threading
from dataclasses import dataclass


@dataclass
class NovaContext:
    """All subsystem references needed by skill executors, API routes, etc."""

    state: object              # State
    sonic: object              # NovaSonic
    vision: object             # NovaVision
    browser: object            # NovaBrowser
    memory: object             # NovaMemory
    feedback: object           # NovaFeedback
    slack_bot: object          # NovaSlack
    tracker: object            # TrackingManager
    face_manager: object       # FaceManager
    face_recognition: object   # FaceRecognition
    skill_manager: object      # SkillManager
    gesture_engine: object     # GestureEngine
    sleep_orchestrator: object # SleepOrchestrator
    mqtt: object               # NovaMQTT
    safety: object             # SafetyManager
    session: object            # SessionState
    emotional_state: object    # EmotionalState
    reachy_mini: object        # ReachyMini
    stop_event: threading.Event = None
    t0: float = 0.0
    gesture_cancel_event: threading.Event = None
