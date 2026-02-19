"""Multi-dimensional emotional state system.

Manages 5 base emotions (joy, sadness, anger, fear, disgust) with natural decay,
event-driven stimulation, wound mechanics for severe trauma, boredom tracking,
and derived mood mapping to feed into the antenna animation system.

Thread-safe: all state access is protected by an internal lock.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "emotions.yaml"

EMOTION_NAMES = ("joy", "sadness", "anger", "fear", "disgust")


@dataclass
class ActiveWound:
    """A wound created by a severe emotional event."""
    id: str
    event: str
    floors: dict[str, float]       # emotion -> minimum floor value
    duration: float                 # seconds the floor is fixed
    heal_rate: float                # floor decay per second after duration
    created_at: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def is_fixed(self) -> bool:
        """True while within the initial fixed-floor duration."""
        return self.age < self.duration

    @property
    def is_healed(self) -> bool:
        """True when all floors have decayed to zero."""
        return all(v <= 0 for v in self.floors.values())

    def tick(self, dt: float):
        """Shrink floors after duration expires."""
        if self.is_fixed:
            return
        for emo in list(self.floors):
            self.floors[emo] = max(0.0, self.floors[emo] - self.heal_rate * dt)

    def reduce(self, amount: float):
        """Chip away at wound floors (from healing events)."""
        for emo in list(self.floors):
            self.floors[emo] = max(0.0, self.floors[emo] - amount)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event": self.event,
            "floors": dict(self.floors),
            "duration": self.duration,
            "age": round(self.age, 1),
            "healing": not self.is_fixed,
        }


class EmotionalState:
    """Thread-safe multi-dimensional emotional state manager."""

    def __init__(self, config_path: str | Path | None = None):
        self._lock = threading.Lock()
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG
        self._config: dict[str, Any] = {}

        # Emotion levels (0.0 - 1.0)
        self._levels: dict[str, float] = {}
        self._baselines: dict[str, float] = {}
        self._decay_rates: dict[str, float] = {}

        # Wounds
        self._wounds: list[ActiveWound] = []

        # Boredom
        self._boredom: float = 0.0

        # Derived mood hysteresis
        self._current_mood: str = "happy"
        self._mood_change_time: float = 0.0

        # Mood override (backward compatibility)
        self._mood_override: str | None = None
        self._mood_override_expires: float = 0.0

        # Transcript trigger cooldowns: event_name -> last_trigger_time
        self._trigger_cooldowns: dict[str, float] = {}

        self._load_config()

    # --- Config loading ---

    def _load_config(self):
        try:
            with open(self._config_path) as f:
                self._config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Emotion config not found at {self._config_path}, using defaults")
            self._config = {}

        emo_cfg = self._config.get("emotions", {})
        for name in EMOTION_NAMES:
            cfg = emo_cfg.get(name, {})
            self._baselines[name] = cfg.get("baseline", 0.0)
            self._decay_rates[name] = cfg.get("decay_rate", 0.03)
            self._levels[name] = self._baselines[name]

        logger.info(f"Emotion system loaded: {len(self._config.get('events', {}))} events, "
                     f"{len(self._config.get('derived_mood_rules', []))} mood rules")

    def reload_config(self):
        """Hot-reload config from disk."""
        with self._lock:
            self._load_config()
        logger.info("Emotion config reloaded")

    # --- Event application ---

    def apply_event(self, event_name: str, intensity: float = 1.0):
        """Apply a named event from config."""
        events = self._config.get("events", {})
        event_cfg = events.get(event_name)
        if event_cfg is None:
            logger.warning(f"Unknown emotion event: {event_name}")
            return

        with self._lock:
            # Apply deltas
            deltas = event_cfg.get("deltas", {})
            for emo, delta in deltas.items():
                if emo in self._levels:
                    self._levels[emo] = max(0.0, min(1.0, self._levels[emo] + delta * intensity))

            # Healing events chip at wounds
            wound_reduction = event_cfg.get("wound_reduction", 0.0)
            if wound_reduction > 0:
                for wound in self._wounds:
                    wound.reduce(wound_reduction * intensity)

            # Severe events create wounds
            if event_cfg.get("severity") == "severe" and "wound" in event_cfg:
                max_wounds = self._config.get("settings", {}).get("max_wounds", 5)
                if len(self._wounds) < max_wounds:
                    wcfg = event_cfg["wound"]
                    wound = ActiveWound(
                        id=str(uuid.uuid4())[:8],
                        event=event_name,
                        floors=dict(wcfg.get("floors", {})),
                        duration=wcfg.get("duration", 300),
                        heal_rate=wcfg.get("heal_rate", 0.001),
                    )
                    self._wounds.append(wound)
                    logger.info(f"Wound created: {wound.id} from {event_name} "
                                f"(floors={wound.floors}, duration={wound.duration}s)")

        logger.debug(f"Event applied: {event_name} (intensity={intensity:.2f})")

    def apply_raw_delta(self, emotion_name: str, delta: float):
        """Direct adjustment to a single emotion (for skill use)."""
        if emotion_name not in self._levels:
            return
        with self._lock:
            self._levels[emotion_name] = max(0.0, min(1.0,
                self._levels[emotion_name] + delta))

    # --- Frame update ---

    def update(self, dt: float):
        """Called each frame (~50Hz). Handles decay, wounds, boredom."""
        with self._lock:
            # Natural decay toward baselines
            for name in EMOTION_NAMES:
                level = self._levels[name]
                baseline = self._baselines[name]
                rate = self._decay_rates[name]
                if abs(level - baseline) > 0.001:
                    if level > baseline:
                        self._levels[name] = max(baseline, level - rate * dt)
                    else:
                        self._levels[name] = min(baseline, level + rate * dt)

            # Wound tick + floor enforcement
            healed = []
            for wound in self._wounds:
                wound.tick(dt)
                if wound.is_healed:
                    healed.append(wound)
                else:
                    # Enforce floors
                    for emo, floor in wound.floors.items():
                        if emo in self._levels and self._levels[emo] < floor:
                            self._levels[emo] = floor

            for wound in healed:
                self._wounds.remove(wound)
                logger.info(f"Wound healed: {wound.id} ({wound.event})")

            # Boredom accumulation/decay
            boredom_cfg = self._config.get("boredom", {})
            joy_thresh = boredom_cfg.get("joy_threshold", 0.15)
            neg_thresh = boredom_cfg.get("negative_threshold", 0.1)
            accum_rate = boredom_cfg.get("accumulate_rate", 0.008)
            decay_rate = boredom_cfg.get("decay_rate", 0.05)

            negatives_low = all(
                self._levels[e] < neg_thresh
                for e in ("sadness", "anger", "fear", "disgust")
            )
            if self._levels["joy"] < joy_thresh and negatives_low:
                self._boredom = min(1.0, self._boredom + accum_rate * dt)
            else:
                self._boredom = max(0.0, self._boredom - decay_rate * dt)

            # Expire mood override
            if self._mood_override and time.time() > self._mood_override_expires:
                self._mood_override = None

    # --- Derived mood ---

    def get_derived_mood(self, voice_state: str = "idle") -> str:
        """Evaluate rules to derive a mood string. Respects hysteresis."""
        with self._lock:
            # Mood override takes priority
            if self._mood_override:
                return self._mood_override

            hysteresis = self._config.get("settings", {}).get("hysteresis", 1.0)
            now = time.time()

            rules = self._config.get("derived_mood_rules", [])
            matched_mood = "calm"  # fallback

            # Find dominant emotion
            dominant = max(EMOTION_NAMES, key=lambda e: self._levels[e])

            for rule in rules:
                conditions = rule.get("conditions", {})
                if self._evaluate_conditions(conditions, voice_state, dominant):
                    matched_mood = rule["mood"]
                    break

            # Apply hysteresis
            if matched_mood != self._current_mood:
                if now - self._mood_change_time >= hysteresis:
                    self._current_mood = matched_mood
                    self._mood_change_time = now
            return self._current_mood

    def _evaluate_conditions(self, conditions: dict, voice_state: str, dominant: str) -> bool:
        """Evaluate a single rule's conditions against current state."""
        if not conditions:
            return True  # empty conditions = always match (fallback)

        for key, value in conditions.items():
            if key == "_voice_state":
                if voice_state != value:
                    return False
            elif key == "_boredom":
                if not self._compare(self._boredom, value):
                    return False
            elif key == "_has_wounds":
                if value and not self._wounds:
                    return False
                if not value and self._wounds:
                    return False
            elif key == "_dominant":
                if dominant != value:
                    return False
            elif key in EMOTION_NAMES:
                if not self._compare(self._levels.get(key, 0.0), value):
                    return False
            else:
                logger.warning(f"Unknown condition key: {key}")
                return False
        return True

    @staticmethod
    def _compare(actual: float, condition) -> bool:
        """Parse and evaluate a comparison like '>= 0.7' or '< 0.2'."""
        if isinstance(condition, (int, float)):
            return actual >= condition
        if isinstance(condition, bool):
            return bool(actual) == condition
        s = str(condition).strip()
        if s.startswith(">="):
            return actual >= float(s[2:])
        elif s.startswith("<="):
            return actual <= float(s[2:])
        elif s.startswith(">"):
            return actual > float(s[1:])
        elif s.startswith("<"):
            return actual < float(s[1:])
        elif s.startswith("=="):
            return abs(actual - float(s[2:])) < 0.001
        else:
            return actual >= float(s)

    # --- Transcript detection ---

    def check_transcript(self, text: str) -> list[str]:
        """Check transcript text against configured triggers. Returns matched event names."""
        triggers = self._config.get("transcript_triggers", [])
        cooldown = self._config.get("settings", {}).get("transcript_cooldown", 5.0)
        now = time.time()
        matched = []

        for trigger in triggers:
            event = trigger["event"]

            # Check cooldown
            last = self._trigger_cooldowns.get(event, 0.0)
            if now - last < cooldown:
                continue

            caps_only = trigger.get("caps_only", False)
            patterns = trigger.get("patterns", [])

            for pattern in patterns:
                if caps_only:
                    # Match only if the pattern appears in the original text as-is (uppercase)
                    if pattern in text:
                        matched.append(event)
                        self._trigger_cooldowns[event] = now
                        break
                else:
                    if pattern.lower() in text.lower():
                        matched.append(event)
                        self._trigger_cooldowns[event] = now
                        break

        return matched

    def get_transcript_triggers(self) -> list[dict]:
        """Return configured transcript triggers."""
        return self._config.get("transcript_triggers", [])

    # --- Override (backward compat) ---

    def set_mood_override(self, mood: str, duration: float = 5.0):
        """Set a direct mood override that expires after duration seconds."""
        with self._lock:
            self._mood_override = mood
            self._mood_override_expires = time.time() + duration

    # --- Getters ---

    def get_levels(self) -> dict[str, float]:
        with self._lock:
            return {k: round(v, 3) for k, v in self._levels.items()}

    def get_boredom(self) -> float:
        with self._lock:
            return round(self._boredom, 3)

    def get_wounds(self) -> list[dict]:
        with self._lock:
            return [w.to_dict() for w in self._wounds]

    def get_full_state(self) -> dict:
        with self._lock:
            return {
                "levels": {k: round(v, 3) for k, v in self._levels.items()},
                "boredom": round(self._boredom, 3),
                "wounds": [w.to_dict() for w in self._wounds],
                "mood": self._current_mood,
                "mood_override": self._mood_override,
            }

    def get_serializable_state(self) -> dict:
        """Full state including wound reconstruction data for session persistence."""
        with self._lock:
            wounds = []
            for w in self._wounds:
                remaining_fixed = max(0.0, w.duration - w.age)
                wounds.append({
                    "id": w.id,
                    "event": w.event,
                    "floors": dict(w.floors),
                    "duration": w.duration,
                    "heal_rate": w.heal_rate,
                    "created_at": w.created_at,
                    "remaining_fixed_seconds": round(remaining_fixed, 1),
                })
            return {
                "levels": {k: round(v, 4) for k, v in self._levels.items()},
                "boredom": round(self._boredom, 4),
                "current_mood": self._current_mood,
                "wounds": wounds,
            }

    def restore_state(self, data: dict, elapsed_seconds: float):
        """Restore emotional state from saved data with time-adjusted decay.

        Applies natural decay for elapsed_seconds toward baselines.
        Reconstructs wounds with time-adjusted healing.
        """
        if not data:
            return

        with self._lock:
            # Restore levels and apply decay for elapsed time
            saved_levels = data.get("levels", {})
            for name in EMOTION_NAMES:
                if name in saved_levels:
                    level = saved_levels[name]
                    baseline = self._baselines.get(name, 0.0)
                    rate = self._decay_rates.get(name, 0.03)
                    # Apply natural decay toward baseline
                    if level > baseline:
                        level = max(baseline, level - rate * elapsed_seconds)
                    elif level < baseline:
                        level = min(baseline, level + rate * elapsed_seconds)
                    self._levels[name] = max(0.0, min(1.0, level))

            # Reset boredom if person is coming back after 60s+
            if elapsed_seconds > 60.0:
                self._boredom = 0.0
            else:
                self._boredom = data.get("boredom", 0.0)

            # Restore mood
            saved_mood = data.get("current_mood", "")
            if saved_mood:
                self._current_mood = saved_mood

            # Reconstruct wounds with time-adjusted healing
            self._wounds.clear()
            for wd in data.get("wounds", []):
                floors = wd.get("floors", {})
                duration = wd.get("duration", 300)
                heal_rate = wd.get("heal_rate", 0.001)
                remaining_fixed = wd.get("remaining_fixed_seconds", 0.0)

                if remaining_fixed > elapsed_seconds:
                    # Still in fixed phase: adjust created_at so remaining time is correct
                    new_remaining = remaining_fixed - elapsed_seconds
                    created_at = time.time() - (duration - new_remaining)
                    wound = ActiveWound(
                        id=wd.get("id", str(uuid.uuid4())[:8]),
                        event=wd.get("event", "unknown"),
                        floors=floors,
                        duration=duration,
                        heal_rate=heal_rate,
                        created_at=created_at,
                    )
                else:
                    # Fixed phase expired offline: apply healing
                    healing_time = elapsed_seconds - remaining_fixed
                    for emo in list(floors):
                        floors[emo] = max(0.0, floors[emo] - heal_rate * healing_time)
                    # Check if fully healed
                    if all(v <= 0 for v in floors.values()):
                        continue  # Discard fully healed wound
                    # Set created_at so it appears past the fixed duration
                    created_at = time.time() - duration - 1.0
                    wound = ActiveWound(
                        id=wd.get("id", str(uuid.uuid4())[:8]),
                        event=wd.get("event", "unknown"),
                        floors=floors,
                        duration=duration,
                        heal_rate=heal_rate,
                        created_at=created_at,
                    )
                self._wounds.append(wound)

        logger.info(
            f"Emotional state restored (elapsed={elapsed_seconds:.0f}s, "
            f"wounds={len(self._wounds)})"
        )

    def get_event_names(self) -> list[str]:
        return list(self._config.get("events", {}).keys())
