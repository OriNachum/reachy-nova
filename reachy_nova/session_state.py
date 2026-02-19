"""Session persistence for Reachy Nova.

Saves and restores session state (conversation, emotions, metadata) across
restarts. Classifies restart type (crash recovery, short break, long absence)
and generates appropriate context injection text for seamless continuity.

Storage: ~/.reachy_nova/session/session.json (atomic writes with .bak backup).
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_DIR = Path.home() / ".reachy_nova" / "session"
SESSION_FILE = SESSION_DIR / "session.json"
SESSION_TMP = SESSION_DIR / "session.json.tmp"
SESSION_BAK = SESSION_DIR / "session.json.bak"

VERSION = 1

# Thresholds for restart classification
CRASH_THRESHOLD = 30.0       # seconds — below this + unclean = crash recovery
SHORT_BREAK_THRESHOLD = 3600.0  # 1 hour


class SessionState:
    """Manages session persistence with atomic writes and restart classification."""

    def __init__(self):
        self._session_id: str = uuid.uuid4().hex[:8]
        self._session_start: float = time.time()
        self._total_sessions: int = 1
        self._restart_type: str = "fresh_start"
        self._elapsed: float = 0.0

        self._last_heartbeat_time: float = 0.0
        self._last_save_time: float = 0.0

        SESSION_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def restart_type(self) -> str:
        return self._restart_type

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def total_sessions(self) -> int:
        return self._total_sessions

    def load(self) -> dict:
        """Read previous session from disk. Try backup on failure. Return {} on any error."""
        for path in (SESSION_FILE, SESSION_BAK):
            try:
                if path.exists():
                    data = json.loads(path.read_text())
                    if isinstance(data, dict) and data.get("version") == VERSION:
                        logger.info(f"Loaded session state from {path}")
                        return data
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        return {}

    def classify_restart(self, prev: dict) -> tuple[str, float]:
        """Classify restart type based on previous session data.

        Returns (restart_type, elapsed_seconds).
        """
        if not prev:
            self._restart_type = "fresh_start"
            self._elapsed = 0.0
            return self._restart_type, self._elapsed

        heartbeat = prev.get("heartbeat", 0.0)
        elapsed = time.time() - heartbeat
        shutdown_clean = prev.get("shutdown_clean", True)
        self._total_sessions = prev.get("total_sessions", 0) + 1
        self._elapsed = elapsed

        if not shutdown_clean and elapsed < CRASH_THRESHOLD:
            self._restart_type = "crash_recovery"
        elif elapsed < SHORT_BREAK_THRESHOLD:
            self._restart_type = "short_break"
        else:
            self._restart_type = "long_absence"

        logger.info(
            f"Restart classified: {self._restart_type} "
            f"(elapsed={elapsed:.0f}s, clean={shutdown_clean}, "
            f"session #{self._total_sessions})"
        )
        return self._restart_type, elapsed

    def mark_started(self):
        """Write shutdown_clean=false immediately as crash detection marker."""
        try:
            data = {
                "version": VERSION,
                "heartbeat": time.time(),
                "shutdown_clean": False,
                "session_start": self._session_start,
                "session_id": self._session_id,
                "total_sessions": self._total_sessions,
            }
            self._atomic_write(data)
            logger.info(f"Session started: {self._session_id} (crash marker set)")
        except Exception as e:
            logger.error(f"Failed to mark session started: {e}")

    def update_heartbeat(self):
        """Update heartbeat timestamp. Internally throttled to every 10s."""
        now = time.time()
        if now - self._last_heartbeat_time < 10.0:
            return
        self._last_heartbeat_time = now
        try:
            if SESSION_FILE.exists():
                data = json.loads(SESSION_FILE.read_text())
                data["heartbeat"] = now
                self._atomic_write(data)
        except Exception as e:
            logger.warning(f"Heartbeat update failed: {e}")

    def save(
        self,
        emotions: dict | None = None,
        conversation: list[dict] | None = None,
        sleep_state: str = "awake",
        face_info: dict | None = None,
    ):
        """Full state save. Internally throttled to every 30s."""
        now = time.time()
        if now - self._last_save_time < 30.0:
            return
        self._last_save_time = now
        self._do_save(emotions, conversation, sleep_state, face_info, shutdown_clean=False)

    def save_shutdown(
        self,
        emotions: dict | None = None,
        conversation: list[dict] | None = None,
        sleep_state: str = "awake",
        face_info: dict | None = None,
    ):
        """Set shutdown_clean=true and save immediately."""
        self._do_save(emotions, conversation, sleep_state, face_info, shutdown_clean=True)
        logger.info("Session saved (clean shutdown)")

    def _do_save(
        self,
        emotions: dict | None,
        conversation: list[dict] | None,
        sleep_state: str,
        face_info: dict | None,
        shutdown_clean: bool,
    ):
        try:
            data = {
                "version": VERSION,
                "heartbeat": time.time(),
                "shutdown_clean": shutdown_clean,
                "session_start": self._session_start,
                "session_id": self._session_id,
                "uptime_seconds": time.time() - self._session_start,
                "total_sessions": self._total_sessions,
                "emotions": emotions or {},
                "conversation": conversation or [],
                "sleep_state": sleep_state,
            }
            if face_info:
                data["last_person_seen"] = face_info.get("name", "")
                data["last_person_seen_time"] = face_info.get("time", 0.0)
            self._atomic_write(data)
        except Exception as e:
            logger.error(f"Session save failed: {e}")

    def _atomic_write(self, data: dict):
        """Write to .tmp then os.replace(). Keep .bak of previous good state."""
        try:
            if SESSION_FILE.exists():
                # Backup current file before overwriting
                try:
                    SESSION_BAK.write_text(SESSION_FILE.read_text())
                except Exception:
                    pass

            SESSION_TMP.write_text(json.dumps(data, indent=2))
            os.replace(str(SESSION_TMP), str(SESSION_FILE))
        except Exception as e:
            # Clean up tmp file if it exists
            try:
                SESSION_TMP.unlink(missing_ok=True)
            except Exception:
                pass
            raise e

    def get_restart_context(
        self,
        restart_type: str,
        elapsed: float,
        prev: dict,
    ) -> str:
        """Build injection text for the given restart scenario."""
        if restart_type == "fresh_start":
            return ""

        if restart_type == "crash_recovery":
            return self._crash_recovery_context(prev)
        elif restart_type == "short_break":
            return self._short_break_context(elapsed, prev)
        elif restart_type == "long_absence":
            return self._long_absence_context(elapsed, prev)
        return ""

    def _crash_recovery_context(self, prev: dict) -> str:
        """Silent recovery — inject last messages, tell Nova to continue naturally."""
        lines = ["[Session context — you just restarted due to a brief interruption.]"]

        conversation = prev.get("conversation", [])
        recent = conversation[-10:] if conversation else []
        if recent:
            lines.append("Recent conversation (continue naturally, do NOT mention any restart):")
            for msg in recent:
                role = msg.get("role", "?")
                text = msg.get("text", "")
                lines.append(f"  {role}: {text}")

        mood = prev.get("emotions", {}).get("current_mood", "")
        if mood:
            lines.append(f"Your mood was: {mood}")

        lines.append("Continue the conversation seamlessly as if nothing happened.")
        return "\n".join(lines)

    def _short_break_context(self, elapsed: float, prev: dict) -> str:
        """Brief acknowledgment — inject messages + mood + wounds."""
        minutes = int(elapsed / 60)
        seconds = int(elapsed % 60)
        if minutes > 0:
            time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            time_str = f"{seconds} second{'s' if seconds != 1 else ''}"

        lines = [
            f"[Session context — you were offline for {time_str}.]",
        ]

        conversation = prev.get("conversation", [])
        recent = conversation[-15:] if conversation else []
        if recent:
            lines.append("Recent conversation before the break:")
            for msg in recent:
                role = msg.get("role", "?")
                text = msg.get("text", "")
                lines.append(f"  {role}: {text}")

        emotions = prev.get("emotions", {})
        mood = emotions.get("current_mood", "")
        if mood:
            lines.append(f"Your mood was: {mood}")

        wounds = emotions.get("wounds", [])
        if wounds:
            wound_strs = [f"{w.get('event', '?')} (healing: {w.get('healing', False)})" for w in wounds]
            lines.append(f"Active emotional wounds: {', '.join(wound_strs)}")

        person = prev.get("last_person_seen", "")
        if person:
            lines.append(f"Last person you saw: {person}")

        lines.append("Greet naturally — briefly acknowledge you're back, then continue.")
        return "\n".join(lines)

    def _long_absence_context(self, elapsed: float, prev: dict) -> str:
        """Warm welcome — summarize topics, mention time passed."""
        hours = elapsed / 3600
        if hours >= 24:
            days = int(hours / 24)
            time_str = f"{days} day{'s' if days != 1 else ''}"
        else:
            time_str = f"{int(hours)} hour{'s' if int(hours) != 1 else ''}"

        lines = [
            f"[Session context — you've been offline for about {time_str}.]",
        ]

        # Summarize topics from conversation instead of full transcript
        conversation = prev.get("conversation", [])
        if conversation:
            topics = self._extract_topics(conversation)
            if topics:
                lines.append(f"Topics from your last session: {', '.join(topics)}")

        total = prev.get("total_sessions", 0)
        if total > 1:
            lines.append(f"This is your session #{self._total_sessions} overall.")

        person = prev.get("last_person_seen", "")
        if person:
            lines.append(f"Last person you interacted with: {person}")

        lines.append("Welcome them back warmly. You can mention how long it's been.")
        return "\n".join(lines)

    @staticmethod
    def _extract_topics(conversation: list[dict], max_topics: int = 5) -> list[str]:
        """Extract rough topic hints from conversation (simple keyword approach)."""
        user_texts = [
            msg["text"] for msg in conversation
            if msg.get("role") == "USER" and msg.get("text")
        ]
        if not user_texts:
            return []

        # Take a sample of user messages spread across the conversation
        if len(user_texts) <= max_topics:
            samples = user_texts
        else:
            step = len(user_texts) / max_topics
            samples = [user_texts[int(i * step)] for i in range(max_topics)]

        # Truncate each to a short snippet
        topics = []
        for text in samples:
            snippet = text[:60].strip()
            if len(text) > 60:
                snippet += "..."
            topics.append(f'"{snippet}"')
        return topics

    def get_session_info(self) -> dict:
        """Return session metadata for API endpoint."""
        return {
            "session_id": self._session_id,
            "restart_type": self._restart_type,
            "elapsed_since_last": round(self._elapsed, 1),
            "uptime": round(time.time() - self._session_start, 1),
            "total_sessions": self._total_sessions,
            "session_start": self._session_start,
        }
