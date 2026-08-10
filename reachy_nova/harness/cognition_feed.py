"""Cognition feed emitter — reachy-mini-cli's export-schema NDJSON contract.

Mirrors ``reachy-mini-cli``'s cognition feed
(``docs/export-schema.md``, ``reachy/export/events.py``): the same three block
types, the same key ordering, the same compact/`ensure_ascii=False` wire
format, so a consumer written against that feed (e.g. the reTerminal bridge,
``~/.claude/skills/reterminal/scripts/reachy-export-bridge.py``) accepts
Reachy Nova's lines unmodified.

Wire format
-----------
One compact JSON object per line, ``t`` and ``ts`` always first:

- ``{"t":"thinking","ts":<float>,"cues":[...],"text":"..."}``
- ``{"t":"message","ts":<float>,"text":"..."}``
- ``{"t":"emotion","ts":<float>,"emoji":"...","pose":{...}|null}``

Only these three block types are ever emitted from this module — a consumer
that sees any other ``t`` value belongs to a different feed (reachy-mini-cli's
runtime feed uses ``sense``/``rule``/``intent``/``motion`` and is intentionally
never mixed with this one).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import IO, Any


class CognitionFeed:
    """Writes cognition-feed NDJSON lines to a file-like target or a path.

    Parameters
    ----------
    target:
        Where lines are written. Either a file-like object with a ``write``
        method (default: ``sys.stdout``) or a filesystem path (``str`` /
        ``os.PathLike``), which is opened in append mode and closed by
        :meth:`close`.

    Thread-safety: every write is serialized behind an internal lock, so
    multiple threads may share one ``CognitionFeed`` instance safely.
    """

    def __init__(self, target: IO[str] | str | os.PathLike | None = None) -> None:
        self._lock = threading.Lock()
        self._owns_file = False
        if target is None:
            self._file: IO[str] = sys.stdout
        elif hasattr(target, "write"):
            self._file = target  # type: ignore[assignment]
        else:
            self._file = open(target, "a", encoding="utf-8")  # noqa: SIM115
            self._owns_file = True

    def thinking(self, text: str, cues: list[str] | None = None, ts: float | None = None) -> None:
        """Emit a ``"thinking"`` block."""
        self._emit(
            {
                "t": "thinking",
                "ts": self._ts(ts),
                "cues": list(cues) if cues is not None else [],
                "text": text,
            }
        )

    def message(self, text: str, ts: float | None = None) -> None:
        """Emit a ``"message"`` block."""
        self._emit({"t": "message", "ts": self._ts(ts), "text": text})

    def emotion(self, emoji: str, pose: dict[str, Any] | None = None, ts: float | None = None) -> None:
        """Emit an ``"emotion"`` block."""
        self._emit({"t": "emotion", "ts": self._ts(ts), "emoji": emoji, "pose": pose})

    def close(self) -> None:
        """Close the underlying file if this instance opened it (path target)."""
        with self._lock:
            if self._owns_file:
                self._file.close()

    @staticmethod
    def _ts(ts: float | None) -> float:
        return time.time() if ts is None else ts

    def _emit(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()
