"""A small, queryable memory of what the harness actually injected (t8).

Every stage of the sensory pipeline already gets one grep-able senselog line
(``sensory_log.stage``), but a log line is not something Nova can read back
mid-conversation. This module is the read seam behind the ``recall_senses``
tool (see ``reachy_nova/harness/tools.py``): a small ring buffer that
:class:`~reachy_nova.harness.bus.NovaBus` appends to for every inject that
actually reaches the voice model, so "why did you move?" / "what did you
feel?" can be answered from what really happened instead of guessed at.

Ordering
--------
:meth:`SenseHistory.recent` returns entries **newest first** — index 0 is
the most recently recorded sense. That is the order a "what just happened"
answer wants to read in: ``recent(1)`` is always "the very last thing", and
a caller that wants the story in chronological order just reverses the
list. (Tested in ``tests/test_harness_sense_history.py``.)

Thread-safety
-------------
:meth:`record` runs on paho's network thread (via ``NovaBus``); :meth:`recent`
runs on whatever thread dispatches the ``recall_senses`` tool call. Both take
the same lock, so the two never race.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

#: Default bound on how many senses are remembered at once — matches the
#: ``recall_senses`` tool's own upper clamp (see ``tools.py``).
DEFAULT_MAXLEN = 20


class SenseHistory:
    """Bounded, thread-safe ring buffer of injected senses.

    Args:
        maxlen: how many most-recent entries to keep; recording past this
            silently drops the oldest (a :class:`collections.deque` with
            ``maxlen`` set does this for us).
        clock: zero-arg monotonic-seconds source, injectable for tests.
            ``None`` uses :func:`time.monotonic`.
    """

    def __init__(
        self, maxlen: int = DEFAULT_MAXLEN, clock: Callable[[], float] | None = None
    ) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._entries: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def record(
        self,
        source: str,
        event_type: str,
        rule: str | None,
        text: str,
        sense_class: str | None,
        voice: str | None,
    ) -> None:
        """Append one entry. Called for every inject that reaches ``on_inject``."""
        entry = {
            "t": self._clock(),
            "source": source,
            "type": event_type,
            "rule": rule,
            "text": text,
            "sense_class": sense_class,
            "voice": voice,
        }
        with self._lock:
            self._entries.append(entry)

    def recent(self, n: int = 5) -> list[dict[str, Any]]:
        """The *n* most recently recorded entries, newest first.

        Each returned entry is a copy carrying an ``age_s`` field computed
        AT READ TIME (``now - t``), so two calls seconds apart report a
        growing age for the same underlying entry.
        """
        now = self._clock()
        with self._lock:
            snapshot = list(self._entries)
        newest_first = list(reversed(snapshot))
        out: list[dict[str, Any]] = []
        for entry in newest_first[: max(0, n)]:
            copy = dict(entry)
            copy["age_s"] = now - copy["t"]
            out.append(copy)
        return out
