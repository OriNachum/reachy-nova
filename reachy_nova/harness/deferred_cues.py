"""Body cues parked while Sonic is generating, delivered with their age (t9).

Today a cue that arrives while Nova Sonic is mid-utterance is **lost, not
late**: ``NovaSonic.inject_text``'s speaking guard logs
``dropped reason=speaking`` and returns, because injecting into a Bedrock
bidirectional stream that is actively generating audio can hang it. A pat
during a ten-second reply therefore never reaches the mind at all.

This module is the small piece of state that turns that drop into a
deferral. It holds **one latest-wins slot per sense class** — the ``sense:``
field of the ``config/nervous-system/rules.yaml`` entry that produced the
cue (``pat``, ``face``, ``sound``, ``vision``; anything unnamed lands in
:data:`OTHER_CLASS`) — so a burst of pats during one reply becomes a single
"you were petted", while a pat and a face are two independent facts that
both survive.

Latest-wins, not a queue
------------------------
A queue would deliver a stale reaction to every cue that arrived during a
long reply — five pats become five "oh, a pat" lines a full utterance later.
The slot keeps only the most recent cue of each class, and replacing a cue
also **refreshes its age**, because the newest pat is the one being reacted
to. Arrival *order* is preserved by first arrival: a replaced pat keeps the
position its first pat claimed, so the drain reads in the order the senses
actually started happening.

Why the age is in the text (spec c32)
-------------------------------------
The slot is drained when Sonic stops **generating**, which — with chunked
playback — is before the human has finished **hearing** the reply. A
deferred reaction is therefore already a beat late by the time it is
audible, and a reaction that reads as if it were happening now would be
plain wrong. :meth:`DeferredCues.render` puts the delay in the text
("just now" under :data:`JUST_NOW_S`, "<n> seconds ago" beyond it) so the
mind can phrase the reaction as the late thing it is.

TTL
---
A cue that waited longer than ``ttl_s`` is no longer worth saying — the
moment has passed. It is dropped on drain with one named senselog line
(``dropped reason=deferred-expired age=...``) rather than silently, so the
journal can always answer "where did that pat go?".

Thread-safety
-------------
:meth:`put` runs on whatever thread called ``inject_text`` (paho's network
thread, the vision leg, a tool executor); :meth:`drain` runs on the Sonic
asyncio thread. Every mutation takes the same lock, so the two never race,
and a cue is never delivered twice: :meth:`drain` empties the slot inside
the lock and works on its own copy afterwards.

stdlib only — no numpy, no boto3, no MQTT. It knows nothing about Sonic.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from ..sensory_log import stage as sensory_stage

#: Senselog stage/source every line from this module carries — the same pair
#: ``inject_text``'s own drop lines use, so one grep covers the whole fate of
#: a cue (deferred, drained, expired) in the journal.
STAGE = "inject"
SOURCE = "speech"

#: How long a parked cue stays worth saying.
DEFAULT_TTL_S = 5.0

#: Sense class for a cue whose rules entry names no ``sense:`` field.
OTHER_CLASS = "other"

#: Under this many seconds the delay is described rather than counted.
JUST_NOW_S = 2.0

#: Cues delivered per speaking->listening transition. There are four sense
#: classes in rules.yaml plus ``other``; the cap keeps a pathological burst
#: from turning one utterance's end into five back-to-back injects.
MAX_DRAIN_PER_TRANSITION = 4

#: Named drop reasons (kept as constants so tests and greps agree).
REASON_EXPIRED = "deferred-expired"
REASON_OVERFLOW = "deferred-overflow"

#: How much of a cue's text rides in a log line — matches ``inject_text``.
_LOG_TEXT_CHARS = 60


@dataclass(frozen=True)
class DeferredCue:
    """One parked cue: what happened, in which sense class, and when."""

    sense_class: str
    text: str
    t: float
    seq: int

    def age(self, now: float) -> float:
        """Seconds since this cue arrived, never negative."""
        return max(0.0, now - self.t)


def normalise_class(sense_class: str | None) -> str:
    """Map a rules ``sense:`` value onto a slot key.

    ``None``, ``""`` and whitespace all mean "this cue names no sense class",
    which is a real case (rules entries without a ``sense:`` field), not an
    error — they share :data:`OTHER_CLASS`.
    """
    if not sense_class:
        return OTHER_CLASS
    cleaned = str(sense_class).strip()
    return cleaned or OTHER_CLASS


class DeferredCues:
    """Thread-safe latest-wins slot of body cues, one per sense class.

    Args:
        ttl_s: how long a parked cue stays deliverable; older cues are
            dropped on :meth:`drain` with a named senselog line.
        clock: zero-arg monotonic-seconds source, injectable for tests.
            ``None`` uses :func:`time.monotonic`.
    """

    def __init__(
        self,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_s = ttl_s
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._slots: dict[str, DeferredCue] = {}
        self._seq = 0
        self._counters = {"deferred": 0, "replaced": 0, "expired": 0, "drained": 0}

    # -- writing -------------------------------------------------------

    def put(self, sense_class: str | None, text: str) -> DeferredCue:
        """Park *text* under *sense_class*, replacing any cue already there.

        Returns the stored cue, so the caller can name the resolved class in
        its own log line without re-deriving it.
        """
        key = normalise_class(sense_class)
        with self._lock:
            self._seq += 1
            previous = self._slots.get(key)
            # A replacement inherits the ORIGINAL arrival position (seq) so
            # drain order stays "the order these senses started happening",
            # but takes a fresh timestamp: the newest pat is the one reacted
            # to, and it should not expire on the first pat's clock.
            cue = DeferredCue(
                sense_class=key,
                text=text,
                t=self._clock(),
                seq=previous.seq if previous is not None else self._seq,
            )
            self._slots[key] = cue
            self._counters["deferred"] += 1
            if previous is not None:
                self._counters["replaced"] += 1
        return cue

    # -- reading -------------------------------------------------------

    def drain(self, now: float | None = None) -> list[DeferredCue]:
        """Empty the slot and return the cues still worth saying, in order.

        Cues older than ``ttl_s`` are dropped here — each with one named
        senselog line — rather than handed back, so the caller only ever
        sees cues it should actually deliver.
        """
        with self._lock:
            when = self._clock() if now is None else now
            cues = sorted(self._slots.values(), key=lambda c: c.seq)
            self._slots.clear()
            survivors = [c for c in cues if c.age(when) <= self.ttl_s]
            expired = [c for c in cues if c.age(when) > self.ttl_s]
            self._counters["expired"] += len(expired)
            self._counters["drained"] += len(survivors)
        # Log outside the lock: sensory_log goes through the logging module,
        # and a handler must never be able to block a put() on another thread.
        for cue in expired:
            self._log_drop(REASON_EXPIRED, cue, cue.age(when))
        return survivors

    def render(self, cue: DeferredCue, now: float | None = None) -> str:
        """The age-aware text a deferred cue is delivered as (spec c32)."""
        when = self._clock() if now is None else now
        age = cue.age(when)
        if age < JUST_NOW_S:
            delay = "just now"
        else:
            # max() keeps rounding from ever producing "1 seconds ago" right
            # next to the "just now" band.
            delay = f"{max(int(JUST_NOW_S), int(round(age)))} seconds ago"
        return f"({delay}, while you were talking: {cue.text})"

    # -- housekeeping --------------------------------------------------

    def clear(self) -> int:
        """Forget every parked cue without delivering it; returns how many.

        Used on session restart: the conversation the cues belonged to is
        gone, so delivering them into a fresh session would be a reaction to
        something the new session never heard.
        """
        with self._lock:
            count = len(self._slots)
            self._slots.clear()
        return count

    def now(self) -> float:
        """This slot's idea of the current time.

        Callers that want to age a drained cue must read the clock through
        here rather than calling :func:`time.monotonic` themselves — the
        clock is injectable, and mixing the two would age a cue against a
        different timeline than the one it was stamped on.
        """
        return self._clock()

    def pending(self) -> int:
        """How many cues are parked right now."""
        with self._lock:
            return len(self._slots)

    def counters(self) -> dict[str, int]:
        """Snapshot of the lifecycle counters (deferred/replaced/expired/drained)."""
        with self._lock:
            return dict(self._counters)

    # -- logging -------------------------------------------------------

    def _log_drop(self, reason: str, cue: DeferredCue, age: float) -> None:
        sensory_stage(
            STAGE,
            SOURCE,
            str(uuid.uuid4()),
            f"dropped reason={reason} age={age:.1f}s class={cue.sense_class} "
            f"text={cue.text[:_LOG_TEXT_CHARS]!r}",
        )

    def log_overflow(self, cue: DeferredCue, age: float) -> None:
        """Name a cue the per-transition cap left undelivered."""
        self._log_drop(REASON_OVERFLOW, cue, age)
