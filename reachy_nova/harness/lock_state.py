"""``LockState`` — the harness's own belief about the runtime's gaze lock (t13).

The runtime does not (yet) mirror lock state into ``<state>/behavior/
state.json`` the way it mirrors ``intents.inhibitions`` (see
:func:`reachy_nova.harness.tools.current_inhibitions`), so this harness keeps
its OWN small, in-process belief instead of inventing a second reader of a
file the runtime does not write. Two things update it:

1. **Our own tool calls.** :class:`~reachy_nova.harness.tools.IntentTools`
   marks the belief right after a ``lock_face``/``release_face`` tool call
   the engine confirmed — never on a refusal or a degraded (unconfirmed)
   result, since neither of those means the body's state actually changed.
2. **The runtime's own bus events.** ``motion/lock-released`` (t13) fires
   whenever the RUNTIME drops a lock on its own — ``reason: requested`` mirrors
   our own ``release_face``, but ``mind-offline``/``max-hold`` are the runtime
   acting without us, and only the bus tells us it happened.

A belief this cheap is worth exactly what it costs to keep current, and no
more: it is read-only local color for :func:`reachy_nova.harness.supervisor.
status` (``locked: bool|None`` — ``None`` when unknown, e.g. from a
fresh-started harness, or a different process reading ``status`` from outside)
and for the one-line drop this module logs when the ENGINE itself restarts
(see :meth:`on_engine_dropped`) — it never gates or blocks a tool call. The
next ``lock_face`` after an engine restart proceeds exactly like any other:
the spool write and the engine's own answer are the only truth that matters.
"""

from __future__ import annotations

import threading

from .. import sensory_log

_STAGE = "supervise"
_SOURCE = "nova"
_EVENT = "lock"

#: The bus event that means the RUNTIME dropped a lock we may believe we hold.
_RELEASE_SOURCE = "motion"
_RELEASE_TYPE = "lock-released"


class LockState:
    """Thread-safe belief about whether Nova's gaze is currently locked."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._locked: bool | None = None

    @property
    def locked(self) -> bool | None:
        """``True``/``False`` if known, ``None`` if unknown (never guessed)."""
        with self._lock:
            return self._locked

    def mark_locked(self) -> None:
        """Record a confirmed ``lock_face`` — call only on the engine's ``ok: true``."""
        with self._lock:
            self._locked = True

    def mark_released(self, reason: str | None = None) -> None:
        """Record a confirmed release — a ``release_face`` the engine confirmed,
        or a ``motion/lock-released`` event the runtime published on its own.
        *reason* (``requested``/``mind-offline``/``max-hold`` from the two call
        sites, or ``None`` when a caller has none to give) is folded into the
        one-line drop this logs — any reason-specific NARRATION already
        happened via the bus's own ``inject_template`` (see rules.yaml's
        ``motion/lock-released`` entry) before this belief update ever runs,
        but this log line is the belief-tracking side, same as
        :meth:`on_engine_dropped`'s.
        """
        with self._lock:
            self._locked = False
        sensory_log.stage(_STAGE, _SOURCE, _EVENT, f"released reason={reason}")

    # -- the bus hook --------------------------------------------------------

    def on_bus_event(self, event: dict) -> None:
        """Wire this as (part of) :class:`~reachy_nova.harness.bus.NovaBus`'s
        ``on_event`` tap — called with EVERY decoded bus payload, so this
        filters to the one event that means anything for the lock belief.
        """
        if not isinstance(event, dict):
            return
        if event.get("source") == _RELEASE_SOURCE and event.get("type") == _RELEASE_TYPE:
            self.mark_released(event.get("reason"))

    # -- the supervisor hook --------------------------------------------------

    def on_engine_dropped(self) -> None:
        """The engine heartbeat just transitioned live -> dropped.

        A lock is a promise the ENGINE keeps; when the engine itself goes
        away, whatever it was doing is gone with it, and a harness that keeps
        believing "locked" after the process it delegated to disappeared would
        be reporting on a robot that no longer exists. Logs exactly ONE named
        line and clears the belief to unknown — but only when we actually
        believed we were locked, so a restart while nothing was locked (the
        overwhelmingly common case) stays silent rather than logging on every
        single engine hiccup.
        """
        with self._lock:
            was_locked = self._locked
            self._locked = None
        if was_locked:
            sensory_log.stage(_STAGE, _SOURCE, _EVENT, "released reason=engine-restart")
