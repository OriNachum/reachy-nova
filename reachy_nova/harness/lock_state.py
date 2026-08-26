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

import math
import os
import threading
import time
from collections.abc import Callable

from .. import sensory_log

_STAGE = "supervise"
_SOURCE = "nova"
_EVENT = "lock"

#: The bus event that means the RUNTIME dropped a lock we may believe we hold.
_RELEASE_SOURCE = "motion"
_RELEASE_TYPE = "lock-released"

#: How long the engine heartbeat must STAY down before a believed lock is
#: dropped. LIVE FINDING L6 (2026-08-26): on a loaded CM4 the heartbeat flaps
#: live/lost about every 2 s, so an edge-triggered clear threw the belief away
#: two seconds after every lock while the RUNTIME lock itself was perfectly
#: fine. The flapping is a pre-existing load problem on the device; what this
#: grace fixes is the harness believing the flap.
DEFAULT_DROP_GRACE_S = 5.0

#: Env override for :data:`DEFAULT_DROP_GRACE_S`, in seconds.
DROP_GRACE_ENV = "NOVA_LOCK_DROP_GRACE_S"


def default_drop_grace_s() -> float:
    """The engine-drop grace, from ``NOVA_LOCK_DROP_GRACE_S`` or the default.

    Parsed defensively: unset, empty, unparseable, NaN or negative resolves to
    :data:`DEFAULT_DROP_GRACE_S`. ``0`` is honoured — it is the old,
    edge-triggered behaviour, and someone asking for it explicitly means it.
    """
    raw = os.environ.get(DROP_GRACE_ENV)
    if raw is None:
        return DEFAULT_DROP_GRACE_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_DROP_GRACE_S
    if value < 0.0 or math.isnan(value):  # negative, or NaN
        return DEFAULT_DROP_GRACE_S
    return value


class LockState:
    """Thread-safe belief about whether Nova's gaze is currently locked.

    Parameters
    ----------
    clock:
        Monotonic seconds source, injectable for tests.
    drop_grace_s:
        How long the engine must stay down before a believed lock is dropped;
        ``None`` resolves :func:`default_drop_grace_s`.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        drop_grace_s: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._locked: bool | None = None
        self._clock = clock
        self.drop_grace_s = (
            default_drop_grace_s() if drop_grace_s is None else float(drop_grace_s)
        )
        #: When the engine went down while we believed we were locked, or
        #: ``None`` when no drop is pending. See :meth:`on_engine_dropped`.
        self._drop_pending_since: float | None = None

    @property
    def locked(self) -> bool | None:
        """``True``/``False`` if known, ``None`` if unknown (never guessed).

        Settles a pending engine drop first, so a caller that only ever reads
        the belief still sees it expire on time.
        """
        self.settle()
        with self._lock:
            return self._locked

    def mark_locked(self) -> None:
        """Record a confirmed ``lock_face`` — call only on the engine's ``ok: true``."""
        with self._lock:
            self._locked = True
            self._drop_pending_since = None

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
            self._drop_pending_since = None
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
        be reporting on a robot that no longer exists.

        But a heartbeat that FLICKERS is not an engine that went away. Live on
        2026-08-26 the heartbeat on a loaded CM4 flapped live/lost roughly
        every 2 s while the runtime lock was held perfectly well, and an
        edge-triggered clear threw the belief away seconds after every lock
        (finding L6). So this only ARMS the drop: the belief survives until
        the engine has stayed down for :attr:`drop_grace_s`, and any
        :meth:`on_engine_live` inside that window cancels it. Nothing is
        logged here — the drop line comes with the actual clear, in
        :meth:`settle`, so exactly one line still names it.

        Arming at all only happens when we actually believed we were locked,
        so a restart while nothing was locked (the overwhelmingly common case)
        stays silent rather than logging on every single engine hiccup.
        """
        with self._lock:
            if self._locked and self._drop_pending_since is None:
                self._drop_pending_since = self._clock()
        self.settle()

    def on_engine_live(self) -> None:
        """The engine heartbeat came back — cancel any pending belief drop.

        Wired from the supervisor's own transition logging
        (:func:`reachy_nova.harness.supervisor._log_engine_transition`), which
        is the one place that already sees both edges.
        """
        with self._lock:
            self._drop_pending_since = None

    def settle(self) -> bool:
        """Clear a believed lock once the engine has been down past the grace.

        Idempotent and cheap: safe to call from the supervisor's poll loop on
        every tick, and from every :attr:`locked` read. Returns whether this
        call was the one that dropped the belief (so the caller can tell a
        no-op tick from the real thing); logs the ONE named line when it was.
        """
        with self._lock:
            since = self._drop_pending_since
            if since is None or (self._clock() - since) < self.drop_grace_s:
                return False
            self._drop_pending_since = None
            was_locked = self._locked
            self._locked = None
        if not was_locked:
            return False
        sensory_log.stage(
            _STAGE, _SOURCE, _EVENT, "released reason=engine-restart"
        )
        return True
