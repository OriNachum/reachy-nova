"""The standing Kiro ACP session unit (task t4) — one warm session, kept alive.

``reachy_nova.kiro_acp.KiroAcpSession`` drives a single ``kiro-cli acp``
turn-taking conversation over a subprocess, but every session it opens costs a
cold start (spawn, ``initialize``, ``session/new``) before the first prompt can
land. Skill-forge and future self-extension work want to hand kiro-cli
prompts one at a time without paying that cost per call, so this module keeps
exactly ONE :class:`~reachy_nova.kiro_acp.KiroAcpSession` warm for the whole
harness lifetime and re-derives it only when it must:

* **dies** — the child process exited (crash, OOM-kill, a stray ``kill``); or
* **hangs** — a prompt has been in flight past a stuck-prompt deadline, the
  backstop for a process that is alive but wedged; or
* **grows too much history** — kiro-cli's own conversation state accumulates
  with every turn, and nothing here assumes a native ACP compaction request
  exists yet, so past ``KIRO_HISTORY_MAX`` prompts the whole session is
  recycled (closed and respawned) rather than compacted in place. See
  :meth:`KiroSessionUnit._compact_history` for the seam a native compaction
  call would slot into.

Restart follows the same shape as every other watchdog in this codebase
(``sleep_orchestrator``'s SDK-call retries, the tee's reconnect backoff in
``hearing.py``): capped exponential backoff, reset after a healthy period, so
a wedged kiro-cli binary is retried with increasing patience rather than
hammered in a tight loop.

Threading model
----------------
:meth:`KiroSessionUnit.prompt` is the only caller-facing entry point; a
``threading.Lock`` (``_call_lock``) serializes concurrent callers so kiro-cli
— a single conversational turn-taker — never sees two prompts in flight at
once, and so a threshold-triggered recycle happens atomically between two
prompts rather than racing a third caller. A SEPARATE, short-held lock
(``_session_lock``) protects only the ``_session`` pointer itself: the
liveness monitor (a daemon thread, ``_monitor_loop``) swaps that pointer on a
dead/stuck session WITHOUT waiting on ``_call_lock`` — a wedged ``prompt()``
call may be sitting inside ``_call_lock`` for the whole stuck-prompt deadline,
and the watchdog must still be able to move the unit onto a fresh session for
the NEXT caller rather than deadlock behind the wedged one.

Never imports ``reachy_mini`` and never references ``set_target`` — enforced
by ``tests/test_harness_boundary.py`` over the whole ``reachy_nova/harness/``
package.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from reachy_nova import sensory_log
from reachy_nova.kiro_acp import KiroAcpError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Senselog identity                                                           #
# --------------------------------------------------------------------------- #

#: ``[SENSE stage=kiro source=nova event=...]`` — every line this module emits.
STAGE = "kiro"
SOURCE = "nova"


def _sense(event: str, detail: str) -> None:
    sensory_log.stage(STAGE, SOURCE, event, detail)


# --------------------------------------------------------------------------- #
# Env-configured defaults                                                     #
# --------------------------------------------------------------------------- #

#: How often the watchdog re-checks liveness/stuck-ness.
MONITOR_INTERVAL_ENV = "KIRO_MONITOR_INTERVAL_S"
DEFAULT_MONITOR_INTERVAL_S = 5.0

#: Prompts served before the session is recycled (closed + respawned). This is
#: the committed history-compaction path — see :meth:`KiroSessionUnit._compact_history`.
HISTORY_MAX_ENV = "KIRO_HISTORY_MAX"
DEFAULT_HISTORY_MAX = 50

#: Capped exponential backoff for session restarts: 1s, 2s, 4s, ... capped at 60s.
DEFAULT_BACKOFF_INITIAL_S = 1.0
DEFAULT_BACKOFF_MAX_S = 60.0

#: How long the session must stay healthy (no restart) before backoff resets
#: back down to ``DEFAULT_BACKOFF_INITIAL_S``.
DEFAULT_BACKOFF_RESET_AFTER_S = 60.0

#: A prompt in flight longer than this is considered wedged, independent of
#: whatever timeout the caller passed to ``prompt()`` — a backstop for a
#: process that is alive but not answering. Set comfortably above
#: ``kiro_acp.DEFAULT_PROMPT_TIMEOUT`` (300s) so the session's OWN timeout
#: (raised inside ``KiroAcpSession.prompt``) is expected to fire first in the
#: ordinary case; this only catches what that doesn't.
DEFAULT_PROMPT_STUCK_DEADLINE_S = 330.0

#: Granularity of the interruptible monitor-loop sleep.
_WAIT_SLICE_S = 0.05

#: Factory building one warm session: takes no arguments, returns an object
#: with the ``KiroAcpSession`` surface this unit relies on (``start()``,
#: ``initialize()``, ``new_session(cwd)``, ``prompt(text, timeout=...)``,
#: ``close()``, an ``alive`` property or an ``is_alive()`` method). Injectable
#: so tests never spawn a real kiro-cli subprocess.
SessionFactory = Callable[[], Any]


class KiroSessionUnit:
    """Keeps one warm :class:`KiroAcpSession` alive for the harness lifetime.

    Mirrors the harness's component protocol (``hearing.TeeHearing``,
    ``speaking.SonicSpeaker``): ``start(stop_event)`` / ``stop()`` /
    ``is_alive()``, plus a ``name`` attribute :func:`supervisor._component_name`
    can pick up.

    Args:
        session_factory: zero-argument callable returning a fresh, unstarted
            session object (real production code passes something that builds
            a :class:`~reachy_nova.kiro_acp.KiroAcpSession`; tests pass a fake).
        cwd: the directory handed to ``new_session`` for every spawned/respawned
            session.
        monitor_interval: seconds between watchdog ticks. Defaults to
            ``KIRO_MONITOR_INTERVAL_S`` env or :data:`DEFAULT_MONITOR_INTERVAL_S`.
        history_max: prompts served before a recycle. Defaults to
            ``KIRO_HISTORY_MAX`` env or :data:`DEFAULT_HISTORY_MAX`.
        backoff_initial_s / backoff_max_s / backoff_reset_after_s: restart
            backoff tuning (test seams — production code leaves these at their
            defaults).
        prompt_stuck_deadline_s: see :data:`DEFAULT_PROMPT_STUCK_DEADLINE_S`.
        env: mapping consulted instead of ``os.environ`` (test convenience).
    """

    #: Picked up by ``supervisor._component_name`` when composed as a component.
    name = "kiro_session"

    def __init__(
        self,
        session_factory: SessionFactory,
        cwd: str,
        *,
        monitor_interval: float | None = None,
        history_max: int | None = None,
        backoff_initial_s: float = DEFAULT_BACKOFF_INITIAL_S,
        backoff_max_s: float = DEFAULT_BACKOFF_MAX_S,
        backoff_reset_after_s: float = DEFAULT_BACKOFF_RESET_AFTER_S,
        prompt_stuck_deadline_s: float = DEFAULT_PROMPT_STUCK_DEADLINE_S,
        env: Mapping[str, str] | None = None,
    ) -> None:
        source: Mapping[str, str] = os.environ if env is None else env

        self._session_factory = session_factory
        self._cwd = cwd

        self._monitor_interval = (
            float(monitor_interval)
            if monitor_interval is not None
            else float(source.get(MONITOR_INTERVAL_ENV, DEFAULT_MONITOR_INTERVAL_S))
        )
        self._history_max = (
            int(history_max)
            if history_max is not None
            else int(source.get(HISTORY_MAX_ENV, DEFAULT_HISTORY_MAX))
        )

        self._backoff_initial = float(backoff_initial_s)
        self._backoff_max = float(backoff_max_s)
        self._backoff_reset_after = float(backoff_reset_after_s)
        self._prompt_stuck_deadline = float(prompt_stuck_deadline_s)

        # Serializes prompt() callers against each other AND against a
        # threshold-triggered recycle (held for the whole call+maybe-recycle).
        self._call_lock = threading.Lock()
        # Guards only the _session pointer swap — never held across I/O, so
        # the watchdog can always move the unit off a wedged session.
        self._session_lock = threading.Lock()
        # Guards the small counters/status fields below.
        self._status_lock = threading.Lock()

        self._session: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._external_stop: threading.Event | None = None

        self._backoff = self._backoff_initial
        self._healthy_since: float | None = None
        self._restarts = 0
        self._prompts_served = 0
        self._recycles = 0
        #: monotonic start time of the in-flight prompt, or None when idle.
        self._prompt_started_at: float | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        """Spawn the first session and start the watchdog thread.

        Blocking: the initial spawn (factory + start + initialize + new_session)
        happens synchronously here, so a caller sees a ready-or-failed unit,
        never a unit that silently never came up.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._external_stop = stop_event
        self._stop.clear()

        session = self._spawn_session()
        with self._session_lock:
            self._session = session

        self._thread = threading.Thread(
            target=self._monitor_loop, name="kiro-session-monitor", daemon=True
        )
        self._thread.start()
        _sense("start", "kiro session unit started (one warm session, watchdog armed)")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the watchdog, join it, and close the current session.

        Best-effort and idempotent, like every other component's ``stop()`` in
        this package — shutdown reports, it never raises.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        with self._session_lock:
            session = self._session
            self._session = None
        if session is not None:
            try:
                session.close()
            except Exception as err:  # noqa: BLE001 - shutdown must not raise
                logger.debug("kiro_session: close() on stop raised: %s", err)
        _sense("stop", "kiro session unit stopped")

    def is_alive(self) -> bool:
        with self._session_lock:
            session = self._session
        return session is not None and self._session_alive(session)

    # -- observability ---------------------------------------------------------

    def status(self) -> dict[str, object]:
        """A small status dict for the ``/api`` dashboard and tests."""
        with self._session_lock:
            session = self._session
        alive = session is not None and self._session_alive(session)
        with self._status_lock:
            return {
                "alive": alive,
                "restarts": self._restarts,
                "prompts_served": self._prompts_served,
                "recycles": self._recycles,
                "history_max": self._history_max,
            }

    # -- the caller-facing entry point -----------------------------------------

    def prompt(self, text: str, timeout: float | None = None) -> str:
        """Send one conversational turn on the current session.

        Serialized against other callers (``_call_lock``) so kiro-cli, a
        single-conversation turn-taker, never sees two prompts in flight.
        Every served prompt counts toward the recycle threshold; the recycle
        itself (when triggered) happens here, before returning, which is the
        "next idle moment" the design calls for — no other caller can be mid
        ``session.prompt()`` while this holds ``_call_lock``.

        Raises :class:`KiroAcpError` if the unit has not been started.
        """
        with self._call_lock:
            with self._session_lock:
                session = self._session
            if session is None:
                raise KiroAcpError("KiroSessionUnit.prompt() called before start()")

            with self._status_lock:
                self._prompt_started_at = time.monotonic()
            try:
                if timeout is not None:
                    result = session.prompt(text, timeout=timeout)
                else:
                    result = session.prompt(text)
            finally:
                with self._status_lock:
                    self._prompt_started_at = None

            recycle_needed = False
            with self._status_lock:
                self._prompts_served += 1
                if self._prompts_served >= self._history_max:
                    recycle_needed = True
            if recycle_needed:
                self._compact_history()
            return result

    # -- session spawn / recycle -----------------------------------------------

    def _spawn_session(self) -> Any:
        """Build, start, initialize and open a session.

        The single choke point for BOTH:

        * **no leaked subprocesses** — if the handshake fails partway
          (``start()`` succeeded but ``initialize()`` or ``new_session()``
          raised, or ``start()`` itself raised after doing partial work), the
          partially-started session is closed before the exception
          propagates. ``close()`` itself raising is swallowed (best-effort,
          logged at DEBUG) so it never masks the original failure.
        * **a watchdog that survives any spawn failure** — every exception
          raised anywhere in this method (factory call, ``start()``,
          ``initialize()``, ``new_session()``) is normalized to
          :class:`KiroAcpError`, chained via ``from err`` so the original
          traceback/type is still visible. Production spawn failures are not
          limited to :class:`KiroAcpError` — e.g. ``kiro-cli`` missing from
          PATH raises ``FileNotFoundError`` out of ``subprocess.Popen`` (this
          fired live under systemd) — and callers here
          (:meth:`_restart_with_backoff`, :meth:`_compact_history`) only
          catch :class:`KiroAcpError`. Normalizing here, rather than
          broadening those catches, keeps the watchdog/recycle code itself
          unchanged. A :class:`KiroAcpError` raised directly by the session is
          re-raised as-is (same type/message, no double-wrapping).
        """
        try:
            session = self._session_factory()
        except KiroAcpError:
            raise
        except Exception as err:  # noqa: BLE001 - normalize to KiroAcpError
            raise KiroAcpError(str(err)) from err

        try:
            session.start()
            session.initialize()
            session.new_session(self._cwd)
        except Exception as err:
            try:
                session.close()
            except Exception as close_err:  # noqa: BLE001 - best-effort cleanup
                logger.debug(
                    "kiro_session: close() on failed spawn raised: %s", close_err
                )
            if isinstance(err, KiroAcpError):
                raise
            raise KiroAcpError(str(err)) from err

        return session

    def _compact_history(self) -> None:
        """History-compaction path: currently a full recycle (close + respawn).

        SEAM: kiro-cli's ACP surface is not assumed to expose a native
        compaction request today. If/when one exists (e.g. a hypothetical
        ``session/compact`` method), swap the body of this method for that
        call instead of close+respawn — :meth:`prompt` does not need to
        change, it only calls this method at the threshold.

        Called from inside :meth:`prompt`, which already holds ``_call_lock``,
        so this runs atomically with respect to other prompt() callers — the
        "next idle moment" the design calls for.
        """
        with self._session_lock:
            old_session = self._session
        _sense(
            "recycle",
            f"recycling session at prompts_served={self._history_max} threshold "
            "(close+respawn — no native kiro compaction method assumed)",
        )
        try:
            new_session = self._spawn_session()
        except KiroAcpError as err:
            # Keep serving the existing session rather than lose it; the next
            # prompt will retry the recycle at the same threshold.
            _sense("recycle", f"recycle failed, keeping existing session: {err}")
            return

        with self._session_lock:
            self._session = new_session
        with self._status_lock:
            self._recycles += 1
            self._prompts_served = 0

        if old_session is not None:
            try:
                old_session.close()
            except Exception as err:  # noqa: BLE001 - best-effort during recycle
                logger.debug("kiro_session: close() on recycle raised: %s", err)

    # -- the watchdog ------------------------------------------------------------

    def _monitor_loop(self) -> None:
        while not self._should_stop():
            self._watchdog_tick()
            self._wait(self._monitor_interval)

    def _watchdog_tick(self) -> None:
        """One liveness/stuck-prompt check, restarting on failure. Testable directly."""
        with self._session_lock:
            session = self._session
        if session is None:
            return

        dead = not self._session_alive(session)
        stuck = self._is_prompt_stuck()
        if dead or stuck:
            reason = "dead-process" if dead else "stuck-prompt"
            with self._status_lock:
                self._healthy_since = None
            _sense("watchdog", f"session unhealthy reason={reason}; restarting")
            self._restart_with_backoff()
            return

        with self._status_lock:
            now = time.monotonic()
            if self._healthy_since is None:
                self._healthy_since = now
            elif now - self._healthy_since >= self._backoff_reset_after:
                if self._backoff != self._backoff_initial:
                    _sense("watchdog", "healthy period elapsed; backoff reset")
                self._backoff = self._backoff_initial

    def _restart_with_backoff(self) -> None:
        """Wait out the current backoff, then respawn — capped, growing.

        Does NOT take ``_call_lock``: a wedged ``prompt()`` call may be
        holding it for the whole stuck-prompt deadline, and the watchdog must
        still be able to move the unit onto a fresh session so the NEXT
        caller is not stuck behind the wedged one forever.
        """
        with self._status_lock:
            self._restarts += 1
            delay = self._backoff
            self._backoff = min(self._backoff * 2.0, self._backoff_max)

        _sense("restart", f"restart #{self._restarts} in {delay:.2f}s (capped exponential backoff)")
        self._wait(delay)
        if self._should_stop():
            return

        try:
            new_session = self._spawn_session()
        except KiroAcpError as err:
            # Backoff already grew; the next watchdog tick tries again.
            _sense("restart", f"restart failed: {err}")
            return

        with self._session_lock:
            old_session = self._session
            self._session = new_session
        with self._status_lock:
            self._prompt_started_at = None

        if old_session is not None:
            try:
                old_session.close()
            except Exception as err:  # noqa: BLE001 - best-effort during restart
                logger.debug("kiro_session: close() on restart raised: %s", err)

    def _is_prompt_stuck(self) -> bool:
        with self._status_lock:
            started = self._prompt_started_at
        if started is None:
            return False
        return (time.monotonic() - started) > self._prompt_stuck_deadline

    @staticmethod
    def _session_alive(session: Any) -> bool:
        """Best-effort liveness read supporting both a property and a method."""
        alive_attr = getattr(session, "alive", None)
        if callable(alive_attr):
            return bool(alive_attr())
        if alive_attr is not None:
            return bool(alive_attr)
        is_alive = getattr(session, "is_alive", None)
        if callable(is_alive):
            return bool(is_alive())
        # Unknown surface: assume alive rather than restart-loop on a session
        # that never told us otherwise.
        return True

    # -- plumbing ----------------------------------------------------------------

    def _should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        return self._external_stop is not None and self._external_stop.is_set()

    def _wait(self, seconds: float) -> None:
        """Sleep, interruptibly: either stop signal cuts it short."""
        deadline = time.monotonic() + seconds
        while not self._should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            self._stop.wait(min(remaining, _WAIT_SLICE_S))
