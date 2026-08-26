"""Timed quiet — a deadline for the robot's mouth, not a mode.

"Be quiet for ten minutes" is a *deadline*, and the difference matters. A mode
is a flag someone has to remember to clear; a deadline expires on its own, so
the failure mode of a forgotten quiet is a robot that starts talking again ten
minutes later rather than one that never speaks again. Everything here follows
from that:

* **Later always wins.** A second request while quiet is armed can only push
  the deadline OUT (``note="extended"``), never pull it in: asking for five
  more minutes in the middle of a thirty-minute quiet obviously means "at least
  five more", not "cut it to five" (``note="kept"``). A repeat of the SAME
  request — the model calling ``stay_silent(2)`` twice inside one second, seen
  live 2026-08-26 18:28:34 — lands on a deadline within
  :data:`SAME_DEADLINE_EPS_S` of the standing one and is ``note="kept"`` too:
  nothing about the quiet changed, so calling it an "extension" tells the
  model a story that did not happen (finding L2). Only :meth:`release` ends a
  quiet early, and it says whether there was anything to end.
* **It survives a restart.** The deadline is persisted atomically (tmp +
  ``os.replace``) to ``<state>/nova-quiet.json`` on every arm and release, and
  reloaded on construction. A harness restart at 02:05 inside a quiet armed at
  02:00 must not loudly reintroduce itself; a deadline already in the past is
  ignored and the stale file removed, so quiet can never outlive its own clock.
  The file also carries ``body`` — whether WE are the ones holding the
  runtime's own mouth shut (:meth:`set_body_latches`) — because a restored
  deadline with forgotten ownership is a mute nobody will ever lift. Files
  written before ``body`` existed still load, owning nothing.
* **The acknowledgement is heard.** :meth:`arm` leaves
  :attr:`pending_first_utterance` set, so the FIRST utterance after arming is
  spoken ("okay, quiet for ten minutes") and the mouth closes behind it. If no
  utterance arrives within ``grace_s`` the mouth closes anyway — a Sonic turn
  that never produces audio must not leave the gate open.

  That grace is a bet on how long Sonic needs to turn a tool result into
  audio, and the first bet was wrong: live on 2026-08-26 the ``stay_silent``
  result landed at 18:28:34 and Sonic's acknowledgement arrived 9 s later, so
  a 2 s grace dropped the one utterance the whole feature exists to let
  through (finding L1). The default is now :data:`DEFAULT_GRACE_S` = 15 s,
  overridable with ``NOVA_QUIET_ACK_GRACE_S`` and parsed defensively (a
  missing, unparseable or non-positive value falls back to the default rather
  than closing the mouth instantly). The reservation is spent by an
  UTTERANCE, never by time alone inside the window: no other utterance can
  slip through first, because the first :meth:`allow_utterance` after
  :meth:`arm` IS the acknowledgement.

Quiet gates the SPEAKER only (see :mod:`reachy_nova.harness.speaking`). The ear
keeps hearing, the mind keeps thinking, sensory events keep flowing: the robot
is quiet, not asleep and not deaf.

The clock is wall time (``time.time``) because the deadline is persisted across
process restarts, and injectable so tests never sleep.

stdlib only; never imports ``reachy_mini``
(``tests/test_harness_boundary.py``).
"""

from __future__ import annotations

import math
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from reachy_nova import sensory_log
from reachy_nova.harness import statedir

STAGE_SUPERVISE = "supervise"
SOURCE = "nova"
EVENT_QUIET = "quiet"

#: How long after arming we still wait for the acknowledgement utterance.
#: 15 s, not the original 2 s: Sonic took 9 s to produce the acknowledgement
#: under inject load on the live robot (2026-08-26, finding L1).
DEFAULT_GRACE_S = 15.0

#: Env override for :data:`DEFAULT_GRACE_S`, in seconds.
GRACE_ENV = "NOVA_QUIET_ACK_GRACE_S"

#: Two deadlines this close together are the SAME deadline (finding L2) — a
#: repeated identical request, not an extension.
SAME_DEADLINE_EPS_S = 1.0


def default_grace_s() -> float:
    """The acknowledgement grace, from ``NOVA_QUIET_ACK_GRACE_S`` or the default.

    Parsed defensively: an unset, empty, unparseable, NaN or non-positive
    value resolves to :data:`DEFAULT_GRACE_S`. A typo in an env var must never
    be the reason the robot swallowed the one thing it was allowed to say.
    """
    raw = os.environ.get(GRACE_ENV)
    if raw is None:
        return DEFAULT_GRACE_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_GRACE_S
    if value <= 0.0 or math.isnan(value):  # non-positive, or NaN
        return DEFAULT_GRACE_S
    return value


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    """tmp + ``os.replace`` — a torn quiet file is a robot with no deadline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def peek_until_iso(path: Path | None = None, now: float | None = None) -> str | None:
    """Read a persisted, still-future quiet deadline as an ISO string.

    Side-effect free (it never removes the file) so the out-of-process
    ``status`` CLI can report another process's quiet without disturbing it.
    Returns ``None`` for absent, unreadable, or already-expired state.
    """
    target = statedir.quiet_state_path() if path is None else path
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        until = float(payload["until"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if until <= (time.time() if now is None else now):
        return None
    return _iso(until)


class QuietState:
    """A persisted, self-expiring quiet deadline. Thread-safe.

    Parameters
    ----------
    clock:
        Wall-clock source (epoch seconds); injectable for tests.
    path:
        Persistence target; defaults to
        :func:`reachy_nova.harness.statedir.quiet_state_path`.
    grace_s:
        How long the acknowledgement window stays open after :meth:`arm`.
        ``None`` (the default) resolves :func:`default_grace_s` at
        construction, so the env override applies without every caller
        having to know about it.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        path: Path | None = None,
        grace_s: float | None = None,
    ) -> None:
        self._clock = clock
        self._path = statedir.quiet_state_path() if path is None else Path(path)
        self.grace_s = default_grace_s() if grace_s is None else float(grace_s)
        self._lock = threading.RLock()
        self._until = 0.0
        #: True between arm() and the acknowledgement utterance (or the grace).
        self.pending_first_utterance = False
        self._pending_deadline = 0.0
        # Who closed the BODY's mouth (the runtime's own voice) for this quiet.
        # Persisted alongside the deadline: a harness restart that restored the
        # deadline but forgot the ownership would either never lift the mute
        # (nobody left who thinks it is theirs) or let the body talk through a
        # quiet the person is still inside. See IntentTools._mute_body.
        self._body = {"added_speak": False, "muted_voice": False}
        self.load()

    # -- persistence --------------------------------------------------------

    def load(self) -> None:
        """Restore a still-future deadline; drop an expired or corrupt file.

        Tolerant of the pre-ownership file shape (``{"until": ...}`` with no
        ``body``): an older file simply comes back owning nothing.
        """
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            until = float(payload["until"])
        except FileNotFoundError:
            return
        except (OSError, ValueError, KeyError, TypeError):
            _safe_unlink(self._path)
            return
        if until <= self._clock():
            # Quiet must never outlive its own clock.
            _safe_unlink(self._path)
            return
        body = payload.get("body")
        with self._lock:
            self._until = until
            if isinstance(body, dict):
                self._body = {
                    "added_speak": bool(body.get("added_speak")),
                    "muted_voice": bool(body.get("muted_voice")),
                }

    def _persist(self) -> None:
        _atomic_write(
            self._path, json.dumps({"until": self._until, "body": dict(self._body)})
        )

    # -- the body's mouth ---------------------------------------------------

    def body_latches(self) -> tuple[bool, bool]:
        """``(added_speak, muted_voice)`` — what WE are holding on the runtime."""
        with self._lock:
            return (self._body["added_speak"], self._body["muted_voice"])

    def set_body_latches(self, added_speak: bool, muted_voice: bool) -> None:
        """Record the body-mute ownership, persisting it with the deadline.

        Only written to disk while a deadline is armed — the file IS the quiet,
        so ownership with no deadline behind it has nothing to survive into.
        """
        with self._lock:
            self._body = {
                "added_speak": bool(added_speak),
                "muted_voice": bool(muted_voice),
            }
            if self._until > 0.0:
                self._persist()

    # -- the deadline -------------------------------------------------------

    @property
    def until(self) -> float | None:
        """The armed deadline (epoch seconds), or ``None`` when not armed."""
        with self._lock:
            return self._until if self._until > 0.0 else None

    def until_iso(self) -> str | None:
        until = self.until
        return None if until is None else _iso(until)

    def arm(self, minutes: float) -> dict[str, object]:
        """Be quiet for *minutes*. Later wins; returns the deadline and a note."""
        with self._lock:
            now = self._clock()
            candidate = now + float(minutes) * 60.0
            was_armed = self._until > now
            if not was_armed:
                note = "armed"
                self._until = candidate
            elif candidate > self._until + SAME_DEADLINE_EPS_S:
                note = "extended"
                self._until = candidate
            else:
                # Either a SHORTER request (later wins, keep ours) or the same
                # request repeated within SAME_DEADLINE_EPS_S (finding L2):
                # the quiet did not actually change, so neither does the note.
                note = "kept"
                self._until = max(self._until, candidate)
            # Even a "kept"/"extended" request is an exchange the robot should
            # be able to answer out loud before the mouth closes again.
            self.pending_first_utterance = True
            self._pending_deadline = now + self.grace_s
            until = self._until
            self._persist()
        if note != "kept":
            sensory_log.stage(
                STAGE_SUPERVISE, SOURCE, EVENT_QUIET, f"{note} until={_iso(until)}"
            )
        return {"until": until, "note": note}

    def _clear(self) -> bool:
        """Drop the deadline and its file. Returns whether it was still future."""
        with self._lock:
            was_armed = self._until > self._clock()
            self._until = 0.0
            self.pending_first_utterance = False
            self._pending_deadline = 0.0
            self._body = {"added_speak": False, "muted_voice": False}
            _safe_unlink(self._path)
        return was_armed

    def release(self, reason: str = "ended") -> dict[str, bool]:
        """End the quiet now. ``was_armed`` False when there was nothing to end."""
        was_armed = self._clear()
        if was_armed:
            sensory_log.stage(
                STAGE_SUPERVISE, SOURCE, EVENT_QUIET, f"released reason={reason}"
            )
        return {"was_armed": was_armed}

    def active(self) -> bool:
        """Is the mouth closed right now? Expiry releases itself, once, named."""
        with self._lock:
            if self._until <= 0.0:
                return False
            if self._clock() < self._until:
                return True
        # Expired: same teardown as a hand-release (file removed), and ONE
        # named line — a quiet that ran its course must be as visible in the
        # log as one ended by hand, and the clear latches it so a polling
        # caller cannot print it twice.
        self._clear()
        sensory_log.stage(
            STAGE_SUPERVISE, SOURCE, EVENT_QUIET, "released reason=expired"
        )
        return False

    def remaining_s(self) -> float:
        if not self.active():
            return 0.0
        with self._lock:
            return max(0.0, self._until - self._clock())

    # -- the speaker's question --------------------------------------------

    def allow_utterance(self) -> bool:
        """May the speaker play an utterance right now?

        True when not quiet, and once more right after :meth:`arm` — the
        acknowledgement — provided it arrives within ``grace_s``.
        """
        if not self.active():
            return True
        with self._lock:
            if not self.pending_first_utterance:
                return False
            self.pending_first_utterance = False
            return self._clock() <= self._pending_deadline
