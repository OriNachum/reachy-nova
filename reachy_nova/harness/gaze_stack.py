"""Gaze stack — the harness's posture layer, single-writer (t8 core).

Three different parts of the harness all want the head at once: the browser
leg wants a "thinking" pose while a browse is in flight, the conversation
layer wants a face lock while someone is actually talking, and the runtime's
own feel-alive base wants to wander when nobody wants anything. Left to
themselves they interleave — a browse ending mid-conversation clears a goal
the conversation is relying on, two producers race the same ``declare_goal``,
and the head ends up somewhere no single component asked for.

So postures here are **layered, lowest first**::

    WANDER  <  BROWSING  <  CONVERSATION

* ``WANDER`` — nothing to do. The runtime's own feel-alive base owns the head;
  the harness declares nothing at all.
* ``BROWSING`` — a browse is in flight. A standing ``declare_goal`` of the
  runtime's ``gaze-hold`` behaviour holds the head up and aside (the aside side
  alternating per browse, so two browses in a row do not look identical).
  ``gaze-hold`` claims only the head channel, so antennas and body yaw keep
  feel-alive's own sway underneath it. A ``gaze-hold`` alone is not enough,
  though — the runtime's own ``orient-to-sound`` and ``nod`` reflexes still
  compete for the head by recency (task t10), so entering this layer also
  merges :data:`BROWSING_INHIBITS` into the runtime's CURRENT inhibited set,
  and leaving it (to WANDER) gives back only the names it added.
* ``CONVERSATION`` — someone is actually talking. The face lock owns the head;
  the browsing goal is deliberately **left standing** beneath it, because the
  lock wins by recency and the goal simply resumes when the lock releases.

**The conversation layer (t9).** Entering it turns the head toward the voice
(``look_at_sound``) and then asks for a standing face lock (``lock_face``,
owned ``"auto"``). The lock is held through Nova's replies AND the listening
gaps — it is given back only when the conversation itself fades. There is no
face-presence belief in this module on purpose: the ENGINE's own refusal
("no face known") IS the presence check, and a second opinion kept here would
be a second thing to drift. A refused (or degraded) lock is simply retried
while the conversation stays live, on the :data:`LOCK_RETRY_BACKOFF_S`
schedule, so a conversation that starts with nobody in frame still locks on
the moment somebody leans in.

A lock the MODEL took (``lock_state.owner == "model"``) is never touched by
this layer: not re-taken, not released. The model asked for that lock
deliberately, and an automatic hold quietly stealing or dropping it would be
the harness overruling the mind it serves.

The desired top layer is a **pure function of two inputs**:
:attr:`~reachy_nova.harness.attention.AttentionState.conversation_live` and
"is the browser busy". Everything else is bookkeeping.

**One writer.** Producers (the browser leg, transcripts, Sonic's speaking
edge, the speaker going idle) only ever set a flag under a lock and wake an
:class:`threading.Event`; they never touch the spool. ONE worker thread
(``nova-gaze``) computes the top layer, and on a transition issues ONLY the
transition ops, serially, each waiting out its own
:meth:`~reachy_nova.harness.tools.IntentTools.execute` round trip. That is
what makes the op list on the spool causally ordered rather than a race
between whichever producer fired last.

The single exception to "only the worker writes" is :meth:`clear_for_result`,
and it is an exception on purpose: the app must be able to guarantee the head
has dropped out of the thinking pose *before* it injects a browse result, and
a promise like that cannot be kept by setting a flag and hoping the worker
gets there first. So it runs on the CALLER's thread — but through the same
serialising op lock the worker uses, so it is still exactly one writer at a
time.

The single-writer rule has exactly two more exceptions, both deliberate and
both taken under the same :attr:`_op_lock`: :meth:`start` and :meth:`stop`
run their hygiene (``release_face`` / ``declare_goal`` None) on the CALLER's
thread, because "the harness leaves the head the way it found it" has to be
true before the worker exists and after it is gone.

stdlib only; never imports ``reachy_mini`` or ``reachy``
(``tests/test_harness_boundary.py``).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from .. import sensory_log
from .attention import AttentionState
from .lock_state import LockState
from .tools import (
    DECLARE_GOAL,
    LOCK_FACE,
    LOOK_AT_SOUND,
    RELEASE_FACE,
    SET_INHIBITION,
    IntentTools,
)
from .tools import current_inhibitions as _runtime_current_inhibitions

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Senselog identity                                                            #
# --------------------------------------------------------------------------- #

STAGE = "gaze"
SOURCE = "nova"
EVENT_LAYER = "layer"
EVENT_OP = "op"
EVENT_LOCK = "lock"

# --------------------------------------------------------------------------- #
# The layers                                                                   #
# --------------------------------------------------------------------------- #

LAYER_WANDER = "wander"
LAYER_BROWSING = "browsing"
LAYER_CONVERSATION = "conversation"

#: The runtime behaviour the browsing layer declares as a standing goal.
GAZE_HOLD_BEHAVIOR = "gaze-hold"

#: Runtime head reflexes the browsing layer keeps off a head it is holding in
#: the thinking pose (task t10). ``orient-to-sound`` and ``nod`` both steal
#: the head from a standing ``gaze-hold`` goal by recency — the same failure
#: mode a face lock has, just from the runtime's own reflex layer instead of
#: another harness component.
BROWSING_INHIBITS = ("orient-to-sound", "nod")

#: The browser states that mean "a browse is in flight". Everything else —
#: ``idle``, ``error``, an unknown string, ``None`` — means it is not, because
#: the failure mode of guessing the other way is a head stuck in the thinking
#: pose forever.
BUSY_STATES = frozenset({"busy"})

#: The transcript role whose lines count as someone talking.
USER_ROLE = "user"

#: The Sonic state whose RISING edge counts as a conversation tick.
SPEAKING_STATE = "speaking"

#: How long a conversation stays "live" for the local fallback used when no
#: :class:`~reachy_nova.harness.attention.AttentionState` is wired in. Same
#: length as :data:`reachy_nova.harness.attention.DEFAULT_WINDOW_S` on
#: purpose — the fallback should behave like the real thing, not like a
#: different policy that only shows up in degraded wiring.
FALLBACK_LIVE_S = 45.0

#: Default worker cadence: how long the worker waits for a producer to wake it
#: before re-computing the desired layer anyway (liveness expires on a clock,
#: not on an event, so the loop must re-check on its own).
DEFAULT_TICK_S = 0.25

#: How long to wait before retrying a ``lock_face`` the engine refused (or
#: could not confirm), in seconds, indexed by attempt. The LAST value repeats
#: for every attempt beyond the tuple, so a conversation held with nobody in
#: frame settles into one quiet probe every 30 s rather than either giving up
#: or hammering the spool. The early values are short because the common
#: refusal is simply "the face detector has not caught up yet", which
#: resolves in a second or two.
LOCK_RETRY_BACKOFF_S = (3.0, 6.0, 12.0, 24.0, 30.0)

#: The lock owner this layer takes, and the one it must never touch.
OWNER_AUTO = "auto"
OWNER_MODEL = "model"

#: Default browsing pose: up, and aside by this much (degrees).
DEFAULT_UP_PITCH_DEG = 10.0
DEFAULT_SIDE_YAW_DEG = 15.0


class GazeStack:
    """The posture layer: one worker, one standing goal, layered priorities.

    Args:
        intents: the :class:`~reachy_nova.harness.tools.IntentTools` every op
            goes through. Only ``execute(tool_name, params) -> str`` is used,
            so tests pass a fake.
        attention: the :class:`~reachy_nova.harness.attention.AttentionState`
            whose ``conversation_live`` decides the conversation layer. When
            ``None`` (or broken), a local "live until now + 45 s" fallback fed
            by :meth:`on_transcript` / :meth:`on_sonic_state` is used instead.
        lock_state: the harness's gaze-lock belief. The conversation layer
            marks it ``owner="auto"`` on a confirmed lock and clears it on a
            confirmed release, and reads ``owner``/``locked`` to stay off a
            lock the MODEL took. ``None`` is fine — the layer then simply has
            no way to tell an auto hold from a model one, and behaves as if
            every lock were its own.
        clock: monotonic-seconds source, injectable for tests.
        tick_s: worker cadence (see :data:`DEFAULT_TICK_S`).
        side_yaw_deg: how far aside the browsing pose looks, in degrees. The
            SIGN alternates per browse; this is the magnitude.
        up_pitch_deg: how far up the browsing pose looks, in degrees.
        current_inhibitions: reads the runtime's CURRENT inhibited set — the
            live set the browsing layer merges its own additions into and
            restores out of (see :data:`BROWSING_INHIBITS`). Defaults to
            :func:`reachy_nova.harness.tools.current_inhibitions`; injectable
            for tests so they never touch a real state dir.
    """

    name = "gaze"

    def __init__(
        self,
        intents: IntentTools,
        attention: AttentionState | None = None,
        lock_state: LockState | None = None,
        clock: Callable[[], float] = time.monotonic,
        tick_s: float = DEFAULT_TICK_S,
        side_yaw_deg: float = DEFAULT_SIDE_YAW_DEG,
        up_pitch_deg: float = DEFAULT_UP_PITCH_DEG,
        current_inhibitions: Callable[[], list[str]] | None = None,
    ) -> None:
        self._intents = intents
        self._attention = attention
        self.lock_state = lock_state
        self._clock = clock
        self.tick_s = max(0.01, float(tick_s))
        self.side_yaw_deg = float(side_yaw_deg)
        self.up_pitch_deg = float(up_pitch_deg)
        self._current_inhibitions = current_inhibitions or _runtime_current_inhibitions

        #: Guards the PRODUCER flags only — held for microseconds, never
        #: across an op, so a hook can never be blocked behind a spool round
        #: trip.
        self._flag_lock = threading.Lock()
        self._browser_busy = False
        self._live_until = float("-inf")
        self._last_sonic_state: str | None = None
        #: When the speaker last went idle (monotonic). Recorded here for
        #: task t9, which needs "has Nova stopped talking?" to decide when a
        #: deferred gaze move is safe; t8 only keeps it current.
        self.last_speaker_idle_at: float | None = None

        #: Serialises every ``intents.execute`` call — the worker's own
        #: transitions AND :meth:`clear_for_result`'s caller-thread clear.
        #: This is the "single writer" invariant, made mechanical.
        self._op_lock = threading.RLock()

        #: The layer currently in force, and whether the browsing goal is
        #: believed to be standing on the runtime. Both are written only under
        #: :attr:`_op_lock`; reads are plain attribute reads (atomic) so
        #: :meth:`status` never waits behind a spool round trip.
        self.layer = LAYER_WANDER
        self.goal_standing = False
        #: Sign of the NEXT browsing aside yaw (+1 first, then alternating).
        self._next_side = 1.0

        #: Names from :data:`BROWSING_INHIBITS` the stack currently believes
        #: IT added to the runtime's inhibited set (as opposed to names that
        #: were already inhibited for some other reason before browsing
        #: started, which are never the stack's to remove). Written only
        #: under :attr:`_op_lock`.
        self._browsing_added: set[str] = set()

        #: Whether an AUTO-owned face lock is believed held right now. Written
        #: only under :attr:`_op_lock` (or by :meth:`on_lock_released`, which
        #: only ever clears it — a clear can never be wrong in the dangerous
        #: direction).
        self.lock_held = False
        #: Per-conversation lock bookkeeping, all reset by
        #: :meth:`_reset_lock_retry`.
        self._lock_attempts = 0
        self._lock_refusals = 0
        self._refusal_logged = False
        self._ever_locked = False
        self._conversation_started_at: float | None = None
        self._next_lock_retry_at: float | None = None
        self._retry_index = 0

        #: Set once the stop hygiene has run, so the caller's :meth:`stop` and
        #: the worker's own exit path cannot both submit it.
        self._hygiene_done = False

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._external_stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self, stop_event: threading.Event) -> None:
        """Run the start hygiene, then spawn the single worker thread (idempotent).

        The hygiene is one ``release_face`` and one ``declare_goal`` None,
        submitted on the CALLER's thread before the worker exists — a harness
        that just restarted has no idea what the previous process left
        standing on the runtime, and both ops are idempotent no-ops when
        nothing is held (``release_face`` answers ``ok: true`` "not locked").
        Starting from a known-clean head is worth two spool round trips.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._external_stop = stop_event
        self._stop.clear()
        with self._op_lock:
            self._hygiene_done = False
            self.lock_held = False
            self._reset_lock_retry()
            self._op(RELEASE_FACE, {}, "start-hygiene release")
            self._op(DECLARE_GOAL, {"goal": None}, "start-hygiene clear goal")
            self.goal_standing = False
        self._thread = threading.Thread(target=self._run, name="nova-gaze", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Run the stop hygiene, then ask the worker to exit and join it.

        The hygiene runs BEFORE the join, on the caller's thread but under
        :attr:`_op_lock`, so it can never race the worker's last tick. It is
        guarded by a one-shot flag shared with the worker's own exit path, so
        a stop that arrives mid-conversation costs exactly one release either
        way — and a second :meth:`stop` submits nothing at all.
        """
        self._stop.set()
        with self._op_lock:
            self._stop_hygiene()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _stop_hygiene(self) -> None:
        """Give back whatever we still hold. Caller holds the op lock; once."""
        if self._hygiene_done:
            return
        self._hygiene_done = True
        if self.lock_held and not self._model_owns_lock():
            result = self._op(RELEASE_FACE, {}, "stop-hygiene release")
            self.lock_held = False
            if self.lock_state is not None and _confirmed(result):
                self._mark_released("auto-stop")
        if self.goal_standing:
            self._op(DECLARE_GOAL, {"goal": None}, "stop-hygiene clear goal")
            self.goal_standing = False

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        return self._external_stop is not None and self._external_stop.is_set()

    # -- producer hooks: set a flag, wake the worker, never block ------------ #

    def on_browser_state(self, state: str) -> None:
        """``"busy"`` raises the browsing layer; anything else lowers it."""
        try:
            busy = isinstance(state, str) and state.strip().lower() in BUSY_STATES
            with self._flag_lock:
                changed = busy != self._browser_busy
                self._browser_busy = busy
            if changed:
                self._wake.set()
        except Exception as exc:  # noqa: BLE001 - a hook must never raise
            logger.warning("gaze stack on_browser_state raised: %s", exc)

    def on_transcript(self, role: str, _text: str) -> None:
        """A USER line is someone talking — note a conversation tick.

        Only the local fallback clock is fed here: when an
        :class:`AttentionState` is wired in, IT is the authority on liveness
        (it hears injects and Nova's own utterances too), and a second opinion
        kept here would be a second thing to drift.
        """
        try:
            if not isinstance(role, str) or role.strip().lower() != USER_ROLE:
                return
            self._note_conversation_tick()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gaze stack on_transcript raised: %s", exc)

    def on_sonic_state(self, state: str) -> None:
        """Nova starting to speak is also a conversation tick (rising edge only)."""
        try:
            current = state.strip().lower() if isinstance(state, str) else None
            with self._flag_lock:
                previous, self._last_sonic_state = self._last_sonic_state, current
            if current == SPEAKING_STATE and previous != SPEAKING_STATE:
                self._note_conversation_tick()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gaze stack on_sonic_state raised: %s", exc)

    def on_speaker_idle(self) -> None:
        """The speaker drained. Recorded for task t9; no posture effect in t8."""
        try:
            with self._flag_lock:
                self.last_speaker_idle_at = self._clock()
            self._wake.set()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gaze stack on_speaker_idle raised: %s", exc)

    def on_lock_released(self, reason: str | None = None) -> None:
        """The RUNTIME dropped the lock (wire this from the bus's
        ``motion/lock-released`` tap).

        Believing a lock we no longer hold is the one failure that cannot fix
        itself: the layer would sit there content while the head wandered off
        mid-sentence. So this only ever CLEARS, submits nothing, and drops the
        retry deadline so the worker re-arms the backoff and tries to take the
        hold back while the conversation is still live.
        """
        try:
            if self._model_owns_lock():
                return
            self.lock_held = False
            self._next_lock_retry_at = None
            sensory_log.stage(
                STAGE, SOURCE, EVENT_LOCK, f"runtime dropped the lock reason={reason}"
            )
            self._wake.set()
        except Exception as exc:  # noqa: BLE001 - a hook must never raise
            logger.warning("gaze stack on_lock_released raised: %s", exc)

    def _note_conversation_tick(self) -> None:
        with self._flag_lock:
            self._live_until = self._clock() + FALLBACK_LIVE_S
        self._wake.set()

    # -- the synchronous escape hatch --------------------------------------- #

    def clear_for_result(self) -> bool:
        """Drop the browsing goal NOW, on the caller's thread. Returns whether
        anything was cleared.

        The app calls this immediately before injecting a browse result, so
        the head is demonstrably out of the thinking pose *before* Nova starts
        talking about what it found. It goes through :attr:`_op_lock` — the
        same lock the worker holds for its own transitions — so "synchronous"
        never means "concurrent with the worker".

        The layer itself is left alone: while the browser is still busy the
        stack stays in ``browsing``, just with nothing standing, so the later
        ``idle`` transition issues no second clear.
        """
        try:
            with self._op_lock:
                if not self.goal_standing:
                    return False
                self._clear_goal("result")
                return True
        except Exception as exc:  # noqa: BLE001 - never raise into the app
            logger.warning("gaze stack clear_for_result raised: %s", exc)
            return False

    # -- reads --------------------------------------------------------------- #

    def conversation_live(self) -> bool:
        """Is a conversation live right now? Attention first, fallback second."""
        if self._attention is not None:
            try:
                return bool(self._attention.conversation_live)
            except Exception as exc:  # noqa: BLE001 - degrade to "not live"
                logger.warning("gaze stack could not read attention: %s", exc)
                return False
        with self._flag_lock:
            return self._clock() < self._live_until

    def browser_busy(self) -> bool:
        with self._flag_lock:
            return self._browser_busy

    def status(self) -> dict:
        """Everything worth knowing about the posture layer, in one dict."""
        due = self._next_lock_retry_at
        return {
            "layer": self.layer,
            "browser_busy": self.browser_busy(),
            "conversation_live": self.conversation_live(),
            "goal_standing": self.goal_standing,
            "browsing_inhibits": sorted(self._browsing_added),
            "lock_held": self.lock_held,
            "lock_attempts": self._lock_attempts,
            "next_lock_retry_s": None if due is None else max(0.0, due - self._clock()),
        }

    # -- the worker ----------------------------------------------------------- #

    def _run(self) -> None:
        while not self._should_stop():
            self._wake.wait(self.tick_s)
            self._wake.clear()
            if self._should_stop():
                break
            try:
                self._settle()
            except Exception as exc:  # noqa: BLE001 - a bad tick never kills the loop
                logger.warning("gaze stack worker tick raised: %s", exc)
        # The worker's own exit path runs the same hygiene as stop(), guarded
        # by the same one-shot flag: a stop_event set from outside never joins
        # this thread, so without this a lock taken mid-conversation would
        # outlive the harness that took it.
        try:
            with self._op_lock:
                self._stop_hygiene()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gaze stack exit hygiene raised: %s", exc)

    def _settle(self) -> None:
        """Compute the desired top layer, issue the transition ops, then let
        the conversation layer's lock retry have its turn."""
        with self._op_lock:
            live = self.conversation_live()
            busy = self.browser_busy()
            if live:
                desired = LAYER_CONVERSATION
            elif busy:
                desired = LAYER_BROWSING
            else:
                desired = LAYER_WANDER
            old = self.layer
            if desired != old:
                reason = f"conversation_live={live} browser_busy={busy}"
                sensory_log.stage(
                    STAGE, SOURCE, EVENT_LAYER, f"{old} -> {desired} reason={reason}"
                )
                self._transition(old, desired)
                self.layer = desired
            if self.layer == LAYER_CONVERSATION:
                self._tick_lock()

    def _transition(self, old: str, new: str) -> None:
        """Issue ONLY the ops this transition needs. Caller holds the op lock."""
        if new == LAYER_CONVERSATION:
            # The lock owns the head by recency; a standing browsing goal is
            # deliberately left in place and resumes when the lock releases.
            self._enter_conversation()
            return
        if old == LAYER_CONVERSATION:
            self._leave_conversation()
        if new == LAYER_BROWSING:
            # It may already be standing — we never cleared it for the lock —
            # in which case there is nothing to re-declare.
            if not self.goal_standing:
                self._declare_browsing_goal()
            self._enter_browsing_inhibits()
            return
        # -> wander
        if self.goal_standing:
            self._clear_goal("wander")
        self._leave_browsing_inhibits()

    # -- the conversation layer ----------------------------------------------- #

    def _enter_conversation(self) -> None:
        """Take the conversation posture: turn toward the voice, then lock on.

        Called by the worker under :attr:`_op_lock`, with :attr:`layer` still
        the OLD layer, so it issues its ops directly through :meth:`_op`.

        ``look_at_sound`` first and ``lock_face`` second is the order a person
        does it in: you turn toward the voice, and only then settle on the
        face. It also gives the runtime's own face detector the best possible
        frame to answer ``lock_face`` from — the head is already pointing the
        right way when the question is asked.
        """
        self._reset_lock_retry()
        self._conversation_started_at = self._clock()
        if self._model_owns_lock():
            # The model asked for this lock itself. Do not re-take it, do not
            # look anywhere, and (see _leave_conversation) do not release it.
            sensory_log.stage(
                STAGE, SOURCE, EVENT_LOCK, "model lock standing — auto hold not taken"
            )
            self.lock_held = False
            return
        self._op(LOOK_AT_SOUND, {}, "look at sound")
        self._attempt_lock("enter")

    def _leave_conversation(self) -> None:
        """Give the conversation posture back: release an AUTO-owned lock.

        Same guarantees as :meth:`_enter_conversation` — under
        :attr:`_op_lock`, with :attr:`layer` still the OLD layer.
        """
        if self._model_owns_lock():
            sensory_log.stage(
                STAGE, SOURCE, EVENT_LOCK, "model lock standing — not released on fade"
            )
            self._reset_lock_retry()
            return
        if self.lock_held:
            result = self._op(RELEASE_FACE, {}, "release face reason=fade")
            if _confirmed(result):
                self._mark_released("auto-fade")
            else:
                # Clear the belief anyway. The runtime drops a standing lock on
                # its own max-hold timer regardless of what we believe, so a
                # belief kept "held" on an unconfirmed release would go stale
                # with nothing left to correct it — and a stale "held" is the
                # one error that suppresses the next lock attempt. The cost of
                # being wrong the other way is one redundant release later.
                sensory_log.stage(
                    STAGE,
                    SOURCE,
                    EVENT_LOCK,
                    "release unconfirmed — clearing the belief anyway",
                )
            self.lock_held = False
        elif self._lock_attempts and not self._ever_locked:
            sensory_log.stage(
                STAGE, SOURCE, EVENT_LOCK, f"lock never held: refusals={self._lock_refusals}"
            )
        self._reset_lock_retry()

    def _tick_lock(self) -> None:
        """One conversation tick: notice a lost lock, or retry a refused one.

        Caller holds the op lock and has already confirmed the layer is
        ``conversation``.
        """
        if self._model_owns_lock():
            return
        if self.lock_held:
            if self._engine_dropped_the_lock():
                self.lock_held = False
                self._next_lock_retry_at = None
            else:
                return
        if self._next_lock_retry_at is None:
            self._schedule_lock_retry()
            return
        if self._clock() >= self._next_lock_retry_at:
            self._attempt_lock("retry")

    def _attempt_lock(self, why: str) -> bool:
        """One ``lock_face``. Returns whether the lock is now believed held."""
        self._lock_attempts += 1
        result = self._op(LOCK_FACE, {}, f"lock face attempt={self._lock_attempts} reason={why}")
        ok = result.get("ok") if isinstance(result, dict) else None
        if ok is True:
            self.lock_held = True
            self._ever_locked = True
            self._next_lock_retry_at = None
            if self.lock_state is not None:
                try:
                    self.lock_state.mark_locked(owner=OWNER_AUTO)
                except Exception as exc:  # noqa: BLE001 - a belief is not worth a crash
                    logger.warning("gaze stack could not mark the lock: %s", exc)
            started = self._conversation_started_at
            waited = 0.0 if started is None else max(0.0, self._clock() - started)
            sensory_log.stage(
                STAGE,
                SOURCE,
                EVENT_LOCK,
                f"locked after={waited:.1f}s attempts={self._lock_attempts}",
            )
            return True
        if ok is False:
            # "no face known" is the ENGINE's presence check answering, not an
            # error: nobody is in frame yet. ONE line per conversation says so;
            # the rest are counted and summarised at fade, because a 45-minute
            # conversation with nobody in view must not cost 90 log lines.
            self._lock_refusals += 1
            if not self._refusal_logged:
                self._refusal_logged = True
                sensory_log.stage(
                    STAGE, SOURCE, EVENT_LOCK, "no face known — retrying with backoff"
                )
        # ok is None (degraded, or the call itself failed): UNKNOWN, never
        # "locked". The belief is left exactly as it was and we retry on the
        # same schedule — see _op's ok=unknown line for what actually happened.
        self._schedule_lock_retry()
        return False

    def _schedule_lock_retry(self) -> None:
        index = min(self._retry_index, len(LOCK_RETRY_BACKOFF_S) - 1)
        self._retry_index += 1
        self._next_lock_retry_at = self._clock() + LOCK_RETRY_BACKOFF_S[index]

    def _reset_lock_retry(self) -> None:
        self._lock_attempts = 0
        self._lock_refusals = 0
        self._refusal_logged = False
        self._ever_locked = False
        self._retry_index = 0
        self._next_lock_retry_at = None
        self._conversation_started_at = None

    def _model_owns_lock(self) -> bool:
        if self.lock_state is None:
            return False
        try:
            return self.lock_state.owner == OWNER_MODEL
        except Exception as exc:  # noqa: BLE001 - degrade to "not the model's"
            logger.warning("gaze stack could not read the lock owner: %s", exc)
            return False

    def _engine_dropped_the_lock(self) -> bool:
        """Has the belief stopped saying ``True`` under us (engine restart)?"""
        if self.lock_state is None:
            return False
        try:
            return self.lock_state.locked is not True
        except Exception as exc:  # noqa: BLE001
            logger.warning("gaze stack could not read the lock belief: %s", exc)
            return False

    def _mark_released(self, reason: str) -> None:
        if self.lock_state is None:
            return
        try:
            self.lock_state.mark_released(reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gaze stack could not mark the release: %s", exc)

    # -- the ops -------------------------------------------------------------- #

    def _declare_browsing_goal(self) -> None:
        """Declare the standing ``gaze-hold`` goal, aside side alternating."""
        yaw = self._next_side * self.side_yaw_deg
        self._next_side = -self._next_side
        result = self._op(
            DECLARE_GOAL,
            {
                "goal": GAZE_HOLD_BEHAVIOR,
                "params": {"pitch": self.up_pitch_deg, "yaw": yaw},
            },
            f"declare gaze-hold yaw={yaw:+g} pitch={self.up_pitch_deg:g}",
        )
        # A refusal means nothing is standing. A degraded ``ok: null`` means
        # the command IS on disk and an engine started a second later still
        # applies it (see tools.py's three result shapes), so it counts as
        # standing — believing otherwise would leave the head held with
        # nothing in the harness willing to clear it.
        self.goal_standing = not _refused(result)

    def _clear_goal(self, why: str) -> None:
        """Clear the standing goal (``declare_goal`` with no goal)."""
        result = self._op(DECLARE_GOAL, {"goal": None}, f"clear goal reason={why}")
        # Mirror image of the declare: only an explicit refusal leaves the
        # goal believed standing, so a later transition tries the clear again.
        self.goal_standing = _refused(result)

    # -- browsing head-reflex inhibits (task t10) ----------------------------- #

    def _enter_browsing_inhibits(self) -> None:
        """Entering (or resuming) BROWSING: add :data:`BROWSING_INHIBITS`.

        Two cases, told apart by :attr:`_browsing_added`:

        * Fresh entry (from WANDER, or first-ever browse): merge the runtime's
          CURRENT inhibited set with :data:`BROWSING_INHIBITS` in one
          ``set_inhibition``, remembering exactly which names were not already
          there — those, and only those, are ours to give back later.
        * Resuming from CONVERSATION: the names may already be standing (a
          face lock re-asserts its own inhibitions, but ``nod`` is the
          harness's own addition and is not guaranteed to survive a
          replacement) — re-read the live set and re-add only what is
          missing, rather than assuming nothing changed underneath us.
        """
        if self._browsing_added:
            self._reassert_browsing_inhibits()
        else:
            self._declare_browsing_inhibits()

    def _declare_browsing_inhibits(self) -> None:
        live = self._read_inhibitions()
        added = [name for name in BROWSING_INHIBITS if name not in live]
        merged = sorted(set(live) | set(BROWSING_INHIBITS))
        result = self._op(
            SET_INHIBITION, {"behaviors": merged}, f"browsing inhibit add={added}"
        )
        # Mirror the goal-declare pattern: anything but an explicit refusal is
        # believed to have taken.
        if not _refused(result):
            self._browsing_added = set(added)

    def _reassert_browsing_inhibits(self) -> None:
        live = self._read_inhibitions()
        missing = [name for name in BROWSING_INHIBITS if name not in live]
        if not missing:
            return
        merged = sorted(set(live) | set(missing))
        result = self._op(
            SET_INHIBITION, {"behaviors": merged}, f"browsing inhibit re-add={missing}"
        )
        if not _refused(result):
            self._browsing_added = self._browsing_added | set(missing)

    def _leave_browsing_inhibits(self) -> None:
        """Leaving BROWSING for WANDER: give back only the names WE added.

        Re-reads the live set first rather than assuming it — a later-wins
        operator change (say, adding ``antenna-sway``) in the meantime must
        survive the restore.
        """
        if not self._browsing_added:
            return
        live = self._read_inhibitions()
        remaining = sorted(set(live) - self._browsing_added)
        result = self._op(
            SET_INHIBITION,
            {"behaviors": remaining},
            f"browsing inhibit remove={sorted(self._browsing_added)}",
        )
        # Mirror the goal-clear pattern: only an explicit refusal leaves the
        # names believed still added, so a later transition retries the give-back.
        if not _refused(result):
            self._browsing_added = set()

    def _read_inhibitions(self) -> list[str]:
        try:
            return list(self._current_inhibitions())
        except Exception as exc:  # noqa: BLE001 - degrade to "nothing known inhibited"
            logger.warning("gaze stack could not read current inhibitions: %s", exc)
            return []

    def _op(self, tool_name: str, params: dict, detail: str) -> dict | None:
        """One serialised tool call; logs its outcome; never raises.

        Returns the parsed result dict, or ``None`` when the call itself
        failed or came back unparseable.
        """
        try:
            raw = self._intents.execute(tool_name, params)
        except Exception as exc:  # noqa: BLE001 - a broken spool is not fatal here
            sensory_log.stage(
                STAGE, SOURCE, EVENT_OP, f"{detail} failed error={type(exc).__name__}: {exc}"
            )
            return None
        result = _parse(raw)
        if result is None:
            sensory_log.stage(STAGE, SOURCE, EVENT_OP, f"{detail} unparseable result={raw!r}")
            return None
        ok = result.get("ok")
        if ok is True:
            sensory_log.stage(STAGE, SOURCE, EVENT_OP, f"{detail} ok=true")
        elif ok is None:
            # The third result shape: on disk, unconfirmed. Named rather than
            # silently treated as success — "did the head actually move?" must
            # be answerable from the log alone.
            sensory_log.stage(
                STAGE,
                SOURCE,
                EVENT_OP,
                f"{detail} ok=unknown submitted={result.get('submitted')}",
            )
        else:
            sensory_log.stage(
                STAGE, SOURCE, EVENT_OP, f"{detail} ok=false error={result.get('error')}"
            )
        return result


def _parse(raw: Any) -> dict | None:
    """``IntentTools.execute`` returns a JSON string; be tolerant anyway."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _refused(result: dict | None) -> bool:
    """Did the engine (or the pre-flight) explicitly refuse this op?

    ``None`` (the call itself blew up) is NOT a refusal: nothing observable
    happened either way, and the conservative reading for a declare is "it may
    well be standing".
    """
    return isinstance(result, dict) and result.get("ok") is False


def _confirmed(result: dict | None) -> bool:
    """Did the engine confirm this op with ``ok: true``? Nothing else counts."""
    return isinstance(result, dict) and result.get("ok") is True
