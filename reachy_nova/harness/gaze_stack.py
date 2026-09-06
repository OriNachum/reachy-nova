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
  feel-alive's own sway underneath it.
* ``CONVERSATION`` — someone is actually talking. The face lock (task **t9**)
  owns the head; the browsing goal is deliberately **left standing** beneath
  it, because the lock wins by recency and the goal simply resumes when the
  lock releases.

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

**Task t9 seam.** :meth:`_enter_conversation` / :meth:`_leave_conversation`
are where the conversation layer's ``look_at_sound``/``lock_face``/
``release_face`` spool calls will live. In this task they only log and keep
state. Both are called by the worker while holding :attr:`_op_lock`, so t9
may issue serialised ops from inside them directly via :meth:`_op`.

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
from .tools import DECLARE_GOAL, IntentTools

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Senselog identity                                                            #
# --------------------------------------------------------------------------- #

STAGE = "gaze"
SOURCE = "nova"
EVENT_LAYER = "layer"
EVENT_OP = "op"

# --------------------------------------------------------------------------- #
# The layers                                                                   #
# --------------------------------------------------------------------------- #

LAYER_WANDER = "wander"
LAYER_BROWSING = "browsing"
LAYER_CONVERSATION = "conversation"

#: The runtime behaviour the browsing layer declares as a standing goal.
GAZE_HOLD_BEHAVIOR = "gaze-hold"

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
        lock_state: the harness's gaze-lock belief. Held for task t9, which
            takes and releases the lock; t8 never touches it.
        clock: monotonic-seconds source, injectable for tests.
        tick_s: worker cadence (see :data:`DEFAULT_TICK_S`).
        side_yaw_deg: how far aside the browsing pose looks, in degrees. The
            SIGN alternates per browse; this is the magnitude.
        up_pitch_deg: how far up the browsing pose looks, in degrees.
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
    ) -> None:
        self._intents = intents
        self._attention = attention
        self.lock_state = lock_state
        self._clock = clock
        self.tick_s = max(0.01, float(tick_s))
        self.side_yaw_deg = float(side_yaw_deg)
        self.up_pitch_deg = float(up_pitch_deg)

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

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._external_stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self, stop_event: threading.Event) -> None:
        """Spawn the single worker thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._external_stop = stop_event
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="nova-gaze", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the worker to exit and join it (within one tick, plus any op)."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

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

    def on_transcript(self, role: str, text: str) -> None:
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
        """The four things worth knowing about the posture layer."""
        return {
            "layer": self.layer,
            "browser_busy": self.browser_busy(),
            "conversation_live": self.conversation_live(),
            "goal_standing": self.goal_standing,
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

    def _settle(self) -> None:
        """Compute the desired top layer and issue only the transition ops."""
        with self._op_lock:
            live = self.conversation_live()
            busy = self.browser_busy()
            desired = (
                LAYER_CONVERSATION
                if live
                else (LAYER_BROWSING if busy else LAYER_WANDER)
            )
            old = self.layer
            if desired == old:
                return
            reason = f"conversation_live={live} browser_busy={busy}"
            sensory_log.stage(STAGE, SOURCE, EVENT_LAYER, f"{old} -> {desired} reason={reason}")
            self._transition(old, desired)
            self.layer = desired

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
            return
        # -> wander
        if self.goal_standing:
            self._clear_goal("wander")

    # -- the t9 seam ---------------------------------------------------------- #

    def _enter_conversation(self) -> None:
        """**Task t9 seam.** Take the conversation posture.

        In t8 this only logs. t9 fills it in with the ``look_at_sound`` +
        ``lock_face`` (owner ``"auto"``) spool calls and their retry schedule.

        It is called by the worker while holding :attr:`_op_lock`, so it may
        issue serialised ops directly via :meth:`_op`, and it may rely on
        :attr:`layer` still being the OLD layer, :attr:`goal_standing` being
        current, and :attr:`lock_state` / :attr:`last_speaker_idle_at` being
        readable.
        """
        sensory_log.stage(STAGE, SOURCE, EVENT_LAYER, "enter conversation (lock deferred to t9)")

    def _leave_conversation(self) -> None:
        """**Task t9 seam.** Give the conversation posture back.

        In t8 this only logs. t9 fills it in with the ``release_face`` call,
        submitted only when the lock's owner is ``"auto"``. Same guarantees as
        :meth:`_enter_conversation`: called under :attr:`_op_lock`, with
        :attr:`layer` still the OLD layer.
        """
        sensory_log.stage(
            STAGE, SOURCE, EVENT_LAYER, "leave conversation (release deferred to t9)"
        )

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
