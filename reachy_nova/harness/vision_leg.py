"""Vision leg — the runtime's rolling camera clip, understood and injected.

On the wireless Reachy Mini the camera belongs to the reachy-mini-cli runtime,
not to us. Its **clip rider** continuously re-encodes a few seconds of video to
ONE overwrite-in-place file and announces that file on the RETAINED MQTT topic
``reachy/state/clip``::

    {"available": true,  "reason": null,
     "path": "/…/clip.mp4", "ts": …, "duration_s": 4.0, "frame_count": 60}

    {"available": false, "reason": "vision-extra-absent",
     "path": null, "ts": …, "duration_s": null, "frame_count": 0}
    # ^ what the device actually reported when this leg was written (2026-08-11)

This module is the consumer of that state, and *only* of that state. It opens
no camera, decodes no frame, imports neither ``cv2`` nor ``reachy_mini``: one
daemon thread wakes on an interval, reads the clip state, and — when there is a
real clip — hands its path to :class:`~reachy_nova.nova_omni.NovaOmni` and
passes the one answer that comes back to the inject callable.

Everything arrives by constructor injection
-------------------------------------------
``get_clip_state`` (the retained payload, however the app obtains it),
``understand`` (``NovaOmni``, or any ``understand``-shaped callable) and
``on_answer`` (wired to ``NovaSonic.inject_text``, **bare**, so that method's
own speaking guard and 3 s throttle stay in the path — this module never names
``force`` and always calls it with exactly one positional argument). The leg
therefore subscribes to nothing itself and is fully testable with three plain
fakes.

Every silence has a name
------------------------
An absent clip is the ORDINARY resting state of a peripheral, not an error:
today's device reports ``available: false`` with ``reason:
"vision-extra-absent"`` and will keep doing so until the vision extra is
installed. So a missing clip costs exactly ONE senselog line — the payload's
own ``reason`` is reused verbatim as the drop name, rather than inventing a
second vocabulary for the same fact — and the next cycle simply tries again.
Drops are **latched**: a line is emitted when the verdict CHANGES, so a
permanently absent camera costs one line per condition, not one per minute,
and a recovery is visible because the next failure logs again.

Two guards are worth their code:

* a retained payload outlives the file it names (the runtime restarts, the
  clip is cleaned up, the broker still replays the last message) — so the path
  is checked for existence and non-emptiness before it is handed on, turning
  a silent Omni→Lite degradation into a named drop;
* the rider overwrites one path forever, so an unchanged ``ts`` means the same
  clip: re-describing it would inject the same sentence twice
  (``skip_unchanged``).

Nothing raised anywhere in a cycle escapes it — not the state getter, not the
context provider, not ``understand``, not the inject. A sense must never kill
its own thread.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from reachy_nova import sensory_log

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Senselog identity                                                            #
# --------------------------------------------------------------------------- #

#: ``[SENSE stage=vision source=nova event=...]`` — every line this leg emits.
STAGE = "vision"
SOURCE = "nova"

#: The understanding-request id this leg identifies itself with. Passed to
#: ``understand`` only when that callable actually accepts an ``event``
#: parameter (today's :class:`~reachy_nova.nova_omni.NovaOmni` mints its own).
EVENT = "vision"

# --------------------------------------------------------------------------- #
# Interval                                                                     #
# --------------------------------------------------------------------------- #

#: Env var overriding the seconds between understanding cycles.
INTERVAL_ENV = "NOVA_VISION_INTERVAL_S"
#: One deep look per minute: Omni is a request-response model with real cost
#: and real latency, and the conversation is Sonic's, not vision's.
DEFAULT_INTERVAL_S = 60.0
#: Granularity of the interruptible sleep — the shutdown-latency ceiling.
_WAIT_SLICE_S = 0.05

# --------------------------------------------------------------------------- #
# Context                                                                      #
# --------------------------------------------------------------------------- #

#: Asked of the clip when the app supplies no context provider.
DEFAULT_CONTEXT = (
    "Describe what has been happening in front of the robot over these few "
    "seconds. Be brief and specific."
)

# --------------------------------------------------------------------------- #
# NAMED drop reasons — every silence this module produces has one              #
# --------------------------------------------------------------------------- #

#: The state getter returned ``None`` — no retained ``reachy/state/clip`` seen.
REASON_NO_CLIP_STATE = "no-clip-state"
#: A retained payload that is not a JSON object.
REASON_BAD_CLIP_STATE = "bad-clip-state"
#: The state getter itself raised (broker down, decode error upstream).
REASON_CLIP_STATE_FAILED = "clip-state-failed"
#: ``available: false`` with no ``reason`` of its own. When the payload DOES
#: carry a reason (e.g. ``vision-extra-absent``) that reason is used verbatim.
REASON_CLIP_UNAVAILABLE = "clip-unavailable"
#: ``available: true`` but ``path`` is null/blank — nothing to read.
REASON_NO_CLIP_PATH = "no-clip-path"
#: The announced path is not a file (a retained payload outliving its clip).
REASON_CLIP_FILE_MISSING = "clip-file-missing"
#: The clip is there but zero bytes — the rider is mid-write or wrote nothing.
REASON_CLIP_FILE_EMPTY = "clip-file-empty"
#: Same ``ts`` as the clip already understood (``skip_unchanged``).
REASON_CLIP_UNCHANGED = "clip-unchanged"
#: The context provider raised; the default context is used instead.
REASON_CONTEXT_FAILED = "context-failed"
#: ``understand`` raised. NovaOmni degrades internally, so this is the layer
#: below that: no client, no credentials, an unreadable clip.
REASON_UNDERSTAND_FAILED = "understand-failed"
#: ``understand`` returned nothing worth saying.
REASON_EMPTY_ANSWER = "empty-answer"
#: The inject callable raised (the Sonic stream is down).
REASON_INJECT_FAILED = "inject-failed"

#: Whitespace in a runtime-supplied reason, collapsed so ``reason=`` stays one
#: grep-able token.
_WHITESPACE = re.compile(r"\s+")


def resolve_interval(raw: str | None = None) -> float:
    """Seconds between cycles: *raw* / ``$NOVA_VISION_INTERVAL_S`` / default.

    Absent, blank, unparseable, zero and negative all resolve to
    :data:`DEFAULT_INTERVAL_S` — a misconfigured interval must never turn the
    leg into a spin loop hammering Bedrock, and never stop it dead either.
    """
    value = os.environ.get(INTERVAL_ENV) if raw is None else raw
    if value is None:
        return DEFAULT_INTERVAL_S
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_S
    if not seconds > 0.0:
        return DEFAULT_INTERVAL_S
    return seconds


class VisionLeg:
    """Turns the runtime's rolling camera clip into one inject per cycle.

    Args:
        get_clip_state: returns the retained ``reachy/state/clip`` payload as a
            mapping, or ``None`` when none has been seen. Called once per
            cycle; may raise (that is a named drop).
        understand: either a :class:`~reachy_nova.nova_omni.NovaOmni` instance
            (its ``understand`` method is used) or any callable with the same
            keyword surface (``clip_path``, ``context``). An ``event`` keyword
            is passed only when the callable accepts one.
        on_answer: **required.** Called with exactly one positional string for
            every answer. Wired to ``NovaSonic.inject_text`` bare, so its
            speaking guard and throttle stay in the path.
        interval_s: seconds between cycles; ``None`` resolves
            :data:`INTERVAL_ENV` then :data:`DEFAULT_INTERVAL_S`.
        context: the text handed to ``understand`` — a fixed string, a
            zero-argument callable evaluated fresh each cycle, or ``None`` for
            :data:`DEFAULT_CONTEXT`.
        skip_unchanged: skip a clip whose ``ts`` equals the last one
            understood (the rider overwrites a single path forever).
    """

    def __init__(
        self,
        get_clip_state: Callable[[], Mapping[str, Any] | None],
        understand: Any,
        on_answer: Callable[[str], None],
        interval_s: float | None = None,
        context: str | Callable[[], str] | None = None,
        *,
        skip_unchanged: bool = True,
    ) -> None:
        if not callable(get_clip_state):
            raise ValueError("VisionLeg requires a callable get_clip_state")
        if not callable(on_answer):
            raise ValueError("VisionLeg requires an on_answer callable (e.g. inject_text)")
        self._get_clip_state = get_clip_state
        self._understand = _understand_callable(understand)
        self._passes_event = _accepts_event(self._understand)
        self._on_answer = on_answer
        self.interval_s = resolve_interval() if interval_s is None else float(interval_s)
        self._context = context
        self._skip_unchanged = bool(skip_unchanged)

        # Counters — the status surface (tests, /api).
        self.cycles = 0
        self.answers = 0
        self.drops = 0
        self.last_answer = ""

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._external_stop: threading.Event | None = None

        # Latches, so a persistent condition costs ONE line, not one per cycle.
        self._latched_reason: str | None = None
        self._context_fault_reported = False
        self._last_clip_ts: Any = None

    # -- lifecycle ---------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        """Start the cycle daemon thread, shutting down on *stop_event*."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._external_stop = stop_event
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision-leg", daemon=True)
        self._thread.start()
        self._sense(
            "start", f"watching the runtime clip state every {self.interval_s:.1f}s"
        )

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the leg to finish and join it (best effort, never raises)."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- the loop ----------------------------------------------------------

    def _run(self) -> None:
        while not self._should_stop():
            try:
                self.cycle()
            except Exception as err:  # cycle() is already total; belt and braces
                logger.warning("vision leg cycle raised: %s", err)
            self._wait(self.interval_s)

    # -- one cycle ---------------------------------------------------------

    def cycle(self) -> bool:
        """Read the clip state once; return whether an answer was injected.

        Total: every failure resolves to a named, latched senselog line and a
        ``False``, never an exception and never a dead thread.
        """
        self.cycles += 1

        state = self._read_state()
        if state is None:  # already named + counted by _read_state
            return False

        verdict = _clip_verdict(state)
        if verdict is not None:
            reason, detail = verdict
            return self._drop(reason, detail)

        clip_path = str(state.get("path"))
        clip_ts = state.get("ts")
        if self._skip_unchanged and clip_ts is not None and clip_ts == self._last_clip_ts:
            return self._drop(REASON_CLIP_UNCHANGED, f"ts={clip_ts} already understood")

        context = self._resolve_context()
        kwargs: dict[str, Any] = {"clip_path": clip_path, "context": context}
        if self._passes_event:
            kwargs["event"] = EVENT

        self._sense(
            "understand",
            f"asking about {clip_path} (duration_s={state.get('duration_s')} "
            f"frames={state.get('frame_count')} context_chars={len(context)})",
        )
        try:
            answer = self._understand(**kwargs)
        except Exception as err:
            return self._drop(REASON_UNDERSTAND_FAILED, str(err))

        text = str(answer or "").strip()
        if not text:
            return self._drop(REASON_EMPTY_ANSWER, f"understand returned {answer!r}")

        try:
            # One positional argument, always: inject_text's speaking guard and
            # 3 s throttle are the harness's flood protection and stay in path.
            self._on_answer(text)
        except Exception as err:
            return self._drop(REASON_INJECT_FAILED, str(err))

        self._last_clip_ts = clip_ts
        self.answers += 1
        self.last_answer = text
        self._latched_reason = None
        self._sense("inject", f"injected the clip description (chars={len(text)})")
        return True

    # -- reading the retained state ----------------------------------------

    def _read_state(self) -> Mapping[str, Any] | None:
        """The clip payload, or ``None`` when it was already named and dropped."""
        try:
            state = self._get_clip_state()
        except Exception as err:
            self._drop(REASON_CLIP_STATE_FAILED, str(err))
            return None
        if state is None:
            self._drop(REASON_NO_CLIP_STATE, "no retained reachy/state/clip yet")
            return None
        if not isinstance(state, Mapping):
            self._drop(
                REASON_BAD_CLIP_STATE,
                f"clip state is {type(state).__name__}, not an object",
            )
            return None
        return state

    # -- context -----------------------------------------------------------

    def _resolve_context(self) -> str:
        """The text for this cycle — a raising provider degrades to the default."""
        provider = self._context
        if provider is None:
            return DEFAULT_CONTEXT
        if callable(provider):
            try:
                text = provider()
            except Exception as err:
                if not self._context_fault_reported:
                    self._context_fault_reported = True
                    self._sense(
                        "context",
                        f"dropped reason={REASON_CONTEXT_FAILED}: {err} "
                        "(using the default context)",
                    )
                return DEFAULT_CONTEXT
            self._context_fault_reported = False
        else:
            text = provider
        text = str(text or "").strip()
        return text or DEFAULT_CONTEXT

    # -- plumbing ----------------------------------------------------------

    def _drop(self, reason: str, detail: str) -> bool:
        """Count a drop and log it ONCE per condition (latched). Always ``False``."""
        self.drops += 1
        if reason != self._latched_reason:
            self._latched_reason = reason
            suffix = f": {detail}" if detail else ""
            self._sense("clip", f"dropped reason={reason}{suffix}")
        return False

    def _sense(self, event: str, detail: str) -> None:
        sensory_log.stage(STAGE, SOURCE, event, detail)

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


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def _clip_verdict(state: Mapping[str, Any]) -> tuple[str, str] | None:
    """``(reason, detail)`` when this payload yields no usable clip, else ``None``."""
    if not state.get("available"):
        return (_named_reason(state.get("reason")), "the runtime reports no clip")

    raw_path = state.get("path")
    path_text = str(raw_path).strip() if raw_path is not None else ""
    if not path_text:
        return (REASON_NO_CLIP_PATH, "available=true but path is null")

    try:
        size = Path(path_text).stat().st_size
    except OSError as err:
        return (REASON_CLIP_FILE_MISSING, f"{path_text} ({err.strerror or err})")
    if size <= 0:
        return (REASON_CLIP_FILE_EMPTY, f"{path_text} is 0 bytes")
    return None


def _named_reason(raw: object) -> str:
    """The runtime's own reason as a single grep-able token, or our fallback."""
    text = _WHITESPACE.sub("-", str(raw or "").strip())
    return text or REASON_CLIP_UNAVAILABLE


def _understand_callable(understand: Any) -> Callable[..., Any]:
    """Accept either a ``NovaOmni`` instance or a bare ``understand`` callable."""
    method = getattr(understand, "understand", None)
    if callable(method):
        return method
    if callable(understand):
        return understand
    raise ValueError(
        "VisionLeg requires a NovaOmni instance or an understand-shaped callable"
    )


def _accepts_event(understand: Callable[..., Any]) -> bool:
    """Does this ``understand`` take an ``event`` keyword?

    Today's :class:`~reachy_nova.nova_omni.NovaOmni` mints its own event id and
    takes no such parameter, so passing one would be a ``TypeError``. Asking
    the signature keeps the leg correct against both shapes instead of pinning
    it to one.
    """
    try:
        parameters = inspect.signature(understand).parameters
    except (TypeError, ValueError):
        return False
    for name, parameter in parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if name == "event" and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False
