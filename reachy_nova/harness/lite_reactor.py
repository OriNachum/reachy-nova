"""Lite reactor — Nova 2 Lite one-line reaction plans, on their own worker (t11).

Sonic owns the live voice; it is a big, slow-to-turn mind. Some cues (a pat,
a recognised face) deserve a reaction sharper than "the plain template line"
but cannot afford to wait on Sonic's own turn-taking. :class:`LiteReactor` is
the fast side channel for those cues: a rules.yaml entry that opts in
(``react: lite``, wired by task t13) hands the bus's rendered cue plus a
little context to Nova 2 Lite (``us.amazon.nova-2-lite-v1:0`` by default, the
same ``messages-v1`` request shape :func:`reachy_nova.nova_omni.NovaOmni._invoke`
uses) and gets back one short line: what to say (if anything), a
vocalisation, and a gesture from the engine's small library.

Everything here runs off the caller's thread and off Sonic's response loop
(spec c28/h20): :meth:`react` only enqueues and returns; ONE worker thread
drains a small bounded queue where a request arriving while the queue is
full evicts the oldest pending one (latest-wins — an old reaction is not
worth acting on once a newer cue has already superseded it). Bedrock's
``invoke_model`` is itself a *blocking* call, so even the worker thread does
not call it directly: each attempt runs on a short-lived helper thread and
the worker ``Event.wait()``s for it up to ``timeout_s`` — a call that never
returns just gets abandoned (it is a daemon thread; nothing joins it), and
the worker moves on to the next queued cue immediately. That is what makes a
30 s hang in Lite cost exactly one timed-out reaction instead of wedging
every reaction behind it.

Delivery is always exactly one call to the caller's ``deliver`` callable, and
the text handed to it is always **raw** — no voice/quiet markers. Task t13's
bus is the ONE place those markers get applied (spec c29); baking them in
here would double the markers once the bus adds its own, and this module has
no business knowing them anyway (``voice: none`` cues never reach here at
all — that filtering is also the bus's job, wired in t13).

Every reaction — planned or fallen back — costs exactly one
``[SENSE stage=react source=lite event=<n>]`` line, so "did Lite actually
answer, or did we fall back to the template, and why" is always answered
from the log alone.

stdlib + boto3 only; never imports ``reachy_mini``, ``paho`` or
``reachy_nova.nova_sonic``.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Any

import boto3

from .. import config
from ..sensory_log import stage as _sense

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Senselog identity                                                            #
# --------------------------------------------------------------------------- #

STAGE = "react"
SOURCE = "lite"

#: NAMED drop reasons — every fallback-to-template this module produces has
#: one, per the task's acceptance criteria.
REASON_TIMEOUT = "lite-timeout"
REASON_ERROR = "lite-error"
REASON_MALFORMED = "lite-malformed"
#: Not a fallback (nothing is delivered for the evicted cue) — the "latest
#: wins" queue superseding a still-pending, older request.
REASON_EVICTED = "lite-evicted"

# --------------------------------------------------------------------------- #
# Reply format                                                                 #
# --------------------------------------------------------------------------- #

VOCALIZE_CHOICES = ("chirp", "trill", "purr", "none")
GESTURE_CHOICES = ("nod", "shake", "antenna-sway", "none")

#: One line, e.g. ``say=Thank you, that feels nice! | vocalize=purr |
#: gesture=antenna-sway``. ``say`` is greedy-but-non-pipe so it may contain
#: spaces and punctuation, never a literal ``|``.
_PLAN_LINE_RE = re.compile(
    r"^say=(?P<say>[^|]*)\|\s*vocalize=(?P<vocalize>"
    + "|".join(VOCALIZE_CHOICES)
    + r")\s*\|\s*gesture=(?P<gesture>"
    + "|".join(GESTURE_CHOICES)
    + r")\s*$",
    re.IGNORECASE,
)

#: maxTokens for the Lite call — a one-line reply needs very little (live
#: finding: a 60-token reply round-tripped in ~1s median, 1.34s max).
MAX_TOKENS = 60

#: Default deadline for one Lite round trip (live finding, 2026-09-06: 0.97 /
#: 1.06 median / 1.34s max for a 60-token reply).
DEFAULT_TIMEOUT_S = 2.0

#: How often the worker wakes to re-check the stop signal while idle.
_POLL_S = 0.05

SYSTEM_PROMPT = (
    "You are the fast reflex layer of a small desk robot, working alongside "
    "its slower voice mind. You are given a brief cue about something that "
    "just happened, plus a little context, and decide on ONE short "
    "in-the-moment reaction. "
    "Reply with EXACTLY one line in this format and nothing else:\n"
    "say=<one short line or none> | vocalize=<chirp|trill|purr|none> | "
    "gesture=<nod|shake|antenna-sway|none>\n"
    "Reply with exactly one line in that format and nothing else."
)


@dataclass(frozen=True)
class ReactionPlan:
    """A parsed, well-formed Lite reply."""

    say: str | None
    vocalize: str
    gesture: str


def parse_plan(reply: str) -> ReactionPlan | None:
    """The first line of *reply* matching the plan format, or ``None``.

    A well-formed first line followed by a "*Reasoning:*" trailer (observed
    live, 2026-09-06) parses fine — later lines are never even looked at once
    a match is found. An empty reply, or one with no line in the required
    format, is malformed.
    """
    if not reply:
        return None
    for line in reply.splitlines():
        match = _PLAN_LINE_RE.match(line.strip())
        if not match:
            continue
        say_raw = match.group("say").strip()
        say = None if not say_raw or say_raw.lower() == "none" else say_raw
        return ReactionPlan(
            say=say,
            vocalize=match.group("vocalize").strip().lower(),
            gesture=match.group("gesture").strip().lower(),
        )
    return None


def render_reaction(cue: str, say: str) -> str:
    """The inject text Sonic receives for a plan with a non-``none`` ``say``."""
    return f"({cue} — you feel like saying: {say})"


def _build_user_text(cue: str, context: dict) -> str:
    """User content: the cue plus all four context parts, always labelled.

    Every label is always present (even when its part is missing from
    *context*) so a caller/test can rely on the four parts always appearing
    in the request, and so an absent part reads as "(none)" rather than
    silently vanishing from the prompt.
    """
    senses = context.get("senses") or []
    memory = context.get("memory") or ""
    mood = context.get("mood") or ""
    exchanges = context.get("exchanges") or []

    lines = [f"Cue: {cue}"]
    lines.append(
        "Recent senses: " + ("; ".join(str(s) for s in senses) if senses else "(none)")
    )
    lines.append(f"Today's memory: {memory or '(none)'}")
    lines.append(f"Mood: {mood or '(none)'}")
    if exchanges:
        rendered = "; ".join(
            f"{e.get('role', '?')}: {e.get('text', '')}" for e in exchanges
        )
    else:
        rendered = "(none)"
    lines.append(f"Recent exchanges: {rendered}")
    return "\n".join(lines)


def _request_body(user_text: str) -> dict:
    """The ``messages-v1`` request body — same schema as ``nova_omni._invoke``."""
    return {
        "schemaVersion": "messages-v1",
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "inferenceConfig": {
            "maxTokens": MAX_TOKENS,
            "topP": 0.9,
            "temperature": 0.7,
        },
    }


class _LiteTimeoutError(Exception):
    """Internal: the helper thread did not finish within ``timeout_s``."""


@dataclass
class _ReactItem:
    cue: str
    template: str
    deliver: Callable[[str], None]
    event: str


class LiteReactor:
    """Supervisor component: one worker thread, one bounded latest-wins queue.

    Args:
        client: an already-built bedrock-runtime client; built lazily on
            first use when omitted (never at construction — matches
            :class:`~reachy_nova.nova_omni.NovaOmni`).
        model_id: defaults to :func:`reachy_nova.config.lite_model_id`.
        timeout_s: seconds allowed for one Lite round trip before it is
            abandoned and the template is delivered instead. Default from the
            live probe in the plan's t7 (median ~1s, max 1.34s for a 60-token
            reply).
        context_provider: zero-arg callable returning
            ``{"senses": [...], "memory": "...", "mood": "...",
            "exchanges": [{"role", "text"}, ...]}``. Any key may be missing;
            a raising or ``None`` provider degrades to an empty context.
        on_gesture: called with the gesture name when a plan names one other
            than ``"none"``. May be ``None``.
        max_queue: bound on pending (not-yet-processed) requests. A new
            request past this bound evicts the oldest pending one.
        clock: zero-arg monotonic-seconds source, injectable for tests.
    """

    name = "lite_reactor"

    def __init__(
        self,
        client: Any = None,
        model_id: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        context_provider: Callable[[], dict] | None = None,
        on_gesture: Callable[[str], None] | None = None,
        max_queue: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self.model_id = model_id or config.lite_model_id()
        self.timeout_s = float(timeout_s)
        self._context_provider = context_provider
        self._on_gesture = on_gesture
        self._clock = clock

        self._queue: queue.Queue[_ReactItem | None] = queue.Queue(maxsize=max(1, max_queue))
        self._enqueue_lock = threading.Lock()
        self._event_seq = count(1)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._external_stop: threading.Event | None = None

        # Status surface (tests, /api).
        self.planned = 0
        self.fallbacks = 0
        self.evicted = 0

    # -- the boto3 client, built lazily -------------------------------------

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=config.region())
        return self._client

    # -- lifecycle -----------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._external_stop = stop_event
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lite-reactor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        return self._external_stop is not None and self._external_stop.is_set()

    # -- public API ------------------------------------------------------------

    def react(self, cue: str, template: str, deliver: Callable[[str], None]) -> None:
        """Enqueue a reaction request; returns immediately (sub-millisecond).

        A full queue evicts the oldest still-pending request (latest wins):
        that request's ``deliver`` is never called and one named drop line
        is logged for it instead.
        """
        if not callable(deliver):
            raise ValueError("LiteReactor.react requires a callable deliver")
        item = _ReactItem(
            cue=cue, template=template, deliver=deliver, event=str(next(self._event_seq))
        )
        with self._enqueue_lock:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                pass
            try:
                evicted = self._queue.get_nowait()
            except queue.Empty:
                evicted = None
            if evicted is not None:
                self.evicted += 1
                _sense(
                    STAGE,
                    SOURCE,
                    evicted.event,
                    f"dropped reason={REASON_EVICTED} cue={evicted.cue!r} "
                    "(superseded by a newer cue before it was processed)",
                )
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # Another producer refilled the freed slot first (concurrent
                # react() callers) — this request is the one that loses out.
                self.evicted += 1
                _sense(
                    STAGE,
                    SOURCE,
                    item.event,
                    f"dropped reason={REASON_EVICTED} cue={item.cue!r} "
                    "(queue still full)",
                )

    # -- the worker ------------------------------------------------------------

    def _run(self) -> None:
        while not self._should_stop():
            try:
                item = self._queue.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            if item is None:
                continue
            try:
                self._handle(item)
            except Exception as exc:  # noqa: BLE001 - _handle is already total
                logger.warning("lite reactor _handle raised: %s", exc)

    def _handle(self, item: _ReactItem) -> None:
        start = self._clock()
        context = self._safe_context()
        try:
            raw_text = self._call_lite(item.cue, context)
        except _LiteTimeoutError:
            self._fallback(item, REASON_TIMEOUT)
            return
        except Exception as exc:  # noqa: BLE001 - the call itself failed
            self._fallback(item, REASON_ERROR, str(exc))
            return

        plan = parse_plan(raw_text)
        if plan is None:
            self._fallback(item, REASON_MALFORMED, f"reply={raw_text!r}")
            return

        latency_ms = (self._clock() - start) * 1000.0
        if plan.say is not None:
            plan_text = render_reaction(item.cue, plan.say)
        else:
            plan_text = item.template

        # The senselog line is emitted BEFORE the deliver/gesture callbacks:
        # tests (and callers generally) learn a reaction happened by observing
        # those callbacks fire, and the log line must already be durable by
        # then rather than racing it on another thread.
        self.planned += 1
        say_disp = plan.say if plan.say is not None else "none"
        _sense(
            STAGE,
            SOURCE,
            item.event,
            f"planned say={say_disp} vocalize={plan.vocalize} "
            f"gesture={plan.gesture} latency={latency_ms:.0f}ms",
        )

        self._safe_deliver(item.deliver, plan_text)
        if plan.gesture != "none" and self._on_gesture is not None:
            self._safe_gesture(plan.gesture)

    def _fallback(self, item: _ReactItem, reason: str, detail: str = "") -> None:
        # Same ordering rule as the success path above: log before deliver.
        self.fallbacks += 1
        suffix = f": {detail}" if detail else ""
        _sense(STAGE, SOURCE, item.event, f"dropped reason={reason}{suffix}")
        self._safe_deliver(item.deliver, item.template)

    # -- the Lite call, bounded by a helper thread ------------------------------

    def _call_lite(self, cue: str, context: dict) -> str:
        """One Lite round trip, abandoned (not cancelled) past ``timeout_s``.

        ``invoke_model`` is a blocking boto3 call, so it runs on a short-lived
        daemon helper thread; this method (running on the reactor's OWN
        worker thread) just waits up to ``timeout_s`` for it. A call that
        never finishes is simply left running in the background — nothing
        joins it — so the worker is free to move straight on to the next
        queued cue.
        """
        body = _request_body(_build_user_text(cue, context))
        result: dict[str, Any] = {}
        done = threading.Event()

        def _work() -> None:
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id, body=json.dumps(body)
                )
                payload = json.loads(response["body"].read())
                text = payload["output"]["message"]["content"][0]["text"]
                result["text"] = text if isinstance(text, str) else ""
            except Exception as exc:  # noqa: BLE001 - handed back to the waiter
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(target=_work, name="lite-reactor-call", daemon=True).start()
        if not done.wait(self.timeout_s):
            raise _LiteTimeoutError(f"Lite call exceeded {self.timeout_s}s")
        if "error" in result:
            raise result["error"]
        return result.get("text", "")

    # -- defensive wrappers ------------------------------------------------------

    def _safe_context(self) -> dict:
        provider = self._context_provider
        if provider is None:
            return {}
        try:
            context = provider()
        except Exception as exc:  # noqa: BLE001 - a bad provider must not wedge the worker
            logger.warning("lite reactor context_provider raised: %s", exc)
            return {}
        return context if isinstance(context, dict) else {}

    def _safe_deliver(self, deliver: Callable[[str], None], text: str) -> None:
        try:
            deliver(text)
        except Exception as exc:  # noqa: BLE001 - a bad callback must not wedge the worker
            logger.warning("lite reactor deliver callback raised: %s", exc)

    def _safe_gesture(self, gesture: str) -> None:
        try:
            self._on_gesture(gesture)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            logger.warning("lite reactor on_gesture callback raised: %s", exc)
