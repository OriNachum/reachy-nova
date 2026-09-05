"""Tests for LiteReactor (t11): one worker, bounded latest-wins queue, timeout
and template fallback for a Nova 2 Lite one-line reaction plan.

Everything here drives :class:`~reachy_nova.harness.lite_reactor.LiteReactor`
with a plain fake bedrock-runtime client (``ScriptedLiteClient``) — no boto3
credentials, no network, no ``reachy_mini``. The four acceptance criteria in
the task map onto the sections below:

1. ``react()`` enqueues and returns immediately; the worker calls the
   injected client with a prompt carrying the cue and all four context
   parts, and delivers a well-formed plan (say text + gesture).
2. A client that hangs does not block ``react()`` or the delivery of OTHER
   cues; the hung plan times out at the configured deadline with a named
   drop, and the template is delivered instead.
3. The plan text is delivered raw; an empty, malformed, or
   "reasoning-trailer" reply parses correctly (first matching line only,
   or a template fallback when nothing matches).
4. A full queue evicts the oldest pending request (latest wins) with a
   named drop; ``stop()`` joins the worker within its timeout.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time

import pytest

from reachy_nova import config
from reachy_nova.harness import lite_reactor
from reachy_nova.harness.lite_reactor import (
    LiteReactor,
    ReactionPlan,
    parse_plan,
    render_reaction,
)

# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def _response_for(text: str) -> dict:
    payload = {"output": {"message": {"content": [{"text": text}]}}}
    return {"body": _FakeBody(json.dumps(payload).encode("utf-8"))}


class ScriptedLiteClient:
    """A fake bedrock-runtime client: per-call, scripted behavior.

    ``behaviors`` is a list; the Nth ``invoke_model`` call behaves per
    ``behaviors[N]`` (clamped to the last entry once exhausted). A behavior
    is one of:

    * ``{"reply": "<text>"}`` -- respond with this text.
    * ``{"error": exc}``      -- raise this exception.
    * ``{"hang": True}``      -- block forever. Runs on a daemon helper
      thread the reactor starts and abandons, so it never keeps the test
      process alive.
    """

    def __init__(self, behaviors: list[dict]) -> None:
        self._behaviors = behaviors
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def invoke_model(self, modelId, body):  # noqa: N803 - boto3's own casing
        with self._lock:
            idx = len(self.calls)
            self.calls.append({"modelId": modelId, "body": json.loads(body)})
        behavior = self._behaviors[min(idx, len(self._behaviors) - 1)]
        if behavior.get("hang"):
            threading.Event().wait()
        if "error" in behavior:
            raise behavior["error"]
        return _response_for(behavior.get("reply", ""))


class Recorder:
    """Records delivered values and signals an Event on every call."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.event = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, value: str) -> None:
        with self._lock:
            self.calls.append(value)
        self.event.set()


def _sense_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "nova.sensory"]


def _drops(caplog: pytest.LogCaptureFixture, reason: str) -> list[str]:
    return [line for line in _sense_lines(caplog) if f"reason={reason}" in line]


def _full_context() -> dict:
    return {
        "senses": ["a pat on the head", "quiet for a while"],
        "memory": "we talked about the weather earlier",
        "mood": "You are in a playful mood.",
        "exchanges": [
            {"role": "user", "text": "hey there"},
            {"role": "assistant", "text": "hi!"},
        ],
    }


@pytest.fixture()
def make_reactor():
    """Build+start a LiteReactor and guarantee it is stopped afterwards."""
    started: list[tuple[LiteReactor, threading.Event]] = []

    def _make(**kwargs) -> LiteReactor:
        stop_event = threading.Event()
        reactor = LiteReactor(**kwargs)
        reactor.start(stop_event)
        started.append((reactor, stop_event))
        return reactor

    yield _make
    for reactor, stop_event in started:
        stop_event.set()
        reactor.stop(timeout=2.0)


# --------------------------------------------------------------------------- #
# 0. Construction defaults                                                     #
# --------------------------------------------------------------------------- #


def test_component_shape_defaults() -> None:
    reactor = LiteReactor()
    assert reactor.name == "lite_reactor"
    assert reactor.timeout_s == lite_reactor.DEFAULT_TIMEOUT_S == 2.0
    assert reactor.model_id == config.lite_model_id()
    assert reactor._client is None, "the client must be built lazily, never at construction"
    assert reactor.is_alive() is False
    assert (reactor.planned, reactor.fallbacks, reactor.evicted) == (0, 0, 0)


# --------------------------------------------------------------------------- #
# 1. react() enqueues and returns immediately; full-context prompt; delivery   #
# --------------------------------------------------------------------------- #


def test_react_returns_immediately_and_delivers_well_formed_plan(
    make_reactor, caplog: pytest.LogCaptureFixture
) -> None:
    client = ScriptedLiteClient(
        [{"reply": "say=Thank you, that feels nice! | vocalize=purr | gesture=antenna-sway"}]
    )
    gestures: list[str] = []
    reactor = make_reactor(
        client=client,
        model_id="test-lite-model",
        timeout_s=1.0,
        context_provider=_full_context,
        on_gesture=gestures.append,
    )
    deliver = Recorder()

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        start = time.monotonic()
        reactor.react("someone is petting you", "You feel a gentle pat.", deliver)
        elapsed = time.monotonic() - start
        assert deliver.event.wait(2.0), "delivery never happened"

    assert elapsed < 0.05, f"react() blocked the caller for {elapsed}s"
    assert deliver.calls == [
        "(someone is petting you — you feel like saying: Thank you, that feels nice!)"
    ]
    assert gestures == ["antenna-sway"]
    assert reactor.planned == 1
    assert reactor.fallbacks == 0

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["modelId"] == "test-lite-model"
    body = call["body"]
    assert body["schemaVersion"] == "messages-v1"
    assert body["inferenceConfig"]["maxTokens"] == lite_reactor.MAX_TOKENS
    user_text = body["messages"][0]["content"][0]["text"]
    assert "someone is petting you" in user_text
    assert "a pat on the head" in user_text  # senses
    assert "weather earlier" in user_text  # memory
    assert "playful mood" in user_text  # mood
    assert "hey there" in user_text and "hi!" in user_text  # exchanges

    assert any(
        "planned" in line and "gesture=antenna-sway" in line and "latency=" in line
        for line in _sense_lines(caplog)
    )


def test_react_with_missing_context_keys_still_carries_the_cue(
    make_reactor, caplog: pytest.LogCaptureFixture
) -> None:
    """``context_provider`` may omit any key — the call must never raise."""
    client = ScriptedLiteClient([{"reply": "say=none | vocalize=none | gesture=none"}])
    reactor = make_reactor(
        client=client, model_id="m", timeout_s=1.0, context_provider=lambda: {"mood": "calm"}
    )
    deliver = Recorder()
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        reactor.react("a face was recognized", "Someone is here.", deliver)
        assert deliver.event.wait(2.0)

    user_text = client.calls[0]["body"]["messages"][0]["content"][0]["text"]
    assert "a face was recognized" in user_text
    assert "calm" in user_text
    assert reactor.planned == 1


# --------------------------------------------------------------------------- #
# 2. A hung Lite call times out without blocking react() or other cues        #
# --------------------------------------------------------------------------- #


def test_hung_lite_call_times_out_without_blocking_other_cues(
    make_reactor, caplog: pytest.LogCaptureFixture
) -> None:
    client = ScriptedLiteClient(
        [
            {"hang": True},
            {"reply": "say=Hello there! | vocalize=none | gesture=none"},
        ]
    )
    reactor = make_reactor(client=client, model_id="m", timeout_s=0.2)
    deliver_a = Recorder()
    deliver_b = Recorder()

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        start = time.monotonic()
        reactor.react("cue-a", "template-a", deliver_a)
        elapsed_react = time.monotonic() - start
        reactor.react("cue-b", "template-b", deliver_b)

        assert deliver_a.event.wait(2.0), "fallback for the hung cue never arrived"
        assert deliver_b.event.wait(2.0), "cue-b was blocked behind the hung cue-a call"

    assert elapsed_react < 0.05, f"react() blocked the caller for {elapsed_react}s"
    assert deliver_a.calls == ["template-a"]
    assert deliver_b.calls == ["(cue-b — you feel like saying: Hello there!)"]
    assert reactor.fallbacks == 1
    assert reactor.planned == 1
    assert _drops(caplog, lite_reactor.REASON_TIMEOUT), _sense_lines(caplog)


def test_lite_call_raising_falls_back_with_named_error_drop(
    make_reactor, caplog: pytest.LogCaptureFixture
) -> None:
    client = ScriptedLiteClient([{"error": RuntimeError("bedrock unavailable")}])
    reactor = make_reactor(client=client, model_id="m", timeout_s=1.0)
    deliver = Recorder()

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        reactor.react("cue", "template text", deliver)
        assert deliver.event.wait(2.0)

    assert deliver.calls == ["template text"]
    assert reactor.fallbacks == 1
    assert _drops(caplog, lite_reactor.REASON_ERROR), _sense_lines(caplog)


# --------------------------------------------------------------------------- #
# 3. Raw delivery, parsing (reasoning trailer / malformed / empty)             #
# --------------------------------------------------------------------------- #


def test_parse_plan_returns_first_matching_line_ignoring_reasoning_trailer() -> None:
    reply = (
        "say=Thank you, that feels nice! | vocalize=purr | gesture=antenna-sway\n"
        "\n"
        "*Reasoning:* The user petted the robot, so a warm response fits.\n"
    )
    assert parse_plan(reply) == ReactionPlan(
        say="Thank you, that feels nice!", vocalize="purr", gesture="antenna-sway"
    )


def test_parse_plan_handles_say_none_and_is_case_insensitive() -> None:
    assert parse_plan("SAY=none | VOCALIZE=Chirp | GESTURE=NOD") == ReactionPlan(
        say=None, vocalize="chirp", gesture="nod"
    )


def test_parse_plan_rejects_fully_malformed_reply() -> None:
    assert parse_plan("I think I'll just purr contentedly.") is None


def test_parse_plan_rejects_empty_reply() -> None:
    assert parse_plan("") is None


def test_render_reaction_shape() -> None:
    assert (
        render_reaction("someone is petting you", "Thank you, that feels nice!")
        == "(someone is petting you — you feel like saying: Thank you, that feels nice!)"
    )


def test_malformed_reply_falls_back_to_template(
    make_reactor, caplog: pytest.LogCaptureFixture
) -> None:
    client = ScriptedLiteClient([{"reply": "I think I'll just purr contentedly."}])
    reactor = make_reactor(client=client, model_id="m", timeout_s=1.0)
    deliver = Recorder()

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        reactor.react("cue", "template text", deliver)
        assert deliver.event.wait(2.0)

    assert deliver.calls == ["template text"]
    assert reactor.fallbacks == 1
    assert _drops(caplog, lite_reactor.REASON_MALFORMED), _sense_lines(caplog)


def test_empty_reply_falls_back_to_template_as_malformed(
    make_reactor, caplog: pytest.LogCaptureFixture
) -> None:
    client = ScriptedLiteClient([{"reply": ""}])
    reactor = make_reactor(client=client, model_id="m", timeout_s=1.0)
    deliver = Recorder()

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        reactor.react("cue", "template text", deliver)
        assert deliver.event.wait(2.0)

    assert deliver.calls == ["template text"]
    assert _drops(caplog, lite_reactor.REASON_MALFORMED), _sense_lines(caplog)


def test_say_none_with_gesture_delivers_template_but_still_fires_gesture(
    make_reactor, caplog: pytest.LogCaptureFixture
) -> None:
    client = ScriptedLiteClient([{"reply": "say=none | vocalize=none | gesture=nod"}])
    gestures: list[str] = []
    reactor = make_reactor(client=client, model_id="m", timeout_s=1.0, on_gesture=gestures.append)
    deliver = Recorder()

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        reactor.react("cue", "template text", deliver)
        assert deliver.event.wait(2.0)

    assert deliver.calls == ["template text"], "say=none must deliver the ORIGINAL template"
    assert gestures == ["nod"]
    assert reactor.planned == 1, "a well-formed reply is planned, not a fallback"
    assert reactor.fallbacks == 0


def test_gesture_not_called_when_plan_gesture_is_none(make_reactor) -> None:
    client = ScriptedLiteClient([{"reply": "say=Hi! | vocalize=none | gesture=none"}])
    gestures: list[str] = []
    reactor = make_reactor(client=client, model_id="m", timeout_s=1.0, on_gesture=gestures.append)
    deliver = Recorder()
    reactor.react("cue", "template text", deliver)
    assert deliver.event.wait(2.0)
    assert gestures == []
    assert deliver.calls == ["(cue — you feel like saying: Hi!)"]


def test_on_gesture_none_is_tolerated(make_reactor) -> None:
    """No gesture callable wired at all must never raise."""
    client = ScriptedLiteClient([{"reply": "say=Hi! | vocalize=none | gesture=nod"}])
    reactor = make_reactor(client=client, model_id="m", timeout_s=1.0, on_gesture=None)
    deliver = Recorder()
    reactor.react("cue", "template text", deliver)
    assert deliver.event.wait(2.0)
    assert deliver.calls == ["(cue — you feel like saying: Hi!)"]


# --------------------------------------------------------------------------- #
# 4. Full-queue eviction and stop()                                            #
# --------------------------------------------------------------------------- #


def test_full_queue_evicts_oldest_pending_request_with_named_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # No start(): the worker never drains the queue, so react() calls alone
    # deterministically exercise the bounded latest-wins behavior.
    reactor = LiteReactor(
        client=ScriptedLiteClient([{"hang": True}]), model_id="m", timeout_s=100.0, max_queue=2
    )
    d1, d2, d3 = Recorder(), Recorder(), Recorder()

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        reactor.react("cue-1", "t1", d1)
        reactor.react("cue-2", "t2", d2)
        reactor.react("cue-3", "t3", d3)  # queue maxsize=2 -> evicts cue-1

    assert reactor.evicted == 1
    evicted_lines = _drops(caplog, lite_reactor.REASON_EVICTED)
    assert evicted_lines and "cue-1" in evicted_lines[0]
    assert d1.calls == [] and d2.calls == [] and d3.calls == [], (
        "eviction must not itself deliver anything for either request"
    )

    remaining = []
    while True:
        try:
            remaining.append(reactor._queue.get_nowait().cue)
        except queue.Empty:
            break
    assert remaining == ["cue-2", "cue-3"]


def test_stop_joins_the_worker_within_its_timeout() -> None:
    stop_event = threading.Event()
    reactor = LiteReactor(
        client=ScriptedLiteClient([{"reply": "say=none | vocalize=none | gesture=none"}]),
        model_id="m",
    )
    reactor.start(stop_event)
    assert reactor.is_alive()

    reactor.stop(timeout=2.0)

    assert not reactor.is_alive()


def test_stop_is_idempotent_and_a_second_start_is_a_no_op_while_alive() -> None:
    stop_event = threading.Event()
    reactor = LiteReactor(client=ScriptedLiteClient([{"reply": ""}]), model_id="m")
    reactor.start(stop_event)
    first_thread = reactor._thread
    reactor.start(stop_event)  # already alive -> no-op, same thread
    assert reactor._thread is first_thread
    reactor.stop(timeout=2.0)
    reactor.stop(timeout=2.0)  # idempotent
    assert not reactor.is_alive()


def test_recent_lines_are_fed_back_so_the_same_cue_varies():
    """Three pats in a row got 'Ah, that feels nice!' three times on the robot (2026-09-06)."""
    from reachy_nova.harness.lite_reactor import _build_user_text

    text = _build_user_text("(someone is petting you)", {}, ["Ah, that feels nice!", "Thank you!"])
    assert "must NOT reuse" in text
    assert "Ah, that feels nice!" in text and "Thank you!" in text
    assert "must NOT reuse" not in _build_user_text("(someone is petting you)", {}, [])
