"""The must-deliver inject path (t1) — a result that must not be lost.

Two live drops on 2026-09-06 motivate this file:

1. **Throttled away.** The browser's *result* inject ("Your web browsing
   finished. Tell the user what you found: ...") arrives milliseconds after
   its own *progress* inject, so the 3 s anti-flood throttle ate it —
   ``dropped reason=throttled interval=0.0s``, three times in one journal.
   The throttle exists to stop a *flood* of body cues; a caller that says
   "this one is the answer" is not a flood.
2. **Dropped into a restart.** A session rotation takes 1–3 s, during which
   ``inject_text`` answers ``dropped-inactive`` and the text is simply gone.

``must_deliver=True`` fixes both: throttle-exempt, parked under the caller's
own sense class if the model happens to be speaking, and queued in a small
bounded FIFO (drained into the next live session, with its age in the text)
if no session is up. A plain inject keeps every existing behaviour.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time

import pytest

from reachy_nova import nova_sonic
from reachy_nova.nova_sonic import NovaSonic

# --------------------------------------------------------------------------
# fakes — the same shapes tests/test_sonic_rotation.py and
# tests/test_harness_deferred_cues.py drive NovaSonic with (no Bedrock).
# --------------------------------------------------------------------------


class _FakeInputStream:
    def __init__(self, sent: list[dict]):
        self._sent = sent

    async def send(self, chunk) -> None:
        self._sent.append(json.loads(chunk.value.bytes_.decode("utf-8"))["event"])

    async def close(self) -> None:
        pass


class _FakeStream:
    """Accepts every input event; answers nothing unless scripted."""

    def __init__(self, events: list[dict] | None = None):
        self.sent: list[dict] = []
        self.input_stream = _FakeInputStream(self.sent)
        self._events = list(events or [])

    async def await_output(self):
        return (None, self)

    async def receive(self):
        await asyncio.sleep(0)
        if not self._events:
            raise ConnectionError("end of script")
        return _chunk(self._events.pop(0))


def _chunk(event: dict):
    class _Value:
        bytes_ = json.dumps({"event": event}).encode("utf-8")

    class _Chunk:
        value = _Value()

    return _Chunk()


class _FakeClient:
    """Hands out a fresh ``_FakeStream`` per session and keeps them all."""

    def __init__(self) -> None:
        self.streams: list[_FakeStream] = []

    async def invoke_model_with_bidirectional_stream(self, _operation_input):
        stream = _FakeStream()
        self.streams.append(stream)
        return stream


class _NoSleepAsyncio:
    """``asyncio`` proxy whose ``sleep`` returns immediately."""

    def __getattr__(self, name):  # pragma: no cover - trivial delegation
        return getattr(asyncio, name)

    async def sleep(self, delay: float = 0, result=None):
        await asyncio.sleep(0)
        return result


class _LiveLoop:
    """A real asyncio loop on its own thread — what ``inject_text`` schedules on.

    ``inject_text``'s success path is ``run_coroutine_threadsafe``, so the
    only way to assert "sent" *and* see what reached the wire is to give it a
    loop that actually runs.
    """

    def __enter__(self) -> asyncio.AbstractEventLoop:
        self._ready = threading.Event()
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(5), "test loop never started"
        return self.loop

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self._ready.set)
        self.loop.run_forever()

    def __exit__(self, *exc) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_SENSE_LINE_RE = re.compile(
    r"^\[SENSE stage=(?P<stage>\S+) source=(?P<source>\S+) event=(?P<event>\S+)\] "
    r"(?P<detail>.*)$"
)


def _sense_details(caplog) -> list[str]:
    out = []
    for record in caplog.records:
        if record.name != "nova.sensory":
            continue
        match = _SENSE_LINE_RE.match(record.getMessage())
        assert match is not None, f"unparseable sensory line: {record.getMessage()!r}"
        out.append(match.group("detail"))
    return out


def _live_sonic(loop, **kwargs) -> NovaSonic:
    """An active NovaSonic wired to a real loop and a fake stream."""
    sonic = NovaSonic(system_prompt="You are Nova.", **kwargs)
    sonic._loop = loop
    sonic._stream = _FakeStream()
    sonic._active = True
    return sonic


def _ready_sonic(**kwargs) -> NovaSonic:
    """Active with a stub loop — enough for the guard/park branches."""
    sonic = NovaSonic(system_prompt="You are Nova.", **kwargs)
    sonic._active = True
    sonic._loop = object()
    return sonic


def _texts(stream: _FakeStream) -> list[str]:
    return [e["textInput"]["content"] for e in stream.sent if "textInput" in e]


def _wait_for_texts(stream: _FakeStream, count: int, timeout: float = 5.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        texts = _texts(stream)
        if len(texts) >= count:
            return texts
        time.sleep(0.01)
    return _texts(stream)


def _drive_responses(sonic: NovaSonic, events: list[dict]) -> list[dict]:
    """Run ``_process_responses`` over a scripted event list; return what was sent."""
    stream = _FakeStream(events)
    sonic._stream = stream
    sonic._active = True

    async def _go() -> None:
        sonic._loop = asyncio.get_running_loop()
        sonic._inject_lock = asyncio.Lock()
        await sonic._process_responses()
        for _ in range(20):
            await asyncio.sleep(0)

    asyncio.run(_go())
    return stream.sent


def _user_texts(sent: list[dict]) -> list[str]:
    return [e["textInput"]["content"] for e in sent if "textInput" in e]


@pytest.fixture
def nosleep(monkeypatch):
    """Skip ``_start_session``'s settle pause without touching anything else."""
    monkeypatch.setattr(nova_sonic, "asyncio", _NoSleepAsyncio())


# ==========================================================================
# 1. the throttle exemption — the live browse-result drop
# ==========================================================================


class TestThrottleExemption:
    def test_a_must_deliver_inject_right_after_a_progress_inject_is_sent(self):
        with _LiveLoop() as loop:
            sonic = _live_sonic(loop)

            assert sonic.inject_text("I'm looking that up now.") == "sent"
            # 0 ms later — exactly the shape the robot journal showed.
            assert (
                sonic.inject_text(
                    "Your web browsing finished. Tell the user what you found: rain.",
                    sense_class="browse",
                    must_deliver=True,
                )
                == "sent"
            )

            texts = _wait_for_texts(sonic._stream, 2)
            assert len(texts) == 2
            assert "rain" in texts[1]

    def test_a_plain_inject_in_the_same_window_is_still_throttled(self):
        with _LiveLoop() as loop:
            sonic = _live_sonic(loop)

            assert sonic.inject_text("first") == "sent"
            assert _wait_for_texts(sonic._stream, 1) == ["first"]
            assert sonic.inject_text("second") == "dropped-throttled"

    def test_a_must_deliver_send_rearms_the_throttle_for_plain_injects(self):
        with _LiveLoop() as loop:
            sonic = _live_sonic(loop)
            sonic._last_inject_time = 0.0

            assert sonic.inject_text("a result", must_deliver=True) == "sent"
            assert _wait_for_texts(sonic._stream, 1) == ["a result"]
            assert sonic.inject_text("a body cue") == "dropped-throttled"

    def test_must_deliver_does_not_bypass_the_speaking_guard(self):
        sonic = _ready_sonic()
        sonic._speaking = True

        assert sonic.inject_text("a result", sense_class="browse", must_deliver=True) == (
            "deferred"
        )

    def test_a_plain_inject_is_unchanged(self, caplog):
        with _LiveLoop() as loop:
            sonic = _live_sonic(loop)
            sonic._last_inject_time = nova_sonic.time.time()

            with caplog.at_level(logging.INFO, logger="nova.sensory"):
                assert sonic.inject_text("a throttled cue", sense_class="pat") == (
                    "dropped-throttled"
                )

            assert "dropped reason=throttled" in _sense_details(caplog)[0]


# ==========================================================================
# 2. the deferred slot — a result gets its own class
# ==========================================================================


class TestMustDeliverIsParkedUnderItsOwnClass:
    def test_a_later_unclassed_cue_cannot_overwrite_the_parked_result(self):
        sonic = _ready_sonic()
        sonic._speaking = True

        sonic.inject_text(
            "Your web browsing finished. Tell the user what you found: rain.",
            sense_class="browse",
            must_deliver=True,
        )
        sonic.inject_text("something else happened")  # unclassed -> 'other'

        classes = {c.sense_class for c in sonic._deferred.drain()}
        assert classes == {"browse", "other"}

    def test_the_parked_result_is_drained_when_the_utterance_ends(self):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        sonic._loop = object()
        sonic._speaking = True

        sonic.inject_text(
            "Your web browsing finished. Tell the user what you found: rain.",
            sense_class="browse",
            must_deliver=True,
        )
        sonic.inject_text("something else happened")

        sent = _drive_responses(sonic, [
            {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
        ])

        texts = _user_texts(sent)
        assert any("rain" in t for t in texts), texts
        assert any("something else happened" in t for t in texts), texts
        assert sonic._deferred.pending() == 0


# ==========================================================================
# 3. the restart retry queue
# ==========================================================================


class TestQueuedWhileInactive:
    def test_an_inactive_must_deliver_inject_is_queued_not_dropped(self):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = False
        sonic._loop = None

        assert sonic.inject_text("a browse result", must_deliver=True) == "queued-inactive"
        assert len(sonic._must_deliver_queue) == 1

    def test_a_plain_inject_while_inactive_is_still_dropped_and_never_queued(self):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = False
        sonic._loop = None

        assert sonic.inject_text("a body cue") == "dropped-inactive"
        assert len(sonic._must_deliver_queue) == 0

    def test_the_queue_is_bounded_and_evicts_the_oldest(self, caplog):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = False
        sonic._loop = None

        for i in range(nova_sonic.MUST_DELIVER_QUEUE_MAX + 1):
            assert sonic.inject_text(f"result {i}", must_deliver=True) == "queued-inactive"

        assert len(sonic._must_deliver_queue) == nova_sonic.MUST_DELIVER_QUEUE_MAX
        texts = [item.text for item in sonic._must_deliver_queue]
        assert "result 0" not in texts, "the OLDEST entry is the one evicted"
        assert texts[-1] == f"result {nova_sonic.MUST_DELIVER_QUEUE_MAX}"

    def test_the_queue_drains_into_the_next_live_session_with_its_age(self, nosleep):
        sonic = NovaSonic(system_prompt="You are Nova.")
        client = _FakeClient()
        # type: ignore[method-assign]
        sonic._init_client = lambda: setattr(sonic, "_client", client)
        sonic._active = False
        sonic._loop = None

        assert (
            sonic.inject_text(
                "Your web browsing finished. Tell the user what you found: rain.",
                sense_class="browse",
                must_deliver=True,
            )
            == "queued-inactive"
        )
        assert not client.streams, "nothing may reach the wire while inactive"

        asyncio.run(sonic._start_session())

        texts = _texts(client.streams[0])
        matching = [t for t in texts if "rain" in t]
        assert len(matching) == 1, texts
        assert "ago" in matching[0]
        assert not sonic._must_deliver_queue

    def test_a_drained_item_is_sent_exactly_once(self, nosleep):
        sonic = NovaSonic(system_prompt="You are Nova.")
        client = _FakeClient()
        # type: ignore[method-assign]
        sonic._init_client = lambda: setattr(sonic, "_client", client)
        sonic._active = False
        sonic._loop = None
        sonic.inject_text("a browse result", must_deliver=True)

        asyncio.run(sonic._start_session())
        sonic._active = False
        asyncio.run(sonic._start_session())

        first = [t for t in _texts(client.streams[0]) if "a browse result" in t]
        second = [t for t in _texts(client.streams[1]) if "a browse result" in t]
        assert len(first) == 1
        assert second == []

    def test_the_drain_is_ordered_and_logged(self, nosleep, caplog):
        sonic = NovaSonic(system_prompt="You are Nova.")
        client = _FakeClient()
        # type: ignore[method-assign]
        sonic._init_client = lambda: setattr(sonic, "_client", client)
        sonic._active = False
        sonic._loop = None
        sonic.inject_text("first result", must_deliver=True)
        sonic.inject_text("second result", must_deliver=True)

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            asyncio.run(sonic._start_session())

        texts = [t for t in _texts(client.streams[0]) if "result" in t]
        assert len(texts) == 2
        assert "first result" in texts[0]
        assert "second result" in texts[1]

        drained = [d for d in _sense_details(caplog) if d.startswith("drained-queued")]
        assert len(drained) == 2
        assert all("age=" in d for d in drained)


# ==========================================================================
# 4. failed and stale sends return the answer to the queue (PR #26 review)
# ==========================================================================


def _queued_sonic(*texts: str) -> NovaSonic:
    """An inactive NovaSonic with *texts* already on the must-deliver queue."""
    sonic = NovaSonic(system_prompt="You are Nova.")
    sonic._active = False
    sonic._loop = None
    for text in texts:
        assert sonic.inject_text(text, sense_class="browse", must_deliver=True) == (
            "queued-inactive"
        )
    return sonic


class _ScriptedWire:
    """A ``_send`` replacement that can be told to fail on a given event type."""

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.sent: list[dict] = []

    async def __call__(self, event: dict) -> None:
        self.sent.append(event)
        if self.fail_on and self.fail_on in event:
            raise ConnectionError("the wire died")

    def texts(self) -> list[str]:
        return [e["textInput"]["content"] for e in self.sent if "textInput" in e]


def _run_send(sonic: NovaSonic, text: str, gen: int, **kwargs) -> bool:
    async def _go() -> bool:
        sonic._inject_lock = asyncio.Lock()
        return await sonic._send_user_text(text, gen, **kwargs)

    return asyncio.run(_go())


def _run_drain(sonic: NovaSonic) -> None:
    async def _go() -> None:
        sonic._inject_lock = asyncio.Lock()
        await sonic._drain_must_deliver()

    asyncio.run(_go())


class TestFailedSendReturnsTheAnswerToTheQueue:
    def test_a_failed_must_deliver_send_requeues_the_original_text(self, caplog):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        wire = _ScriptedWire(fail_on="textInput")
        sonic._send = wire  # type: ignore[method-assign]

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            ok = _run_send(
                sonic,
                "the browse answer",
                sonic._session_gen,
                must_deliver=True,
                sense_class="browse",
            )

        assert ok is False
        assert [item.text for item in sonic._must_deliver_queue] == ["the browse answer"]
        details = _sense_details(caplog)
        assert any(
            "dropped reason=send-failed" in d and "requeued=true" in d for d in details
        ), details

    def test_the_requeued_answer_is_sent_once_with_its_age_on_the_next_drain(self):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        sonic._send = _ScriptedWire(fail_on="textInput")  # type: ignore[method-assign]
        assert (
            _run_send(sonic, "the browse answer", sonic._session_gen, must_deliver=True)
            is False
        )

        good = _ScriptedWire()
        sonic._send = good  # type: ignore[method-assign]
        _run_drain(sonic)

        matching = [t for t in good.texts() if "the browse answer" in t]
        assert len(matching) == 1, good.texts()
        assert "ago" in matching[0], matching
        assert not sonic._must_deliver_queue

        # ...and exactly once: a second drain has nothing left to send.
        _run_drain(sonic)
        assert len([t for t in good.texts() if "the browse answer" in t]) == 1

    def test_a_plain_send_that_fails_is_unchanged_and_never_queues(self):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        sonic._send = _ScriptedWire(fail_on="textInput")  # type: ignore[method-assign]

        assert _run_send(sonic, "a body cue", sonic._session_gen) is False
        assert not sonic._must_deliver_queue

    def test_a_successful_send_returns_true_and_queues_nothing(self):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        sonic._send = _ScriptedWire()  # type: ignore[method-assign]

        assert (
            _run_send(sonic, "the answer", sonic._session_gen, must_deliver=True) is True
        )
        assert not sonic._must_deliver_queue


class TestStaleGenerationReturnsTheAnswerToTheQueue:
    def test_a_must_deliver_send_under_a_rotated_generation_requeues(self, caplog):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        wire = _ScriptedWire()
        sonic._send = wire  # type: ignore[method-assign]
        gen = sonic._session_gen
        sonic._session_gen += 1  # the session rotated before the coroutine ran

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            ok = _run_send(sonic, "the browse answer", gen, must_deliver=True)

        assert ok is False
        assert wire.sent == [], "nothing may reach a rotated session"
        assert [item.text for item in sonic._must_deliver_queue] == ["the browse answer"]
        details = _sense_details(caplog)
        assert any(
            "dropped reason=stale-session" in d and "requeued=true" in d for d in details
        ), details

    def test_a_plain_send_under_a_rotated_generation_is_still_discarded(self):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        wire = _ScriptedWire()
        sonic._send = wire  # type: ignore[method-assign]
        gen = sonic._session_gen
        sonic._session_gen += 1

        assert _run_send(sonic, "a body cue", gen) is False
        assert wire.sent == []
        assert not sonic._must_deliver_queue

    def test_the_active_inject_path_carries_the_must_deliver_marker(self):
        """``inject_text`` still answers "sent", but the coroutine can retry."""
        with _LiveLoop() as loop:
            sonic = _live_sonic(loop)
            wire = _ScriptedWire(fail_on="textInput")
            sonic._send = wire  # type: ignore[method-assign]

            assert (
                sonic.inject_text("the browse answer", sense_class="browse", must_deliver=True)
                == "sent"
            )

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not sonic._must_deliver_queue:
                time.sleep(0.01)
            assert [item.text for item in sonic._must_deliver_queue] == ["the browse answer"]

    def test_a_first_attempt_send_carries_the_bare_text(self):
        """The age annotation is applied on a drain only, never on attempt one."""
        with _LiveLoop() as loop:
            sonic = _live_sonic(loop)
            assert sonic.inject_text("the browse answer", must_deliver=True) == "sent"
            assert _wait_for_texts(sonic._stream, 1) == ["the browse answer"]


class TestDeferredMustDeliverCuesSurviveARotation:
    def _speaking_sonic(self) -> NovaSonic:
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        sonic._loop = object()
        sonic._speaking = True
        return sonic

    def test_a_deferred_cue_remembers_that_it_must_be_delivered(self):
        sonic = self._speaking_sonic()
        sonic.inject_text("the browse answer", sense_class="browse", must_deliver=True)
        sonic.inject_text("a pat", sense_class="pat")

        cues = {c.sense_class: c for c in sonic._deferred.drain()}
        assert cues["browse"].must_deliver is True
        assert cues["pat"].must_deliver is False

    def test_a_deferred_must_deliver_cue_drained_under_a_rotated_gen_requeues(self):
        sonic = self._speaking_sonic()
        sonic.inject_text("the browse answer", sense_class="browse", must_deliver=True)
        sonic.inject_text("a pat", sense_class="pat")
        wire = _ScriptedWire()
        sonic._send = wire  # type: ignore[method-assign]

        gen = sonic._session_gen

        async def _go() -> None:
            sonic._loop = asyncio.get_running_loop()
            sonic._inject_lock = asyncio.Lock()
            sonic._session_gen += 1  # rotated before the drain ran
            await sonic._drain_deferred(gen)

        asyncio.run(_go())

        assert wire.sent == [], "a rotated session takes nothing"
        assert [item.text for item in sonic._must_deliver_queue] == ["the browse answer"]
        assert sonic._deferred.pending() == 0, "the plain cue is discarded, not left parked"

    def test_a_deferred_must_deliver_cue_whose_send_fails_requeues(self):
        sonic = self._speaking_sonic()
        sonic.inject_text("the browse answer", sense_class="browse", must_deliver=True)
        sonic._send = _ScriptedWire(fail_on="textInput")  # type: ignore[method-assign]

        gen = sonic._session_gen

        async def _go() -> None:
            sonic._loop = asyncio.get_running_loop()
            sonic._inject_lock = asyncio.Lock()
            await sonic._drain_deferred(gen)

        asyncio.run(_go())

        assert [item.text for item in sonic._must_deliver_queue] == ["the browse answer"], (
            "the ORIGINAL text goes back on the queue, not the age-rendered one"
        )


class TestDrainCountsOnlyRealDeliveries:
    def test_a_failing_drain_does_not_rearm_the_throttle(self):
        sonic = _queued_sonic("the browse answer")
        sonic._active = True
        sonic._send = _ScriptedWire(fail_on="textInput")  # type: ignore[method-assign]
        sonic._last_inject_time = 0.0

        _run_drain(sonic)

        assert sonic._last_inject_time == 0.0, "nothing was delivered, nothing to re-arm"
        assert [item.text for item in sonic._must_deliver_queue] == ["the browse answer"]

    def test_an_item_that_keeps_failing_is_dropped_as_exhausted(self, caplog):
        sonic = _queued_sonic("the browse answer")
        sonic._active = True
        wire = _ScriptedWire(fail_on="textInput")
        sonic._send = wire  # type: ignore[method-assign]

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            for _ in range(nova_sonic.MUST_DELIVER_MAX_ATTEMPTS):
                _run_drain(sonic)

        assert not sonic._must_deliver_queue
        details = _sense_details(caplog)
        exhausted = [d for d in details if "dropped reason=must-deliver-exhausted" in d]
        assert len(exhausted) == 1, details

        attempts = len(wire.texts())
        _run_drain(sonic)
        assert len(wire.texts()) == attempts, "an exhausted item is never sent again"

    def test_a_recovering_wire_delivers_before_exhaustion(self):
        sonic = _queued_sonic("the browse answer")
        sonic._active = True
        sonic._send = _ScriptedWire(fail_on="textInput")  # type: ignore[method-assign]
        _run_drain(sonic)
        _run_drain(sonic)

        good = _ScriptedWire()
        sonic._send = good  # type: ignore[method-assign]
        _run_drain(sonic)

        assert len([t for t in good.texts() if "the browse answer" in t]) == 1
        assert not sonic._must_deliver_queue
