"""Deferred body cues: parked while Sonic generates, delivered with their age (t9).

Today a body cue that arrives while Nova Sonic is generating an utterance is
dropped outright — ``inject_text``'s speaking guard logs
``dropped reason=speaking`` and returns — so a pat during a ten-second reply
never reaches the mind at all. It is lost, not late.

:mod:`reachy_nova.harness.deferred_cues` turns that fate into a deferral: one
latest-wins slot per sense class (``pat``, ``face``, ``sound``, ``vision``,
plus ``other`` for anything unnamed), a short TTL, and a drain the moment the
utterance ends. The two halves are tested separately:

1. **The slot** (``DeferredCues``) — pure, clock-injectable, no Bedrock:
   replace-within-a-class, independence between classes, arrival-order drain,
   TTL expiry with a named senselog line, and the age-aware :meth:`render`.
2. **The wiring** (``NovaSonic``) — ``inject_text``'s speaking branch fills
   the slot instead of dropping, and the end of the utterance (both the
   ASSISTANT ``contentEnd`` and the 4 s speaking watchdog) drains it through
   the *same* inject coroutine — contentStart / textInput / contentEnd under
   ``_inject_lock``, with the stale-session check — bypassing the speaking
   guard and the 3 s throttle but re-arming the throttle afterwards.

Why the age is in the text (spec c32): with chunked playback Sonic finishes
GENERATING before the human has finished HEARING, so a deferred reaction is
already a beat late by the time it plays. Naming the delay keeps it coherent.

No network anywhere: the slot needs nothing, and ``NovaSonic`` is driven the
way ``tests/test_sonic_resilience.py`` drives it — a fake ``_send`` and a
scripted response stream.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import threading

import pytest

from reachy_nova import nova_sonic
from reachy_nova.harness import deferred_cues as dc
from reachy_nova.harness.deferred_cues import DeferredCues
from reachy_nova.nova_sonic import NovaSonic

_SENSE_LINE_RE = re.compile(
    r"^\[SENSE stage=(?P<stage>\S+) source=(?P<source>\S+) event=(?P<event>\S+)\] (?P<detail>.*)$"
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class _Clock:
    """A movable monotonic clock."""

    def __init__(self, now: float = 1_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _sense_details(caplog) -> list[str]:
    out = []
    for record in caplog.records:
        if record.name != "nova.sensory":
            continue
        match = _SENSE_LINE_RE.match(record.getMessage())
        assert match is not None, f"unparseable sensory line: {record.getMessage()!r}"
        assert match.group("stage") == "inject"
        assert match.group("source") == "speech"
        assert match.group("event"), "expected a non-empty event id"
        out.append(match.group("detail"))
    return out


# ==========================================================================
# 1. the slot itself
# ==========================================================================


class TestPutAndReplace:
    def test_a_cue_is_parked_and_comes_back_on_drain(self):
        slot = DeferredCues(clock=_Clock())

        slot.put("pat", "(someone is petting you)")

        cues = slot.drain()
        assert [(c.sense_class, c.text) for c in cues] == [
            ("pat", "(someone is petting you)")
        ]

    def test_a_second_cue_of_the_same_class_replaces_the_first(self):
        slot = DeferredCues(clock=_Clock())

        slot.put("pat", "first pat")
        slot.put("pat", "second pat")

        cues = slot.drain()
        assert [c.text for c in cues] == ["second pat"]
        assert slot.counters()["replaced"] == 1
        assert slot.counters()["deferred"] == 2

    def test_classes_are_independent(self):
        slot = DeferredCues(clock=_Clock())

        slot.put("pat", "a pat")
        slot.put("face", "a face")
        slot.put("pat", "another pat")

        cues = slot.drain()
        assert [(c.sense_class, c.text) for c in cues] == [
            ("pat", "another pat"),
            ("face", "a face"),
        ]

    def test_drain_preserves_arrival_order_not_class_order(self):
        slot = DeferredCues(clock=_Clock())

        slot.put("vision", "one")
        slot.put("sound", "two")
        slot.put("face", "three")

        assert [c.sense_class for c in slot.drain()] == ["vision", "sound", "face"]

    def test_replacing_keeps_the_original_arrival_position(self):
        """Latest-wins on the TEXT, not a re-queue: order is by first arrival."""
        slot = DeferredCues(clock=_Clock())

        slot.put("pat", "pat one")
        slot.put("face", "face one")
        slot.put("pat", "pat two")

        assert [c.text for c in slot.drain()] == ["pat two", "face one"]

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_an_unnamed_class_maps_to_other(self, raw):
        slot = DeferredCues(clock=_Clock())

        slot.put(raw, "a cue with no sense class")

        assert [c.sense_class for c in slot.drain()] == [dc.OTHER_CLASS]
        assert dc.OTHER_CLASS == "other"

    def test_every_unnamed_cue_shares_the_other_slot(self):
        slot = DeferredCues(clock=_Clock())

        slot.put(None, "first")
        slot.put("", "second")

        assert [c.text for c in slot.drain()] == ["second"]


class TestDrain:
    def test_drain_clears_the_slot(self):
        slot = DeferredCues(clock=_Clock())
        slot.put("pat", "a pat")

        assert len(slot.drain()) == 1
        assert slot.drain() == []
        assert slot.pending() == 0

    def test_pending_counts_parked_cues(self):
        slot = DeferredCues(clock=_Clock())
        assert slot.pending() == 0

        slot.put("pat", "a pat")
        slot.put("pat", "another pat")
        assert slot.pending() == 1

        slot.put("face", "a face")
        assert slot.pending() == 2

    def test_clear_drops_everything_without_delivering(self):
        slot = DeferredCues(clock=_Clock())
        slot.put("pat", "a pat")
        slot.put("face", "a face")

        assert slot.clear() == 2
        assert slot.drain() == []
        assert slot.counters()["drained"] == 0

    def test_counters_track_the_whole_lifecycle(self):
        clock = _Clock()
        slot = DeferredCues(ttl_s=5.0, clock=clock)

        slot.put("pat", "old")
        clock.advance(10.0)
        slot.put("face", "fresh")
        slot.put("face", "fresher")

        slot.drain()

        assert slot.counters() == {
            "deferred": 3,
            "replaced": 1,
            "expired": 1,
            "drained": 1,
        }


class TestExpiry:
    def test_a_cue_older_than_the_ttl_is_dropped(self, caplog):
        clock = _Clock()
        slot = DeferredCues(ttl_s=5.0, clock=clock)
        slot.put("pat", "a pat nobody heard about")
        clock.advance(5.5)

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            assert slot.drain() == []

        details = _sense_details(caplog)
        assert len(details) == 1
        assert "dropped reason=deferred-expired" in details[0]
        assert "age=" in details[0]

    def test_a_cue_inside_the_ttl_survives(self, caplog):
        clock = _Clock()
        slot = DeferredCues(ttl_s=5.0, clock=clock)
        slot.put("pat", "a fresh pat")
        clock.advance(4.9)

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            assert [c.text for c in slot.drain()] == ["a fresh pat"]

        assert _sense_details(caplog) == []

    def test_expiry_is_per_cue_not_per_slot(self, caplog):
        clock = _Clock()
        slot = DeferredCues(ttl_s=5.0, clock=clock)
        slot.put("pat", "the old one")
        clock.advance(6.0)
        slot.put("face", "the new one")

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            cues = slot.drain()

        assert [c.text for c in cues] == ["the new one"]
        assert len(_sense_details(caplog)) == 1

    def test_replacing_refreshes_the_age(self):
        clock = _Clock()
        slot = DeferredCues(ttl_s=5.0, clock=clock)
        slot.put("pat", "first")
        clock.advance(4.0)
        slot.put("pat", "second")
        clock.advance(4.0)

        assert [c.text for c in slot.drain()] == ["second"]

    def test_an_explicit_now_overrides_the_clock(self, caplog):
        clock = _Clock()
        slot = DeferredCues(ttl_s=5.0, clock=clock)
        slot.put("pat", "a pat")

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            assert slot.drain(now=clock.now + 99.0) == []

        assert len(_sense_details(caplog)) == 1


class TestRender:
    def test_under_two_seconds_reads_as_just_now(self):
        clock = _Clock()
        slot = DeferredCues(clock=clock)
        cue = slot.put("pat", "(someone is petting you)")

        assert slot.render(cue, clock.now + 0.4) == (
            "(just now, while you were talking: (someone is petting you))"
        )

    def test_two_seconds_and_over_names_the_delay(self):
        clock = _Clock()
        slot = DeferredCues(clock=clock)
        cue = slot.put("pat", "(someone is petting you)")

        assert slot.render(cue, clock.now + 3.0) == (
            "(3 seconds ago, while you were talking: (someone is petting you))"
        )

    def test_the_boundary_is_two_seconds(self):
        clock = _Clock()
        slot = DeferredCues(clock=clock)
        cue = slot.put("pat", "hi")

        assert "just now" in slot.render(cue, clock.now + 1.999)
        assert "just now" not in slot.render(cue, clock.now + 2.0)

    def test_a_named_delay_is_never_under_two_seconds(self):
        """Rounding must never produce '1 seconds ago' next to the 'just now' band."""
        clock = _Clock()
        slot = DeferredCues(clock=clock)
        cue = slot.put("pat", "hi")

        assert slot.render(cue, clock.now + 2.0).startswith("(2 seconds ago,")

    def test_now_defaults_to_the_injected_clock(self):
        clock = _Clock()
        slot = DeferredCues(clock=clock)
        cue = slot.put("pat", "hi")
        clock.advance(4.0)

        assert slot.render(cue) == "(4 seconds ago, while you were talking: hi)"

    def test_the_rendered_text_differs_from_the_immediate_one(self):
        clock = _Clock()
        slot = DeferredCues(clock=clock)
        cue = slot.put("pat", "(someone is petting you)")

        assert slot.render(cue, clock.now) != cue.text


class TestThreadSafety:
    def test_concurrent_puts_and_drains_never_lose_the_lock(self):
        slot = DeferredCues(ttl_s=1000.0, clock=_Clock())
        drained: list[str] = []
        errors: list[BaseException] = []

        def _putter(name: str) -> None:
            try:
                for i in range(200):
                    slot.put(name, f"{name}-{i}")
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        def _drainer() -> None:
            try:
                for _ in range(200):
                    drained.extend(c.text for c in slot.drain())
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [
            threading.Thread(target=_putter, args=("pat",)),
            threading.Thread(target=_putter, args=("face",)),
            threading.Thread(target=_drainer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert not any(t.is_alive() for t in threads)
        drained.extend(c.text for c in slot.drain())
        assert len(set(drained)) == len(drained)  # no cue delivered twice


# ==========================================================================
# 2. the NovaSonic wiring
# ==========================================================================


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
        # A real stream suspends here; suspending too is what lets a task the
        # response loop scheduled actually start while the loop keeps reading.
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


class _NoSleepAsyncio:
    """``asyncio`` proxy whose ``sleep`` returns immediately."""

    def __getattr__(self, name):  # pragma: no cover - trivial delegation
        return getattr(asyncio, name)

    async def sleep(self, delay: float = 0, result=None):
        await asyncio.sleep(0)
        return result


def _ready_sonic(**kwargs) -> NovaSonic:
    """A NovaSonic marked active with a stub loop — enough for ``inject_text``."""
    sonic = NovaSonic(system_prompt="You are Nova.", **kwargs)
    sonic._active = True
    sonic._loop = object()
    return sonic


def _event_names(sent: list[dict]) -> list[str]:
    return [next(iter(e.keys()), "unknown") for e in sent]


def _drive_responses(sonic: NovaSonic, events: list[dict]) -> list[dict]:
    """Run ``_process_responses`` over a scripted event list; return what was sent."""
    stream = _FakeStream(events)
    sonic._stream = stream
    sonic._active = True

    async def _go() -> None:
        sonic._loop = asyncio.get_running_loop()
        sonic._inject_lock = asyncio.Lock()
        await sonic._process_responses()
        # let any task the response loop scheduled actually run
        for _ in range(20):
            await asyncio.sleep(0)

    asyncio.run(_go())
    return stream.sent


class TestInjectTextDefersInsteadOfDropping:
    def test_a_cue_arriving_while_speaking_is_parked_not_dropped(self, caplog):
        sonic = _ready_sonic()
        sonic._speaking = True

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sonic.inject_text("(someone is petting you)", sense_class="pat")

        details = _sense_details(caplog)
        assert len(details) == 1
        assert details[0].startswith("deferred ")
        assert "class=pat" in details[0]
        assert "(someone is petting you)" in details[0]
        assert "dropped reason=speaking" not in details[0]
        assert sonic._deferred.pending() == 1

    def test_a_second_cue_of_the_same_class_replaces_the_first(self):
        sonic = _ready_sonic()
        sonic._speaking = True

        sonic.inject_text("first pat", sense_class="pat")
        sonic.inject_text("second pat", sense_class="pat")

        assert [c.text for c in sonic._deferred.drain()] == ["second pat"]

    def test_classes_are_independent(self):
        sonic = _ready_sonic()
        sonic._speaking = True

        sonic.inject_text("a pat", sense_class="pat")
        sonic.inject_text("a face", sense_class="face")

        assert [c.sense_class for c in sonic._deferred.drain()] == ["pat", "face"]

    def test_a_cue_without_a_class_lands_in_the_other_slot(self, caplog):
        sonic = _ready_sonic()
        sonic._speaking = True

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sonic.inject_text("something happened")

        assert "class=other" in _sense_details(caplog)[0]
        assert [c.sense_class for c in sonic._deferred.drain()] == ["other"]

    def test_the_ttl_is_constructor_configurable(self):
        sonic = _ready_sonic(deferred_ttl_s=0.25)
        assert sonic._deferred.ttl_s == 0.25
        assert _ready_sonic()._deferred.ttl_s == dc.DEFAULT_TTL_S == 5.0


class TestUnchangedInjectBehaviour:
    def test_force_still_bypasses_the_speaking_guard_and_defers_nothing(self, caplog):
        sonic = _ready_sonic()
        sonic._speaking = True

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            try:
                sonic.inject_text("urgent", force=True)
            except Exception:  # pragma: no cover - stub loop scheduling
                pass

        assert sonic._deferred.pending() == 0
        assert not any(d.startswith("deferred ") for d in _sense_details(caplog))

    def test_the_three_second_throttle_still_drops_when_not_speaking(self, caplog):
        sonic = _ready_sonic()
        sonic._speaking = False
        sonic._last_inject_time = nova_sonic.time.time()

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sonic.inject_text("a throttled cue", sense_class="pat")

        details = _sense_details(caplog)
        assert len(details) == 1
        assert "dropped reason=throttled" in details[0]
        assert sonic._deferred.pending() == 0, "a throttled cue is dropped, not deferred"

    def test_an_inactive_sonic_defers_nothing(self):
        sonic = _ready_sonic()
        sonic._active = False
        sonic._speaking = True

        sonic.inject_text("a pat", sense_class="pat")

        assert sonic._deferred.pending() == 0


class TestDrainOnUtteranceEnd:
    def _speaking_sonic(self) -> NovaSonic:
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        sonic._loop = object()
        sonic._speaking = True
        return sonic

    def test_the_assistant_content_end_drains_the_slot_through_the_inject_path(self):
        sonic = self._speaking_sonic()
        sonic.inject_text("(someone is petting you)", sense_class="pat")

        sent = _drive_responses(sonic, [
            {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
        ])

        assert _event_names(sent) == ["contentStart", "textInput", "contentEnd"]
        start = sent[0]["contentStart"]
        assert start["role"] == "USER"
        assert start["type"] == "TEXT"
        assert sent[1]["textInput"]["contentName"] == start["contentName"]
        assert sent[2]["contentEnd"]["contentName"] == start["contentName"]
        text = sent[1]["textInput"]["content"]
        assert "while you were talking" in text
        assert "(someone is petting you)" in text
        assert sonic._deferred.pending() == 0

    def test_the_drained_text_carries_the_age(self):
        clock = _Clock()
        sonic = self._speaking_sonic()
        sonic._deferred = DeferredCues(clock=clock)
        sonic.inject_text("(someone is petting you)", sense_class="pat")
        clock.advance(3.0)

        sent = _drive_responses(sonic, [
            {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
        ])

        assert sent[1]["textInput"]["content"] == (
            "(3 seconds ago, while you were talking: (someone is petting you))"
        )

    def test_several_cues_drain_in_arrival_order(self):
        sonic = self._speaking_sonic()
        sonic.inject_text("a pat", sense_class="pat")
        sonic.inject_text("a face", sense_class="face")
        sonic.inject_text("a sound", sense_class="sound")

        sent = _drive_responses(sonic, [
            {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
        ])

        texts = [e["textInput"]["content"] for e in sent if "textInput" in e]
        assert len(texts) == 3
        assert "a pat" in texts[0] and "a face" in texts[1] and "a sound" in texts[2]

    def test_at_most_four_cues_drain_per_transition(self, caplog):
        sonic = self._speaking_sonic()
        for i, klass in enumerate(["pat", "face", "sound", "vision", "other"]):
            sonic.inject_text(f"cue {i}", sense_class=klass)

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sent = _drive_responses(sonic, [
                {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
            ])

        texts = [e["textInput"]["content"] for e in sent if "textInput" in e]
        assert len(texts) == dc.MAX_DRAIN_PER_TRANSITION == 4
        assert "cue 4" not in " ".join(texts)
        assert any("deferred-overflow" in d for d in _sense_details(caplog))

    def test_an_expired_cue_is_dropped_and_never_sent(self, caplog):
        clock = _Clock()
        sonic = self._speaking_sonic()
        sonic._deferred = DeferredCues(ttl_s=5.0, clock=clock)
        sonic.inject_text("a stale pat", sense_class="pat")
        clock.advance(6.0)
        sonic.inject_text("a fresh face", sense_class="face")

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sent = _drive_responses(sonic, [
                {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
            ])

        texts = [e["textInput"]["content"] for e in sent if "textInput" in e]
        assert len(texts) == 1
        assert "a fresh face" in texts[0]
        expired = [d for d in _sense_details(caplog) if "reason=deferred-expired" in d]
        assert len(expired) == 1
        assert "age=" in expired[0]

    def test_an_empty_slot_sends_nothing_at_all(self):
        sonic = self._speaking_sonic()

        sent = _drive_responses(sonic, [
            {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
        ])

        assert sent == []

    def test_a_user_content_end_does_not_drain(self):
        sonic = self._speaking_sonic()
        sonic.inject_text("a pat", sense_class="pat")

        sent = _drive_responses(sonic, [
            {"contentEnd": {"type": "TEXT", "role": "USER", "stopReason": "END_TURN"}},
        ])

        assert sent == []
        assert sonic._deferred.pending() == 1

    def test_the_drain_bypasses_the_three_second_throttle(self):
        """The cue already waited out the whole utterance — it must not be throttled."""
        sonic = self._speaking_sonic()
        sonic.inject_text("a pat", sense_class="pat")
        sonic._last_inject_time = nova_sonic.time.time()  # a fresh throttle window

        sent = _drive_responses(sonic, [
            {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
        ])

        assert "textInput" in _event_names(sent)

    def test_the_throttle_is_re_armed_after_the_drain(self, caplog):
        """The drain protects the injects that follow it, exactly like a normal inject."""
        sonic = self._speaking_sonic()
        sonic._last_inject_time = 0.0
        sonic.inject_text("a pat", sense_class="pat")

        _drive_responses(sonic, [
            {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
        ])

        assert nova_sonic.time.time() - sonic._last_inject_time < 1.0
        sonic._loop = object()
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sonic.inject_text("a follow-up", sense_class="face")
        assert any("dropped reason=throttled" in d for d in _sense_details(caplog))

    def test_a_stale_session_generation_discards_the_drain(self):
        sonic = self._speaking_sonic()
        sonic.inject_text("a pat", sense_class="pat")

        stream = _FakeStream()
        sonic._stream = stream

        async def _go() -> None:
            sonic._loop = asyncio.get_running_loop()
            sonic._inject_lock = asyncio.Lock()
            sonic._on_speaking_ended()
            sonic._session_gen += 1  # the session restarted before the drain ran
            for _ in range(20):
                await asyncio.sleep(0)

        asyncio.run(_go())

        assert stream.sent == []


class TestDrainDoesNotBlockTheResponseLoop:
    def test_the_response_loop_keeps_going_while_the_drain_is_stuck(self):
        """The drain is scheduled on the loop, never awaited inside the branch."""
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        sonic._loop = object()
        sonic._speaking = True
        sonic.inject_text("a pat", sense_class="pat")

        sent: list[str] = []

        async def _go() -> None:
            sonic._loop = asyncio.get_running_loop()
            sonic._inject_lock = asyncio.Lock()
            gate = asyncio.Event()  # never set: the drain's first send hangs

            async def _hanging_send(event: dict) -> None:
                name = next(iter(event.keys()), "unknown")
                sent.append(name)
                if name == "contentStart":
                    await gate.wait()

            sonic._send = _hanging_send  # type: ignore[method-assign]
            sonic._stream = _FakeStream([
                {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
                {"textOutput": {"role": "USER", "content": "hello there"}},
            ])
            await sonic._process_responses()

        asyncio.run(_go())

        # The response loop processed the event AFTER the contentEnd even though
        # the drain it scheduled is still blocked on its very first send.
        assert sonic.last_user_text == "hello there"
        assert "textInput" not in sent, "the drain must not have completed"

class TestBothSpeakingEndSitesDrain:
    """The speaking state ends in two places; one hook covers both."""

    def test_the_hook_delivers_through_the_normal_inject_path(self):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        sonic._loop = object()
        sonic._speaking = True
        sonic.inject_text("a pat", sense_class="pat")
        stream = _FakeStream()

        async def _go() -> None:
            sonic._loop = asyncio.get_running_loop()
            sonic._inject_lock = asyncio.Lock()
            sonic._stream = stream
            sonic._on_speaking_ended()
            for _ in range(20):
                await asyncio.sleep(0)

        asyncio.run(_go())

        assert _event_names(stream.sent) == ["contentStart", "textInput", "contentEnd"]
        assert "a pat" in stream.sent[1]["textInput"]["content"]

    def test_the_content_end_branch_calls_the_hook(self):
        source = inspect.getsource(NovaSonic._process_responses)
        assert "_on_speaking_ended()" in source

    def test_the_speaking_watchdog_calls_the_hook(self):
        source = inspect.getsource(NovaSonic._run_loop)
        assert "_on_speaking_ended()" in source

    def test_the_hook_is_a_no_op_with_an_empty_slot(self):
        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._active = True
        stream = _FakeStream()

        async def _go() -> None:
            sonic._loop = asyncio.get_running_loop()
            sonic._inject_lock = asyncio.Lock()
            sonic._stream = stream
            sonic._on_speaking_ended()
            for _ in range(20):
                await asyncio.sleep(0)

        asyncio.run(_go())

        assert stream.sent == []


class TestRestartClearsTheSlot:
    def test_the_run_loop_restart_path_clears_parked_cues(self, monkeypatch):
        monkeypatch.setattr(nova_sonic, "asyncio", _NoSleepAsyncio())

        stream = _FakeStream([])  # receive() raises at once -> the stream "dies"

        class _FakeClient:
            async def invoke_model_with_bidirectional_stream(self, _operation_input):
                return stream

        sonic = NovaSonic(system_prompt="You are Nova.")
        sonic._init_client = lambda: setattr(sonic, "_client", _FakeClient())  # type: ignore[method-assign]
        stop = threading.Event()

        async def _stop_instead_of_waiting(_stop_event, _delay):
            stop.set()

        sonic._interruptible_wait = _stop_instead_of_waiting  # type: ignore[method-assign]

        async def _go() -> None:
            sonic._loop = asyncio.get_running_loop()
            # park a cue as soon as the session is live, then let the stream die
            sonic._deferred.put("pat", "a pat nobody will hear")
            await sonic._run_loop(stop)

        asyncio.run(_go())

        assert sonic._deferred.pending() == 0
