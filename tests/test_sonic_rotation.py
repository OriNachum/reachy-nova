"""Proactive session rotation with conversation-history replay (t12).

Two facts measured on the robot on 2026-09-05/06 drive this file:

1. **The stream has an 8-minute ceiling.** The one session that outlived the
   liveness window (because Ori kept talking) started 21:51:29 and died at
   21:59:30 — 480.5 s — with Bedrock's "Model has timed out in processing the
   request". The harness then waited the 3 s base backoff and came back with
   no memory of the conversation at all: every restart was "CLEAN: fresh
   client, fresh UUIDs, system prompt only".
2. **Sonic can speak unprompted on a fresh session.** The first utterance of
   that boot arrived 33 s after session start with no user speech in front of
   it. A rotation every seven minutes that re-greets the room is worse than
   the amnesia it fixes.

So the session now rotates itself *before* the ceiling (c12) and carries the
conversation across (c11): AWS's Nova 2 input-events page says history "can be
included only once, after the system prompt and before audio streaming
begins", as TEXT content blocks with a USER or ASSISTANT role. The rotation
waits for an idle moment — listening, no tool call in flight, not speaking,
the speaker's queue empty — with a hard deadline shortly before the ceiling so
nothing in flight is ever cut by our own timer (c31), and a healthy rotation
pays no backoff at all: delay 0.

The tests drive the two real seams — ``_start_session`` against a fake client
for the replay, ``_run_loop`` against a fake clock for the timer — rather than
asserting on a re-implementation of either.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import types

import pytest

from reachy_nova import nova_sonic
from reachy_nova.nova_sonic import NovaSonic

# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class _FakeClock:
    """A wall clock and a monotonic clock that can be moved independently."""

    def __init__(self, wall: float = 1_700_000_000.0, mono: float = 1_000.0):
        self.wall = wall
        self.mono = mono

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds

    def as_module(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(time=self.time, monotonic=self.monotonic)


class _FastAsyncio:
    """``asyncio`` proxy whose ``sleep`` advances the fake clock instead of waiting."""

    def __init__(self, clock: _FakeClock):
        self._clock = clock

    def __getattr__(self, name):  # pragma: no cover - trivial delegation
        return getattr(asyncio, name)

    async def sleep(self, delay: float = 0, result=None):
        self._clock.advance(delay or 0)
        await asyncio.sleep(0)
        return result


class _NoSleepAsyncio:
    """``asyncio`` proxy whose ``sleep`` returns immediately.

    ``_start_session`` ends with a 0.5 s settle pause before it accepts
    traffic; real behaviour worth keeping, dead weight in a unit test.
    """

    def __getattr__(self, name):  # pragma: no cover - trivial delegation
        return getattr(asyncio, name)

    async def sleep(self, delay: float = 0, result=None):
        await asyncio.sleep(0)
        return result


class _FakeInputStream:
    def __init__(self, sent: list[dict]):
        self._sent = sent
        self.closed = False

    async def send(self, chunk) -> None:
        self._sent.append(json.loads(chunk.value.bytes_.decode("utf-8"))["event"])

    async def close(self) -> None:
        self.closed = True


class _FakeStream:
    """Accepts every input event and never answers — a healthy, quiet session."""

    def __init__(self, on_first_receive=None):
        self.sent: list[dict] = []
        self.input_stream = _FakeInputStream(self.sent)
        self._on_first_receive = on_first_receive
        self._received_once = False
        self._silent = asyncio.Event()

    async def await_output(self):
        return (None, self)

    async def receive(self):
        if not self._received_once:
            self._received_once = True
            if self._on_first_receive is not None:
                self._on_first_receive()
        await self._silent.wait()  # never set: no response event, ever

    def event_types(self) -> list[str]:
        return [next(iter(e.keys()), "unknown") for e in self.sent]


class _FakeClient:
    """Hands out a fresh ``_FakeStream`` per session and keeps them all."""

    def __init__(self, first_stream_hook=None):
        self.streams: list[_FakeStream] = []
        self._first_stream_hook = first_stream_hook

    async def invoke_model_with_bidirectional_stream(self, _operation_input):
        hook = self._first_stream_hook if not self.streams else None
        stream = _FakeStream(on_first_receive=hook)
        self.streams.append(stream)
        return stream


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _make_sonic(client: _FakeClient | None = None, **kwargs) -> NovaSonic:
    sonic = NovaSonic(system_prompt="You are Nova, a small curious robot.", **kwargs)
    if client is not None:
        sonic._init_client = lambda: setattr(sonic, "_client", client)  # type: ignore[method-assign]
    return sonic


def _start_session(sonic: NovaSonic) -> None:
    asyncio.run(sonic._start_session())


def _messages(caplog, level: int) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "reachy_nova.nova_sonic" and r.levelno == level
    ]


def _rotation_lines(caplog) -> list[str]:
    """Journal lines that a `grep rotation` would surface with a replay count."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "reachy_nova.nova_sonic"
        and "rotation" in r.getMessage()
        and "replay=" in r.getMessage()
    ]


def _text_blocks(stream: _FakeStream) -> list[dict]:
    """Every TEXT content block on the wire as ``{role, interactive, content}``.

    Reconstructed by walking the send log, so a block only appears here if its
    ``contentStart``/``textInput``/``contentEnd`` really were sent, in order,
    against the same content name.
    """
    open_blocks: dict[str, dict] = {}
    blocks: list[dict] = []
    for event in stream.sent:
        if "contentStart" in event:
            cs = event["contentStart"]
            if cs.get("type") == "TEXT":
                open_blocks[cs["contentName"]] = {
                    "role": cs.get("role"),
                    "interactive": cs.get("interactive"),
                    "config": cs.get("textInputConfiguration"),
                    "content": None,
                    "ended": False,
                }
        elif "textInput" in event:
            name = event["textInput"]["contentName"]
            if name in open_blocks:
                open_blocks[name]["content"] = event["textInput"]["content"]
        elif "contentEnd" in event:
            name = event["contentEnd"]["contentName"]
            block = open_blocks.pop(name, None)
            if block is not None:
                block["ended"] = True
                blocks.append(block)
    return blocks


def _history_blocks(stream: _FakeStream) -> list[dict]:
    """The replayed history — every TEXT block that is not the system prompt."""
    return [b for b in _text_blocks(stream) if b["role"] != "SYSTEM"]


def _audio_start_index(stream: _FakeStream) -> int:
    for index, event in enumerate(stream.sent):
        if "contentStart" in event and event["contentStart"].get("type") == "AUDIO":
            return index
    raise AssertionError("no audio contentStart in the send log")


def _run_loop_for(sonic: NovaSonic, clock: _FakeClock, fake_seconds: float) -> None:
    """Drive ``_run_loop`` until ``fake_seconds`` of simulated time have passed.

    The loop moves the clock itself (its 0.1 s tick sleep and its restart sleep
    both go through ``_FastAsyncio``), so the stopper only watches the clock.
    """
    stop = threading.Event()

    async def _stopper() -> None:
        deadline = clock.mono + fake_seconds
        while clock.mono < deadline:
            await asyncio.sleep(0)
        stop.set()

    async def _drive() -> None:
        sonic._loop = asyncio.get_running_loop()
        helper = asyncio.create_task(_stopper())
        try:
            await asyncio.wait_for(sonic._run_loop(stop), timeout=60)
        finally:
            helper.cancel()

    asyncio.run(_drive())


@pytest.fixture(autouse=True)
def _no_ambient_rotation_env(monkeypatch):
    """A developer's exported knobs must never decide what these tests assert."""
    for name in (
        "NOVA_SONIC_ROTATE_S",
        "NOVA_SONIC_ROTATE_DEADLINE_S",
        "NOVA_SONIC_LIVENESS_S",
        "NOVA_SONIC_ENDPOINTING",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def nosleep(monkeypatch):
    """Skip ``_start_session``'s settle pause without touching anything else."""
    monkeypatch.setattr(nova_sonic, "asyncio", _NoSleepAsyncio())


@pytest.fixture
def faketime(monkeypatch):
    """Swap the module's ``time`` and ``asyncio`` for fake-clock-driven ones."""
    clock = _FakeClock()
    monkeypatch.setattr(nova_sonic, "time", clock.as_module())
    monkeypatch.setattr(nova_sonic, "asyncio", _FastAsyncio(clock))
    return clock


TWO_BLOCKS = [
    {"role": "USER", "text": "(earlier today we talked about: the garden)"},
    {"role": "ASSISTANT", "text": "The tomatoes are winning."},
]


# --------------------------------------------------------------------------
# 1. history replay — what crosses the wire, and where
# --------------------------------------------------------------------------


class TestHistoryReplay:
    def test_blocks_land_between_the_system_prompt_and_the_audio(self, nosleep):
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(TWO_BLOCKS))

        _start_session(sonic)

        stream = client.streams[0]
        assert [next(iter(e)) for e in stream.sent] == [
            "sessionStart",
            "promptStart",
            "contentStart",  # system prompt
            "textInput",
            "contentEnd",
            "contentStart",  # history block 1
            "textInput",
            "contentEnd",
            "contentStart",  # history block 2
            "textInput",
            "contentEnd",
            "contentStart",  # audio channel
        ]
        roles = [e["contentStart"].get("role") for e in stream.sent if "contentStart" in e]
        assert roles == ["SYSTEM", "USER", "ASSISTANT", "USER"]

    def test_each_block_is_a_non_interactive_plain_text_content(self, nosleep):
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(TWO_BLOCKS))

        _start_session(sonic)

        blocks = _history_blocks(client.streams[0])
        assert [b["role"] for b in blocks] == ["USER", "ASSISTANT"]
        assert [b["content"] for b in blocks] == [b["text"] for b in TWO_BLOCKS]
        assert all(b["interactive"] is False for b in blocks)
        assert all(b["config"] == {"mediaType": "text/plain"} for b in blocks)
        assert all(b["ended"] for b in blocks), "every block must be closed"

    def test_the_system_prompt_stays_interactive_and_first(self, nosleep):
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(TWO_BLOCKS))

        _start_session(sonic)

        system = _text_blocks(client.streams[0])[0]
        assert system["role"] == "SYSTEM"
        assert system["interactive"] is True
        assert system["content"] == sonic.system_prompt

    def test_no_provider_sends_no_history(self, nosleep):
        client = _FakeClient()
        sonic = _make_sonic(client)

        _start_session(sonic)

        assert _history_blocks(client.streams[0]) == []
        assert [next(iter(e)) for e in client.streams[0].sent] == [
            "sessionStart",
            "promptStart",
            "contentStart",
            "textInput",
            "contentEnd",
            "contentStart",
        ]

    def test_an_empty_provider_sends_no_history_and_still_starts(self, nosleep):
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: [])

        _start_session(sonic)

        assert _history_blocks(client.streams[0]) == []
        assert sonic.state == "listening"
        assert sonic._active is True

    def test_a_raising_provider_is_logged_and_treated_as_empty(self, nosleep, caplog):
        def _boom() -> list[dict]:
            raise RuntimeError("ledger unreadable")

        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=_boom)

        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            _start_session(sonic)

        assert _history_blocks(client.streams[0]) == []
        assert sonic.state == "listening", "a bad provider must not cost the session"
        warnings = [m for m in _messages(caplog, logging.WARNING) if "history" in m]
        assert len(warnings) == 1
        assert "ledger unreadable" in warnings[0]

    def test_blocks_past_the_cap_are_dropped(self, nosleep):
        many = [{"role": "USER", "text": f"line {i}"} for i in range(12)]
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(many))

        _start_session(sonic)

        blocks = _history_blocks(client.streams[0])
        assert len(blocks) == 8, "history_max_blocks defaults to 8"
        assert [b["content"] for b in blocks] == [f"line {i}" for i in range(8)]

    def test_the_cap_is_a_constructor_kwarg(self, nosleep):
        many = [{"role": "USER", "text": f"line {i}"} for i in range(12)]
        client = _FakeClient()
        sonic = _make_sonic(
            client, history_provider=lambda: list(many), history_max_blocks=3
        )

        _start_session(sonic)

        assert len(_history_blocks(client.streams[0])) == 3

    def test_an_unsupported_role_is_skipped_with_a_warning(self, nosleep, caplog):
        blocks = [
            {"role": "USER", "text": "kept"},
            {"role": "SYSTEM", "text": "a second system prompt"},
            {"role": "TOOL", "text": "not conversation"},
            {"role": "ASSISTANT", "text": "also kept"},
        ]
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(blocks))

        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            _start_session(sonic)

        sent = _history_blocks(client.streams[0])
        assert [b["content"] for b in sent] == ["kept", "also kept"]
        warnings = [m for m in _messages(caplog, logging.WARNING) if "history" in m]
        assert len(warnings) == 2
        assert "SYSTEM" in warnings[0]
        assert "TOOL" in warnings[1]

    def test_lower_case_roles_are_accepted(self, nosleep):
        blocks = [
            {"role": "user", "text": "hello"},
            {"role": "assistant", "text": "hi"},
        ]
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(blocks))

        _start_session(sonic)

        assert [b["role"] for b in _history_blocks(client.streams[0])] == [
            "USER",
            "ASSISTANT",
        ]

    def test_one_info_line_names_the_replay_count(self, nosleep, caplog):
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(TWO_BLOCKS))

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _start_session(sonic)

        named = [m for m in _messages(caplog, logging.INFO) if "history replayed" in m]
        assert named == ["history replayed blocks=2"]

    def test_a_session_with_no_history_still_says_so(self, nosleep, caplog):
        client = _FakeClient()
        sonic = _make_sonic(client)

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _start_session(sonic)

        named = [m for m in _messages(caplog, logging.INFO) if "history replayed" in m]
        assert named == ["history replayed blocks=0"]

    def test_each_block_gets_its_own_content_name(self, nosleep):
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(TWO_BLOCKS))

        _start_session(sonic)

        names = [
            e["contentStart"]["contentName"]
            for e in client.streams[0].sent
            if "contentStart" in e
        ]
        assert len(set(names)) == len(names), "content names must not collide"

    def test_nothing_is_sent_between_the_last_block_and_the_audio(self, nosleep):
        """c31: a fresh session must not open its mouth first."""
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(TWO_BLOCKS))

        _start_session(sonic)

        stream = client.streams[0]
        audio_at = _audio_start_index(stream)
        last_end = max(
            i for i, e in enumerate(stream.sent[:audio_at]) if "contentEnd" in e
        )
        assert stream.sent[last_end + 1 : audio_at] == []
        assert audio_at == last_end + 1

    def test_the_provider_is_called_once_per_session(self, nosleep):
        calls: list[int] = []

        def _provider() -> list[dict]:
            calls.append(1)
            return list(TWO_BLOCKS)

        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=_provider)

        _start_session(sonic)
        _start_session(sonic)

        assert len(calls) == 2
        assert len(_history_blocks(client.streams[1])) == 2


# --------------------------------------------------------------------------
# 2. the rotation timer — when a healthy session is allowed to be replaced
# --------------------------------------------------------------------------


class TestRotationEnv:
    def test_interval_defaults_to_420(self):
        assert nova_sonic._rotate_interval_s() == 420.0

    def test_interval_is_overridable(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "90")
        assert nova_sonic._rotate_interval_s() == 90.0

    def test_garbage_interval_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "soon")
        assert nova_sonic._rotate_interval_s() == 420.0

    @pytest.mark.parametrize("raw", ["0", "-1"])
    def test_zero_or_negative_disables_rotation(self, monkeypatch, raw):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", raw)
        assert nova_sonic._rotate_interval_s() == 0.0

    def test_deadline_defaults_to_470(self):
        assert nova_sonic._rotate_deadline_s() == 470.0

    def test_deadline_is_overridable(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_DEADLINE_S", "300")
        assert nova_sonic._rotate_deadline_s() == 300.0

    def test_garbage_deadline_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_DEADLINE_S", "")
        assert nova_sonic._rotate_deadline_s() == 470.0

    def test_read_at_call_time(self, monkeypatch):
        """``load_dotenv()`` order must never matter (see ``_liveness_window``)."""
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "60")
        assert nova_sonic._rotate_interval_s() == 60.0
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "120")
        assert nova_sonic._rotate_interval_s() == 120.0


class TestRotationDue:
    def _idle_session(self, **kwargs) -> NovaSonic:
        sonic = _make_sonic(**kwargs)
        sonic._arm_watchdogs(1_000.0, 500.0)
        sonic.state = "listening"
        sonic._speaking = False
        sonic._current_tool_use = None
        return sonic

    def test_a_young_session_is_never_rotated(self):
        sonic = self._idle_session()
        assert sonic._rotation_due(500.0 + 419.0) is None

    def test_an_idle_session_past_the_interval_rotates(self):
        sonic = self._idle_session()
        due = sonic._rotation_due(500.0 + 421.0)
        assert due == pytest.approx(421.0)

    def test_a_speaking_session_waits(self):
        sonic = self._idle_session()
        sonic.state = "speaking"
        sonic._speaking = True
        assert sonic._rotation_due(500.0 + 421.0) is None

    def test_a_thinking_session_waits(self):
        sonic = self._idle_session()
        sonic.state = "thinking"
        assert sonic._rotation_due(500.0 + 421.0) is None

    def test_a_tool_call_in_flight_waits(self):
        sonic = self._idle_session()
        sonic._current_tool_use = {"toolName": "look_at", "toolUseId": "x", "content": ""}
        assert sonic._rotation_due(500.0 + 421.0) is None

    def test_a_stale_speaking_flag_waits(self):
        """Belt and braces: the flag is checked as well as the state."""
        sonic = self._idle_session()
        sonic._speaking = True
        assert sonic._rotation_due(500.0 + 421.0) is None

    def test_a_busy_speaker_waits(self):
        sonic = self._idle_session(speaker_idle=lambda: False)
        assert sonic._rotation_due(500.0 + 421.0) is None

    def test_an_idle_speaker_rotates(self):
        sonic = self._idle_session(speaker_idle=lambda: True)
        assert sonic._rotation_due(500.0 + 421.0) == pytest.approx(421.0)

    def test_a_missing_speaker_callable_counts_as_idle(self):
        sonic = self._idle_session(speaker_idle=None)
        assert sonic._rotation_due(500.0 + 421.0) == pytest.approx(421.0)

    def test_a_raising_speaker_callable_is_not_idle(self):
        def _boom() -> bool:
            raise RuntimeError("speaker gone")

        sonic = self._idle_session(speaker_idle=_boom)
        assert sonic._rotation_due(500.0 + 421.0) is None
        # ...but the hard deadline still saves the session.
        assert sonic._rotation_due(500.0 + 471.0) == pytest.approx(471.0)

    @pytest.mark.parametrize(
        "busy",
        [
            {"state": "speaking", "_speaking": True},
            {"_current_tool_use": {"toolName": "t", "toolUseId": "i", "content": ""}},
        ],
    )
    def test_the_hard_deadline_rotates_regardless(self, busy):
        sonic = self._idle_session(speaker_idle=lambda: False)
        for name, value in busy.items():
            setattr(sonic, name, value)
        assert sonic._rotation_due(500.0 + 469.0) is None
        assert sonic._rotation_due(500.0 + 471.0) == pytest.approx(471.0)

    def test_the_interval_and_deadline_follow_the_env(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "10")
        monkeypatch.setenv("NOVA_SONIC_ROTATE_DEADLINE_S", "20")
        sonic = self._idle_session(speaker_idle=lambda: False)
        assert sonic._rotation_due(500.0 + 9.0) is None
        assert sonic._rotation_due(500.0 + 11.0) is None  # busy speaker
        assert sonic._rotation_due(500.0 + 21.0) == pytest.approx(21.0)

    def test_rotation_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "0")
        sonic = self._idle_session()
        assert sonic._rotation_due(500.0 + 100_000.0) is None

    def test_an_unarmed_session_is_never_due(self):
        sonic = _make_sonic()
        assert sonic._session_start_mono is None
        assert sonic._rotation_due(500.0) is None


class TestRotationIntegration:
    def test_an_idle_session_rotates_and_replays(self, faketime, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "10")
        monkeypatch.setenv("NOVA_SONIC_ROTATE_DEADLINE_S", "20")
        client = _FakeClient()
        sonic = _make_sonic(
            client,
            history_provider=lambda: list(TWO_BLOCKS),
            speaker_idle=lambda: True,
        )

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, faketime, fake_seconds=14.0)

        assert len(client.streams) == 2, "expected exactly one rotation"
        assert len(_history_blocks(client.streams[1])) == 2

        lines = _rotation_lines(caplog)
        assert len(lines) == 1, lines
        assert "delay=0" in lines[0]
        assert "replay=2" in lines[0]
        assert "age=" in lines[0]

    def test_a_busy_speaker_holds_the_rotation_off(self, faketime, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "10")
        monkeypatch.setenv("NOVA_SONIC_ROTATE_DEADLINE_S", "60")
        client = _FakeClient()
        sonic = _make_sonic(
            client,
            history_provider=lambda: list(TWO_BLOCKS),
            speaker_idle=lambda: False,
        )

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, faketime, fake_seconds=30.0)

        assert len(client.streams) == 1, "a busy speaker must not be cut off"
        assert _rotation_lines(caplog) == []

    def test_the_hard_deadline_rotates_a_busy_session(self, faketime, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "10")
        monkeypatch.setenv("NOVA_SONIC_ROTATE_DEADLINE_S", "20")
        client = _FakeClient()
        sonic = _make_sonic(
            client,
            history_provider=lambda: list(TWO_BLOCKS),
            speaker_idle=lambda: False,
        )

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, faketime, fake_seconds=25.0)

        assert len(client.streams) == 2
        lines = _rotation_lines(caplog)
        assert len(lines) == 1
        assert "delay=0" in lines[0]

    def test_the_rotation_pays_no_backoff(self, faketime, monkeypatch):
        """delay=0 is a claim about the clock, not only about the log line."""
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "10")
        monkeypatch.setenv("NOVA_SONIC_ROTATE_DEADLINE_S", "20")
        client = _FakeClient()
        sonic = _make_sonic(client, speaker_idle=lambda: True)
        seen: list[float] = []
        original = NovaSonic._start_session

        async def _record(self):  # noqa: ANN001
            seen.append(faketime.mono)
            await original(self)

        monkeypatch.setattr(NovaSonic, "_start_session", _record)

        _run_loop_for(sonic, faketime, fake_seconds=14.0)

        assert len(seen) == 2, seen
        # Session 1 starts, arms at +0.5s (the settle pause), rotates at ~10.5s
        # and the next start begins immediately: no 3s base backoff in between.
        assert seen[1] - seen[0] < 12.0
        assert sonic._restart_attempt == 0, "a rotation must not escalate the backoff"

    def test_rotation_disabled_never_rotates(self, faketime, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_SONIC_ROTATE_S", "0")
        client = _FakeClient()
        sonic = _make_sonic(client, speaker_idle=lambda: True)

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, faketime, fake_seconds=60.0)

        assert len(client.streams) == 1
        assert _rotation_lines(caplog) == []

    def test_the_default_interval_leaves_a_short_session_alone(self, faketime, caplog):
        """420 s by default: a two-minute session is nowhere near a rotation."""
        client = _FakeClient()
        sonic = _make_sonic(client, speaker_idle=lambda: True)

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, faketime, fake_seconds=120.0)

        assert len(client.streams) == 1
        assert _rotation_lines(caplog) == []


# --------------------------------------------------------------------------
# 3. every restart replays — the replay lives in _start_session, so it must
# --------------------------------------------------------------------------


class TestEveryRestartReplays:
    def test_the_immediate_restart_path_replays(self, faketime, caplog):
        """``request_immediate_restart`` is the network-change path (t8)."""
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(TWO_BLOCKS))
        client._first_stream_hook = lambda: sonic.request_immediate_restart(
            "network joined"
        )

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _run_loop_for(sonic, faketime, fake_seconds=20.0)

        assert len(client.streams) == 2, "the immediate restart must have happened"
        fresh = client.streams[1]
        blocks = _history_blocks(fresh)
        assert [b["content"] for b in blocks] == [b["text"] for b in TWO_BLOCKS]
        assert all(b["interactive"] is False for b in blocks)

        replayed = [m for m in _messages(caplog, logging.INFO) if "history replayed" in m]
        assert replayed == ["history replayed blocks=2", "history replayed blocks=2"]

    def test_a_fresh_session_sends_nothing_between_history_and_audio(self, faketime):
        """c31: no assistant-initiating text on a replayed session."""
        client = _FakeClient()
        sonic = _make_sonic(client, history_provider=lambda: list(TWO_BLOCKS))
        client._first_stream_hook = lambda: sonic.request_immediate_restart(
            "network joined"
        )

        _run_loop_for(sonic, faketime, fake_seconds=20.0)

        fresh = client.streams[1]
        audio_at = _audio_start_index(fresh)
        # (the session's own closing contentEnd lives past the audio start,
        # so only the pre-audio window is interesting here)
        last_end = max(
            i for i, e in enumerate(fresh.sent[:audio_at]) if "contentEnd" in e
        )
        assert fresh.sent[last_end + 1 : audio_at] == []
        assert audio_at == last_end + 1

        # ...and the whole pre-audio handshake is exactly what we intended.
        assert [next(iter(e)) for e in fresh.sent[: audio_at + 1]] == [
            "sessionStart",
            "promptStart",
            "contentStart",
            "textInput",
            "contentEnd",
            "contentStart",
            "textInput",
            "contentEnd",
            "contentStart",
            "textInput",
            "contentEnd",
            "contentStart",
        ]

    def test_an_immediate_restart_without_a_provider_is_unchanged(self, faketime):
        """The t8 network path keeps its old shape when nothing is remembered."""
        client = _FakeClient()
        sonic = _make_sonic(client)
        client._first_stream_hook = lambda: sonic.request_immediate_restart(
            "network joined"
        )

        _run_loop_for(sonic, faketime, fake_seconds=20.0)

        assert len(client.streams) == 2
        fresh = client.streams[1]
        text_inputs = [e["textInput"] for e in fresh.sent if "textInput" in e]
        assert len(text_inputs) == 1
        assert text_inputs[0]["content"] == sonic.system_prompt
