"""Session configuration and event visibility on the Nova Sonic stream (t2).

Two things the harness never told Bedrock, and one thing Bedrock never told
the harness:

1. **Turn detection.** Nova 2 Sonic's ``sessionStart`` accepts
   ``turnDetectionConfiguration.endpointingSensitivity`` (``HIGH``,
   ``MEDIUM``, ``LOW``); the harness sent only ``inferenceConfiguration``, so
   every reply waited on the service default's idea of a finished sentence.
   ``NOVA_SONIC_ENDPOINTING`` now selects it, defaulting to ``HIGH`` — the
   fast end of the trade ("detects pauses quickly, enabling faster responses
   but may cut off slower speakers").
2. **contentEnd.** On the robot (2026-09-05) Sonic's ASSISTANT ``contentEnd``
   never ended the speaking state: every one of five consecutive utterances
   was flushed by the 4 s speaking watchdog instead. Whether the event never
   arrives, arrives with another ``role``/``type``, or carries a
   ``stopReason`` the code ignores is not knowable from the logs as they
   were, so every ``contentEnd`` now names all three at INFO.

The tests drive the two real seams — ``_start_session`` against a fake
client, ``_process_responses`` against a scripted event stream — rather than
asserting on a re-implementation of either.
"""

from __future__ import annotations

import asyncio
import json
import logging
import types

import pytest

from reachy_nova import nova_sonic
from reachy_nova.nova_sonic import NovaSonic

# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class _NoSleepAsyncio:
    """``asyncio`` proxy whose ``sleep`` returns immediately.

    ``_start_session`` ends with a 0.5 s settle pause before it accepts
    traffic; that pause is real behaviour worth keeping and dead weight in a
    unit test, so it is skipped rather than waited on. Everything else is
    delegated to the real module.
    """

    def __getattr__(self, name):  # pragma: no cover - trivial delegation
        return getattr(asyncio, name)

    async def sleep(self, delay: float = 0, result=None):
        await asyncio.sleep(0)
        return result


class _FakeInputStream:
    def __init__(self, sent: list[dict]):
        self._sent = sent

    async def send(self, chunk) -> None:
        self._sent.append(json.loads(chunk.value.bytes_.decode("utf-8"))["event"])

    async def close(self) -> None:  # pragma: no cover - not exercised here
        pass


class _FakeStream:
    """Accepts every input event and never answers — enough for a session start."""

    def __init__(self):
        self.sent: list[dict] = []
        self.input_stream = _FakeInputStream(self.sent)


class _FakeClient:
    """Hands out a fresh ``_FakeStream`` per session and keeps them all."""

    def __init__(self):
        self.streams: list[_FakeStream] = []

    async def invoke_model_with_bidirectional_stream(self, _operation_input):
        stream = _FakeStream()
        self.streams.append(stream)
        return stream


class _ScriptedStream:
    """Replays a fixed list of response events, then dies.

    "Dies" means ``receive()`` raises, which is what actually ends
    ``_process_responses``'s loop (a hang is indistinguishable from a healthy
    quiet connection — that shape belongs to the liveness tests).
    """

    def __init__(self, events: list[dict]):
        self._events = list(events)

    async def await_output(self):
        return (None, self)

    async def receive(self):
        if not self._events:
            raise ConnectionError("end of script")
        return _chunk(self._events.pop(0))


def _chunk(event: dict):
    """Wrap one Sonic output event in the shape ``_process_responses`` unwraps."""

    class _Value:
        bytes_ = json.dumps({"event": event}).encode("utf-8")

    class _Chunk:
        value = _Value()

    return _Chunk()


def _make_sonic(client: _FakeClient | None = None, **kwargs) -> NovaSonic:
    sonic = NovaSonic(system_prompt="You are Nova, a small curious robot.", **kwargs)
    if client is not None:
        sonic._init_client = lambda: setattr(sonic, "_client", client)  # type: ignore[method-assign]
    return sonic


def _start_session(sonic: NovaSonic) -> None:
    asyncio.run(sonic._start_session())


def _drive_responses(sonic: NovaSonic, events: list[dict]) -> None:
    """Run ``_process_responses`` over a scripted event list and return."""
    sonic._stream = _ScriptedStream(events)
    sonic._active = True
    asyncio.run(sonic._process_responses())


def _messages(caplog, level: int) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "reachy_nova.nova_sonic" and r.levelno == level
    ]


@pytest.fixture(autouse=True)
def _no_ambient_endpointing(monkeypatch):
    """A developer's exported ``NOVA_SONIC_ENDPOINTING`` must not flip the default."""
    monkeypatch.delenv("NOVA_SONIC_ENDPOINTING", raising=False)


@pytest.fixture
def nosleep(monkeypatch):
    """Skip ``_start_session``'s settle pause without touching anything else."""
    monkeypatch.setattr(nova_sonic, "asyncio", _NoSleepAsyncio())


# --------------------------------------------------------------------------
# 1. endpointing sensitivity — env parsing
# --------------------------------------------------------------------------


class TestEndpointingSensitivity:
    def test_unset_means_high(self):
        assert nova_sonic._endpointing_sensitivity() == "HIGH"

    @pytest.mark.parametrize("value", ["HIGH", "MEDIUM", "LOW"])
    def test_each_documented_value_passes_through(self, monkeypatch, value):
        monkeypatch.setenv("NOVA_SONIC_ENDPOINTING", value)
        assert nova_sonic._endpointing_sensitivity() == value

    @pytest.mark.parametrize("raw,expected", [("low", "LOW"), ("Medium", "MEDIUM"), (" high ", "HIGH")])
    def test_case_and_whitespace_insensitive(self, monkeypatch, raw, expected):
        monkeypatch.setenv("NOVA_SONIC_ENDPOINTING", raw)
        assert nova_sonic._endpointing_sensitivity() == expected

    def test_empty_means_the_default_without_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_SONIC_ENDPOINTING", "")
        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            assert nova_sonic._endpointing_sensitivity() == "HIGH"
        assert _messages(caplog, logging.WARNING) == []

    def test_unrecognised_value_warns_and_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_SONIC_ENDPOINTING", "VERY-HIGH")
        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            assert nova_sonic._endpointing_sensitivity() == "HIGH"
        warnings = _messages(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert "NOVA_SONIC_ENDPOINTING" in warnings[0]
        assert "VERY-HIGH" in warnings[0]
        assert "HIGH" in warnings[0]

    def test_read_at_call_time(self, monkeypatch):
        """No import-time capture — ``load_dotenv()`` order must not matter."""
        assert nova_sonic._endpointing_sensitivity() == "HIGH"
        monkeypatch.setenv("NOVA_SONIC_ENDPOINTING", "LOW")
        assert nova_sonic._endpointing_sensitivity() == "LOW"


# --------------------------------------------------------------------------
# 2. endpointing sensitivity — what sessionStart actually carries
# --------------------------------------------------------------------------


def _session_start(stream: _FakeStream) -> dict:
    starts = [e["sessionStart"] for e in stream.sent if "sessionStart" in e]
    assert len(starts) == 1, "exactly one sessionStart per session"
    return starts[0]


class TestSessionStartPayload:
    def test_default_payload_asks_for_high(self, nosleep):
        client = _FakeClient()
        sonic = _make_sonic(client)

        _start_session(sonic)

        turn = _session_start(client.streams[0])["turnDetectionConfiguration"]
        assert turn == {"endpointingSensitivity": "HIGH"}

    def test_env_value_reaches_the_payload(self, nosleep, monkeypatch):
        monkeypatch.setenv("NOVA_SONIC_ENDPOINTING", "low")
        client = _FakeClient()
        sonic = _make_sonic(client)

        _start_session(sonic)

        turn = _session_start(client.streams[0])["turnDetectionConfiguration"]
        assert turn == {"endpointingSensitivity": "LOW"}

    def test_unrecognised_value_sends_high_and_warns(self, nosleep, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_SONIC_ENDPOINTING", "aggressive")
        client = _FakeClient()
        sonic = _make_sonic(client)

        with caplog.at_level(logging.WARNING, logger="reachy_nova.nova_sonic"):
            _start_session(sonic)

        turn = _session_start(client.streams[0])["turnDetectionConfiguration"]
        assert turn == {"endpointingSensitivity": "HIGH"}
        warnings = [m for m in _messages(caplog, logging.WARNING) if "ENDPOINTING" in m]
        assert len(warnings) == 1

    def test_one_info_line_names_the_value_in_force(self, nosleep, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_SONIC_ENDPOINTING", "MEDIUM")
        client = _FakeClient()
        sonic = _make_sonic(client)

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _start_session(sonic)

        named = [m for m in _messages(caplog, logging.INFO) if "endpointing=" in m]
        assert len(named) == 1, "exactly one endpointing line per session start"
        assert "endpointing=MEDIUM" in named[0]

    def test_every_session_start_names_it_again(self, nosleep, caplog):
        """Each restart is a fresh session, so each one re-reads and re-logs."""
        client = _FakeClient()
        sonic = _make_sonic(client)

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _start_session(sonic)
            _start_session(sonic)

        named = [m for m in _messages(caplog, logging.INFO) if "endpointing=" in m]
        assert len(named) == 2
        assert all("endpointing=HIGH" in m for m in named)

    def test_inference_configuration_is_untouched(self, nosleep):
        client = _FakeClient()
        sonic = _make_sonic(client)

        _start_session(sonic)

        payload = _session_start(client.streams[0])
        assert payload["inferenceConfiguration"] == {
            "maxTokens": 1024,
            "topP": 0.9,
            "temperature": 0.7,
        }

    def test_the_rest_of_the_handshake_is_unchanged(self, nosleep):
        """sessionStart, promptStart, the SYSTEM content, then the audio channel."""
        client = _FakeClient()
        sonic = _make_sonic(client)

        _start_session(sonic)

        stream = client.streams[0]
        assert [next(iter(e)) for e in stream.sent] == [
            "sessionStart",
            "promptStart",
            "contentStart",
            "textInput",
            "contentEnd",
            "contentStart",
        ]
        roles = [e["contentStart"].get("role") for e in stream.sent if "contentStart" in e]
        assert roles == ["SYSTEM", "USER"]


# --------------------------------------------------------------------------
# 3. contentEnd visibility
# --------------------------------------------------------------------------


def _content_end_lines(caplog) -> list[str]:
    return [m for m in _messages(caplog, logging.INFO) if m.startswith("contentEnd ")]


class TestContentEndLogging:
    def test_assistant_audio_end_names_type_role_and_stop_reason(self, caplog):
        sonic = _make_sonic()

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _drive_responses(sonic, [
                {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
            ])

        assert _content_end_lines(caplog) == [
            "contentEnd type=AUDIO role=ASSISTANT stopReason=END_TURN"
        ]

    def test_every_content_end_is_logged(self, caplog):
        sonic = _make_sonic()

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _drive_responses(sonic, [
                {"contentEnd": {"type": "TEXT", "role": "USER", "stopReason": "END_TURN"}},
                {"contentEnd": {"type": "TEXT", "role": "ASSISTANT", "stopReason": "PARTIAL_TURN"}},
                {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "INTERRUPTED"}},
            ])

        assert _content_end_lines(caplog) == [
            "contentEnd type=TEXT role=USER stopReason=END_TURN",
            "contentEnd type=TEXT role=ASSISTANT stopReason=PARTIAL_TURN",
            "contentEnd type=AUDIO role=ASSISTANT stopReason=INTERRUPTED",
        ]

    def test_missing_fields_still_produce_one_line(self, caplog):
        """The interesting case is a contentEnd that names *less* than expected."""
        sonic = _make_sonic()

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _drive_responses(sonic, [{"contentEnd": {}}])

        lines = _content_end_lines(caplog)
        assert len(lines) == 1
        assert "type=" in lines[0] and "role=" in lines[0] and "stopReason=" in lines[0]

    def test_assistant_end_still_ends_the_speaking_state(self, caplog):
        """Logging is additive: the branch's existing effect is unchanged."""
        states: list[str] = []
        sonic = _make_sonic(on_state_change=states.append)
        sonic._speaking = True

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _drive_responses(sonic, [
                {"contentEnd": {"type": "AUDIO", "role": "ASSISTANT", "stopReason": "END_TURN"}},
            ])

        assert sonic._speaking is False
        assert states == ["listening"]
        assert len(_content_end_lines(caplog)) == 1

    def test_tool_end_still_fires_the_tool_callback(self, caplog):
        """A TOOL contentEnd is logged AND still completes the tool use."""
        calls: list[tuple[str, str, dict]] = []
        sonic = _make_sonic(on_tool_use=lambda name, tid, params: calls.append((name, tid, params)))

        with caplog.at_level(logging.INFO, logger="reachy_nova.nova_sonic"):
            _drive_responses(sonic, [
                {"contentStart": {"type": "TOOL", "toolUseId": "tu-1"}},
                {"toolUse": {"toolName": "look_at", "toolUseId": "tu-1", "content": '{"target": "left"}'}},
                {"contentEnd": {"type": "TOOL", "role": "TOOL", "stopReason": "TOOL_USE"}},
            ])

        assert calls == [("look_at", "tu-1", {"target": "left"})]
        assert _content_end_lines(caplog) == [
            "contentEnd type=TOOL role=TOOL stopReason=TOOL_USE"
        ]

    def test_a_user_end_does_not_end_the_speaking_state(self):
        """Unchanged behaviour — only ASSISTANT flips the guard."""
        sonic = _make_sonic()
        sonic._speaking = True

        _drive_responses(sonic, [
            {"contentEnd": {"type": "TEXT", "role": "USER", "stopReason": "END_TURN"}},
        ])

        assert sonic._speaking is True


# --------------------------------------------------------------------------
# 4. the module still imports cleanly under a hostile environment
# --------------------------------------------------------------------------


def test_endpointing_helper_is_module_level(monkeypatch):
    """The helper is a plain function, callable without an instance."""
    assert isinstance(nova_sonic._endpointing_sensitivity, types.FunctionType)
