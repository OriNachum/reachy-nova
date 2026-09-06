"""Eyes — naming a dead camera the runtime itself never mentions (t11).

Live finding (2026-09-06): ``reachy/events/sense/snapshot`` published
``frame_available: false`` in 251/251 and 435/435 samples while the runtime's
own availability report said the camera senses were "available" — and the
harness had no line naming this for two days.

Everything here runs with a fake clock and, for :class:`EyesComponent`, an
injected client factory: no broker, no network, no real time.sleep.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from reachy_nova.harness import eyes, supervisor


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """A throwaway REACHY_STATE_DIR so status() touches no real state."""
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    (tmp_path / "behavior").mkdir(parents=True, exist_ok=True)
    return tmp_path


def fake_msg(payload) -> SimpleNamespace:
    if isinstance(payload, dict):
        payload = json.dumps(payload, separators=(",", ":"))
    if isinstance(payload, str):
        payload = payload.encode()
    return SimpleNamespace(topic=eyes.SNAPSHOT_TOPIC, payload=payload)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, delta: float) -> None:
        self.t += delta


# --------------------------------------------------------------------------- #
# Eyes — the pure state machine                                               #
# --------------------------------------------------------------------------- #


def test_initial_state_is_unknown():
    assert eyes.Eyes().state == "unknown"


def test_true_from_unknown_goes_live_silently(caplog):
    e = eyes.Eyes(dead_after_s_value=60.0)
    with caplog.at_level("INFO"):
        e.note(True, now=0.0)
    assert e.state == "live"
    assert "[SENSE" not in caplog.text


def test_60s_of_false_latches_exactly_one_dropped_line(caplog):
    clock = FakeClock()
    e = eyes.Eyes(dead_after_s_value=60.0, clock=clock)
    with caplog.at_level("INFO"):
        for _ in range(61):
            e.note(False, now=clock.t)
            clock.advance(1.0)
    assert e.state == "dead"
    dropped_lines = [
        line for line in caplog.text.splitlines() if "dropped reason=no-frames" in line
    ]
    assert len(dropped_lines) == 1
    assert "after=60s" in dropped_lines[0]


def test_first_true_after_dead_restores_exactly_once_then_state_live(caplog):
    clock = FakeClock()
    e = eyes.Eyes(dead_after_s_value=60.0, clock=clock)
    for _ in range(61):
        e.note(False, now=clock.t)
        clock.advance(1.0)
    assert e.state == "dead"

    with caplog.at_level("INFO"):
        e.note(True, now=clock.t)
        # A second True while already live logs nothing further.
        e.note(True, now=clock.t)

    assert e.state == "live"
    restored_lines = [line for line in caplog.text.splitlines() if "restored after=" in line]
    assert len(restored_lines) == 1


def test_repeated_false_while_dead_logs_nothing_further(caplog):
    clock = FakeClock()
    e = eyes.Eyes(dead_after_s_value=60.0, clock=clock)
    for _ in range(61):
        e.note(False, now=clock.t)
        clock.advance(1.0)
    assert e.state == "dead"

    with caplog.at_level("INFO"):
        for _ in range(20):
            e.note(False, now=clock.t)
            clock.advance(1.0)

    dropped_lines = [
        line for line in caplog.text.splitlines() if "dropped reason=no-frames" in line
    ]
    assert dropped_lines == []


def test_full_cycle_dead_then_restored_then_dead_again(caplog):
    """60s False -> one dropped line; True -> one restored line, state live;
    60s False again -> one more dropped line (t11 acceptance criterion)."""
    clock = FakeClock()
    e = eyes.Eyes(dead_after_s_value=60.0, clock=clock)

    with caplog.at_level("INFO"):
        for _ in range(61):
            e.note(False, now=clock.t)
            clock.advance(1.0)
    assert e.state == "dead"
    assert len([ln for ln in caplog.text.splitlines() if "dropped reason=no-frames" in ln]) == 1

    caplog.clear()
    with caplog.at_level("INFO"):
        e.note(True, now=clock.t)
    assert e.state == "live"
    assert len([ln for ln in caplog.text.splitlines() if "restored after=" in ln]) == 1

    caplog.clear()
    with caplog.at_level("INFO"):
        for _ in range(61):
            e.note(False, now=clock.t)
            clock.advance(1.0)
    assert e.state == "dead"
    assert len([ln for ln in caplog.text.splitlines() if "dropped reason=no-frames" in ln]) == 1


def test_dead_after_s_env_parsed_defensively(monkeypatch):
    monkeypatch.setenv(eyes.DEAD_AFTER_ENV, "not-a-number")
    assert eyes.dead_after_s() == eyes.DEFAULT_DEAD_AFTER_S
    monkeypatch.setenv(eyes.DEAD_AFTER_ENV, "-5")
    assert eyes.dead_after_s() == eyes.DEFAULT_DEAD_AFTER_S
    monkeypatch.setenv(eyes.DEAD_AFTER_ENV, "0")
    assert eyes.dead_after_s() == eyes.DEFAULT_DEAD_AFTER_S
    monkeypatch.setenv(eyes.DEAD_AFTER_ENV, "  ")
    assert eyes.dead_after_s() == eyes.DEFAULT_DEAD_AFTER_S
    monkeypatch.delenv(eyes.DEAD_AFTER_ENV, raising=False)
    assert eyes.dead_after_s() == eyes.DEFAULT_DEAD_AFTER_S
    monkeypatch.setenv(eyes.DEAD_AFTER_ENV, "30")
    assert eyes.dead_after_s() == 30.0


def test_false_streak_resets_after_a_restoration_before_relatching():
    """A False stretch shorter than dead_after_s, after a restoration, must
    not immediately re-latch — the streak clock starts over on every True."""
    clock = FakeClock()
    e = eyes.Eyes(dead_after_s_value=60.0, clock=clock)
    for _ in range(61):
        e.note(False, now=clock.t)
        clock.advance(1.0)
    assert e.state == "dead"
    e.note(True, now=clock.t)
    assert e.state == "live"

    for _ in range(30):
        e.note(False, now=clock.t)
        clock.advance(1.0)
    assert e.state == "live"  # only 30s in — not dead yet


# --------------------------------------------------------------------------- #
# EyesComponent — the ~1Hz subscriber, no broker                              #
# --------------------------------------------------------------------------- #


class RecordingClient:
    def __init__(self) -> None:
        self.on_connect = None
        self.on_message = None
        self.subscriptions: list[str] = []
        self.connected_to: tuple | None = None
        self.loop_started = False
        self.loop_stopped = False
        self.disconnected = False

    def connect_async(self, host, port, keepalive=60):
        self.connected_to = (host, port)

    def loop_start(self):
        self.loop_started = True

    def loop_stop(self):
        self.loop_stopped = True

    def subscribe(self, topic, qos=0):
        self.subscriptions.append(topic)

    def disconnect(self):
        self.disconnected = True


def make_component(**kwargs) -> tuple[eyes.EyesComponent, RecordingClient]:
    client = RecordingClient()
    component = eyes.EyesComponent(client_factory=lambda: client, **kwargs)
    return component, client


def test_start_opens_its_own_subscription_and_never_raises():
    component, client = make_component()
    component.start(threading.Event())
    assert client.loop_started is True
    # _on_connect drives the actual subscribe call (paho semantics).
    component._on_connect(client)
    assert eyes.SNAPSHOT_TOPIC in client.subscriptions


def test_stop_disconnects_and_is_idempotent():
    component, client = make_component()
    component.start(threading.Event())
    component.stop()
    component.stop()
    assert client.disconnected is True


def test_one_hz_sampling_50_messages_in_one_second_yield_one_note():
    clock = FakeClock()
    component, client = make_component(clock=clock)
    component.start(threading.Event())
    for _ in range(50):
        component._on_message(client, None, fake_msg({"frame_available": False}))
    assert component.eyes.state == "unknown"  # only one note ever landed; well below 60s
    # Confirm exactly one sample was processed: advance past the dead_after_s
    # window using only ONE more (unthrottled, later) call and check it latches.
    clock.advance(eyes.SAMPLE_INTERVAL_S)
    for _ in range(50):
        component._on_message(client, None, fake_msg({"frame_available": False}))
        clock.advance(0.0)
    # Two processed samples total (t=0 and t=1.0) is nowhere near 60s of
    # continuous False, so state is still not dead — this asserts the ignored
    # 98 messages truly never reached Eyes.note.
    assert component.eyes.state == "unknown"


def test_decodes_json_and_notes_frame_available_true():
    clock = FakeClock()
    component, client = make_component(clock=clock)
    component.start(threading.Event())
    component._on_message(client, None, fake_msg({"frame_available": True, "t": "sense"}))
    assert component.eyes.state == "live"


def test_missing_frame_available_key_defaults_false():
    clock = FakeClock()
    component, client = make_component(clock=clock)
    component.start(threading.Event())
    component._on_message(client, None, fake_msg({"t": "sense"}))
    assert component.eyes.state == "unknown"  # one False note, nowhere near 60s


def test_non_json_payload_is_ignored_not_raised():
    clock = FakeClock()
    component, client = make_component(clock=clock)
    component.start(threading.Event())
    component._on_message(client, None, fake_msg(b"not json at all"))
    assert component.eyes.state == "unknown"


def test_broker_unreachable_factory_raises_degrades_to_unknown(caplog):
    def exploding_factory():
        raise OSError("no route to host")

    component = eyes.EyesComponent(client_factory=exploding_factory)
    with caplog.at_level("INFO"):
        component.start(threading.Event())  # must not raise
    assert component.eyes.state == "unknown"
    lines = [ln for ln in caplog.text.splitlines() if "component absent" in ln]
    assert len(lines) == 1
    assert "reason=broker-unreachable" in lines[0]


def test_broker_unreachable_connect_failure_degrades_to_unknown(caplog):
    class ExplodingClient(RecordingClient):
        def connect_async(self, host, port, keepalive=60):
            raise OSError("connection refused")

    client = ExplodingClient()
    component = eyes.EyesComponent(client_factory=lambda: client)
    with caplog.at_level("INFO"):
        component.start(threading.Event())  # must not raise
    assert component.eyes.state == "unknown"
    lines = [ln for ln in caplog.text.splitlines() if "component absent" in ln]
    assert len(lines) == 1


# --------------------------------------------------------------------------- #
# supervisor.status()'s new "eyes" field                                      #
# --------------------------------------------------------------------------- #


def test_status_reports_eyes_unknown_when_no_eyes_state_is_given(state_dir):
    assert supervisor.status()["eyes"] == "unknown"


def test_status_reports_the_eyes_current_state(state_dir):
    clock = FakeClock()
    e = eyes.Eyes(dead_after_s_value=60.0, clock=clock)
    assert supervisor.status(eyes_state=e)["eyes"] == "unknown"

    for _ in range(61):
        e.note(False, now=clock.t)
        clock.advance(1.0)
    assert supervisor.status(eyes_state=e)["eyes"] == "dead"

    e.note(True, now=clock.t)
    assert supervisor.status(eyes_state=e)["eyes"] == "live"


def test_find_eyes_state_discovers_it_on_a_component():
    e = eyes.Eyes()
    component = SimpleNamespace(eyes=e)
    assert supervisor._find_eyes_state([SimpleNamespace(), component]) is e


def test_find_eyes_state_returns_none_when_absent():
    assert supervisor._find_eyes_state([SimpleNamespace()]) is None
