"""Chaos: wifi down/up — the broker session flaps under a live NovaBus.

On the robot this is ``nmcli radio wifi off`` then ``on``: the localhost
broker itself survives, but a mesh/remote-broker deployment (and any broker
restart) drops the MQTT session mid-conversation. paho owns the reconnect;
what THIS suite pins is everything the harness must do around it, driven
through a fake client's recorded calls and hand-fired callbacks:

* the drop is a named ``[SENSE stage=bus ... event=disconnect]`` line;
* on reconnect ``_on_connect`` re-issues every subscription and re-publishes
  the retained ``nova/harness/state`` online payload (the fake client's
  recorded, observable effects);
* the broker's retained-message REPLAY on reconnect is dropped named
  (``reason=not-an-event-topic``) and never becomes an inject — the module
  docstring's rule 1, exercised in the exact situation it exists for;
* ``runtime_online`` tracks the retained ``reachy/state/online`` false/true
  flips (the runtime's Last Will firing during the flap, then its recovery);
* after the flap a fresh runtime event still routes through rules.yaml to an
  inject — the conversation works again, same bus instance.
"""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import pytest

from reachy_nova.harness import bus


def fake_msg(topic: str, payload) -> SimpleNamespace:
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, separators=(",", ":"))
    if isinstance(payload, str):
        payload = payload.encode()
    return SimpleNamespace(topic=topic, payload=payload)


class FlappyClient:
    """Records every paho call NovaBus makes; the test plays the network."""

    def __init__(self) -> None:
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None
        self.will: tuple | None = None
        self.subscriptions: list[str] = []
        self.published: list[tuple[str, str, bool]] = []
        self.loop_started = False
        self.loop_stopped = False
        self.disconnected = False

    def will_set(self, topic, payload, qos=0, retain=False):
        self.will = (topic, payload, qos, retain)

    def reconnect_delay_set(self, min_delay=1, max_delay=120):
        self.reconnect_delays = (min_delay, max_delay)

    def connect_async(self, host, port, keepalive=60):
        self.connected_to = (host, port)

    def loop_start(self):
        self.loop_started = True

    def loop_stop(self):
        self.loop_stopped = True

    def subscribe(self, topic, qos=0):
        self.subscriptions.append(topic)

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, retain))

    def disconnect(self):
        self.disconnected = True


RULE_FIRE = {
    "t": "rule",
    "ts": 1718362800.3,
    "tick": 15,
    "action": "fire",
    "rule": "hear",
    "kind": "react",
    "field": "speech",
    "op": "is_true",
    "reason": "fired",
    "behavior": "nod",
    "disable": [],
}


def bus_lines(caplog: pytest.LogCaptureFixture, stage: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "nova.sensory" and f"stage={stage}" in r.getMessage()
    ]


def online_state_publishes(client: FlappyClient) -> list[tuple[str, str, bool]]:
    return [
        p
        for p in client.published
        if p[0] == bus.HARNESS_STATE_TOPIC and '"status":"online"' in p[1]
    ]


def test_wifi_flap_full_arc_disconnect_replay_hygiene_and_recovery(caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    injects: list[str] = []
    client = FlappyClient()
    nb = bus.NovaBus(
        on_inject=injects.append,
        broker="localhost:1883",
        client_factory=lambda: client,
    )
    stop = threading.Event()
    nb.start(stop)
    assert client.loop_started
    assert client.will is not None and client.will[0] == bus.HARNESS_STATE_TOPIC

    # --- Phase 1: the session comes up; the conversation works.
    client.on_connect(client, None, None, 0)
    assert nb.connected is True
    first_subs = list(client.subscriptions)
    assert set(first_subs) == {
        "reachy/events/rule/#",
        "reachy/events/intent/#",
        "reachy/events/motion/#",
        bus.RUNTIME_ONLINE_TOPIC,
    }
    assert len(online_state_publishes(client)) == 1
    assert online_state_publishes(client)[0][2] is True  # retained
    nb.on_message(client, None, fake_msg("reachy/state/online", b"true"))
    assert nb.runtime_online is True
    nb.on_message(client, None, fake_msg("reachy/events/rule/fire", RULE_FIRE))
    assert len(injects) == 1, "the pre-flap conversation never worked"

    # --- Phase 2: wifi drops. Named, non-fatal, paho retries.
    client.on_disconnect(client, None, None, 7)
    assert nb.connected is False
    drop_lines = [m for m in bus_lines(caplog, "bus") if "event=disconnect" in m]
    assert drop_lines and "paho will retry with backoff" in drop_lines[0], (
        "the wifi drop was not a named [SENSE] line"
    )
    # The runtime's Last Will fires while we are away; delivered on reconnect.

    # --- Phase 3: wifi returns; paho re-fires on_connect on the same client.
    client.on_connect(client, None, None, 0)
    assert nb.connected is True
    # Observable recovery on the fake client: every filter re-subscribed...
    assert client.subscriptions == first_subs + first_subs, "no resubscribe on reconnect"
    # ...and the retained harness-state online payload re-published.
    assert len(online_state_publishes(client)) == 2
    connect_lines = [m for m in bus_lines(caplog, "bus") if "event=connect" in m]
    assert len(connect_lines) == 2, "the reconnect was not a named [SENSE] line"

    # The broker now REPLAYS retained state; it must be dropped NAMED and
    # never become a cue (the reconnect-replay hazard rules.yaml rule 1).
    nb.on_message(client, None, fake_msg("reachy/state/head", {"pitch": 4.2}))
    replay_drops = [
        m
        for m in bus_lines(caplog, "route")
        if f"dropped reason={bus.REASON_NOT_AN_EVENT}" in m
    ]
    assert replay_drops, "the retained replay was not a named reason= drop"
    assert len(injects) == 1, "a retained replay leaked into the conversation"

    # The runtime's Last Will (offline) then its own recovery (online).
    nb.on_message(client, None, fake_msg("reachy/state/online", b"false"))
    assert nb.runtime_online is False
    nb.on_message(client, None, fake_msg("reachy/state/online", b"true"))
    assert nb.runtime_online is True
    bus_events = "\n".join(bus_lines(caplog, "bus"))
    assert "event=runtime-offline" in bus_events
    assert bus_events.count("event=runtime-online") >= 1

    # --- Phase 4: recovery proven end to end — a fresh event injects again,
    #     on the SAME NovaBus instance.
    nb.on_message(client, None, fake_msg("reachy/events/rule/fire", RULE_FIRE))
    assert len(injects) == 2, "the post-flap conversation never recovered"

    stop.set()
    nb.join(timeout=2.0)
    assert client.disconnected and client.loop_stopped


def test_a_session_takeover_reconnect_is_named_reconnect(caplog) -> None:
    """paho can re-fire on_connect while the session is still marked live
    (network path change without a delivered disconnect — the wifi came back
    before the keepalive noticed it left). That is the one path that logs
    ``event=reconnect``, and it must still resubscribe."""
    caplog.set_level(logging.INFO, logger="nova.sensory")
    client = FlappyClient()
    nb = bus.NovaBus(
        on_inject=lambda text: None,
        broker="localhost:1883",
        client_factory=lambda: client,
    )
    stop = threading.Event()
    nb.start(stop)
    client.on_connect(client, None, None, 0)
    subs_after_first = len(client.subscriptions)

    client.on_connect(client, None, None, 0)  # takeover: no disconnect between

    assert any(
        "event=reconnect" in m for m in bus_lines(caplog, "bus")
    ), "a takeover reconnect was not named"
    assert len(client.subscriptions) == 2 * subs_after_first
    stop.set()
    nb.join(timeout=2.0)
