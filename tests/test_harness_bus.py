"""Harness bus subscriber + nervous-system prioritization (t6).

Everything here runs with **no broker and no network**: the module's pure
helpers (``parse_broker_url``, ``topic_filters``, ``resolve_sources``,
``route_event``) are separated from the paho client on purpose, and the
message path is exercised by calling :meth:`NovaBus.on_message` directly
with a fake ``msg`` object (``.topic`` + ``.payload``), exactly as paho
would. The only "client" that appears is a hand-rolled recorder injected
through ``client_factory``.

The wire contract under test is reachy-mini-cli 0.48.0's
(``docs/export-schema.md``, ``reachy/export/mqtt.py``):

- events: ``reachy/events/{source}/{type}``, QoS 0, NOT retained, where
  source is the runtime block type (``sense``/``rule``/``intent``/``motion``)
  and type is that block's action (``sense`` publishes as ``sense/snapshot``)
- retained availability: ``reachy/state/online`` = ``true``/``false``
- payloads: compact JSON with ``t`` and ``ts`` first
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from reachy_nova.harness import bus

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "config" / "nervous-system" / "rules.yaml"
MOSQUITTO_CONF = REPO_ROOT / "config" / "mosquitto" / "mosquitto.conf"


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


def fake_msg(topic: str, payload) -> SimpleNamespace:
    """A stand-in for paho's MQTTMessage: just ``.topic`` and ``.payload``."""
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, separators=(",", ":"))
    if isinstance(payload, str):
        payload = payload.encode()
    return SimpleNamespace(topic=topic, payload=payload)


class RecordingClient:
    """Records the paho calls :class:`NovaBus` makes, performs no I/O."""

    def __init__(self) -> None:
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None
        self.will: tuple | None = None
        self.subscriptions: list[str] = []
        self.published: list[tuple[str, str, bool]] = []
        self.connected_to: tuple[str, int] | None = None
        self.loop_started = False
        self.loop_stopped = False
        self.disconnected = False
        self.reconnect_delays: tuple | None = None

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


class Recorder:
    """Collects injects (and optionally raw events) from the bus."""

    def __init__(self) -> None:
        self.injects: list[str] = []
        self.events: list[dict] = []

    def on_inject(self, text: str) -> None:
        self.injects.append(text)

    def on_event(self, event: dict) -> None:
        self.events.append(event)


def make_bus(recorder: Recorder, **kwargs) -> bus.NovaBus:
    return bus.NovaBus(
        on_inject=recorder.on_inject,
        on_event=recorder.on_event,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# parse_broker_url                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mqtt://192.168.1.162:1883", ("192.168.1.162", 1883)),
        ("mqtts://broker.local:8883", ("broker.local", 8883)),
        ("192.168.1.162:1883", ("192.168.1.162", 1883)),
        ("localhost:1884", ("localhost", 1884)),
        ("localhost", ("localhost", 1883)),
        ("mqtt://localhost", ("localhost", 1883)),
        ("mqtt://localhost/", ("localhost", 1883)),
        ("  localhost:1883  ", ("localhost", 1883)),
        ("", ("localhost", 1883)),
    ],
)
def test_parse_broker_url_accepts_every_documented_form(raw, expected):
    assert bus.parse_broker_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["localhost:not-a-port", "localhost:", "localhost:0", "localhost:99999", "localhost:-5"],
)
def test_parse_broker_url_bad_port_defaults_and_never_raises(raw):
    """A typo in REACHY_MQTT_URL degrades the bus; it never stops the harness."""
    host, port = bus.parse_broker_url(raw)
    assert host == "localhost"
    assert port == bus.DEFAULT_PORT == 1883


def test_broker_url_reads_reachy_mqtt_url_with_a_loopback_default(monkeypatch):
    monkeypatch.delenv("REACHY_MQTT_URL", raising=False)
    assert bus.broker_url() == bus.DEFAULT_BROKER_URL == "localhost:1883"
    monkeypatch.setenv("REACHY_MQTT_URL", "mqtt://robot.local:1884")
    assert bus.broker_url() == "mqtt://robot.local:1884"
    assert bus.parse_broker_url(bus.broker_url()) == ("robot.local", 1884)


# --------------------------------------------------------------------------- #
# sources + topic filters                                                      #
# --------------------------------------------------------------------------- #


def test_resolve_sources_defaults_to_decisions_and_drops_the_sense_flood(monkeypatch):
    """Upstream measured 187 cues/40 s from `sense` alone — off by default."""
    monkeypatch.delenv("NOVA_BUS_SOURCES", raising=False)
    assert bus.resolve_sources(None) == ("rule", "intent", "motion", "pat", "face", "vision")
    assert "sense" not in bus.resolve_sources(None)


def test_topic_filters_include_pat_face_vision_by_default(monkeypatch):
    """t7: pat/face/vision are discrete senses, not the high-rate snapshot —
    they're on by default alongside rule/intent/motion."""
    monkeypatch.delenv("NOVA_BUS_SOURCES", raising=False)
    filters = bus.topic_filters(bus.resolve_sources(None))
    assert "reachy/events/pat/#" in filters
    assert "reachy/events/face/#" in filters
    assert "reachy/events/vision/#" in filters


def test_resolve_sources_parses_a_comma_list_and_drops_blanks():
    assert bus.resolve_sources("rule, sense ,,motion") == ("rule", "sense", "motion")


def test_topic_filters_are_events_only_never_the_retained_state_tree():
    filters = bus.topic_filters(("rule", "intent", "motion"))
    assert filters == (
        "reachy/events/rule/#",
        "reachy/events/intent/#",
        "reachy/events/motion/#",
    )
    assert bus.topic_filters(("*",)) == ("reachy/events/#",)
    for topic in filters + bus.topic_filters(("*",)):
        assert topic.startswith("reachy/events/")
        assert "reachy/state" not in topic


# --------------------------------------------------------------------------- #
# route_event against the REAL rules.yaml                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_rules() -> dict:
    return bus.load_rules(RULES_PATH)


def test_load_rules_finds_the_repo_rules_yaml_by_default():
    cfg = bus.load_rules()
    assert "rules" in cfg and "default" in cfg
    assert cfg["rules"]["rule/fire"]["inject_template"]


def test_rules_yaml_covers_the_runtime_bus_namespace(real_rules):
    """The runtime's own source/type pairs are explicit, not defaulted."""
    rules = real_rules["rules"]
    for key in (
        "rule/fire",
        "rule/suppress",
        "intent/applied",
        "intent/blocked",
        "motion/goto",
        "sense/snapshot",
        "pat/level1",
        "pat/level2",
        "pat/detected",
        "face/recognized",
        "face/unknown",
        "vision/description",
    ):
        assert key in rules, f"rules.yaml is missing runtime pair {key}"
        assert "priority" in rules[key]
        assert "urgency" in rules[key]
        assert "llm_evaluate" in rules[key]


def test_route_event_renders_the_template_from_the_payload(real_rules):
    payload = {
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
    text, reason = bus.route_event(real_rules, "rule", "fire", payload)
    assert reason == bus.REASON_INJECT
    assert text and "hear" in text


def test_route_event_missing_template_keys_render_empty_not_keyerror(real_rules):
    text, reason = bus.route_event(real_rules, "rule", "fire", {"t": "rule"})
    assert reason == bus.REASON_INJECT
    assert isinstance(text, str)
    assert "{" not in text  # nothing left unrendered


def test_route_event_without_a_template_reports_no_template(real_rules):
    text, reason = bus.route_event(real_rules, "rule", "suppress", {"rule": "hear"})
    assert text is None
    assert reason == bus.REASON_NO_TEMPLATE


def test_route_event_unknown_pair_falls_back_to_the_default_rule(real_rules):
    text, reason = bus.route_event(real_rules, "quantum", "entangle", {})
    assert text is None
    assert reason == bus.REASON_NO_TEMPLATE


def test_route_event_priority_of_returns_the_matched_rule_metadata(real_rules):
    fire = bus.rule_for(real_rules, "rule", "fire")
    assert fire["priority"] == "NORMAL"
    fallback = bus.rule_for(real_rules, "quantum", "entangle")
    assert fallback == real_rules["default"]


# --------------------------------------------------------------------------- #
# t7 — pat/face/vision runtime senses                                         #
# --------------------------------------------------------------------------- #


def test_route_event_pat_level1_names_the_touch(real_rules):
    text, reason = bus.route_event(real_rules, "pat", "level1", {"level": 1})
    assert reason == bus.REASON_INJECT
    assert text and "pat" in text.lower()


def test_route_event_pat_level2_names_the_touch(real_rules):
    text, reason = bus.route_event(real_rules, "pat", "level2", {"level": 2})
    assert reason == bus.REASON_INJECT
    assert text and "pat" in text.lower()


def test_route_event_face_recognized_names_who_was_seen(real_rules):
    text, reason = bus.route_event(real_rules, "face", "recognized", {"name": "Ori"})
    assert reason == bus.REASON_INJECT
    assert text and "Ori" in text


def test_route_event_vision_description_names_what_was_seen(real_rules):
    text, reason = bus.route_event(real_rules, "vision", "description", {"description": "a red mug"})
    assert reason == bus.REASON_INJECT
    assert text and "red mug" in text


def test_route_event_unknown_pat_type_falls_back_to_the_default_rule(real_rules):
    text, reason = bus.route_event(real_rules, "pat", "mystery", {})
    assert text is None
    assert reason == bus.REASON_NO_TEMPLATE


def test_route_event_unknown_face_type_falls_back_to_the_default_rule(real_rules):
    text, reason = bus.route_event(real_rules, "face", "mystery", {})
    assert text is None
    assert reason == bus.REASON_NO_TEMPLATE


# --------------------------------------------------------------------------- #
# on_message — the whole path, with a fake msg and no broker                    #
# --------------------------------------------------------------------------- #


def test_on_message_routes_a_rule_fire_through_rules_yaml_to_an_inject():
    """Acceptance: a fake reachy/events event reaches a recorded inject."""
    rec = Recorder()
    nb = make_bus(rec)
    nb.on_message(
        None,
        None,
        fake_msg(
            "reachy/events/rule/fire",
            {
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
            },
        ),
    )
    assert len(rec.injects) == 1
    assert "hear" in rec.injects[0]
    assert rec.events and rec.events[0]["rule"] == "hear"


def test_on_message_sense_snapshot_is_observed_but_never_injected():
    rec = Recorder()
    nb = make_bus(rec, sources="sense,rule")
    nb.on_message(
        None,
        None,
        fake_msg(
            "reachy/events/sense/snapshot",
            {
                "t": "sense",
                "ts": 1718362800.0,
                "tick": 1,
                "doa": None,
                "speech": False,
                "rms": None,
                "pat": None,
                "face": None,
                "frame_available": False,
            },
        ),
    )
    assert rec.injects == []
    assert len(rec.events) == 1


def test_on_message_routes_a_pat_event_to_an_inject_naming_the_touch():
    """Acceptance: a pat event pushed through NovaBus produces a Sonic inject
    whose text names the touch."""
    rec = Recorder()
    nb = make_bus(rec, sources="pat")
    nb.on_message(
        None,
        None,
        fake_msg("reachy/events/pat/level1", {"t": "pat", "ts": 1.0, "level": 1}),
    )
    assert len(rec.injects) == 1
    assert "pat" in rec.injects[0].lower()
    assert rec.events and rec.events[0]["source"] == "pat"


def test_on_message_routes_a_face_recognized_event_naming_who_was_seen():
    rec = Recorder()
    nb = make_bus(rec, sources="face")
    nb.on_message(
        None,
        None,
        fake_msg("reachy/events/face/recognized", {"t": "face", "ts": 1.0, "name": "Ori"}),
    )
    assert len(rec.injects) == 1
    assert "Ori" in rec.injects[0]


def test_on_message_unknown_pat_type_is_a_named_drop_not_an_inject(caplog):
    rec = Recorder()
    nb = make_bus(rec, sources="pat")
    with caplog.at_level("INFO", logger="nova.sensory"):
        nb.on_message(
            None,
            None,
            fake_msg("reachy/events/pat/mystery", {"t": "pat", "ts": 1.0}),
        )
    assert rec.injects == []
    assert any(f"reason={bus.REASON_NO_TEMPLATE}" in r.getMessage() for r in caplog.records)


def test_on_message_unknown_face_type_is_a_named_drop_not_an_inject(caplog):
    rec = Recorder()
    nb = make_bus(rec, sources="face")
    with caplog.at_level("INFO", logger="nova.sensory"):
        nb.on_message(
            None,
            None,
            fake_msg("reachy/events/face/mystery", {"t": "face", "ts": 1.0}),
        )
    assert rec.injects == []
    assert any(f"reason={bus.REASON_NO_TEMPLATE}" in r.getMessage() for r in caplog.records)


def test_on_message_skips_an_unknown_block_type_gracefully():
    rec = Recorder()
    nb = make_bus(rec)
    nb.on_message(
        None,
        None,
        fake_msg("reachy/events/telepathy/hunch", {"t": "telepathy", "ts": 1.0}),
    )
    assert rec.injects == []
    assert rec.events == []


def test_on_message_tolerates_a_non_json_payload():
    rec = Recorder()
    nb = make_bus(rec)
    nb.on_message(None, None, fake_msg("reachy/events/rule/fire", b"not json at all"))
    assert rec.injects == []
    assert rec.events == []


def test_on_message_never_treats_retained_state_as_a_fresh_cue():
    """reachy/state/* is a last-value tree: it must never become an inject."""
    rec = Recorder()
    nb = make_bus(rec)
    nb.on_message(None, None, fake_msg("reachy/state/doa", {"doa": 1.2}))
    assert rec.injects == []
    assert rec.events == []


def test_runtime_online_tracks_the_retained_availability_topic():
    rec = Recorder()
    nb = make_bus(rec)
    assert nb.runtime_online is False
    nb.on_message(None, None, fake_msg("reachy/state/online", b"true"))
    assert nb.runtime_online is True
    nb.on_message(None, None, fake_msg("reachy/state/online", b"false"))
    assert nb.runtime_online is False
    assert rec.injects == []  # availability is state, not a sense


def test_on_message_is_exception_safe_when_the_inject_callback_raises():
    def boom(_text: str) -> None:
        raise RuntimeError("sonic is down")

    nb = bus.NovaBus(on_inject=boom)
    nb.on_message(
        None,
        None,
        fake_msg("reachy/events/rule/fire", {"t": "rule", "ts": 0.0, "rule": "hear"}),
    )  # must not propagate onto paho's network thread


# --------------------------------------------------------------------------- #
# Lifecycle with an injected client (still no broker)                          #
# --------------------------------------------------------------------------- #


def test_start_sets_the_last_will_subscribes_events_and_publishes_online(monkeypatch):
    monkeypatch.setenv("REACHY_MQTT_URL", "mqtt://127.0.0.1:1883")
    client = RecordingClient()
    rec = Recorder()
    nb = make_bus(rec, client_factory=lambda: client)
    stop_event = threading.Event()
    nb.start(stop_event)

    # Last Will is armed BEFORE connect.
    assert client.will is not None
    will_topic, will_payload, _qos, will_retain = client.will
    assert will_topic == bus.HARNESS_STATE_TOPIC
    assert will_retain is True
    assert json.loads(will_payload)["status"] == "offline"
    assert client.connected_to == ("127.0.0.1", 1883)
    assert client.loop_started is True

    client.on_connect(client, None, None, 0, None)
    assert set(client.subscriptions) == {
        "reachy/events/rule/#",
        "reachy/events/intent/#",
        "reachy/events/motion/#",
        "reachy/events/pat/#",
        "reachy/events/face/#",
        "reachy/events/vision/#",
        bus.RUNTIME_ONLINE_TOPIC,
    }
    online = [p for p in client.published if p[0] == bus.HARNESS_STATE_TOPIC]
    assert online and json.loads(online[-1][1])["status"] == "online"
    assert online[-1][2] is True  # retained
    nb.stop()


def test_stop_publishes_offline_and_tears_the_client_down():
    client = RecordingClient()
    nb = make_bus(Recorder(), client_factory=lambda: client)
    nb.start(threading.Event())
    client.on_connect(client, None, None, 0, None)
    nb.stop()
    assert json.loads(client.published[-1][1])["status"] == "offline"
    assert client.published[-1][2] is True
    assert client.disconnected is True
    assert client.loop_stopped is True


def test_stop_event_stops_the_bus_without_an_explicit_stop_call():
    client = RecordingClient()
    nb = make_bus(Recorder(), client_factory=lambda: client)
    stop_event = threading.Event()
    nb.start(stop_event)
    stop_event.set()
    nb.join(timeout=5.0)
    assert client.disconnected is True


def test_lifecycle_transitions_log_one_sense_bus_line_each(caplog):
    client = RecordingClient()
    nb = make_bus(Recorder(), client_factory=lambda: client)
    with caplog.at_level("INFO", logger="nova.sensory"):
        nb.start(threading.Event())
        client.on_connect(client, None, None, 0, None)
        client.on_disconnect(client, None, None, 7, None)
        nb.on_message(None, None, fake_msg(bus.RUNTIME_ONLINE_TOPIC, b"true"))
        nb.stop()
    lines = [r.getMessage() for r in caplog.records]
    bus_lines = [line for line in lines if "[SENSE stage=bus source=nova" in line]
    assert any("event=connect" in line for line in bus_lines)
    assert any("event=disconnect" in line for line in bus_lines)
    assert any("event=runtime-online" in line for line in bus_lines)


def test_a_dropped_event_names_its_reason_on_the_sense_log(caplog):
    rec = Recorder()
    nb = make_bus(rec)
    with caplog.at_level("INFO", logger="nova.sensory"):
        nb.on_message(
            None,
            None,
            fake_msg("reachy/events/rule/suppress", {"t": "rule", "ts": 0.0, "rule": "hear"}),
        )
    assert any(f"reason={bus.REASON_NO_TEMPLATE}" in r.getMessage() for r in caplog.records)
    assert rec.injects == []


def test_start_never_raises_when_the_client_cannot_connect():
    class ExplodingClient(RecordingClient):
        def connect_async(self, host, port, keepalive=60):
            raise OSError("no route to host")

    nb = make_bus(Recorder(), client_factory=ExplodingClient)
    nb.start(threading.Event())  # degrades, never raises
    assert nb.connected is False
    nb.stop()


# --------------------------------------------------------------------------- #
# Broker config                                                                #
# --------------------------------------------------------------------------- #


def test_mosquitto_listener_binds_loopback_only():
    """allow_anonymous is only acceptable while the listener is localhost-only."""
    text = MOSQUITTO_CONF.read_text()
    listeners = [
        line.strip() for line in text.splitlines() if line.strip().startswith("listener ")
    ]
    assert listeners, "no listener directive in mosquitto.conf"
    for line in listeners:
        assert "127.0.0.1" in line, f"listener is not loopback-bound: {line!r}"


def test_rules_yaml_still_parses_as_yaml():
    with open(RULES_PATH) as handle:
        raw = yaml.safe_load(handle)
    assert isinstance(raw["rules"], dict)
