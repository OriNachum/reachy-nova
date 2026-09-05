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
from reachy_nova.harness.quiet import QuietState
from reachy_nova.harness.sense_history import SenseHistory

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
# Retained clip state (t10 — the vision leg's read seam)                       #
# --------------------------------------------------------------------------- #

_CLIP_AVAILABLE = {
    "available": True,
    "reason": None,
    "path": "/tmp/clip.mp4",
    "ts": 1718362800.0,
    "duration_s": 4.0,
    "frame_count": 60,
}

_CLIP_UNAVAILABLE = {
    "available": False,
    "reason": "vision-extra-absent",
    "path": None,
    "ts": 1718362801.0,
    "duration_s": None,
    "frame_count": 0,
}


def test_clip_state_starts_none_and_caches_the_retained_payload():
    rec = Recorder()
    nb = make_bus(rec)
    assert nb.clip_state() is None
    nb.on_message(None, None, fake_msg(bus.CLIP_STATE_TOPIC, _CLIP_AVAILABLE))
    state = nb.clip_state()
    assert state is not None
    assert state["available"] is True
    assert state["path"] == "/tmp/clip.mp4"
    # State, never a cue: no inject, no raw-event tap.
    assert rec.injects == []
    assert rec.events == []


def test_clip_state_returns_a_copy_not_the_cache():
    nb = make_bus(Recorder())
    nb.on_message(None, None, fake_msg(bus.CLIP_STATE_TOPIC, _CLIP_AVAILABLE))
    state = nb.clip_state()
    state["path"] = "/mutated"
    assert nb.clip_state()["path"] == "/tmp/clip.mp4"


def test_clip_state_bad_payload_is_a_named_drop_that_keeps_the_last_good(caplog):
    nb = make_bus(Recorder())
    nb.on_message(None, None, fake_msg(bus.CLIP_STATE_TOPIC, _CLIP_AVAILABLE))
    with caplog.at_level("INFO", logger="nova.sensory"):
        nb.on_message(None, None, fake_msg(bus.CLIP_STATE_TOPIC, b"not json"))
        nb.on_message(None, None, fake_msg(bus.CLIP_STATE_TOPIC, b'["a", "list"]'))
    assert any(
        f"reason={bus.REASON_BAD_PAYLOAD}" in r.getMessage() for r in caplog.records
    )
    assert nb.clip_state()["path"] == "/tmp/clip.mp4"  # last good survives


def test_clip_state_availability_transitions_log_once_each(caplog):
    nb = make_bus(Recorder())
    with caplog.at_level("INFO", logger="nova.sensory"):
        nb.on_message(None, None, fake_msg(bus.CLIP_STATE_TOPIC, _CLIP_UNAVAILABLE))
        nb.on_message(None, None, fake_msg(bus.CLIP_STATE_TOPIC, _CLIP_UNAVAILABLE))
        nb.on_message(None, None, fake_msg(bus.CLIP_STATE_TOPIC, _CLIP_AVAILABLE))
    lines = [
        r.getMessage()
        for r in caplog.records
        if "event=clip-state" in r.getMessage() and "dropped" not in r.getMessage()
    ]
    assert len(lines) == 2  # one per TRANSITION, not one per retained replay
    assert "unavailable" in lines[0] and "vision-extra-absent" in lines[0]
    assert "available" in lines[1] and "/tmp/clip.mp4" in lines[1]


# --------------------------------------------------------------------------- #
# Per-rule rule/fire overrides (t10)                                           #
# --------------------------------------------------------------------------- #


def test_rule_for_per_rule_override_key_wins_and_falls_back():
    cfg = {
        "rules": {
            "rule/fire": {"priority": "NORMAL", "inject_template": "generic {rule}"},
            "rule/fire:pat-acknowledge": {
                "priority": "HIGH",
                "inject_template": "someone pets you",
            },
        },
        "default": {"priority": "LOW"},
    }
    override = bus.rule_for(cfg, "rule", "fire", {"rule": "pat-acknowledge"})
    assert override["inject_template"] == "someone pets you"
    fallback = bus.rule_for(cfg, "rule", "fire", {"rule": "anything-else"})
    assert fallback["inject_template"] == "generic {rule}"
    # No payload / no rule name — the generic entry, exactly as before.
    assert bus.rule_for(cfg, "rule", "fire")["inject_template"] == "generic {rule}"
    assert bus.rule_for(cfg, "rule", "fire", {"rule": ""})["inject_template"] == "generic {rule}"


def test_route_event_rule_fire_pat_acknowledge_reads_as_touch(real_rules):
    """The deployed pat path: pat reaches Nova as pat-acknowledge's rule/fire."""
    text, reason = bus.route_event(
        real_rules, "rule", "fire", {"t": "rule", "rule": "pat-acknowledge"}
    )
    assert reason == bus.REASON_INJECT
    assert "pet" in text.lower()
    assert "reflex fired" not in text  # the sensory override, not the generic line


def test_route_event_rule_fire_nova_face_noticed_reads_as_sight(real_rules):
    """The harness's standing face rule: its fire is the face sense on the bus."""
    text, reason = bus.route_event(
        real_rules, "rule", "fire", {"t": "rule", "rule": "nova-face-noticed"}
    )
    assert reason == bus.REASON_INJECT
    assert "face" in text.lower()
    assert "reflex fired" not in text


def test_route_event_rule_fire_unknown_rule_keeps_the_generic_template(real_rules):
    """t6: the generic rule/fire template dropped "reflex fired" narration
    in favor of quiet situational context, but still names the rule that
    fired and still always injects (voice: silent, not dropped).

    Uses a rule id with no per-rule override — NOT "look-toward-sound",
    which t14 gave its own `rule/fire:look-toward-sound` override (see
    test_rules_voice.py) precisely so it stops falling through to this
    generic template.
    """
    text, reason = bus.route_event(
        real_rules, "rule", "fire", {"t": "rule", "rule": "some-unmapped-behavior"}
    )
    assert reason == bus.REASON_INJECT
    assert "some-unmapped-behavior" in text
    assert "reflex" not in text.lower()
    assert text.endswith(bus.VOICE_MARKERS["silent"])


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
        bus.CLIP_STATE_TOPIC,
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


# --------------------------------------------------------------------------- #
# SenseHistory wiring (t8) — every inject that reaches on_inject is recorded  #
# --------------------------------------------------------------------------- #


def _custom_rules_path(tmp_path):
    cfg = {
        "rules": {
            "pat/level1": {
                "priority": "HIGH",
                "urgency": "IMMEDIATE",
                "inject_template": "someone is petting you",
                "sense": "touch",
                "voice": "brief",
            },
            "face/recognized": {
                "priority": "NORMAL",
                "urgency": "DEFERRABLE",
                "inject_template": "{name} is looking at you",
            },
            "rule/fire": {
                "priority": "LOW",
                "urgency": "DEFERRABLE",
                "inject_template": "a reflex fired: {rule}",
            },
        },
        "default": {"priority": "NORMAL", "urgency": "DEFERRABLE"},
    }
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_history_records_a_routed_inject_with_source_type_and_text(tmp_path):
    rec = Recorder()
    history = SenseHistory()
    nb = make_bus(rec, rules_path=_custom_rules_path(tmp_path), history=history)
    nb.on_message(
        None, None, fake_msg("reachy/events/pat/level1", {"t": "pat", "ts": 1.0, "level": 1})
    )
    (entry,) = history.recent()
    assert entry["source"] == "pat"
    assert entry["type"] == "level1"
    assert entry["text"] == rec.injects[0]
    assert entry["sense_class"] == "touch"
    assert entry["voice"] == "brief"


def test_history_records_three_routed_senses_in_order():
    rec = Recorder()
    history = SenseHistory()
    nb = make_bus(rec, history=history, sources="pat,face,rule")
    nb.on_message(
        None, None, fake_msg("reachy/events/pat/level1", {"t": "pat", "ts": 1.0, "level": 1})
    )
    nb.on_message(
        None,
        None,
        fake_msg("reachy/events/face/recognized", {"t": "face", "ts": 2.0, "name": "Ori"}),
    )
    nb.on_message(
        None,
        None,
        fake_msg(
            "reachy/events/rule/fire",
            {"t": "rule", "ts": 3.0, "rule": "hear-something-different"},
        ),
    )
    entries = history.recent(3)
    # newest first — the third routed event is entries[0]
    assert [e["source"] for e in entries] == ["rule", "face", "pat"]
    assert [e["type"] for e in entries] == ["fire", "recognized", "level1"]
    # timestamps are strictly increasing in recording order, so newest-first
    # is descending.
    ts = [e["t"] for e in entries]
    assert ts == sorted(ts, reverse=True)
    assert len(ts) == len(set(ts))


def test_history_does_not_record_a_no_template_drop():
    rec = Recorder()
    history = SenseHistory()
    nb = make_bus(rec, history=history, sources="sense")
    nb.on_message(
        None,
        None,
        fake_msg(
            "reachy/events/sense/snapshot",
            {"t": "sense", "ts": 1.0, "tick": 1, "doa": None},
        ),
    )
    assert rec.injects == []
    assert history.recent() == []


def test_history_does_not_record_a_deduped_suppressed_inject(tmp_path):
    rec = Recorder()
    history = SenseHistory()
    clock_state = {"t": 0.0}
    nb = make_bus(
        rec,
        rules_path=_custom_rules_path(tmp_path),
        history=history,
        clock=lambda: clock_state["t"],
    )
    msg = fake_msg("reachy/events/pat/level1", {"t": "pat", "ts": 1.0, "level": 1})
    nb.on_message(None, None, msg)
    clock_state["t"] += 1.0  # well inside the default 10s dedupe window
    nb.on_message(None, None, msg)
    assert len(rec.injects) == 1
    assert len(history.recent(10)) == 1


# --------------------------------------------------------------------------- #
# voice: none (t14) — recorded in SenseHistory, never delivered to Sonic     #
# --------------------------------------------------------------------------- #


def _none_rules_path(tmp_path):
    cfg = {
        "rules": {
            "rule/fire": {
                "priority": "NORMAL",
                "urgency": "NOW",
                "inject_template": "(body cue: {rule})",
                "sense": "sound",
                "voice": "none",
            },
            "intent/applied": {
                "priority": "NORMAL",
                "urgency": "DEFERRABLE",
                "inject_template": "Your standing intention '{name}' is now in effect.",
                "voice": "none",
            },
        },
        "default": {"priority": "NORMAL", "urgency": "DEFERRABLE"},
    }
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_voice_none_never_reaches_on_inject_but_is_recorded(tmp_path):
    rec = Recorder()
    history = SenseHistory()
    nb = make_bus(rec, rules_path=_none_rules_path(tmp_path), history=history, sources="intent")
    nb.on_message(
        None,
        None,
        fake_msg(
            "reachy/events/intent/applied",
            {"t": "intent", "ts": 1.0, "name": "set_inhibition"},
        ),
    )
    assert rec.injects == []
    (entry,) = history.recent()
    assert entry["source"] == "intent"
    assert entry["type"] == "applied"
    assert entry["voice"] == "none"
    assert "set_inhibition" in entry["text"]


def test_voice_none_muted_line_logs_once_per_key_across_three_events(caplog, tmp_path):
    clock_state = {"t": 0.0}
    rec = Recorder()
    history = SenseHistory()
    nb = make_bus(
        rec,
        rules_path=_none_rules_path(tmp_path),
        history=history,
        sources="rule",
        clock=lambda: clock_state["t"],
    )
    with caplog.at_level("INFO", logger="nova.sensory"):
        for i in range(3):
            # Each event clears the dedupe window (10s default) so all three
            # are genuinely-distinct fires, each recorded in history — the
            # muted senselog line is latched separately and must still only
            # appear once.
            clock_state["t"] += 20.0
            nb.on_message(
                None,
                None,
                fake_msg(
                    "reachy/events/rule/fire",
                    {"t": "rule", "ts": clock_state["t"], "rule": "look-toward-sound"},
                ),
            )
    assert rec.injects == []
    assert len(history.recent(10)) == 3
    muted_lines = [r for r in caplog.records if "muted voice=none" in r.getMessage()]
    assert len(muted_lines) == 1


def test_voice_none_respects_the_dedupe_window_for_history_too(tmp_path):
    """A rapid repeat within the dedupe window is suppressed before it ever
    reaches history — `none` still shares `_deliver`'s dedupe reservation,
    just without ever calling on_inject."""
    clock_state = {"t": 0.0}
    rec = Recorder()
    history = SenseHistory()
    nb = make_bus(
        rec,
        rules_path=_none_rules_path(tmp_path),
        history=history,
        sources="intent",
        clock=lambda: clock_state["t"],
    )
    msg = fake_msg(
        "reachy/events/intent/applied", {"t": "intent", "ts": 1.0, "name": "set_inhibition"}
    )
    nb.on_message(None, None, msg)
    clock_state["t"] += 1.0  # well inside the default 10s dedupe window
    nb.on_message(None, None, msg)
    assert rec.injects == []
    assert len(history.recent(10)) == 1


def test_voice_silent_regression_still_injects(tmp_path):
    """Regression guard: `silent` is unaffected by the `none` addition — it
    still always reaches on_inject, just carrying the quiet marker (unlike
    `none`, which reaches SenseHistory only)."""
    cfg = {
        "rules": {
            "intent/declare": {
                "priority": "NORMAL",
                "urgency": "BACKGROUND",
                "inject_template": "a standing goal was declared",
                "voice": "silent",
            },
        },
        "default": {"priority": "NORMAL", "urgency": "DEFERRABLE"},
    }
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(cfg))
    rec = Recorder()
    history = SenseHistory()
    nb = make_bus(rec, rules_path=path, history=history, sources="intent")
    nb.on_message(
        None, None, fake_msg("reachy/events/intent/declare", {"t": "intent", "ts": 1.0})
    )
    assert rec.injects == ["a standing goal was declared" + bus.VOICE_MARKERS["silent"]]
    assert len(history.recent()) == 1


def test_bus_with_no_history_wired_never_raises():
    rec = Recorder()
    nb = make_bus(rec, sources="pat")
    nb.on_message(
        None, None, fake_msg("reachy/events/pat/level1", {"t": "pat", "ts": 1.0, "level": 1})
    )
    assert len(rec.injects) == 1
    assert nb.history is None


# --------------------------------------------------------------------------- #
# The quiet marker (t12) — every inject carries it while quiet is armed       #
# --------------------------------------------------------------------------- #


class _FakeClock:
    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now


def _armed_quiet(tmp_path, minutes: float = 10.0):
    q = QuietState(clock=_FakeClock(), path=tmp_path / "nova-quiet.json")
    q.arm(minutes)
    return q


def _pat(nb):
    nb.on_message(
        None, None, fake_msg("reachy/events/pat/level1", {"t": "pat", "ts": 1.0, "level": 1})
    )


def test_quiet_marker_is_appended_to_a_free_rule_inject_while_armed(tmp_path):
    rec = Recorder()
    quiet = _armed_quiet(tmp_path)
    nb = make_bus(rec, rules_path=_custom_rules_path(tmp_path), quiet=quiet, sources="face")
    nb.on_message(
        None,
        None,
        fake_msg("reachy/events/face/recognized", {"t": "face", "ts": 2.0, "name": "Ori"}),
    )
    # face/recognized in the custom rules carries no ``voice`` field at all
    # (i.e. ``free``) — the quiet marker rides on top regardless.
    assert rec.injects == ["Ori is looking at you" + bus.QUIET_MARKER]


def test_quiet_marker_is_gone_after_release(tmp_path):
    rec = Recorder()
    quiet = _armed_quiet(tmp_path)
    nb = make_bus(rec, rules_path=_custom_rules_path(tmp_path), quiet=quiet, sources="face")
    quiet.release("test")
    nb.on_message(
        None,
        None,
        fake_msg("reachy/events/face/recognized", {"t": "face", "ts": 2.0, "name": "Ori"}),
    )
    assert rec.injects == ["Ori is looking at you"]


def test_quiet_marker_rides_on_top_of_a_voice_marker(tmp_path):
    rec = Recorder()
    quiet = _armed_quiet(tmp_path)
    nb = make_bus(rec, rules_path=_custom_rules_path(tmp_path), quiet=quiet)
    _pat(nb)
    (text,) = rec.injects
    assert text.endswith(bus.QUIET_MARKER)
    assert bus.VOICE_MARKERS["brief"] in text


def test_quiet_marker_reaches_the_sense_history_text_too(tmp_path):
    rec = Recorder()
    history = SenseHistory()
    quiet = _armed_quiet(tmp_path)
    nb = make_bus(
        rec, rules_path=_custom_rules_path(tmp_path), quiet=quiet, history=history
    )
    _pat(nb)
    (entry,) = history.recent()
    assert entry["text"] == rec.injects[0]


def test_bus_without_a_quiet_state_never_marks_an_inject(tmp_path):
    rec = Recorder()
    nb = make_bus(rec, rules_path=_custom_rules_path(tmp_path))
    _pat(nb)
    assert bus.QUIET_MARKER not in rec.injects[0]


def test_quiet_marker_text_says_do_not_speak():
    assert "quiet mode" in bus.QUIET_MARKER
    assert "do not speak" in bus.QUIET_MARKER


# --------------------------------------------------------------------------- #
# Lite reactor routing (t13) — react: lite hands the BASE cue to a reactor;   #
# markers are applied by the bus, on delivery, whichever thread that lands on #
# --------------------------------------------------------------------------- #


class FakeReactor:
    """Stand-in for :class:`~reachy_nova.harness.lite_reactor.LiteReactor`.

    Records every ``react()`` call as ``(cue, template)``. By default
    ``deliver`` fires synchronously with ``plan_text`` (matching the
    acceptance criteria's "use a fake reactor that calls deliver
    synchronously with a plan string"); with ``auto_deliver=False`` the
    caller controls timing via ``pending_deliver`` (used to prove a deliver
    firing on another thread still works).
    """

    def __init__(self, plan_text: str | None = None, auto_deliver: bool = True) -> None:
        self.calls: list[tuple[str, str]] = []
        self.plan_text = plan_text
        self.auto_deliver = auto_deliver
        self.pending_deliver = None

    def react(self, cue: str, template: str, deliver) -> None:
        self.calls.append((cue, template))
        if self.auto_deliver:
            deliver(self.plan_text if self.plan_text is not None else template)
        else:
            self.pending_deliver = deliver


def _lite_rules_path(tmp_path, **overrides):
    entry = {
        "priority": "NORMAL",
        "urgency": "NOW",
        "inject_template": "someone is petting you",
        "sense": "pat",
        "voice": "brief",
        "react": "lite",
    }
    entry.update(overrides)
    cfg = {
        "rules": {"pat/level1": entry},
        "default": {"priority": "NORMAL", "urgency": "DEFERRABLE"},
    }
    path = tmp_path / "rules-lite.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _pat_level1_msg():
    return fake_msg("reachy/events/pat/level1", {"t": "pat", "ts": 1.0, "level": 1})


def test_lite_tier_entry_hands_the_base_text_to_the_reactor(tmp_path):
    """The reactor gets the RENDERED TEMPLATE with no voice marker at all —
    not the fully-marked text route_event would have produced."""
    rec = Recorder()
    reactor = FakeReactor(plan_text="thanks for the pat")
    nb = bus.NovaBus(
        on_inject=rec.on_inject,
        rules_path=_lite_rules_path(tmp_path),
        sources="pat",
        reactor=reactor,
    )
    nb.on_message(None, None, _pat_level1_msg())
    assert reactor.calls == [("someone is petting you", "someone is petting you")]


def test_lite_tier_delivered_plan_carries_the_entrys_voice_marker(tmp_path):
    rec = Recorder()
    reactor = FakeReactor(plan_text="thanks for the pat")
    nb = bus.NovaBus(
        on_inject=rec.on_inject,
        rules_path=_lite_rules_path(tmp_path),
        sources="pat",
        reactor=reactor,
    )
    nb.on_message(None, None, _pat_level1_msg())
    assert rec.injects == ["thanks for the pat" + bus.VOICE_MARKERS["brief"]]


def test_lite_tier_records_the_delivered_plan_text_in_history(tmp_path):
    rec = Recorder()
    history = SenseHistory()
    reactor = FakeReactor(plan_text="thanks for the pat")
    nb = bus.NovaBus(
        on_inject=rec.on_inject,
        rules_path=_lite_rules_path(tmp_path),
        sources="pat",
        reactor=reactor,
        history=history,
    )
    nb.on_message(None, None, _pat_level1_msg())
    (entry,) = history.recent()
    assert entry["text"] == rec.injects[0] == "thanks for the pat" + bus.VOICE_MARKERS["brief"]
    assert entry["sense_class"] == "pat"
    assert entry["voice"] == "brief"


def test_lite_tier_dedupe_suppresses_reactor_call_and_inject_for_a_repeat(tmp_path):
    clock_state = {"t": 0.0}
    rec = Recorder()
    reactor = FakeReactor(plan_text="thanks")
    nb = bus.NovaBus(
        on_inject=rec.on_inject,
        rules_path=_lite_rules_path(tmp_path),
        sources="pat",
        reactor=reactor,
        clock=lambda: clock_state["t"],
    )
    nb.on_message(None, None, _pat_level1_msg())
    clock_state["t"] += 1.0  # well inside the default 10s dedupe window
    nb.on_message(None, None, _pat_level1_msg())
    assert len(reactor.calls) == 1
    assert len(rec.injects) == 1


def test_voice_none_entry_with_react_lite_never_calls_the_reactor(tmp_path):
    """react: lite is documented to never apply to voice: none — structurally
    enforced because _handle_message routes voice: none to _deliver_muted,
    which never touches the reactor at all."""
    rec = Recorder()
    reactor = FakeReactor(plan_text="thanks")
    nb = bus.NovaBus(
        on_inject=rec.on_inject,
        rules_path=_lite_rules_path(tmp_path, voice="none"),
        sources="pat",
        reactor=reactor,
    )
    nb.on_message(None, None, _pat_level1_msg())
    assert reactor.calls == []
    assert rec.injects == []


def test_entries_without_react_lite_ignore_a_wired_reactor(tmp_path):
    """A rule that doesn't opt in renders byte-identically even when a
    reactor is wired — react: lite is per-entry, never global."""
    rec = Recorder()
    reactor = FakeReactor(plan_text="should never be used")
    nb = make_bus(rec, rules_path=_custom_rules_path(tmp_path), reactor=reactor)
    _pat(nb)
    assert reactor.calls == []
    assert rec.injects == ["someone is petting you" + bus.VOICE_MARKERS["brief"]]


def test_react_lite_without_a_wired_reactor_renders_as_today(tmp_path):
    rec = Recorder()
    nb = bus.NovaBus(
        on_inject=rec.on_inject,
        rules_path=_lite_rules_path(tmp_path),
        sources="pat",
    )
    nb.on_message(None, None, _pat_level1_msg())
    assert rec.injects == ["someone is petting you" + bus.VOICE_MARKERS["brief"]]


def test_reactor_deliver_from_another_thread_results_in_exactly_one_inject_and_history_record(
    tmp_path,
):
    rec = Recorder()
    history = SenseHistory()
    reactor = FakeReactor(auto_deliver=False)
    nb = bus.NovaBus(
        on_inject=rec.on_inject,
        rules_path=_lite_rules_path(tmp_path),
        sources="pat",
        reactor=reactor,
        history=history,
    )
    nb.on_message(None, None, _pat_level1_msg())
    assert reactor.pending_deliver is not None

    t = threading.Thread(target=lambda: reactor.pending_deliver("thanks (from a thread)"))
    t.start()
    t.join(timeout=5)

    assert rec.injects == ["thanks (from a thread)" + bus.VOICE_MARKERS["brief"]]
    assert len(history.recent()) == 1


def test_reactor_deliver_on_inject_raise_rolls_back_the_dedupe_reservation(tmp_path):
    clock_state = {"t": 0.0}

    class FlakyRecorder:
        def __init__(self) -> None:
            self.calls = 0
            self.injects: list[str] = []

        def on_inject(self, text: str) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("sonic is not ready")
            self.injects.append(text)

    flaky = FlakyRecorder()
    reactor = FakeReactor(plan_text="thanks")
    nb = bus.NovaBus(
        on_inject=flaky.on_inject,
        rules_path=_lite_rules_path(tmp_path),
        sources="pat",
        reactor=reactor,
        clock=lambda: clock_state["t"],
    )
    nb.on_message(None, None, _pat_level1_msg())  # raises, rolled back
    clock_state["t"] += 0.01  # still well inside the window
    nb.on_message(None, None, _pat_level1_msg())  # must NOT be suppressed

    assert flaky.calls == 2
    assert flaky.injects == ["thanks" + bus.VOICE_MARKERS["brief"]]


def test_lite_tier_route_senselog_line_when_cue_handed_to_reactor(caplog, tmp_path):
    rec = Recorder()
    reactor = FakeReactor(plan_text="thanks")
    nb = bus.NovaBus(
        on_inject=rec.on_inject,
        rules_path=_lite_rules_path(tmp_path),
        sources="pat",
        reactor=reactor,
    )
    with caplog.at_level("INFO", logger="nova.sensory"):
        nb.on_message(None, None, _pat_level1_msg())
    route_lines = [
        r.getMessage()
        for r in caplog.records
        if "[SENSE stage=route source=nova event=pat/level1]" in r.getMessage()
        and "lite" in r.getMessage()
    ]
    assert len(route_lines) == 1


def test_on_inject_accepting_sense_class_receives_the_rules_sense(tmp_path):
    calls: list[tuple[str, str | None]] = []

    class FakeSonic:
        def inject_text(
            self, text: str, force: bool = False, sense_class: str | None = None
        ) -> None:
            calls.append((text, sense_class))

    sonic = FakeSonic()
    nb = bus.NovaBus(on_inject=sonic.inject_text, rules_path=_custom_rules_path(tmp_path))
    _pat(nb)
    assert calls == [("someone is petting you" + bus.VOICE_MARKERS["brief"], "touch")]


def test_on_inject_accepting_sense_class_is_also_used_on_the_lite_path(tmp_path):
    calls: list[tuple[str, str | None]] = []

    class FakeSonic:
        def inject_text(
            self, text: str, force: bool = False, sense_class: str | None = None
        ) -> None:
            calls.append((text, sense_class))

    sonic = FakeSonic()
    reactor = FakeReactor(plan_text="thanks")
    nb = bus.NovaBus(
        on_inject=sonic.inject_text,
        rules_path=_lite_rules_path(tmp_path),
        sources="pat",
        reactor=reactor,
    )
    nb.on_message(None, None, _pat_level1_msg())
    assert calls == [("thanks" + bus.VOICE_MARKERS["brief"], "pat")]


def test_on_inject_with_a_single_positional_parameter_still_works(tmp_path):
    """The memory leg's shape (and every existing test's Recorder): called
    with exactly one positional argument, never a sense_class keyword."""
    calls: list[str] = []

    def on_inject(text: str) -> None:
        calls.append(text)

    nb = bus.NovaBus(on_inject=on_inject, rules_path=_custom_rules_path(tmp_path))
    _pat(nb)
    assert calls == ["someone is petting you" + bus.VOICE_MARKERS["brief"]]
