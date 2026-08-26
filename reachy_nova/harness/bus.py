"""The harness's READ seam: reachy-mini-cli's MQTT event bus, prioritized.

The symbolic runtime (``reachy-mini-cli behavior engine run``) publishes its
own decisions on a localhost broker, and this module is Nova's ear on that
bus. Nothing here imports ``reachy_mini`` or touches the SDK — the runtime
owns the robot, the harness only listens and speaks.

The wire contract (verified against reachy-mini-cli 0.48.0,
``docs/export-schema.md`` + ``reachy/export/mqtt.py``)
---------------------------------------------------------------------------

- **Broker**: ``REACHY_MQTT_URL``, accepting ``mqtt://host:1883``,
  ``host:1883`` and a bare ``host``; default ``localhost:1883``. A bad port
  warns and falls back — a typo must degrade the bus, never stop the harness.
- **Events**: ``reachy/events/{source}/{type}``, QoS 0, **not retained**.
  ``{source}`` is the runtime block type (``sense``/``rule``/``intent``/
  ``motion``, plus the discrete ``pat``/``face``/``vision`` senses — see
  below) and ``{type}`` is that block's action — ``rule/fire``,
  ``rule/suppress``, ``intent/{declare,update,clear,applied,blocked}``,
  ``motion/{admit,evict,goto}``; a ``sense`` snapshot carries no action and
  publishes as ``sense/snapshot``.
- **Discrete pat/face/vision senses (t7)**: alongside the continuous
  ``sense/snapshot`` fields of the same name, the runtime is expected to
  publish discrete events under their own block types — ``pat/level1``,
  ``pat/level2``, ``pat/detected``, ``face/recognized``, ``face/unknown``,
  ``vision/description`` — routed through the same
  ``config/nervous-system/rules.yaml``. The exact topic/type names are
  **not yet live-confirmed** on the device (pat is live, face waits on the
  vision extra); a later task confirms and adjusts ``rules.yaml`` — a
  config-only change, nothing here.
- **Retained state**: ``reachy/state/{key}`` plus ``reachy/state/online``
  (``true``/``false``, backed by the runtime's own Last Will).
- **Payloads**: compact JSON, ``t`` and ``ts`` first. An unknown ``t`` is
  skipped, not guessed at — the runtime feed documents forward-compatible
  extensions.

Two subscription rules are load-bearing
---------------------------------------

1. **Cue filters live under ``reachy/events/`` only.** ``reachy/state/#`` is
   a RETAINED, last-value tree: subscribing to it for cues would replay the
   robot's last-known pose on every reconnect as if it had just happened.
   :func:`topic_filters` cannot name that tree by construction. The two
   retained topics this module *does* subscribe are read as STATE, never as
   cues: ``reachy/state/online`` is availability, and ``reachy/state/clip``
   (the clip rider's rolling-camera-clip announcement, t9's vision leg) is
   cached and exposed via :meth:`NovaBus.clip_state` — neither ever becomes
   an inject.
2. **``sense`` is off by default.** Upstream measured the unfiltered sense
   stream flooding a consumer at 187 cues in ~40 s with zero rule fires in
   the mix (``reachy-mini-cli`` ``scripts/embody_bus_feed.py``). What Nova
   cannot work out on its own is what the runtime DECIDED, which is exactly
   what ``rule``/``intent``/``motion`` carry. ``NOVA_BUS_SOURCES`` overrides
   (``*`` for everything).

Prioritization
--------------

Every bus event is keyed as ``"<source>/<type>"`` — the same key shape
``config/nervous-system/rules.yaml`` already uses for Nova's own senses — and
looked up there. A rule with an ``inject_template`` renders it against the
event payload and calls ``on_inject``; a rule without one (every
LOW/BACKGROUND decision) produces no inject and exactly one named
``[SENSE ... reason=no-template]`` line, so a dropped sense is never silent.
``llm_evaluate`` is carried as metadata only — no model is called here.

Threading
---------

paho's ``loop_start()`` owns the network thread (automatic reconnect with
backoff via ``reconnect_delay_set``); callbacks arrive on it and are wrapped
so a raising consumer can never kill it. :meth:`NovaBus.start` also arms a
watcher thread on the caller's ``stop_event`` so shutdown needs no extra
wiring. Every pure helper (:func:`parse_broker_url`, :func:`topic_filters`,
:func:`route_event`, :meth:`NovaBus.on_message`) is callable with no broker
and no paho import at all.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml

from reachy_nova import sensory_log
from reachy_nova.harness.quiet import QuietState
from reachy_nova.harness.sense_history import SenseHistory

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Senselog identity                                                           #
# --------------------------------------------------------------------------- #

#: ``[SENSE stage=bus source=nova event=...]`` — lifecycle lines.
STAGE_BUS = "bus"
#: ``[SENSE stage=route source=nova event=<key>]`` — per-event verdicts.
STAGE_ROUTE = "route"
#: ``[SENSE stage=inject source=nova event=<key>]`` — a rendered inject.
STAGE_INJECT = "inject"
SOURCE = "nova"

# --------------------------------------------------------------------------- #
# Wire contract constants                                                     #
# --------------------------------------------------------------------------- #

#: Env var naming the broker (shared with the runtime, same spelling).
BROKER_URL_ENV = "REACHY_MQTT_URL"
#: Loopback default — the broker binds 127.0.0.1 (``config/mosquitto/``).
DEFAULT_BROKER_URL = "localhost:1883"
#: Fallback when a broker URL names a host but no (usable) port.
DEFAULT_PORT = 1883

#: Root of the runtime's events tree. Cue filters never leave it.
EVENTS_PREFIX = "reachy/events/"
#: The runtime's RETAINED availability topic (its own Last Will flips it).
RUNTIME_ONLINE_TOPIC = "reachy/state/online"
#: The clip rider's RETAINED clip-state topic (reachy-mini-cli
#: ``reachy/behavior/clip_rider.py``). Read as state for the vision leg —
#: cached, exposed via :meth:`NovaBus.clip_state`, never routed as a cue.
CLIP_STATE_TOPIC = "reachy/state/clip"

#: Env var overriding which runtime sources the harness subscribes.
SOURCES_ENV = "NOVA_BUS_SOURCES"
#: Decisions, not perception — see the module docstring on the sense flood.
#: ``pat``/``face``/``vision`` are discrete senses (t7), not the high-rate
#: ``sense/snapshot`` stream, so they are safe to subscribe by default.
DEFAULT_SOURCES = "rule,intent,motion,pat,face,vision"
#: Runtime block types this harness understands; anything else is skipped.
#: ``pat``/``face``/``vision`` mirror the runtime's discrete touch/sight
#: senses (t7) — exact topic names are provisional, see the module docstring.
KNOWN_BLOCK_TYPES = frozenset({"sense", "rule", "intent", "motion", "pat", "face", "vision"})

#: The harness's OWN retained availability topic (Nova's namespace, not the
#: runtime's — the two never share a tree).
HARNESS_STATE_TOPIC = "nova/harness/state"
#: At-most-once, matching the runtime's policy on every topic.
QOS = 0
KEEPALIVE_S = 60
RECONNECT_MIN_DELAY_S = 1
RECONNECT_MAX_DELAY_S = 30

# --------------------------------------------------------------------------- #
# Named verdicts — every routed event resolves to exactly one, verbatim       #
# --------------------------------------------------------------------------- #

REASON_INJECT = "inject"
REASON_NO_TEMPLATE = "no-template"
REASON_TEMPLATE_FAILED = "template-failed"
REASON_UNKNOWN_BLOCK = "unknown-block-type"
REASON_BAD_PAYLOAD = "bad-payload"
REASON_NOT_AN_EVENT = "not-an-event-topic"

#: Where the nervous-system rules live when no path is injected.
DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "nervous-system" / "rules.yaml"
)
#: Env override for the rules file (an installed harness may relocate it).
RULES_PATH_ENV = "NOVA_RULES_PATH"


# --------------------------------------------------------------------------- #
# Pure helpers — no network, no paho import, trivially testable               #
# --------------------------------------------------------------------------- #


def broker_url(env: dict[str, str] | None = None) -> str:
    """The configured broker URL, unparsed. Defaults to loopback."""
    source = os.environ if env is None else env
    return source.get(BROKER_URL_ENV, DEFAULT_BROKER_URL) or DEFAULT_BROKER_URL


def parse_broker_url(url: str | None, *, default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    """Split a ``REACHY_MQTT_URL``-shaped value into ``(host, port)``.

    All the forms an operator actually writes are accepted:
    ``mqtt://host:1883``, ``host:1883`` and a bare ``host``. A non-numeric or
    out-of-range port warns and falls back to *default_port* instead of
    raising — this runs at startup, where a typo must degrade the bus, never
    stop the harness from booting (same rule the runtime applies in
    ``reachy/export/events_client.py``).
    """
    text = "" if url is None else str(url).strip()
    if "://" in text:
        text = text.split("://", 1)[1]
    text = text.strip("/")
    if not text:
        return "localhost", default_port
    host, _, port_text = text.rpartition(":")
    if not host:  # no colon at all — the whole string is the host
        return text or "localhost", default_port
    try:
        port = int(port_text)
    except ValueError:
        logger.warning("bus: unparseable port in %r; using %d", url, default_port)
        return host, default_port
    if not 0 < port < 65536:
        logger.warning("bus: out-of-range port in %r; using %d", url, default_port)
        return host, default_port
    return host, port


def resolve_sources(
    raw: str | Iterable[str] | None, env: dict[str, str] | None = None
) -> tuple[str, ...]:
    """Resolve the subscribed runtime sources.

    ``None`` consults ``NOVA_BUS_SOURCES`` and then :data:`DEFAULT_SOURCES`
    (decisions only). Blank entries — a stray comma, stray whitespace — are
    dropped.
    """
    if raw is None:
        source_env = os.environ if env is None else env
        raw = source_env.get(SOURCES_ENV) or DEFAULT_SOURCES
    if isinstance(raw, str):
        parts: Iterable[str] = raw.split(",")
    else:
        parts = raw
    return tuple(part.strip() for part in parts if str(part).strip())


def topic_filters(sources: Iterable[str]) -> tuple[str, ...]:
    """Subscription filters for *sources* — under ``reachy/events/`` by construction.

    There is no input that makes this function name ``reachy/state/#``, which
    is what keeps a retained last-value message from ever replaying into a
    cue on reconnect. ``"*"`` widens to the whole events tree and no further.
    """
    resolved = tuple(sources)
    if "*" in resolved:
        return (f"{EVENTS_PREFIX}#",)
    return tuple(f"{EVENTS_PREFIX}{source}/#" for source in resolved)


def parse_event_topic(topic: str) -> tuple[str, str] | None:
    """``reachy/events/rule/fire`` -> ``("rule", "fire")``; anything else -> ``None``."""
    if not topic.startswith(EVENTS_PREFIX):
        return None
    parts = topic[len(EVENTS_PREFIX) :].split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def load_rules(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/nervous-system/rules.yaml`` (or *path*).

    A missing or unreadable file is not fatal: the harness falls back to an
    empty rule set with a permissive ``default``, so the bus still runs (and
    says so on the senselog) rather than refusing to start.
    """
    resolved = Path(path or os.environ.get(RULES_PATH_ENV) or DEFAULT_RULES_PATH)
    try:
        with open(resolved) as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as err:
        sensory_log.stage(STAGE_BUS, SOURCE, "rules", f"unreadable {resolved}: {err}")
        raw = {}
    rules = raw.get("rules") or {}
    default = raw.get("default") or {
        "priority": "NORMAL",
        "urgency": "DEFERRABLE",
        "llm_evaluate": True,
    }
    return {"rules": rules, "default": default}


def rule_for(
    rules_cfg: dict[str, Any],
    source: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The rules.yaml entry governing ``"<source>/<type>"``, or the default.

    When *payload* names a runtime rule (a ``rule/fire`` event's ``rule``
    field), a per-rule override key ``"<source>/<type>:<rule>"`` wins over the
    generic entry — how ``rule/fire:pat-acknowledge`` gets a sensory inject
    ("someone is petting you") instead of the generic "a reflex fired". An
    absent override falls straight back to the generic key.
    """
    rules = rules_cfg.get("rules") or {}
    if isinstance(payload, dict):
        rule_name = payload.get("rule")
        if isinstance(rule_name, str) and rule_name:
            override = rules.get(f"{source}/{event_type}:{rule_name}")
            if isinstance(override, dict):
                return override
    entry = rules.get(f"{source}/{event_type}")
    if isinstance(entry, dict):
        return entry
    return rules_cfg.get("default") or {}


#: Marker text ``route_event`` appends to a rendered inject, keyed by the
#: rule entry's optional ``voice`` field. ``free`` (and any absent/unknown
#: value) appends nothing. See rules.yaml's header comment for the field.
VOICE_MARKERS: dict[str, str] = {
    "silent": " (quiet: do not speak about this)",
    "brief": " (react briefly if at all)",
    "free": "",
}

#: Appended to EVERY rendered inject while a timed quiet is armed (task t12),
#: on top of whatever ``VOICE_MARKERS`` entry the rule's own ``voice`` field
#: already added. Deliberately unconditional: a quiet is a promise made to a
#: person out loud, so it has to outrank a rule's own opinion about how
#: chatty its event is — a ``voice: free`` pat is exactly the event most
#: likely to talk through a quiet otherwise.
#:
#: It is a marker, not a gate. The event still reaches Nova, still lands in
#: the sense history, and the speaker's own quiet gate
#: (:mod:`reachy_nova.harness.speaking`) is what actually keeps the mouth
#: shut — this only stops the model from composing a reply it will never be
#: allowed to say.
QUIET_MARKER = " (quiet mode: do not speak)"


def route_event(
    rules_cfg: dict[str, Any],
    source: str,
    event_type: str,
    payload: dict[str, Any] | None,
) -> tuple[str | None, str]:
    """Decide what (if anything) a bus event should say to Nova.

    Returns ``(inject_text | None, reason)`` where *reason* is one of the
    named verdicts above. The template is rendered with ``str.format_map``
    over a ``defaultdict(str)``, so a field the runtime omitted renders empty
    instead of raising — a partial sentence beats a lost sense.

    When the rule entry carries a ``voice`` field, its marker (see
    ``VOICE_MARKERS``) is appended to the rendered text — a hint to Nova
    about how much it should say about the event, not whether the event
    happened at all. Absent or unrecognized values behave like ``free``
    (no marker).
    """
    rule = rule_for(rules_cfg, source, event_type, payload)
    template = rule.get("inject_template")
    if not template:
        return None, REASON_NO_TEMPLATE
    fields: dict[str, Any] = payload if isinstance(payload, dict) else {}
    try:
        text = str(template).format_map(defaultdict(str, fields))
    except (IndexError, ValueError, TypeError) as err:
        logger.warning("bus: bad inject_template for %s/%s: %s", source, event_type, err)
        return None, REASON_TEMPLATE_FAILED
    text = text.strip()
    if not text:
        return None, REASON_NO_TEMPLATE
    text += VOICE_MARKERS.get(rule.get("voice", "free"), "")
    return text, REASON_INJECT


#: Env var overriding the per-key inject dedupe window (seconds).
DEDUPE_WINDOW_ENV = "NOVA_SENSE_DEDUPE_S"
#: Default dedupe window — comfortably past the ~8s gap observed live
#: between a runtime's shipped reflex fire and a Kiro-authored overlay
#: rule's own fire for the same physical pat.
DEFAULT_DEDUPE_WINDOW_S = 10.0


def dedupe_window_s(env: dict[str, str] | None = None) -> float:
    """The per-key inject dedupe window in seconds (default 10s).

    Reads ``NOVA_SENSE_DEDUPE_S``, parsed defensively: unset, blank,
    non-numeric or non-positive values all fall back to the default rather
    than raising or (worse) silently disabling the window.
    """
    source_env = os.environ if env is None else env
    raw = source_env.get(DEDUPE_WINDOW_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_DEDUPE_WINDOW_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "bus: unparseable %s=%r; using default %.1fs",
            DEDUPE_WINDOW_ENV,
            raw,
            DEFAULT_DEDUPE_WINDOW_S,
        )
        return DEFAULT_DEDUPE_WINDOW_S
    if value <= 0:
        logger.warning(
            "bus: non-positive %s=%r; using default %.1fs",
            DEDUPE_WINDOW_ENV,
            raw,
            DEFAULT_DEDUPE_WINDOW_S,
        )
        return DEFAULT_DEDUPE_WINDOW_S
    return value


def dedupe_key_for(
    source: str,
    event_type: str,
    payload: dict[str, Any] | None,
    rule: dict[str, Any],
) -> str:
    """The dedupe key for one routed event: the rule's ``sense`` class if it
    names one, else the exact resolved key (``"<source>/<type>:<rule>"`` when
    the payload names a runtime rule, else ``"<source>/<type>"``).

    Deliberately NEVER guesses a class from a rule name — two differently
    named, un-classed rule/fire events must dedupe independently (a generic
    reflex fire is not assumed related to another just because both are
    unclassed). Only an explicit ``sense:`` field in rules.yaml collapses
    two distinctly-named events into one dedupe bucket.
    """
    sense = rule.get("sense")
    if isinstance(sense, str) and sense:
        return sense
    if isinstance(payload, dict):
        rule_name = payload.get("rule")
        if isinstance(rule_name, str) and rule_name:
            return f"{source}/{event_type}:{rule_name}"
    return f"{source}/{event_type}"


def harness_state_payload(status: str) -> str:
    """The retained ``nova/harness/state`` body (also used for the Last Will)."""
    return json.dumps({"status": status, "ts": time.time()}, separators=(",", ":"))


def _default_client_factory() -> Any:
    """Build a real paho client. Imported lazily so tests need no broker/paho."""
    import paho.mqtt.client as paho_mqtt

    return paho_mqtt.Client(paho_mqtt.CallbackAPIVersion.VERSION2, client_id="reachy-nova-harness")


# --------------------------------------------------------------------------- #
# The subscriber                                                              #
# --------------------------------------------------------------------------- #


class NovaBus:
    """Subscribes the runtime's event bus and prioritizes what Nova hears.

    Args:
        on_inject: called with one sentence of context whenever a bus event
            matches a rules.yaml entry carrying an ``inject_template``. Wired
            to ``NovaSonic.inject_text`` in the app.
        on_event: optional raw-event tap, called with every decoded payload
            (including ones that produce no inject) for dashboards/logging.
        sources: runtime sources to subscribe; ``None`` reads
            ``NOVA_BUS_SOURCES`` and falls back to decisions only.
        broker: broker URL; ``None`` reads ``REACHY_MQTT_URL``.
        rules_path: nervous-system rules file; ``None`` uses the repo's.
        client_factory: zero-arg paho-client builder, injectable for tests.
        clock: zero-arg monotonic-seconds source for the dedupe window;
            ``None`` uses :func:`time.monotonic`. Injectable for tests.
        quiet: optional :class:`~reachy_nova.harness.quiet.QuietState` (t12).
            While it is armed, every rendered inject carries
            :data:`QUIET_MARKER`. ``None`` (the default) never marks anything.
        history: optional :class:`~reachy_nova.harness.sense_history.SenseHistory`
            (t8). When wired, every inject that actually reaches *on_inject*
            (post-dedupe — a suppressed duplicate is never recorded) is also
            appended here, so the ``recall_senses`` tool can answer "why did
            you move?" from what really happened.
    """

    def __init__(
        self,
        on_inject: Callable[[str], None],
        on_event: Callable[[dict], None] | None = None,
        sources: str | Iterable[str] | None = None,
        broker: str | None = None,
        rules_path: str | Path | None = None,
        client_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] | None = None,
        history: SenseHistory | None = None,
        quiet: QuietState | None = None,
    ) -> None:
        self._on_inject = on_inject
        self._on_event = on_event
        self.history = history
        self.quiet = quiet
        self.sources = resolve_sources(sources)
        self.broker = broker if broker is not None else broker_url()
        self.host, self.port = parse_broker_url(self.broker)
        self.rules = load_rules(rules_path)
        self._client_factory = client_factory or _default_client_factory
        self._client: Any | None = None
        self._stop_event: threading.Event | None = None
        self._watcher: threading.Thread | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._runtime_online = False
        self._stopped = False
        # Per-key inject dedupe (t7) — collapses two distinct rule/fire
        # events for the same physical sense (e.g. a shipped reflex and a
        # Kiro overlay rule both firing on one pat) into a single inject.
        self._clock = clock or time.monotonic
        self._dedupe_window_s = dedupe_window_s()
        self._dedupe_lock = threading.Lock()
        self._last_inject_at: dict[str, float] = {}
        # Retained reachy/state/clip cache — the vision leg's read seam.
        self._clip_lock = threading.Lock()
        self._clip_state: dict[str, Any] | None = None
        self._clip_available: bool | None = None

    # -- read-only status ---------------------------------------------------

    @property
    def connected(self) -> bool:
        """Is a broker session live right now?"""
        return self._connected

    @property
    def runtime_online(self) -> bool:
        """Is the symbolic runtime publishing? (retained ``reachy/state/online``)"""
        return self._runtime_online

    def clip_state(self) -> dict[str, Any] | None:
        """The latest retained ``reachy/state/clip`` payload, or ``None``.

        Returns a copy, so a caller can never mutate the cache under the paho
        thread. This is the ``get_clip_state`` the vision leg is wired with —
        state read on demand, never a cue.
        """
        with self._clip_lock:
            return None if self._clip_state is None else dict(self._clip_state)

    # -- lifecycle ----------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        """Arm the Last Will, connect asynchronously and start paho's loop.

        Never raises: an unreachable broker (or a missing paho) resolves to a
        named senselog line and an inert bus, exactly like ``nova_mqtt.py``.
        """
        self._stop_event = stop_event
        try:
            client = self._client_factory()
        except Exception as err:  # paho missing, bad client id, ...
            sensory_log.stage(STAGE_BUS, SOURCE, "start", f"no client: {err}")
            return
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self.on_message
        try:
            client.will_set(HARNESS_STATE_TOPIC, harness_state_payload("offline"), QOS, True)
            client.reconnect_delay_set(
                min_delay=RECONNECT_MIN_DELAY_S, max_delay=RECONNECT_MAX_DELAY_S
            )
            client.connect_async(self.host, self.port, KEEPALIVE_S)
            client.loop_start()
        except Exception as err:
            sensory_log.stage(
                STAGE_BUS, SOURCE, "start", f"connect failed {self.host}:{self.port}: {err}"
            )
            self._client = client
            return
        self._client = client
        sensory_log.stage(
            STAGE_BUS,
            SOURCE,
            "start",
            f"connecting to {self.host}:{self.port} sources={list(self.sources)}",
        )
        self._watcher = threading.Thread(
            target=self._watch_stop, name="nova-bus-stop", daemon=True
        )
        self._watcher.start()

    def _watch_stop(self) -> None:
        if self._stop_event is None:
            return
        self._stop_event.wait()
        self.stop()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the stop-watcher to finish (test/shutdown convenience)."""
        if self._watcher is not None:
            self._watcher.join(timeout)

    def stop(self) -> None:
        """Publish a clean ``offline`` state and tear the client down. Idempotent."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            client = self._client
        if client is None:
            return
        try:
            client.publish(HARNESS_STATE_TOPIC, harness_state_payload("offline"), QOS, True)
            client.disconnect()
            client.loop_stop()
            sensory_log.stage(STAGE_BUS, SOURCE, "stop", "disconnected cleanly")
        except Exception as err:
            sensory_log.stage(STAGE_BUS, SOURCE, "stop", f"unclean shutdown: {err}")
        self._connected = False

    # -- paho callbacks -----------------------------------------------------

    def _on_connect(self, client, _userdata=None, _flags=None, reason_code=0, _props=None) -> None:
        was_connected = self._connected
        self._connected = True
        try:
            for topic in topic_filters(self.sources):
                client.subscribe(topic, QOS)
            client.subscribe(RUNTIME_ONLINE_TOPIC, QOS)
            client.subscribe(CLIP_STATE_TOPIC, QOS)
            client.publish(HARNESS_STATE_TOPIC, harness_state_payload("online"), QOS, True)
        except Exception as err:
            sensory_log.stage(STAGE_BUS, SOURCE, "connect", f"subscribe failed: {err}")
            return
        sensory_log.stage(
            STAGE_BUS,
            SOURCE,
            "reconnect" if was_connected else "connect",
            f"{self.host}:{self.port} rc={reason_code} filters={list(topic_filters(self.sources))}",
        )

    def _on_disconnect(
        self, _client=None, _userdata=None, _flags=None, reason_code=0, _props=None
    ) -> None:
        self._connected = False
        sensory_log.stage(
            STAGE_BUS,
            SOURCE,
            "disconnect",
            f"lost broker {self.host}:{self.port} rc={reason_code} — paho will retry with backoff",
        )

    # -- the message path (callable directly with a fake msg) ---------------

    def on_message(self, _client, _userdata, msg) -> None:
        """Route one bus message. Never raises onto paho's network thread."""
        try:
            self._handle_message(msg)
        except Exception as err:  # a consumer callback blowing up must not kill the loop
            logger.warning("bus: message handling failed: %s", err, exc_info=True)
            sensory_log.stage(STAGE_ROUTE, SOURCE, "error", f"dropped reason=handler-error: {err}")

    def _handle_message(self, msg) -> None:
        topic = getattr(msg, "topic", "") or ""
        raw = getattr(msg, "payload", b"") or b""

        if topic == RUNTIME_ONLINE_TOPIC:
            self._handle_runtime_online(raw)
            return

        if topic == CLIP_STATE_TOPIC:
            self._handle_clip_state(raw)
            return

        parsed_topic = parse_event_topic(topic)
        if parsed_topic is None:
            # Retained state and anything else off the events tree are never cues.
            sensory_log.stage(
                STAGE_ROUTE, SOURCE, topic, f"dropped reason={REASON_NOT_AN_EVENT}"
            )
            return
        source, event_type = parsed_topic
        key = f"{source}/{event_type}"

        try:
            payload = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        except (ValueError, UnicodeDecodeError) as err:
            sensory_log.stage(
                STAGE_ROUTE, SOURCE, key, f"dropped reason={REASON_BAD_PAYLOAD}: {err}"
            )
            return
        if not isinstance(payload, dict):
            sensory_log.stage(STAGE_ROUTE, SOURCE, key, f"dropped reason={REASON_BAD_PAYLOAD}")
            return

        block_type = payload.get("t", source)
        if block_type not in KNOWN_BLOCK_TYPES:
            # Forward-compatible extension: skip it, name it, never guess.
            sensory_log.stage(
                STAGE_ROUTE, SOURCE, key, f"dropped reason={REASON_UNKNOWN_BLOCK} t={block_type}"
            )
            return

        if self._on_event is not None:
            observed = dict(payload)
            observed.setdefault("source", source)
            observed.setdefault("type", event_type)
            try:
                self._on_event(observed)
            except Exception as err:
                logger.warning("bus: on_event callback failed: %s", err, exc_info=True)

        text, reason = route_event(self.rules, source, event_type, payload)
        if text is None:
            sensory_log.stage(STAGE_ROUTE, SOURCE, key, f"dropped reason={reason}")
            return
        # Marked BEFORE the history record so "what Nova was told" and "what
        # Nova remembers being told" can never drift apart.
        text = self._mark_quiet(text)

        rule = rule_for(self.rules, source, event_type, payload)

        dedupe_key = dedupe_key_for(source, event_type, payload, rule)
        now = self._clock()
        with self._dedupe_lock:
            last_at = self._last_inject_at.get(dedupe_key)
            if last_at is not None and (now - last_at) < self._dedupe_window_s:
                age = now - last_at
                sensory_log.stage(
                    STAGE_INJECT,
                    SOURCE,
                    "dedupe",
                    f"suppressed key={dedupe_key} age={age:.2f}s "
                    f"window={self._dedupe_window_s:.1f}s",
                )
                return
            self._last_inject_at[dedupe_key] = now

        if self.history is not None:
            rule_name = payload.get("rule") if isinstance(payload, dict) else None
            self.history.record(
                source,
                event_type,
                rule_name,
                text,
                rule.get("sense"),
                rule.get("voice"),
            )

        sensory_log.stage(
            STAGE_INJECT,
            SOURCE,
            key,
            f"injecting priority={rule.get('priority')} urgency={rule.get('urgency')} "
            f"chars={len(text)}",
        )
        try:
            self._on_inject(text)
        except Exception as err:
            logger.warning("bus: inject callback failed: %s", err, exc_info=True)
            sensory_log.stage(STAGE_INJECT, SOURCE, key, f"dropped reason=inject-failed: {err}")

    def _mark_quiet(self, text: str) -> str:
        """Append :data:`QUIET_MARKER` while a timed quiet is armed."""
        if self.quiet is None or not self.quiet.active():
            return text
        return text + QUIET_MARKER

    def _handle_clip_state(self, raw: bytes | str) -> None:
        """Cache the retained clip payload for :meth:`clip_state`. Never a cue.

        A bad payload is a named drop that keeps the last good state (the
        vision leg's own guards re-check the file the payload names). The
        ``available`` flag is latched so a rider that republishes the same
        verdict costs one line per TRANSITION, not one per retained replay.
        """
        try:
            payload = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        except (ValueError, UnicodeDecodeError) as err:
            sensory_log.stage(
                STAGE_BUS, SOURCE, "clip-state", f"dropped reason={REASON_BAD_PAYLOAD}: {err}"
            )
            return
        if not isinstance(payload, dict):
            sensory_log.stage(
                STAGE_BUS, SOURCE, "clip-state", f"dropped reason={REASON_BAD_PAYLOAD}"
            )
            return
        with self._clip_lock:
            self._clip_state = payload
        available = bool(payload.get("available"))
        if available != self._clip_available:
            self._clip_available = available
            detail = (
                f"clip available at {payload.get('path')}"
                if available
                else f"clip unavailable reason={payload.get('reason')}"
            )
            sensory_log.stage(STAGE_BUS, SOURCE, "clip-state", detail)

    def _handle_runtime_online(self, raw: bytes | str) -> None:
        text = (raw.decode() if isinstance(raw, bytes) else str(raw)).strip().lower()
        online = text in {"true", "1", "online"}
        if online == self._runtime_online:
            return
        self._runtime_online = online
        sensory_log.stage(
            STAGE_BUS,
            SOURCE,
            "runtime-online" if online else "runtime-offline",
            f"symbolic runtime is {'up' if online else 'down'} (retained {RUNTIME_ONLINE_TOPIC})",
        )
