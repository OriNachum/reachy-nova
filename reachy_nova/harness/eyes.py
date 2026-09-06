"""``Eyes`` — names a dead camera the runtime itself never mentions (t11).

Live finding (2026-09-06): the runtime publishes ``reachy/events/sense/snapshot``
at 50 Hz with a ``frame_available`` boolean; on the robot it read ``false`` in
251/251 and 435/435 samples while the runtime's own availability report said
the camera senses were "available". Nothing in the harness named this for two
days — :class:`Eyes` is the fix: a small, pure state machine that only speaks
once per TRANSITION (dead / restored), never once per sample, so a genuinely
dark camera produces exactly one line, not a flood.

:class:`EyesComponent` is the supervisor-shaped wrapper: it opens its own
small paho-mqtt subscription (deliberately separate from
:class:`~reachy_nova.harness.bus.NovaBus` — ``sense/snapshot`` is off that
bus's default subscription by design, see ``bus.py``'s module docstring on
the sense flood), samples it down to about 1 Hz, and feeds :class:`Eyes`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from .. import sensory_log
from . import bus

logger = logging.getLogger(__name__)

_STAGE = "vision"
_SOURCE = "runtime"
_EVENT = "frames"

#: The runtime's high-rate camera-availability snapshot topic. Deliberately
#: NOT part of NovaBus's own subscription set (see ``bus.py``) — this module
#: opens its own connection rather than widening that one's default sources.
SNAPSHOT_TOPIC = "reachy/events/sense/snapshot"

#: Env override for how long continuous ``frame_available=False`` must persist
#: before the camera is declared dead.
DEAD_AFTER_ENV = "NOVA_EYES_DEAD_AFTER_S"
DEFAULT_DEAD_AFTER_S = 60.0

#: How often :class:`EyesComponent` actually processes a message — the topic
#: itself arrives at ~50 Hz, and processing every message would be exactly
#: the flood ``bus.py`` already refuses to subscribe to by default.
SAMPLE_INTERVAL_S = 1.0

KEEPALIVE_S = 60


def dead_after_s(env: dict[str, str] | None = None) -> float:
    """How long continuous no-frames must persist before "dead" (default 60s).

    Reads :data:`DEAD_AFTER_ENV`, parsed defensively: unset, blank,
    non-numeric or non-positive values all fall back to the default rather
    than raising or silently disabling the latch.
    """
    import os

    source_env = os.environ if env is None else env
    raw = source_env.get(DEAD_AFTER_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_DEAD_AFTER_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "eyes: unparseable %s=%r; using default %.1fs",
            DEAD_AFTER_ENV,
            raw,
            DEFAULT_DEAD_AFTER_S,
        )
        return DEFAULT_DEAD_AFTER_S
    if value <= 0:
        logger.warning(
            "eyes: non-positive %s=%r; using default %.1fs",
            DEAD_AFTER_ENV,
            raw,
            DEFAULT_DEAD_AFTER_S,
        )
        return DEFAULT_DEAD_AFTER_S
    return value


def _fmt_seconds(value: float) -> str:
    """``60.0`` -> ``"60"``; ``60.5`` -> ``"60.5"`` — no trailing noise."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


# --------------------------------------------------------------------------- #
# The pure state machine                                                      #
# --------------------------------------------------------------------------- #


class Eyes:
    """Pure state machine over a stream of ``frame_available`` booleans.

    State is one of ``"unknown"`` (no note yet), ``"live"`` or ``"dead"``.
    Nothing here touches the network — :meth:`note` is called by
    :class:`EyesComponent` with whatever it decoded off the bus, or directly
    by a test with an injected clock.

    A latched line is emitted on exactly two transitions:

    - continuous ``frame_available=False`` for :attr:`dead_after_s` (default
      60s) latches state ``"dead"`` and logs ONE
      ``sensory_log.stage("vision", "runtime", "frames", "dropped
      reason=no-frames after=<n>s")`` line. Further ``False`` notes while
      already dead log nothing.
    - the first ``True`` after ``"dead"`` latches state ``"live"`` and logs
      ONE ``sensory_log.stage("vision", "runtime", "frames", "restored
      after=<n>s")`` line, where ``<n>`` is how long frames were actually
      missing (since the continuous-False stretch began, not since the
      60s latch fired).

    A ``True`` note from ``"unknown"`` moves to ``"live"`` SILENTLY (decision:
    the harness has no baseline yet to call this a "restoration" of anything —
    the very first frame the process ever observes is not news). A later
    ``False`` stretch, after a restoration, latches again after another full
    :attr:`dead_after_s` — the false-streak clock resets on every ``True``.
    """

    def __init__(
        self,
        dead_after_s_value: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.dead_after_s = (
            dead_after_s_value if dead_after_s_value is not None else dead_after_s()
        )
        self._clock = clock or time.monotonic
        self._state = "unknown"
        self._false_since: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """``"unknown"`` | ``"live"`` | ``"dead"``."""
        return self._state

    def note(self, frame_available: bool, now: float | None = None) -> None:
        """Feed one observed ``frame_available`` sample."""
        t = now if now is not None else self._clock()
        with self._lock:
            if frame_available:
                self._note_true(t)
            else:
                self._note_false(t)

    def _note_true(self, t: float) -> None:
        was_dead = self._state == "dead"
        downtime = None if self._false_since is None else t - self._false_since
        self._false_since = None
        self._state = "live"
        if was_dead:
            age = downtime if downtime is not None else 0.0
            sensory_log.stage(_STAGE, _SOURCE, _EVENT, f"restored after={age:.2f}s")

    def _note_false(self, t: float) -> None:
        if self._false_since is None:
            self._false_since = t
        if self._state == "dead":
            return  # already latched — repeated False logs nothing
        elapsed = t - self._false_since
        if elapsed >= self.dead_after_s:
            self._state = "dead"
            sensory_log.stage(
                _STAGE,
                _SOURCE,
                _EVENT,
                f"dropped reason=no-frames after={_fmt_seconds(self.dead_after_s)}s",
            )


# --------------------------------------------------------------------------- #
# The supervisor component                                                    #
# --------------------------------------------------------------------------- #


def _default_client_factory() -> Any:
    """Build a real paho client. Imported lazily so tests need no broker/paho."""
    import paho.mqtt.client as paho_mqtt

    return paho_mqtt.Client(
        paho_mqtt.CallbackAPIVersion.VERSION2, client_id="reachy-nova-harness-eyes"
    )


class EyesComponent:
    """Supervisor component: an own-connection ~1Hz reader of the snapshot topic.

    Args:
        eyes: the :class:`Eyes` state machine to feed. ``None`` builds a
            default one (``dead_after_s()`` from the environment).
        broker: broker URL; ``None`` reads the same env
            :func:`reachy_nova.harness.bus.broker_url` reads
            (``REACHY_MQTT_URL``), so this never needs its own knob.
        client_factory: zero-arg paho-client builder, injectable for tests —
            the seam that keeps every test socket-free. Raising here (or a
            subsequent connect failure) degrades to state ``"unknown"`` with
            one named senselog line; :meth:`start` never raises.
        sample_interval_s: minimum gap between processed messages — the
            topic itself arrives at ~50 Hz; only about one sample per second
            actually reaches :meth:`Eyes.note`.
        clock: zero-arg monotonic-seconds source used for the sampling gate
            (NOT the same clock ``eyes`` uses internally). ``None`` uses
            :func:`time.monotonic`. Injectable for tests.
    """

    name = "eyes"

    def __init__(
        self,
        eyes: Eyes | None = None,
        broker: str | None = None,
        client_factory: Callable[[], Any] | None = None,
        sample_interval_s: float = SAMPLE_INTERVAL_S,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.eyes = eyes if eyes is not None else Eyes()
        self._broker = broker
        self._client_factory = client_factory or _default_client_factory
        self._sample_interval_s = sample_interval_s
        self._clock = clock or time.monotonic
        self._client: Any | None = None
        self._stop_event: threading.Event | None = None
        self._watcher: threading.Thread | None = None
        self._last_processed_at: float | None = None
        self._stopped = False
        self._lock = threading.Lock()

    @property
    def eyes_state(self) -> str:
        """Convenience passthrough — ``self.eyes.state``."""
        return self.eyes.state

    # -- lifecycle ------------------------------------------------------- #

    def start(self, stop_event: threading.Event) -> None:
        """Open our own subscription. Never raises — an unreachable broker
        degrades to state ``"unknown"`` with one named senselog line."""
        self._stop_event = stop_event
        try:
            client = self._client_factory()
        except Exception as err:  # paho missing, broker unreachable, ...
            sensory_log.stage(
                _STAGE, _SOURCE, _EVENT, f"component absent reason=broker-unreachable detail={err}"
            )
            return
        client.on_message = self._on_message
        try:
            client.on_connect = self._on_connect
        except AttributeError:
            pass
        broker = self._broker if self._broker is not None else bus.broker_url()
        host, port = bus.parse_broker_url(broker)
        try:
            client.connect_async(host, port, KEEPALIVE_S)
            client.loop_start()
        except Exception as err:
            sensory_log.stage(
                _STAGE,
                _SOURCE,
                _EVENT,
                f"component absent reason=broker-unreachable detail={err}",
            )
            return
        self._client = client
        self._watcher = threading.Thread(
            target=self._watch_stop, name="nova-eyes-stop", daemon=True
        )
        self._watcher.start()

    def _on_connect(self, client, _userdata=None, _flags=None, reason_code=0, _props=None) -> None:
        try:
            client.subscribe(SNAPSHOT_TOPIC, 0)
        except Exception as err:
            sensory_log.stage(
                _STAGE, _SOURCE, _EVENT, f"component absent reason=broker-unreachable detail={err}"
            )

    def _watch_stop(self) -> None:
        if self._stop_event is None:
            return
        self._stop_event.wait()
        self.stop()

    def stop(self) -> None:
        """Disconnect. Idempotent, never raises."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            client = self._client
        if client is None:
            return
        try:
            client.disconnect()
            client.loop_stop()
        except Exception:
            pass

    # -- the message path (callable directly with a fake msg) ------------ #

    def _on_message(self, _client, _userdata, msg) -> None:
        try:
            self._handle_message(msg)
        except Exception as err:  # a raising handler must never kill paho's loop
            logger.warning("eyes: message handling failed: %s", err, exc_info=True)

    def _handle_message(self, msg) -> None:
        now = self._clock()
        if (
            self._last_processed_at is not None
            and (now - self._last_processed_at) < self._sample_interval_s
        ):
            return
        self._last_processed_at = now

        raw = getattr(msg, "payload", b"") or b""
        try:
            payload = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self.eyes.note(bool(payload.get("frame_available", False)))


def build_component() -> EyesComponent:
    """Factory used by :func:`reachy_nova.harness.supervisor.build_components`."""
    return EyesComponent()
