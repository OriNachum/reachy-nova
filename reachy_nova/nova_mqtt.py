"""MQTT nervous system bridge for Reachy Nova.

Publishes state changes and events, subscribes to inject commands.
"""

import json
import logging
import os
import time
import uuid
from collections.abc import Callable

logger = logging.getLogger(__name__)


class NovaMQTT:
    """MQTT client wrapper for the Nova nervous system."""

    def __init__(self):
        self._client = None
        self._available = False
        self._inject_handler: Callable[[str], None] | None = None

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        """Connect to MQTT broker. Fails silently if unavailable."""
        try:
            import paho.mqtt.client as paho_mqtt

            def _on_connect(client, userdata, flags, rc, properties=None):
                self._available = True
                client.subscribe("nova/inject")
                logger.info(f"MQTT connected (rc={rc}) — subscribed to nova/inject")

            def _on_disconnect(client, userdata, flags, rc, properties=None):
                self._available = False
                logger.warning(f"MQTT disconnected (rc={rc}) — falling back to direct mode")

            self._client = paho_mqtt.Client(
                paho_mqtt.CallbackAPIVersion.VERSION2,
                client_id="reachy-nova-app",
            )
            self._client.on_connect = _on_connect
            self._client.on_disconnect = _on_disconnect
            self._client.reconnect_delay_set(min_delay=1, max_delay=30)
            self._client.connect(
                os.environ.get("MQTT_BROKER", "localhost"),
                int(os.environ.get("MQTT_PORT", "1883")),
                keepalive=60,
            )
            self._client.loop_start()
            self._available = True
            logger.info("MQTT started — Nervous System integration active")
        except Exception as e:
            logger.warning(f"MQTT not available — direct callbacks only: {e}")

    def stop(self) -> None:
        if self._available and self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
                logger.info("MQTT disconnected")
            except Exception:
                pass

    def publish_event(self, source: str, event_type: str, payload: dict) -> None:
        """Publish a pure data event to MQTT."""
        if not self._available:
            return
        event = {
            "event_id": str(uuid.uuid4()),
            "type": event_type,
            "source": source,
            "payload": payload,
            "timestamp": time.time(),
        }
        topic = f"nova/events/{source}/{event_type}"
        self._client.publish(topic, json.dumps(event))

    def publish_state(self, key: str, value: dict) -> None:
        """Publish retained state to MQTT."""
        if not self._available:
            return
        self._client.publish(f"nova/state/{key}", json.dumps(value), retain=True)

    def register_inject_handler(self, callback: Callable[[str], None]) -> None:
        """Register handler for nova/inject topic messages.

        The callback receives the raw JSON payload dict's "text" field.
        """
        self._inject_handler = callback
        if self._client is not None:
            def _on_inject(client, userdata, msg):
                try:
                    data = json.loads(msg.payload.decode())
                    text = data.get("text", "")
                    if text and self._inject_handler:
                        self._inject_handler(text)
                except Exception as e:
                    logger.warning(f"Inject message error: {e}")

            self._client.message_callback_add("nova/inject", _on_inject)
            logger.info("Registered nova/inject handler — Nervous System can drive voice")

    def on_state_change(self, changed: dict) -> None:
        """State change callback — auto-publish mood changes."""
        if not self._available:
            return
        if "mood" in changed:
            self.publish_state("mood", {"mood": changed["mood"]})
