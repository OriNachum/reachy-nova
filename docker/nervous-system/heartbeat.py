"""Heartbeat — publishes periodic tick events to MQTT."""

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Heartbeat:
    """Publishes periodic time events to MQTT."""

    def __init__(self, mqtt_client, interval: int = 60):
        self.client = mqtt_client
        self.interval = interval
        self._thread = None

    def start(self, stop_event: threading.Event):
        def _run():
            tick_count = 0
            while not stop_event.is_set():
                stop_event.wait(timeout=self.interval)
                if stop_event.is_set():
                    break
                tick_count += 1
                event = {
                    "event_id": str(uuid.uuid4()),
                    "type": "tick",
                    "source": "heartbeat",
                    "payload": {
                        "interval": self.interval,
                        "tick_count": tick_count,
                        "wall_clock": datetime.now(timezone.utc).isoformat(),
                    },
                    "timestamp": time.time(),
                }
                self.client.publish("nova/heartbeat", json.dumps(event))
                logger.debug(f"Heartbeat tick #{tick_count}")

        self._thread = threading.Thread(target=_run, daemon=True, name="heartbeat")
        self._thread.start()
        logger.info(f"Heartbeat started (interval={self.interval}s)")
