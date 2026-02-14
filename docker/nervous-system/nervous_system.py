"""Nervous System — centralized event evaluator for Reachy Nova.

Subscribes to all nova/events/#, applies rules, optionally consults an LLM,
and publishes inject/queue/ignore decisions to nova/inject.
"""

import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import yaml

from heartbeat import Heartbeat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("nervous-system")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class NovaEvent:
    """Pure data event received from any subsystem."""
    event_id: str
    type: str
    source: str
    payload: dict
    timestamp: float


@dataclass
class Rule:
    """Rule determining how an event is handled."""
    priority: str = "NORMAL"
    urgency: str = "DEFERRABLE"
    llm_evaluate: bool = True
    inject_template: str | None = None

    def render_template(self, payload: dict) -> str | None:
        if not self.inject_template:
            return None
        try:
            return self.inject_template.format(**payload)
        except KeyError:
            # Template references keys not in payload — return raw template
            return self.inject_template


# ---------------------------------------------------------------------------
# Rules engine
# ---------------------------------------------------------------------------

class RulesEngine:
    """Loads rules from YAML and resolves (source, type) lookups."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        self._default = Rule(**raw.get("default", {}))
        self._rules: dict[str, Rule] = {}
        for key, cfg in raw.get("rules", {}).items():
            self._rules[key] = Rule(**cfg)

        logger.info(f"Loaded {len(self._rules)} rules from {config_path}")

    def lookup(self, source: str, event_type: str) -> Rule:
        """Resolve rule: exact match → wildcard source → default."""
        exact = f"{source}/{event_type}"
        if exact in self._rules:
            return self._rules[exact]

        wildcard = f"{source}/*"
        if wildcard in self._rules:
            return self._rules[wildcard]

        return self._default


# ---------------------------------------------------------------------------
# LLM sub-agent
# ---------------------------------------------------------------------------

LLM_PROMPT_TEMPLATE = """You are the interrupt evaluator for a social robot called Reachy Mini.
The robot is currently in a voice conversation with a human.

Current robot state:
- Voice: {voice_state}
- Mood: {mood}
- Engagement level: {engagement_level}/1.0
- Time since last inject: {seconds_since_last_inject}s

Recent event history (last {event_window_size}):
{recent_events_summary}

New event to evaluate:
- Source: {source}
- Type: {type}
- Baseline priority: {priority}
- Baseline urgency: {urgency}
- Content: {payload_summary}

Given everything above, should this event interrupt the current conversation?

Rules:
- INJECT: Important enough to break into the conversation now
- QUEUE: Worth mentioning, but wait for a pause or idle moment
- IGNORE: Not worth mentioning at all

Reply with exactly one word: INJECT, QUEUE, or IGNORE."""


class LLMSubAgent:
    """Evaluates ambiguous events using Nova 2 Lite."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._client = None
        if enabled:
            try:
                import boto3
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                )
                logger.info("LLM sub-agent initialized (Nova 2 Lite)")
            except Exception as e:
                logger.warning(f"LLM sub-agent disabled — boto3 init failed: {e}")
                self.enabled = False

    def evaluate(self, event: NovaEvent, rule: Rule, context: dict) -> str:
        """Ask Nova 2 Lite for a final interrupt decision.

        Returns: "inject", "queue", or "ignore".
        """
        if not self.enabled or not self._client:
            return self._urgency_fallback(rule.urgency)

        try:
            prompt = LLM_PROMPT_TEMPLATE.format(
                voice_state=context.get("voice_state", "unknown"),
                mood=context.get("mood", "unknown"),
                engagement_level=context.get("engagement_level", 0.0),
                seconds_since_last_inject=context.get("seconds_since_last_inject", 999),
                event_window_size=len(context.get("recent_events", [])),
                recent_events_summary=self._format_recent(context.get("recent_events", [])),
                source=event.source,
                type=event.type,
                priority=rule.priority,
                urgency=rule.urgency,
                payload_summary=json.dumps(event.payload)[:500],
            )

            body = {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 10, "temperature": 0.1, "topP": 0.9},
            }
            response = self._client.invoke_model(
                modelId="us.amazon.nova-2-lite-v1:0",
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            answer = result["output"]["message"]["content"][0]["text"].strip().upper()

            if "INJECT" in answer:
                return "inject"
            elif "IGNORE" in answer:
                return "ignore"
            return "queue"

        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e} — using urgency fallback")
            return self._urgency_fallback(rule.urgency)

    @staticmethod
    def _urgency_fallback(urgency: str) -> str:
        if urgency in ("IMMEDIATE", "NOW"):
            return "inject"
        elif urgency == "DEFERRABLE":
            return "queue"
        return "ignore"

    @staticmethod
    def _format_recent(events: list) -> str:
        if not events:
            return "(none)"
        lines = []
        for e in events[-10:]:
            lines.append(f"  - [{e.get('source', '?')}/{e.get('type', '?')}] {json.dumps(e.get('payload', {}))[:100]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Queue manager
# ---------------------------------------------------------------------------

class QueueManager:
    """Manages deferred events, draining them when the robot is idle."""

    def __init__(self, mqtt_client, max_size: int = 50):
        self.client = mqtt_client
        self.queue: deque[tuple[str, NovaEvent, Rule]] = deque(maxlen=max_size)
        self._voice_state = "idle"
        self._engagement = 0.0

    def enqueue(self, event: NovaEvent, rule: Rule):
        self.queue.append((rule.priority, event, rule))
        logger.info(f"Queued event {event.source}/{event.type} (queue size: {len(self.queue)})")

    def update_state(self, voice_state: str, engagement: float):
        self._voice_state = voice_state
        self._engagement = engagement
        if voice_state == "idle" and engagement < 0.3 and self.queue:
            self._drain_one()

    def _drain_one(self):
        if not self.queue:
            return
        _, event, rule = self.queue.popleft()
        text = rule.render_template(event.payload)
        if not text:
            return
        inject_msg = {
            "text": text,
            "event_id": event.event_id,
            "priority": rule.priority,
        }
        self.client.publish("nova/inject", json.dumps(inject_msg))
        logger.info(f"Drained queued event {event.source}/{event.type} → nova/inject")


# ---------------------------------------------------------------------------
# Nervous System orchestrator
# ---------------------------------------------------------------------------

class NervousSystem:
    """Central event evaluator and router."""

    def __init__(self):
        import paho.mqtt.client as paho_mqtt

        # Configuration
        broker = os.environ.get("MQTT_BROKER", "localhost")
        port = int(os.environ.get("MQTT_PORT", "1883"))
        rules_path = os.environ.get("RULES_CONFIG", "/app/config/rules.yaml")
        heartbeat_interval = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))
        llm_enabled = os.environ.get("LLM_ENABLED", "true").lower() == "true"

        # MQTT client
        self.mqtt = paho_mqtt.Client(
            paho_mqtt.CallbackAPIVersion.VERSION2,
            client_id="nova-nervous-system",
        )
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_message

        logger.info(f"Connecting to MQTT broker at {broker}:{port}")
        self.mqtt.connect(broker, port, keepalive=60)

        # Subsystems
        self.rules = RulesEngine(rules_path)
        self.llm = LLMSubAgent(enabled=llm_enabled)
        self.queue_mgr = QueueManager(self.mqtt)
        self.heartbeat = Heartbeat(self.mqtt, interval=heartbeat_interval)

        # State tracking
        self._state: dict = {
            "voice_state": "idle",
            "mood": "happy",
            "engagement_level": 0.0,
        }
        self._recent_events: list[dict] = []
        self._last_inject_time: float = 0.0
        self._stop_event = threading.Event()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info(f"Connected to MQTT broker (rc={rc})")
        client.subscribe("nova/events/#")
        client.subscribe("nova/state/#")
        client.subscribe("nova/heartbeat")
        logger.info("Subscribed to nova/events/#, nova/state/#, nova/heartbeat")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Bad message on {msg.topic}: {e}")
            return

        topic = msg.topic

        if topic.startswith("nova/state/"):
            key = topic.split("/", 2)[2]
            self._handle_state(key, payload)
        elif topic == "nova/heartbeat":
            self._handle_heartbeat(payload)
        elif topic.startswith("nova/events/"):
            self._handle_event(topic, payload)

    def _handle_state(self, key: str, payload: dict):
        """Update internal state tracking."""
        if key == "voice":
            state = payload.get("state", "idle")
            self._state["voice_state"] = state
            # Compute engagement from voice state
            engagement = {"speaking": 0.8, "listening": 0.5, "thinking": 0.6}.get(state, 0.0)
            self._state["engagement_level"] = engagement
            self.queue_mgr.update_state(state, engagement)
        elif key == "mood":
            self._state["mood"] = payload.get("mood", "happy")
        logger.debug(f"State update: {key} = {payload}")

    def _handle_heartbeat(self, payload: dict):
        """Log heartbeat ticks."""
        tick = payload.get("payload", {}).get("tick_count", "?")
        logger.debug(f"Heartbeat tick #{tick}")

    def _handle_event(self, topic: str, payload: dict):
        """Process an incoming event through the rules pipeline."""
        # Parse event
        try:
            event = NovaEvent(
                event_id=payload.get("event_id", str(uuid.uuid4())),
                type=payload.get("type", "unknown"),
                source=payload.get("source", "unknown"),
                payload=payload.get("payload", {}),
                timestamp=payload.get("timestamp", time.time()),
            )
        except Exception as e:
            logger.warning(f"Failed to parse event from {topic}: {e}")
            return

        # Track recent events (keep last 20)
        self._recent_events.append({
            "source": event.source,
            "type": event.type,
            "payload": event.payload,
            "timestamp": event.timestamp,
        })
        if len(self._recent_events) > 20:
            self._recent_events = self._recent_events[-20:]

        # Look up rule
        rule = self.rules.lookup(event.source, event.type)
        logger.info(
            f"Event {event.source}/{event.type} → "
            f"priority={rule.priority} urgency={rule.urgency} llm={rule.llm_evaluate}"
        )

        if rule.llm_evaluate:
            # Run LLM evaluation in a daemon thread to avoid blocking MQTT loop
            threading.Thread(
                target=self._evaluate_with_llm,
                args=(event, rule),
                daemon=True,
            ).start()
        else:
            decision = self._urgency_to_decision(rule.urgency)
            self._execute_decision(decision, event, rule)

    def _evaluate_with_llm(self, event: NovaEvent, rule: Rule):
        """Run LLM evaluation in background thread."""
        context = {
            **self._state,
            "seconds_since_last_inject": time.time() - self._last_inject_time,
            "recent_events": self._recent_events,
        }
        decision = self.llm.evaluate(event, rule, context)
        logger.info(f"LLM decision for {event.source}/{event.type}: {decision}")
        self._execute_decision(decision, event, rule)

    def _execute_decision(self, decision: str, event: NovaEvent, rule: Rule):
        """Execute inject/queue/ignore decision."""
        if decision == "inject":
            text = rule.render_template(event.payload)
            if text:
                inject_msg = {
                    "text": text,
                    "event_id": event.event_id,
                    "priority": rule.priority,
                }
                self.mqtt.publish("nova/inject", json.dumps(inject_msg))
                self._last_inject_time = time.time()
                logger.info(f"INJECT {event.source}/{event.type}: {text[:80]}")
            else:
                logger.info(f"INJECT {event.source}/{event.type}: (no template)")
        elif decision == "queue":
            self.queue_mgr.enqueue(event, rule)
        else:
            logger.debug(f"IGNORE {event.source}/{event.type}")

    @staticmethod
    def _urgency_to_decision(urgency: str) -> str:
        if urgency in ("IMMEDIATE", "NOW"):
            return "inject"
        elif urgency == "DEFERRABLE":
            return "queue"
        return "ignore"

    def run(self):
        """Start the nervous system main loop."""
        self.heartbeat.start(self._stop_event)
        logger.info("Nervous System is running")

        # MQTT network loop (blocking)
        self.mqtt.loop_start()

        # Wait for shutdown signal
        self._stop_event.wait()

        logger.info("Shutting down...")
        self.mqtt.loop_stop()
        self.mqtt.disconnect()
        logger.info("Nervous System stopped")

    def shutdown(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ns = NervousSystem()

    def _signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        ns.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    ns.run()


if __name__ == "__main__":
    main()
