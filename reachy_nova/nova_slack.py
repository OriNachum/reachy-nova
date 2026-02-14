"""Nova Slack - Slack integration for Reachy Nova using Socket Mode.

Connects to Slack via slack-bolt (Socket Mode, no public URL needed).
Features an interrupt gate that decides whether incoming messages should
interrupt the robot's current voice engagement, be queued for later, or ignored.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import boto3

logger = logging.getLogger(__name__)

LLM_MODEL = "us.amazon.nova-2-lite-v1:0"

URGENCY_KEYWORDS = {"urgent", "emergency", "help", "asap", "critical", "important"}


@dataclass
class SlackEvent:
    """Normalized Slack event."""

    type: str  # message, reaction, reply, mention
    channel: str
    user: str
    text: str
    ts: str  # Slack timestamp (also message ID)
    thread_ts: str | None = None
    emoji: str | None = None
    is_mention: bool = False
    is_dm: bool = False
    raw: dict = field(default_factory=dict)


class InterruptGate:
    """Three-tier decision engine for incoming Slack messages.

    1. Fast rules: @mentions / DMs -> interrupt, urgency keywords -> interrupt, robot idle -> interrupt
    2. Short/empty messages -> ignore
    3. Ambiguous -> call Nova 2 Lite via Bedrock
    """

    def __init__(self, bot_user_id: str | None = None):
        self.bot_user_id = bot_user_id
        self._voice_state = "idle"
        self._engagement_level = 0.0  # 0.0 = idle, 1.0 = deeply engaged
        self._bedrock_client = None
        try:
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
        except Exception as e:
            logger.warning(f"InterruptGate: Bedrock client init failed: {e}")

    def update_context(self, voice_state: str, engagement_level: float) -> None:
        self._voice_state = voice_state
        self._engagement_level = max(0.0, min(1.0, engagement_level))

    def evaluate(self, event: SlackEvent) -> str:
        """Evaluate whether to interrupt, queue, or ignore.

        Returns: "interrupt", "queue", or "ignore"
        """
        # Fast rule: @mentions and DMs always interrupt
        if event.is_mention or event.is_dm:
            return "interrupt"

        # Fast rule: urgency keywords interrupt
        lower = event.text.lower()
        if any(kw in lower for kw in URGENCY_KEYWORDS):
            return "interrupt"

        # Fast rule: robot idle -> interrupt
        if self._voice_state == "idle" and self._engagement_level < 0.3:
            return "interrupt"

        # Short/empty messages -> ignore
        if len(event.text.strip()) < 5:
            return "ignore"

        # Reactions without text -> ignore
        if event.type == "reaction" and not event.text:
            return "ignore"

        # Ambiguous -> LLM evaluation
        return self._llm_evaluate(event)

    def _llm_evaluate(self, event: SlackEvent) -> str:
        """Use Nova 2 Lite to decide on ambiguous messages."""
        if not self._bedrock_client:
            return "queue"

        try:
            prompt = (
                f"You are an interrupt gate for a social robot. "
                f"The robot is currently in voice state '{self._voice_state}' "
                f"with engagement level {self._engagement_level:.1f}/1.0.\n\n"
                f"A Slack message arrived:\n"
                f"Channel: {event.channel}\n"
                f"User: {event.user}\n"
                f"Text: {event.text[:500]}\n\n"
                f"Should the robot be interrupted to hear this message? "
                f"Reply with exactly one word: INTERRUPT, QUEUE, or IGNORE."
            )
            body = {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {
                    "maxTokens": 10,
                    "temperature": 0.1,
                    "topP": 0.9,
                },
            }
            response = self._bedrock_client.invoke_model(
                modelId=LLM_MODEL, body=json.dumps(body)
            )
            result = json.loads(response["body"].read())
            answer = result["output"]["message"]["content"][0]["text"].strip().upper()

            if "INTERRUPT" in answer:
                return "interrupt"
            elif "IGNORE" in answer:
                return "ignore"
            else:
                return "queue"
        except Exception as e:
            logger.warning(f"InterruptGate LLM error: {e}")
            return "queue"


class NovaSlack:
    """Manages Slack integration via Socket Mode."""

    def __init__(
        self,
        on_event: Callable[[SlackEvent], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
        on_interrupt: Callable[[SlackEvent], None] | None = None,
        channel_ids: list[str] | None = None,
    ):
        self.on_event = on_event
        self.on_state_change = on_state_change
        self.on_interrupt = on_interrupt
        self.channel_ids = set(channel_ids or [])

        self._recent_messages: deque[SlackEvent] = deque(maxlen=50)
        self._queued_messages: deque[SlackEvent] = deque(maxlen=20)
        self._thread: threading.Thread | None = None
        self._gate: InterruptGate | None = None
        self._app = None  # Slack Bolt App
        self._client = None  # Slack WebClient
        self._bot_user_id: str | None = None
        self._stop_event: threading.Event | None = None

        self.state = "idle"  # idle, connected, error
        self.last_event: SlackEvent | None = None

    def _set_state(self, state: str) -> None:
        self.state = state
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception:
                pass

    def _handle_event(self, event: SlackEvent) -> None:
        """Process an incoming Slack event through the interrupt gate."""
        self._recent_messages.append(event)
        self.last_event = event

        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

        if not self._gate:
            return

        decision = self._gate.evaluate(event)
        logger.info(f"[Slack] Gate decision for '{event.text[:50]}': {decision}")

        if decision == "interrupt":
            if self.on_interrupt:
                try:
                    self.on_interrupt(event)
                except Exception as e:
                    logger.error(f"on_interrupt callback error: {e}")
        elif decision == "queue":
            self._queued_messages.append(event)

    def update_context(self, voice_state: str, engagement_level: float) -> None:
        """Feed robot state into the interrupt gate."""
        if self._gate:
            self._gate.update_context(voice_state, engagement_level)

    def execute(self, params: dict) -> str:
        """Blocking skill executor for Sonic tool_use.

        Actions: send_message, read_messages, read_queued, reply_to_thread, add_reaction
        """
        action = params.get("action", "read_messages")
        channel = params.get("channel", "")

        # Default to first configured channel
        if not channel and self.channel_ids:
            channel = next(iter(self.channel_ids))

        if action == "send_message":
            text = params.get("text", "")
            if not text:
                return "[No message text provided]"
            return self._send_message(channel, text)

        elif action == "read_messages":
            count = min(int(params.get("count", 10)), 50)
            messages = list(self._recent_messages)[-count:]
            if not messages:
                return "[No recent Slack messages]"
            lines = [f"[{m.user}] {m.text}" for m in messages]
            return f"[{len(lines)} recent Slack messages]\n" + "\n".join(lines)

        elif action == "read_queued":
            messages = list(self._queued_messages)
            self._queued_messages.clear()
            if not messages:
                return "[No queued Slack messages]"
            lines = [f"[{m.user}] {m.text}" for m in messages]
            return f"[{len(lines)} queued Slack messages]\n" + "\n".join(lines)

        elif action == "reply_to_thread":
            text = params.get("text", "")
            thread_ts = params.get("thread_ts", "")
            if not text or not thread_ts:
                return "[Need text and thread_ts for thread reply]"
            return self._send_message(channel, text, thread_ts=thread_ts)

        elif action == "add_reaction":
            emoji = params.get("emoji", "thumbsup")
            ts = params.get("ts", "")
            if not ts:
                # React to the most recent message
                if self._recent_messages:
                    ts = self._recent_messages[-1].ts
                    channel = channel or self._recent_messages[-1].channel
                else:
                    return "[No message to react to]"
            return self._add_reaction(channel, ts, emoji)

        else:
            return f"[Unknown Slack action: {action}]"

    def _send_message(
        self, channel: str, text: str, thread_ts: str | None = None
    ) -> str:
        """Send a message to a Slack channel."""
        if not self._client:
            return "[Slack not connected]"
        if not channel:
            return "[No channel specified]"
        try:
            kwargs = {"channel": channel, "text": text}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            result = self._client.chat_postMessage(**kwargs)
            return f"[Message sent to {channel}]"
        except Exception as e:
            logger.error(f"Slack send error: {e}")
            return f"[Slack send error: {e}]"

    def _add_reaction(self, channel: str, ts: str, emoji: str) -> str:
        """Add an emoji reaction to a message."""
        if not self._client:
            return "[Slack not connected]"
        try:
            self._client.reactions_add(channel=channel, timestamp=ts, name=emoji)
            return f"[Reacted with :{emoji}: in {channel}]"
        except Exception as e:
            logger.error(f"Slack reaction error: {e}")
            return f"[Slack reaction error: {e}]"

    def queue_task(self, action: str, **kwargs) -> None:
        """Fire-and-forget outbound action (runs in background thread)."""

        def _run():
            try:
                params = {"action": action, **kwargs}
                self.execute(params)
            except Exception as e:
                logger.error(f"Slack queue_task error: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def add_reaction(self, channel: str, timestamp: str, emoji: str) -> None:
        """Convenience method to add a reaction."""
        self.queue_task("add_reaction", channel=channel, ts=timestamp, emoji=emoji)

    def start(self, stop_event: threading.Event) -> None:
        """Start the Slack bot in Socket Mode in a daemon thread."""
        bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
        app_token = os.environ.get("SLACK_APP_TOKEN", "")

        if not bot_token or not app_token:
            logger.warning(
                "Slack integration disabled: SLACK_BOT_TOKEN and/or SLACK_APP_TOKEN not set"
            )
            return

        self._stop_event = stop_event
        self._thread = threading.Thread(
            target=self._run_bot,
            args=(bot_token, app_token, stop_event),
            name="nova-slack",
            daemon=True,
        )
        self._thread.start()
        logger.info("Nova Slack thread started")

    def _run_bot(
        self, bot_token: str, app_token: str, stop_event: threading.Event
    ) -> None:
        """Run the Slack Bolt app in Socket Mode."""
        try:
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler

            app = App(token=bot_token)
            self._app = app
            self._client = app.client

            # Get bot user ID for mention detection
            try:
                auth = app.client.auth_test()
                self._bot_user_id = auth.get("user_id", "")
                logger.info(f"Slack bot user ID: {self._bot_user_id}")
            except Exception as e:
                logger.warning(f"Could not get bot user ID: {e}")

            self._gate = InterruptGate(bot_user_id=self._bot_user_id)

            # --- Event handlers ---

            @app.event("message")
            def handle_message(event, say):
                self._on_message_event(event)

            @app.event("reaction_added")
            def handle_reaction(event, say):
                self._on_reaction_event(event)

            @app.event("app_mention")
            def handle_mention(event, say):
                self._on_mention_event(event)

            self._set_state("connected")
            logger.info("Slack bot connected via Socket Mode")

            # Start Socket Mode handler (blocking)
            handler = SocketModeHandler(app, app_token)
            handler.connect()

            # Wait for stop signal
            while not stop_event.is_set():
                stop_event.wait(timeout=1.0)

            handler.close()
            logger.info("Slack bot disconnected")

        except ImportError:
            logger.error(
                "slack-bolt not installed. Run: pip install slack-bolt"
            )
            self._set_state("error")
        except Exception as e:
            logger.error(f"Slack bot error: {e}", exc_info=True)
            self._set_state("error")

    def _on_message_event(self, event: dict) -> None:
        """Handle a Slack message event."""
        # Skip bot's own messages
        if event.get("bot_id") or event.get("user") == self._bot_user_id:
            return

        channel = event.get("channel", "")

        # Filter to configured channels (if set)
        if self.channel_ids and channel not in self.channel_ids:
            return

        # Detect DMs (channel type 'im')
        is_dm = event.get("channel_type") == "im"

        # Detect @mentions in text
        is_mention = (
            self._bot_user_id is not None
            and f"<@{self._bot_user_id}>" in event.get("text", "")
        )

        slack_event = SlackEvent(
            type="message",
            channel=channel,
            user=event.get("user", "unknown"),
            text=event.get("text", ""),
            ts=event.get("ts", ""),
            thread_ts=event.get("thread_ts"),
            is_mention=is_mention,
            is_dm=is_dm,
            raw=event,
        )
        self._handle_event(slack_event)

    def _on_reaction_event(self, event: dict) -> None:
        """Handle a reaction_added event."""
        channel = event.get("item", {}).get("channel", "")
        if self.channel_ids and channel not in self.channel_ids:
            return

        slack_event = SlackEvent(
            type="reaction",
            channel=channel,
            user=event.get("user", "unknown"),
            text="",
            ts=event.get("item", {}).get("ts", ""),
            emoji=event.get("reaction", ""),
            raw=event,
        )
        self._handle_event(slack_event)

    def _on_mention_event(self, event: dict) -> None:
        """Handle an app_mention event."""
        channel = event.get("channel", "")
        if self.channel_ids and channel not in self.channel_ids:
            return

        slack_event = SlackEvent(
            type="mention",
            channel=channel,
            user=event.get("user", "unknown"),
            text=event.get("text", ""),
            ts=event.get("ts", ""),
            thread_ts=event.get("thread_ts"),
            is_mention=True,
            raw=event,
        )
        self._handle_event(slack_event)
