"""Nova Slack - Slack integration for Reachy Nova using Socket Mode.

Connects to Slack via slack-bolt (Socket Mode, no public URL needed).
Publishes events to MQTT; the Nervous System handles interrupt decisions.
"""

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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


class NovaSlack:
    """Manages Slack integration via Socket Mode."""

    def __init__(
        self,
        on_event: Callable[[SlackEvent], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
        channel_ids: list[str] | None = None,
    ):
        self.on_event = on_event
        self.on_state_change = on_state_change
        self.channel_ids = set(channel_ids or [])

        self._recent_messages: deque[SlackEvent] = deque(maxlen=50)
        self._thread: threading.Thread | None = None
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
        """Process an incoming Slack event — publish via on_event callback."""
        self._recent_messages.append(event)
        self.last_event = event

        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

    def execute(self, params: dict) -> str:
        """Blocking skill executor for Sonic tool_use.

        Actions: send_message, read_messages, reply_to_thread, add_reaction
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
