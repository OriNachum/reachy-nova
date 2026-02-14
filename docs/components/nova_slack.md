# Nova Slack Documentation

This documentation covers the `NovaSlack` component, which enables the robot to interact with a Slack workspace.

## Overview

`NovaSlack` connects to Slack using Socket Mode (via `slack-bolt`), allowing the robot to receive events without exposing a public URL. It features an intelligent **Interrupt Gate** that decides whether incoming messages should interrupt the robot's current activity, be queued for later, or be ignored.

**File:** `reachy_nova/nova_slack.py`

## Class Structure

### `NovaSlack`

The main class managing the Slack connection and event handling.

#### Constructor

```python
NovaSlack(
    on_event: Callable[[SlackEvent], None] | None = None,
    on_state_change: Callable[[str], None] | None = None,
    on_interrupt: Callable[[SlackEvent], None] | None = None,
    channel_ids: list[str] | None = None,
)
```

-   **on_event**: Callback for all processed Slack events.
-   **on_state_change**: Callback when connection state changes (`idle`, `connected`, `error`).
-   **on_interrupt**: Callback when a message passes the interrupt gate.
-   **channel_ids**: Optional list of channel IDs to filter events.

#### Key Methods

-   `start(stop_event: threading.Event)`: Starts the Socket Mode client in a daemon thread.
-   `update_context(voice_state: str, engagement_level: float)`: Updates the interrupt gate with the robot's current state.
-   `execute(params: dict)`: Performs actions like sending messages, reading history, or reacting.
-   `queue_task(action: str, **kwargs)`: Runs `execute` in a background thread.

### `InterruptGate`

Depending on the robot's state and the message content, the gate determines how to handle an incoming event.

#### Decision Tiers

1.  **Fast Rules** (Immediate Interrupt):
    -   @Mentions or Direct Messages.
    -   Urgency keywords (e.g., "urgent", "help", "asap").
    -   Robot is `idle` and engagement level is low (< 0.3).
2.  **Ignore Rules**:
    -   Short/Empty messages (< 5 chars).
    -   Reactions without text.
3.  **LLM Evaluation** (Ambiguous Cases):
    -   Uses Amazon Bedrock **Nova 2 Lite** to decide.
    -   Prompt includes the message and the robot's current state (`voice_state`, `engagement_level`).
    -   Model returns `INTERRUPT`, `QUEUE`, or `IGNORE`.

### `SlackEvent`

A normalized data class for Slack events:

-   `type`: `message`, `reaction`, `reply`, `mention`
-   `channel`, `user`, `text`, `ts` (timestamp)
-   `is_mention`, `is_dm` flags

## Capabilities

The `execute` method supports the following actions, typically triggered by the `SlackSkill` via Nova Sonic:

-   **`send_message`**: Post a message to a channel.
-   **`read_messages`**: Get recent messages from the history buffer.
-   **`read_queued`**: Retrieve messages that were queued by the interrupt gate.
-   **`reply_to_thread`**: Reply to a specific message thread.
-   **`add_reaction`**: Add an emoji reaction to a message.

## Setup

Requires the following environment variables:
-   `SLACK_BOT_TOKEN`: Bot User OAuth Token (`xoxb-...`)
-   `SLACK_APP_TOKEN`: App-Level Token (`xapp-...`)
-   `AWS_DEFAULT_REGION`: AWS Region for Bedrock (default: `us-east-1`)

## Usage Example

```python
def handle_slack_interrupt(event: SlackEvent):
    print(f"Interrupting for: {event.text}")
    # Logic to pause current task and speak the message

slack = NovaSlack(
    on_interrupt=handle_slack_interrupt,
    channel_ids=["C12345678"]
)
slack.start(stop_event)

# In main loop, keep context updated
slack.update_context(voice_state="speaking", engagement_level=0.8)
```
