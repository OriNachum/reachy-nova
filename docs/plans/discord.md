# Discord Integration for Reachy Nova

## Context
Reachy Nova currently integrates voice (Nova Sonic), vision (Nova Pro), and browser (Nova Act). This adds a Discord subsystem so the robot can participate in Discord conversations — listening to channel events and sending messages — with an intelligent interrupt gate that uses rule-based checks + Amazon Nova Lite to decide whether to interrupt the robot's current voice engagement.

## New Files

### 1. `reachy_nova/nova_discord.py`
Core Discord subsystem following the NovaBrowser pattern:

- **`DiscordEvent`** dataclass — normalized event for messages, reactions, replies, mentions
- **`InterruptGate`** — three-tier decision engine:
  1. Fast rules: @mentions → interrupt, urgency keywords ("urgent", "emergency", "help") → interrupt, robot idle → interrupt
  2. Short/empty messages → ignore
  3. Ambiguous → call `amazon.nova-lite-v1:0` via Bedrock (wrapped in `asyncio.to_thread()` to avoid blocking the async loop)
- **`NovaDiscord`** class:
  - Constructor takes callbacks: `on_event`, `on_state_change`, `on_interrupt`
  - `start(stop_event)` runs discord.py bot in its own daemon thread + asyncio event loop
  - `_recent_messages` (deque, maxlen=50) stores all channel messages
  - `_queued_messages` (deque, maxlen=20) stores gate-deferred messages
  - `execute(params)` — blocking skill executor path (same pattern as `NovaBrowser.execute()`)
  - `queue_task(action, **kwargs)` — fire-and-forget outbound messages
  - `update_context(voice_state, engagement_level)` — called from main loop to feed the interrupt gate
  - Graceful degradation: if `DISCORD_BOT_TOKEN` not set, `start()` returns immediately with a warning

### 2. `reachy_nova/skills/discord/SKILL.md`
Skill definition with actions: `send_message`, `read_messages`, `read_queued`, `reply_to_thread`

## Modified Files

### 3. `reachy_nova/main.py`
- Import `NovaDiscord`, `DiscordEvent`
- Add to `app_state`: `discord_state`, `discord_last_event`, `discord_queued_count`
- Wire callbacks:
  - `on_discord_event` → updates state
  - `on_discord_state` → updates state
  - `on_discord_interrupt` → sets mood "surprised", calls `sonic.inject_text()` with Discord message context
- Parse `DISCORD_CHANNEL_IDS` from env
- Instantiate `NovaDiscord` with callbacks
- Register `discord` skill executor that delegates to `discord_bot.execute(params)`, defaulting to first configured channel
- Update system prompt to mention Discord capability
- Call `discord_bot.start(stop_event)` alongside other subsystems
- Feed `update_context()` in main loop
- Add API endpoints: `GET /api/discord/state`, `POST /api/discord/send`

### 4. `pyproject.toml`
Add `discord.py>=2.3.0` to dependencies

### 5. `.env.sample`
Add `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_IDS`

## Data Flow

```
Discord Channel → on_message → InterruptGate.evaluate()
  ├─ "interrupt" → on_interrupt → sonic.inject_text() → Robot speaks it
  ├─ "queue"     → _queued_messages → read later via "discord" skill
  └─ "ignore"    → dropped

Nova Sonic tool_use("discord") → skill executor → discord_bot.execute()
  → _task_queue → bot async loop → Discord API → result → send_tool_result()
```

## Verification
1. `uv sync` — installs discord.py
2. Set `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_IDS` in `.env`
3. Run `uv run python -m reachy_nova.main` — verify "Discord bot connected" in logs
4. Send a message in a monitored channel — verify it appears in logs with interrupt gate decision
5. Send an @mention — verify robot speaks the Discord message via inject_text
6. Test skill via voice: "read my Discord messages" — verify tool use triggers read_messages
7. Test send: "send hello to Discord" — verify message appears in channel
8. Test with no token: remove DISCORD_BOT_TOKEN, verify graceful degradation (warning, no crash)
