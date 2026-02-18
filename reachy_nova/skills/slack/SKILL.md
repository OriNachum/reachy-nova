---
name: slack
description: >
  Check your Slack messages, send a message, reply to a thread, or react
  with an emoji.
metadata:
  author: reachy-nova
  version: "1.0"
---

# Slack Skill

Interact with Slack channels — send messages, read recent messages,
reply to threads, and add emoji reactions.

## Parameters
- action (string, required): One of "send_message", "read_messages", "reply_to_thread", "add_reaction"
- text (string, optional): Message text (for send_message/reply_to_thread)
- channel (string, optional): Slack channel ID (defaults to first configured channel)
- thread_ts (string, optional): Thread timestamp for reply_to_thread
- emoji (string, optional): Emoji name for add_reaction (e.g. "thumbsup")
- ts (string, optional): Message timestamp for add_reaction
- count (number, optional): Number of messages to read (default 10, max 50)

## Examples
- "read my Slack messages" → action: read_messages
- "send hello everyone to Slack" → action: send_message, text: "hello everyone"
- "reply to that thread saying thanks" → action: reply_to_thread, text: "thanks"
- "react to that with thumbsup" → action: add_reaction, emoji: "thumbsup"
