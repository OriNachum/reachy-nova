---
name: memory
description: >
  Think back and recall what you know — about people, facts, or things you've
  been told. You can also commit new things to memory.
metadata:
  author: reachy-nova
  version: "1.0"
---

# Memory Skill

Access persistent memory across sessions. You can query knowledge, store new
facts, or retrieve background context about the user and world.

## Parameters
- query (string, required): What to recall, look up, or store
- mode (string, optional): "query" (default), "store", or "context"

## Examples
- "What do you know about Ori?"
- "Remember that the user likes espresso"
- "What projects am I working on?"
- "Store: meeting with team at 3pm tomorrow"
