---
name: forge
description: >
  Forge a brand new skill you don't have yet, or improve one you already
  have, by describing what you want in plain language. The request goes to
  a coder model that writes and statically validates the skill; once it
  passes validation it activates automatically (no admin gate) — you won't
  be able to use it in this same turn, but you'll be told the moment it's
  ready.
metadata:
  author: reachy-nova
  version: "1.0"
---

# Forge Skill

Ask the skill-forge to write you a new capability, or improve an existing
one, from a natural-language goal. Dispatch runs in the background and
returns immediately — the forge later reports success (staged, then
activated) or rejection through its own events, never by blocking this
conversation turn.

## Parameters

- goal (string, required): What new capability to forge, in plain language
- improve (string, optional): Existing executor.py source to improve/iterate on

## Examples

- "learn to wave hello" -> goal: "wave hello when greeted"
- "you should be better at telling jokes" -> goal: "tell a short joke on request"
- "improve the wave skill, it's too slow" -> goal: "make waving faster", improve: "<previous executor.py source>"
