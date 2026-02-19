---
name: gesture
description: >
  Nod, shake your head, tilt curiously, look up to think, or react to touch
  with nuzzling, purring, or enjoying. Use gestures to physically express yourself.
metadata:
  author: reachy-nova
  version: "1.2"
---

# Gesture Skill

Animate the robot's head with expressive gestures. Tracking pauses during
the animation and resumes automatically when finished.

## Parameters
- gesture (string, required): One of "yes", "no", "curious", "pondering", "boredom", "nuzzle", "purr", "enjoy"

## Gestures
- **yes** — Nod the head up and down (pitch oscillation, ~1.2s)
- **no** — Shake the head side to side (yaw oscillation, ~1.5s)
- **curious** — Tilt/roll the head to the side like a curious dog (~1.8s)
- **pondering** — Look up and to the side diagonally, thinking pose (~2.3s)
- **boredom** — Slow look away and down, sighing motion (~5.5s)
- **nuzzle** — Side-to-side yaw oscillation with subtle roll, like a cat nuzzling (~2.5s). Sets mood to "excited".
- **purr** — Slow pitch+roll wobble, leaning slightly down, deep contentment (~3s). Sets mood to "happy".
- **enjoy** — Brief lean into touch, short nuzzle, settle back (~2s). Sets mood to "happy".

## Examples
- "nod your head" -> gesture: yes
- "do you agree?" -> gesture: yes
- "shake your head" -> gesture: no
- "that's not right" -> gesture: no
- "that's interesting" -> gesture: curious
- "hmm really?" -> gesture: curious
- "let me think about that" -> gesture: pondering
- "searching the web for..." -> gesture: pondering
- "that's a tough question" -> gesture: pondering
- "this is boring" -> gesture: boredom
- "i'm tired of this" -> gesture: boredom
- (someone pets you) -> gesture: nuzzle
- "that feels nice" -> gesture: enjoy
- (prolonged head scratching) -> gesture: purr
- "mmm that's so good" -> gesture: purr
- (gentle touch on head) -> gesture: enjoy
