---
name: gesture
description: >
  Nod, shake your head, tilt curiously, or look up to think. Use gestures
  to physically express yourself.
metadata:
  author: reachy-nova
  version: "1.1"
---

# Gesture Skill

Animate the robot's head with expressive gestures. Tracking pauses during
the animation and resumes automatically when finished.

## Parameters
- gesture (string, required): One of "yes", "no", "curious", "pondering", "boredom"

## Gestures
- **yes** — Nod the head up and down (pitch oscillation, ~1.2s)
- **no** — Shake the head side to side (yaw oscillation, ~1.5s)
- **curious** — Tilt/roll the head to the side like a curious dog (~1.8s)
- **pondering** — Look up and to the side diagonally, thinking pose (~2.3s)
- **boredom** — Slow look away and down, sighing motion (~5.5s)

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
