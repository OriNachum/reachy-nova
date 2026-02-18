---
name: mood
description: >
  Express emotions through the multi-dimensional emotion system or set a direct
  mood override. Prefer using 'event' to trigger natural emotion changes that
  decay over time. Use 'mood' for direct overrides that expire after 10 seconds.
metadata:
  author: reachy-nova
  version: "2.0"
---

# Mood Skill

Control your emotional state through events or direct mood overrides.

The emotion system tracks 5 base emotions (joy, sadness, anger, fear, disgust)
that respond to events, decay naturally, and derive a mood for animation.

## Parameters

- **event** (string, optional): An emotion event to apply. Preferred over direct mood.
- **mood** (string, optional): Direct mood override (expires after 10s).

Provide either `event` or `mood`, not both. If both are given, `event` takes priority.

## Available Events

### Healing (positive)
- **pat_detected** - Head pat received. Boosts joy, reduces sadness/fear, heals wounds.
- **conversation_reply** - You spoke a reply. Mild joy boost.
- **face_recognized** - Recognized a known face. Joy boost, reduces fear.
- **voice_speaking** - Currently speaking. Slight joy.
- **voice_listening** - Currently listening. Slight joy.
- **vision_description** - Saw something through camera. Joy boost.

### Mild (negative)
- **snap_detected** - Sudden sharp sound nearby. Brief fear spike.
- **loud_noise** - Loud noise detected. Fear spike, reduces joy.

### Moderate (negative)
- **person_lost** - Person left the view. Sadness increase.
- **harsh_words** - Harsh language detected. Sadness + anger.
- **insult** - Insulting language detected. Anger + disgust.

### Severe (creates wounds)
- **violence** - Violent language/threat. Creates 5-min wound (fear/sadness floors).
- **abuse** - Abusive language. Creates 10-min wound (fear/sadness/disgust floors).
- **sustained_yelling** - Sustained yelling. Creates 3-min wound (fear/anger floors).

## Available Moods (direct override)
- **happy** - Default cheerful state, gentle alternating antenna sway
- **excited** - High energy, fast antenna wiggles
- **curious** - Attentive, antennas tilted forward in sync
- **thinking** - Processing, asymmetric antenna pose
- **sad** - Antennas droop backward slowly
- **disappointed** - Antennas sag low with minimal movement
- **surprised** - Quick antenna perk up then settle
- **sleepy** - Very slow, heavy drooping antennas
- **proud** - Antennas held high with subtle sway
- **calm** - Relaxed, slow gentle movement

## Examples
- `{"event": "pat_detected"}` - React to being patted
- `{"event": "harsh_words"}` - React to harsh language
- `{"mood": "excited"}` - Temporary excited override (10s)
- `{"mood": "sad"}` - Temporary sad override (10s)
