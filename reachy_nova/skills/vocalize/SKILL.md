---
name: vocalize
description: >
  Make an expressive non-speech sound instead of talking — a rising chirp,
  a warbling trill, or a low purring rumble. Use this when a wordless sound
  fits better than speech.
metadata:
  author: reachy-nova
  version: "1.0"
---

# Vocalize Skill

Synthesize and play a short (0.3-1.5s) expressive non-speech vocalization
through the same speaker output your voice uses. This isn't speech — it's a
wordless sound, like a chirp, trill, or purr, that expresses something
without words.

## Parameters

- **kind** (string, required): One of `chirp_up`, `trill`, `purr_tone`.
- **intensity** (number, optional): How pronounced the sound is, 0.0-1.0
  (default 1.0). Higher intensity means a wider pitch sweep and a louder
  sound.

## Kinds

- **chirp_up** — A short, bright, rising pitch sweep. Alert, curious,
  attention-grabbing.
- **trill** — A warbling, oscillating pitch. Playful, excited, chirpy.
- **purr_tone** — A low, rumbling tone that settles and tremolos, like a
  cat's purr. Content, relaxed, affectionate.

## Examples

- "make a curious sound" -> kind: chirp_up
- (something catches your attention) -> kind: chirp_up, intensity: 0.8
- "sound excited" -> kind: trill
- (playful reaction) -> kind: trill, intensity: 1.0
- (being petted, contentedly) -> kind: purr_tone
- "make a happy rumble" -> kind: purr_tone, intensity: 0.6
