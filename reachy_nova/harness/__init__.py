"""Reachy Nova harness — an on-device AI peripheral over reachy-mini-cli's symbolic runtime.

The harness attaches to the runtime ONLY through its documented seams:

- **read**: MQTT bus ``reachy/events/#`` + ``reachy/state/#`` (localhost broker)
- **hear**: the audio-tee Unix socket (``f32le`` mono, JSON header line)
- **act**: the intents spool (atomic JSON files) + the rules overlay (``rules.toml``)
- **speak**: the daemon HTTP media route (upload + play_sound) until the
  streaming speaker feed (reachy-mini-cli#162) ships

It never imports ``reachy_mini`` and contains zero motion code — enforced by
``tests/test_harness_boundary.py``.
"""
