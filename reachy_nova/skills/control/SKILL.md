---
name: control
description: >
  Move your head or body to look in a specific direction, or turn tracking
  on and off.
metadata:
  author: reachy-nova
  version: "1.0"
---

# Control Skill

Direct control of head and body movement with safety-managed collision avoidance.

## Parameters
- action (string, required): One of "move_head", "move_body", "enable_tracking", "disable_tracking"
- yaw (number, optional): Head yaw angle in degrees, -45 to 45 (negative=right, positive=left). For move_head.
- pitch (number, optional): Head pitch angle in degrees, -15 to 25 (negative=down, positive=up). For move_head.
- body_yaw (number, optional): Body rotation in degrees, -25 to 25. For move_body.
- duration (number, optional): Duration in seconds for body rotation (default 1.5, range 0.5-5). For move_body.

## Examples
- "look to your left" -> action: move_head, yaw: 40, pitch: 0
- "look up" -> action: move_head, yaw: 0, pitch: 20
- "look down and to the right" -> action: move_head, yaw: -30, pitch: -10
- "turn your body left" -> action: move_body, body_yaw: 20
- "look at me again" -> action: enable_tracking
- "stop moving" -> action: disable_tracking
