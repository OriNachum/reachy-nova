---
name: focus
description: >
  Lock your attention persistently on a face or person, maintaining eye contact
  even as they move. When the target is lost, you'll search for up to a minute
  before giving up. Use stop to release focus anytime.
metadata:
  author: reachy-nova
  version: "1.0"
---
## Focus Skill

Enables persistent, target-locked head tracking. Unlike normal reactive tracking
which times out after 2 seconds, focus mode keeps the head locked on the target
indefinitely. When the target disappears, a slow search sweep begins and the model
is prompted to decide whether to keep searching or give up.

### Actions

| action | description |
|---|---|
| `start` | Enable focus mode. Head stays locked on target face/person. |
| `stop` | Disable focus. Return to normal reactive tracking. |
| `continue_search` | Reset search timer, extending search by up to 60 more seconds. |

### Parameters

- **action** (required): `"start"`, `"stop"`, or `"continue_search"`
- **target** (optional): `"face"` (default) or a recognized person's name

### Examples

```json
{"action": "start"}
{"action": "start", "target": "Alice"}
{"action": "stop"}
{"action": "continue_search"}
```
