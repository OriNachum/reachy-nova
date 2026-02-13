# Skills System Documentation

This documentation covers the `SkillManager` and the skills system, which enables Reachy Nova to use tools (function calling) dynamically.

## Overview

 The skills system allows the robot to "know" what it can do and expose these capabilities to the Nova Sonic model as tools. Skills are defined by metadata files (`SKILL.md`) and implemented via Python executors.

**File:** `reachy_nova/skills.py`

## Class Structure

### `SkillManager`

The central registry for skills.

#### Constructor

```python
SkillManager(skills_dir: Path | None = None)
```

-   **skills_dir**: Directory to scan for skills (default: `reachy_nova/skills/`).

#### Key Methods

-   `discover()`: Scans the `skills_dir` for subdirectories containing `SKILL.md`. Parses metadata (name, description) from these files.
-   `register_executor(name: str, executor: Callable, input_schema: dict)`:
    -   Associates a Python function with a skill name.
    -   If the skill uses `SKILL.md`, this attaches the code to the metadata.
    -   If no `SKILL.md` exists, a programmatic skill is created.
-   `get_tool_specs()`: Returns a list of tool definitions formatted for Nova Sonic's API.
-   `execute(tool_name: str, params: dict)`: Runs the registered executor for a skill and returns the output string.

### Skill Definition (`SKILL.md`)

Skills are defined in markdown files with YAML frontmatter.

**Example:** `reachy_nova/skills/look/SKILL.md`

```markdown
---
name: look
description: Listen using your camera...
metadata:
  input_schema: ...
---

# Look Skill
Detailed instructions...
```

### Tool Use Flow

1.  **Discovery**: `SkillManager.discover()` loads available skills.
2.  **Registration**: `main.py` registers executors (e.g., `vision.analyze_latest` for the "look" skill).
3.  **Configuration**: `NovaSonic` is initialized with `skill_manager.get_tool_specs()`.
4.  **Invocation**:
    -   Nova Sonic decides to use a tool and sends a `toolUse` event.
    -   `NovaSonic` accumulates the event data.
    -   `NovaSonic` calls `main.py`'s `on_tool_use`.
5.  **Execution**:
    -   `main.py` runs `skill_manager.execute()` in a background thread.
    -   The result is sent back to Nova Sonic via `sonic.send_tool_result()`.

## Available Skills

### `look`
-   **Description**: Look through the camera to see the surroundings.
-   **Parameters**: `query` (string) - What to look for.
-   **Implementation**: Calls `NovaVision.analyze_latest()`.
