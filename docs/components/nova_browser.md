# Nova Browser Documentation

This documentation covers the `NovaBrowser` component, which enables browser automation capabilities for the robot.

## Overview

The `NovaBrowser` allows Reachy Nova to perform complex browser actions based on natural language commands. It leverages the `nova-act` library for browser control and automation.

**File:** `reachy_nova/nova_browser.py`

## Class Structure

### `NovaBrowser`

The main class managing browser tasks.

#### Constructor

```python
NovaBrowser(
    on_result: Callable[[str], None] | None = None,
    on_screenshot: Callable[[str], None] | None = None,
    on_state_change: Callable[[str], None] | None = None,
    headless: bool = False,
)
```

-   **on_result**: Callback when a task is completed (text summary).
-   **on_screenshot**: Callback with base64-encoded screenshot from the browser.
-   **on_state_change**: Callback when the browser state (`idle`, `busy`, `error`) changes.
-   **headless**: Whether to run the browser in the background (default: `False`).

#### Key Methods

-   `start(stop_event: threading.Event)`: Launches the background thread that processes the task queue.
-   `queue_task(instruction: str, url: str | None = None)`: Adds a new automation request.
    -   `instruction`: What the user wants to do (e.g., "Find a recipe for pizza").
    -   `url`: The starting URL (default: "https://www.google.com").
-   `capture_screenshot()`: Takes a screenshot of the current page.

### Automation Workflow

1.  **Request Queue**: Incoming tasks are placed in a thread-safe `queue.Queue`.
2.  **Worker Loop**: The background thread (`_run_loop`) continuously checks for new tasks.
3.  **Task Execution** (`_execute_task`):
    -   Starts a new `NovaAct` session if needed.
    -   Uses `NovaAct.act()` to execute the user's instructions (e.g., searching, clicking).
    -   Captures a screenshot at the end of the action sequence.
    -   Calls `on_result` with the outcome.
4.  **Result Injection**: The result is passed back to the main application, effectively "telling" the robot what happened so it can respond verbally.

### Dependencies

-   **nova-act**: External library for browser automation using Playwright and LLMs.
-   **Playwright**: Underlying browser driver.

### Concurrency

Browser automation is inherently slow. The component uses a dedicated worker thread to ensure that automation steps (like page loads) do not block the main application loop.

### Progress Narration and Duplicate-Task Dedupe (issue #8)

`_emit_progress` is the single emission point for progress narration and
always logs a `[Browser progress] ...` line, but the `on_progress` callback
itself only fires for two phases — `PROGRESS_PHASE_START` and
`PROGRESS_PHASE_WORKING` — each worded as a status, never a request (e.g.
`"Status: your browser is working on '<instruction>' — no action needed."`).
This matters because a fresh Sonic session (after a mid-flight restart)
could otherwise read a bare `"Working on: X..."` line as an instruction and
call the browse tool again. The callback is additionally rate-limited to at
most once per `PROGRESS_RATE_LIMIT_S` (10 s) per task, tracked via an
injectable `clock` (defaults to `time.monotonic`) and reset at the start of
each `_execute_task` call; `"Done! Reading results..."` is a `logger.info`
line only and never reaches `on_progress`.

`queue_task` normalises the instruction (lower-cased, collapsed whitespace)
and drops a duplicate — logging one `sensory_log.stage("act", "browser",
"queue_task", "dropped reason=duplicate ...")` line and returning
`{"ok": True, "queued": False, "duplicate": True, ...}` — when it matches
either the currently running task (regardless of how long it's been
running) or any queued/recently-enqueued task within `DEDUPE_WINDOW_S`
(300 s). A non-duplicate call returns `{"ok": True, "queued": True, ...}`;
the disabled-flag early return returns `{"ok": False, "queued": False,
"reason": "nova-act-disabled"}`. The `_act_on_agentcore` NovaAct/
`browser_session`/`act_get` usage itself is unchanged by any of this.

## Usage Example

```python
def handle_browser_result(text):
    print(f"Browser did: {text}")
    sonic.inject_text(f"[Browser: {text}] Tell the user what happened.")

def handle_screenshot(b64):
    print(f"Captured screen (base64 length: {len(b64)})")

browser = NovaBrowser(
    on_result=handle_browser_result,
    on_screenshot=handle_screenshot,
    headless=True,
)
browser.start(stop_event)

# In response to voice command
browser.queue_task("Search for the weather in Paris")
```
