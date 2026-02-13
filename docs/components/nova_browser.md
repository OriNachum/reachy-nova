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
