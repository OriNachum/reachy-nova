"""Tests for status-shaped browser progress cues and their 10s rate limit
(issue #8, t2).

Live (2026-09-06), the old progress narration fired several messages within
the same second: most got dropped by the voice model's 3 s inject throttle,
and "Working on: X..." landed as a message a *fresh* Sonic session (after a
mid-flight restart) could read as a request, calling the browse tool again.

The fix collapses narration to at most one status cue per phase ("start",
"working"), worded as a status ("Status: ... — no action needed."), rate
limited to at most one ``on_progress`` call per 10 s per task, with
"Done! Reading results..." demoted to a log line only.

The local (non-agentcore) execution path is exercised by injecting a fake
``nova_act`` module into ``sys.modules`` — never the real package — so
``_execute_task``'s local branch runs both the "start" and "working"
``_emit_progress`` call sites without ever touching Playwright/NovaAct.
"""

from __future__ import annotations

import logging
import types

import pytest

from reachy_nova.nova_browser import NovaBrowser


class _FakeResult:
    success = True
    parsed_response = "The answer is 42."


class _FakeNovaAct:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def go_to_url(self, url):
        pass

    def act(self, instruction, max_steps=None):
        return _FakeResult()

    def stop(self):
        pass


@pytest.fixture
def fake_nova_act_module(monkeypatch):
    """Install a fake ``nova_act`` module so the local execution path's
    ``from nova_act import NovaAct`` succeeds without the real package."""
    module = types.ModuleType("nova_act")
    module.NovaAct = _FakeNovaAct
    monkeypatch.setitem(__import__("sys").modules, "nova_act", module)
    return module


def _make_browser(monkeypatch, clock, act_enabled=True):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1" if act_enabled else "0")
    monkeypatch.delenv("NOVA_ACT_BROWSER", raising=False)  # local surface
    seen = []
    browser = NovaBrowser(on_progress=seen.append, clock=clock)
    # Skip the real Workflow context manager entirely — irrelevant to
    # progress-cue behavior and would otherwise import nova_act.types.workflow.
    monkeypatch.setattr(browser, "_ensure_workflow", lambda: None)
    return browser, seen


def test_single_run_emits_at_most_one_cue_per_phase_worded_as_status(
    monkeypatch, caplog, fake_nova_act_module
):
    """A whole task runs inside one second on the local path (start + working)."""
    fake_now = [1000.0]
    browser, seen = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    with caplog.at_level(logging.INFO):
        result_text = browser._execute_task({"instruction": "find the weather", "url": None})

    assert result_text == "The answer is 42."
    # Both start and working land within the same (fake) second; the 10s
    # per-task rate limit means only the FIRST on_progress call goes through.
    assert len(seen) == 1
    for message in seen:
        assert message.startswith("Status:")
        assert "no action needed" in message
    assert "Done! Reading results" not in " ".join(seen)

    # But every phase call still logs, rate-limited or not.
    progress_lines = [
        r.getMessage() for r in caplog.records if "[Browser progress]" in r.getMessage()
    ]
    assert any("starting on 'find the weather'" in line for line in progress_lines)
    assert any("working on 'find the weather'" in line for line in progress_lines)
    assert any("Done! Reading results" in line for line in progress_lines)


def test_second_phase_call_inside_rate_limit_window_is_suppressed_but_logged(monkeypatch, caplog):
    fake_now = [0.0]
    browser, seen = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    with caplog.at_level(logging.INFO):
        browser._emit_progress(browser._phase_message(browser.PROGRESS_PHASE_START, "task a"))
        fake_now[0] = 5.0  # inside the 10s window
        browser._emit_progress(browser._phase_message(browser.PROGRESS_PHASE_WORKING, "task a"))

    assert len(seen) == 1
    assert seen[0].startswith("Status:")
    progress_lines = [
        r.getMessage() for r in caplog.records if "[Browser progress]" in r.getMessage()
    ]
    assert len(progress_lines) == 2


def test_phase_call_after_rate_limit_window_elapses_is_delivered(monkeypatch):
    fake_now = [0.0]
    browser, seen = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    browser._emit_progress(browser._phase_message(browser.PROGRESS_PHASE_START, "task a"))
    fake_now[0] = 10.1  # outside the 10s window
    browser._emit_progress(browser._phase_message(browser.PROGRESS_PHASE_WORKING, "task a"))

    assert len(seen) == 2


def test_new_task_resets_the_rate_limit_window(monkeypatch, fake_nova_act_module):
    """_execute_task resets the per-task rate limit so a later task's own
    start cue is not swallowed by a previous task's recent emission."""
    fake_now = [0.0]
    browser, seen = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    browser._execute_task({"instruction": "first task", "url": None})
    assert len(seen) == 1

    fake_now[0] = 1.0  # well inside 10s of the previous task's emission
    browser._nova = None  # force a fresh NovaAct() so "start" logic re-runs cleanly
    browser._execute_task({"instruction": "second task", "url": None})
    # The new task's own "start" cue still gets through because the per-task
    # rate limit window is reset at the top of _execute_task.
    assert len(seen) == 2
    assert "second task" in seen[-1]


def test_done_reading_results_never_reaches_on_progress(monkeypatch, caplog, fake_nova_act_module):
    fake_now = [100.0]
    browser, seen = _make_browser(monkeypatch, clock=lambda: fake_now[0])

    with caplog.at_level(logging.INFO):
        browser._execute_task({"instruction": "check something", "url": None})

    assert all("Done! Reading results" not in message for message in seen)
    assert any("Done! Reading results" in r.getMessage() for r in caplog.records)
