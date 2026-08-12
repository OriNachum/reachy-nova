"""Tests for the NOVA_ACT_ENABLED flag gating reachy_nova.nova_browser (t15).

Nova Act (browser automation via nova_act/playwright) must be default-off.
With the flag off, NOT ONE code path in this module may import ``nova_act``
or ``playwright`` — those packages may not even be installed. This is
verified by asserting they never land in ``sys.modules`` after exercising
every entry point (construction, start, queue_task, execute).

When the flag is on, we do NOT want to actually import nova_act/playwright
in this test (it can open real browser processes) — instead we monkeypatch
the worker seam (``_run_loop``) to prove the enabled path is actually taken
(a worker thread is spun up) without touching the real automation code.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time

import pytest

from reachy_nova import nova_browser, sensory_log
from reachy_nova.nova_browser import NovaBrowser, act_enabled

_SENSE_LINE_RE = re.compile(
    r"^\[SENSE stage=(?P<stage>\S+) source=(?P<source>\S+) event=(?P<event>\S+)\] (?P<detail>.*)$"
)


def _sensory_records(caplog):
    return [r for r in caplog.records if r.name == "nova.sensory"]


def _assert_no_nova_act_or_playwright_imported():
    assert "nova_act" not in sys.modules
    assert "playwright" not in sys.modules


class TestActEnabledFlagParsing:
    """act_enabled() env parsing (0/1/true/false/absent/case variants)."""

    def test_absent_defaults_off(self, monkeypatch):
        monkeypatch.delenv("NOVA_ACT_ENABLED", raising=False)
        assert act_enabled() is False

    def test_explicit_zero_is_off(self, monkeypatch):
        monkeypatch.setenv("NOVA_ACT_ENABLED", "0")
        assert act_enabled() is False

    def test_explicit_false_is_off(self, monkeypatch):
        monkeypatch.setenv("NOVA_ACT_ENABLED", "false")
        assert act_enabled() is False

    def test_empty_string_is_off(self, monkeypatch):
        monkeypatch.setenv("NOVA_ACT_ENABLED", "")
        assert act_enabled() is False

    def test_garbage_value_is_off(self, monkeypatch):
        monkeypatch.setenv("NOVA_ACT_ENABLED", "maybe")
        assert act_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "True", "YES", "On", "TRUE"])
    def test_truthy_values_are_on(self, monkeypatch, value):
        monkeypatch.setenv("NOVA_ACT_ENABLED", value)
        assert act_enabled() is True

    @pytest.mark.parametrize("value", ["no", "off", "2", "-1", "disabled"])
    def test_other_values_are_off(self, monkeypatch, value):
        monkeypatch.setenv("NOVA_ACT_ENABLED", value)
        assert act_enabled() is False


class TestFlagOffZeroImport:
    """Flag off (default): zero nova_act/playwright import across every entry point."""

    def test_module_import_alone_does_not_import_nova_act_or_playwright(self):
        # nova_browser itself is already imported (module-level, above) —
        # confirm that alone didn't drag in nova_act/playwright.
        _assert_no_nova_act_or_playwright_imported()

    def test_construct_start_queue_execute_do_not_import(self, monkeypatch, caplog):
        monkeypatch.delenv("NOVA_ACT_ENABLED", raising=False)
        stop_event = threading.Event()
        browser = NovaBrowser()

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            browser.start(stop_event)
            browser.queue_task("do something")
            result = browser.execute("do something else")

        _assert_no_nova_act_or_playwright_imported()
        assert browser._thread is None, "no worker thread should be started when disabled"
        assert isinstance(result, str) and result  # execute() returns immediately, no hang

    def test_start_logs_sense_drop_line(self, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_ACT_ENABLED", "0")
        browser = NovaBrowser()

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            browser.start(threading.Event())

        records = _sensory_records(caplog)
        assert len(records) == 1
        match = _SENSE_LINE_RE.match(records[0].getMessage())
        assert match is not None, f"unparseable sensory log line: {records[0].getMessage()!r}"
        assert match.group("stage") == "act"
        assert match.group("source") == "browser"
        assert match.group("detail") == "dropped reason=nova-act-disabled"

    def test_queue_task_logs_sense_drop_line_and_does_not_enqueue(self, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_ACT_ENABLED", "false")
        browser = NovaBrowser()

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            browser.queue_task("go to example.com")

        records = _sensory_records(caplog)
        assert len(records) == 1
        detail = _SENSE_LINE_RE.match(records[0].getMessage()).group("detail")
        assert detail == "dropped reason=nova-act-disabled"
        assert browser._task_queue.empty()

    def test_execute_logs_sense_drop_line_and_returns_immediately(self, monkeypatch, caplog):
        monkeypatch.setenv("NOVA_ACT_ENABLED", "off")
        browser = NovaBrowser()

        start = time.monotonic()
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            result = browser.execute("go to example.com")
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, "execute() must not block waiting on a never-run worker"
        records = _sensory_records(caplog)
        assert len(records) == 1
        detail = _SENSE_LINE_RE.match(records[0].getMessage()).group("detail")
        assert detail == "dropped reason=nova-act-disabled"
        assert "disabled" in result.lower()


class TestFlagOnTakesEnabledPathWithoutRealImport:
    """Flag on: the enabled path (worker thread) is actually taken.

    We never let the test import the real nova_act/playwright — the worker
    entry point (_run_loop) is monkeypatched to a stub that just records it
    ran, proving start() took the "enabled" branch instead of the no-op.
    """

    def test_start_spins_up_worker_thread_when_enabled(self, monkeypatch):
        monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
        ran = threading.Event()
        seen_stop_event = []

        def fake_run_loop(self, stop_event):
            seen_stop_event.append(stop_event)
            ran.set()

        monkeypatch.setattr(NovaBrowser, "_run_loop", fake_run_loop)

        browser = NovaBrowser()
        stop_event = threading.Event()
        browser.start(stop_event)

        assert ran.wait(timeout=2.0), "enabled path should start a worker thread"
        assert browser._thread is not None
        assert browser._thread.name == "nova-browser"
        assert seen_stop_event == [stop_event]
        browser._thread.join(timeout=2.0)

    def test_enabling_does_not_itself_import_nova_act(self, monkeypatch):
        # Enabling the flag alone (without the worker thread actually
        # reaching NovaAct construction) must not import nova_act either —
        # the import only happens inside _execute_task, on a real task.
        monkeypatch.setenv("NOVA_ACT_ENABLED", "true")
        assert act_enabled() is True
        _assert_no_nova_act_or_playwright_imported()


# --------------------------------------------------------------------------- #
# Execution surface (NOVA_ACT_BROWSER) — deviation d5: AgentCore first        #
# --------------------------------------------------------------------------- #


def test_browser_surface_defaults_to_local(monkeypatch):
    from reachy_nova import nova_browser

    monkeypatch.delenv("NOVA_ACT_BROWSER", raising=False)
    assert nova_browser.browser_surface() == "local"
    monkeypatch.setenv("NOVA_ACT_BROWSER", "nonsense")
    assert nova_browser.browser_surface() == "local"


def test_browser_surface_agentcore(monkeypatch):
    from reachy_nova import nova_browser

    monkeypatch.setenv("NOVA_ACT_BROWSER", " AgentCore ")
    assert nova_browser.browser_surface() == "agentcore"


def test_agentcore_surface_routes_execute_to_the_hosted_session(monkeypatch):
    """On the agentcore surface, _execute_task never touches nova_act locally —
    the whole act runs through _act_on_agentcore (mocked here)."""
    from reachy_nova.nova_browser import NovaBrowser

    monkeypatch.setenv("NOVA_ACT_BROWSER", "agentcore")
    browser = NovaBrowser()
    calls = []

    class _Result:
        success = True
        parsed_response = "The answer is 42."

    monkeypatch.setattr(
        browser, "_act_on_agentcore", lambda instr, url: calls.append((instr, url)) or _Result()
    )
    spoken = []
    browser.on_result = spoken.append
    text = browser._execute_task({"instruction": "find the answer", "url": None})
    assert calls == [("find the answer", None)]
    assert text == "The answer is 42."
    assert spoken == ["The answer is 42."]
    assert import_forbidden_absent()


def import_forbidden_absent():
    import sys

    return "nova_act" not in sys.modules and "playwright" not in sys.modules


def test_act_max_steps_default_and_env(monkeypatch):
    from reachy_nova import nova_browser

    monkeypatch.delenv("NOVA_ACT_MAX_STEPS", raising=False)
    assert nova_browser.act_max_steps() == 30
    monkeypatch.setenv("NOVA_ACT_MAX_STEPS", "45")
    assert nova_browser.act_max_steps() == 45
    for bad in ("0", "-3", "many"):
        monkeypatch.setenv("NOVA_ACT_MAX_STEPS", bad)
        assert nova_browser.act_max_steps() == 30
