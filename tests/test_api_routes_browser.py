"""Tests for /api/browser/task using NovaBrowser's typed queue_task result
(PR #26 review, finding 7).

Before this change, submit_browser_task() always returned status="queued"
and always updated ctx.state.browser_task, even when queue_task() actually
discarded the request as a duplicate (or NOVA_ACT_ENABLED was off). An API
client repeating an active/recent instruction was told a new task existed
and the dashboard's shared state was churned for a request that never ran.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_nova import api_routes
from reachy_nova.nova_browser import NovaBrowser
from reachy_nova.state import State


def _make_client(monkeypatch, *, enabled: bool = True):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1" if enabled else "0")
    state = State()
    browser = NovaBrowser()
    ctx = SimpleNamespace(state=state, browser=browser, t0=0.0)
    app = FastAPI()
    api_routes.register_routes(app, ctx)
    return TestClient(app), state, browser


def test_first_post_queues_second_identical_post_is_duplicate(monkeypatch):
    client, state, _browser = _make_client(monkeypatch)

    first = client.post("/api/browser/task", json={"instruction": "search for the weather"})
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["status"] == "queued"
    assert first_body["queued"] is True
    assert first_body["duplicate"] is False
    assert state.get("browser_task") == "search for the weather"

    second = client.post("/api/browser/task", json={"instruction": "search for the weather"})
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["status"] == "duplicate"
    assert second_body["queued"] is False
    assert second_body["duplicate"] is True

    # State was only touched by the first (actually queued) request.
    assert state.get("browser_task") == "search for the weather"


def test_duplicate_does_not_re_update_state(monkeypatch):
    client, state, _browser = _make_client(monkeypatch)

    client.post("/api/browser/task", json={"instruction": "search for the weather"})
    state.update(browser_task="something else entirely")

    duplicate = client.post("/api/browser/task", json={"instruction": "search for the weather"})
    assert duplicate.json()["status"] == "duplicate"
    # A duplicate must never touch state — the sentinel from above survives.
    assert state.get("browser_task") == "something else entirely"


def test_disabled_reports_disabled_and_never_touches_state(monkeypatch):
    client, state, _browser = _make_client(monkeypatch, enabled=False)

    response = client.post("/api/browser/task", json={"instruction": "search for the weather"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "disabled"
    assert body["queued"] is False
    assert body["duplicate"] is False
    assert state.get("browser_task") == ""


def test_legacy_instruction_key_preserved_for_dashboard_js(monkeypatch):
    client, _state, _browser = _make_client(monkeypatch)

    response = client.post("/api/browser/task", json={"instruction": "search for the weather"})
    body = response.json()
    assert body["instruction"] == "search for the weather"
