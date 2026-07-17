"""Tests for reachy_nova.skill_forge.SkillForge.

Covers the dispatch -> stage -> validate -> forge/* event lifecycle. The HTTP
layer is always mocked via the injected ``transport`` callable (or, for a
couple of tests that exercise the real default transport wiring, via
``urllib.request.urlopen``) — no test ever talks to a real network.
"""

import json
import logging
import threading
import time
import urllib.error

import pytest

from reachy_nova import skill_forge
from reachy_nova.skill_forge import SkillForge

GOOD_REPLY_CONTENT = (
    "```SKILL.md\n"
    "---\n"
    "name: wave-hello\n"
    "description: Wave at whoever is in view.\n"
    "---\n"
    "\n"
    "# Wave Hello\n"
    "\n"
    "Waves a greeting.\n"
    "```\n"
    "\n"
    "```executor.py\n"
    "def execute(params, ctx):\n"
    "    ctx.gesture('wave')\n"
    "    return '[waved]'\n"
    "```\n"
)


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class _RecordingPublisher:
    """Stub publish callback that records every (event_type, payload) call."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def __call__(self, event_type: str, payload: dict) -> None:
        with self._lock:
            self.calls.append((event_type, payload))

    def wait_for(self, event_type: str, timeout: float = 5.0) -> tuple[str, dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for call in self.calls:
                    if call[0] == event_type:
                        return call
            time.sleep(0.01)
        raise AssertionError(f"event {event_type!r} never published; got {self.calls}")


def _ok_validator(skill_dir):
    return True, []


def _rejecting_validator(reasons):
    def _validator(skill_dir):
        return False, list(reasons)

    return _validator


@pytest.fixture(autouse=True)
def _clean_forge_env(monkeypatch):
    """Every test controls FORGE_* env vars explicitly."""
    monkeypatch.delenv("FORGE_BASE_URL", raising=False)
    monkeypatch.delenv("FORGE_MODEL", raising=False)
    monkeypatch.delenv("FORGE_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# endpoint not configured
# ---------------------------------------------------------------------------


def test_dispatch_without_endpoint_rejects_immediately(tmp_path):
    """No FORGE_BASE_URL -> forge/rejected reason 'endpoint not configured'."""
    publisher = _RecordingPublisher()
    calls = []
    forge = SkillForge(
        publish=publisher,
        validator=_ok_validator,
        staging_root=tmp_path,
        transport=lambda *a, **kw: calls.append((a, kw)) or _chat_response(GOOD_REPLY_CONTENT),
    )

    thread = forge.dispatch("wave at people", {"who": "ori"})
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert calls == []  # transport never invoked
    event_type, payload = publisher.wait_for("forge/rejected")
    assert "endpoint not configured" in payload.get("reason", "")
    assert list(tmp_path.iterdir()) == []  # nothing staged


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_dispatch_success_stages_skill_and_emits_forge_staged(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()
    seen_calls = []

    def transport(url, payload, headers, timeout):
        seen_calls.append((url, payload, headers, timeout))
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(
        publish=publisher,
        validator=_ok_validator,
        staging_root=tmp_path,
        transport=transport,
    )

    thread = forge.dispatch("wave at people", {"who": "ori"})
    thread.join(timeout=5)

    event_type, payload = publisher.wait_for("forge/staged")
    assert payload["name"] == "wave-hello"
    staged_dir = tmp_path / "wave-hello"
    assert staged_dir.is_dir()
    assert (staged_dir / "SKILL.md").read_text().startswith("---")
    assert "def execute(params, ctx):" in (staged_dir / "executor.py").read_text()
    assert payload["path"] == str(staged_dir)

    # dispatch() reached the injected transport with the right shape.
    assert len(seen_calls) == 1
    url, req_payload, headers, timeout = seen_calls[0]
    assert url.startswith("http://forge.local/v1")
    assert req_payload["model"] == "qwen3"  # FORGE_MODEL default
    assert any("wave at people" in m.get("content", "") for m in req_payload["messages"])
    assert timeout == skill_forge.DEFAULT_TIMEOUT


def test_dispatch_uses_forge_model_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    monkeypatch.setenv("FORGE_MODEL", "qwen3-coder-30b")
    publisher = _RecordingPublisher()
    seen_calls = []

    def transport(url, payload, headers, timeout):
        seen_calls.append(payload)
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=transport)
    forge.dispatch("wave", {}).join(timeout=5)

    assert seen_calls[0]["model"] == "qwen3-coder-30b"


def test_dispatch_sends_bearer_auth_when_api_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    monkeypatch.setenv("FORGE_API_KEY", "secret-token")
    publisher = _RecordingPublisher()
    seen_headers = []

    def transport(url, payload, headers, timeout):
        seen_headers.append(headers)
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=transport)
    forge.dispatch("wave", {}).join(timeout=5)

    assert seen_headers[0].get("Authorization") == "Bearer secret-token"


def test_dispatch_omits_auth_header_without_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()
    seen_headers = []

    def transport(url, payload, headers, timeout):
        seen_headers.append(headers)
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=transport)
    forge.dispatch("wave", {}).join(timeout=5)

    assert "Authorization" not in seen_headers[0]


# ---------------------------------------------------------------------------
# dispatch() never blocks the caller
# ---------------------------------------------------------------------------


def test_dispatch_returns_immediately_without_waiting_for_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()
    release = threading.Event()

    def slow_transport(url, payload, headers, timeout):
        release.wait(timeout=5)
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=slow_transport)

    start = time.time()
    thread = forge.dispatch("wave", {})
    elapsed = time.time() - start

    assert elapsed < 1.0  # returned long before slow_transport unblocks
    assert isinstance(thread, threading.Thread)
    assert publisher.calls == []  # nothing published yet — still in flight

    release.set()
    thread.join(timeout=5)
    publisher.wait_for("forge/staged")


# ---------------------------------------------------------------------------
# failure paths: unreachable / timeout / non-200 / garbage reply
# ---------------------------------------------------------------------------


def test_dispatch_unreachable_endpoint_rejects_and_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    def failing_transport(url, payload, headers, timeout):
        raise ConnectionRefusedError("connection refused")

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=failing_transport)

    with caplog.at_level(logging.WARNING):
        forge.dispatch("wave", {}).join(timeout=5)

    event_type, payload = publisher.wait_for("forge/rejected")
    assert "unreachable" in payload["reason"] or "connection refused" in payload["reason"]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert list(tmp_path.iterdir()) == []


def test_dispatch_timeout_rejects_with_timeout_reason(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    def timing_out_transport(url, payload, headers, timeout):
        raise TimeoutError("timed out")

    forge = SkillForge(
        publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=timing_out_transport
    )

    with caplog.at_level(logging.WARNING):
        forge.dispatch("wave", {}).join(timeout=5)

    event_type, payload = publisher.wait_for("forge/rejected")
    assert "timed out" in payload["reason"] or "timeout" in payload["reason"].lower()
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_dispatch_non_200_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    def http_error_transport(url, payload, headers, timeout):
        raise urllib.error.HTTPError(url, 500, "Internal Server Error", hdrs=None, fp=None)

    forge = SkillForge(
        publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=http_error_transport
    )

    forge.dispatch("wave", {}).join(timeout=5)
    publisher.wait_for("forge/rejected")


def test_dispatch_garbage_reply_shape_rejects(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    def garbage_transport(url, payload, headers, timeout):
        return {"nonsense": True}

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=garbage_transport)

    with caplog.at_level(logging.WARNING):
        forge.dispatch("wave", {}).join(timeout=5)

    event_type, payload = publisher.wait_for("forge/rejected")
    assert "unparseable" in payload["reason"].lower()
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_dispatch_reply_with_no_fences_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    def no_fence_transport(url, payload, headers, timeout):
        return _chat_response("Sure! Here is your skill, no code blocks though.")

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=no_fence_transport)

    forge.dispatch("wave", {}).join(timeout=5)
    event_type, payload = publisher.wait_for("forge/rejected")
    assert payload["reason"]
    assert list(tmp_path.iterdir()) == []


def test_dispatch_reply_with_empty_fence_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()
    content = "```SKILL.md\n```\n\n```executor.py\ndef execute(params, ctx):\n    pass\n```\n"

    def transport(url, payload, headers, timeout):
        return _chat_response(content)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=transport)

    forge.dispatch("wave", {}).join(timeout=5)
    event_type, payload = publisher.wait_for("forge/rejected")
    assert "SKILL.md" in payload["reason"]


def test_dispatch_missing_name_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()
    content = (
        "```SKILL.md\n---\ndescription: no name here\n---\nbody\n```\n\n"
        "```executor.py\ndef execute(params, ctx):\n    pass\n```\n"
    )

    def transport(url, payload, headers, timeout):
        return _chat_response(content)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=transport)

    forge.dispatch("wave", {}).join(timeout=5)
    event_type, payload = publisher.wait_for("forge/rejected")
    assert "name" in payload["reason"].lower()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# path traversal / sanitization
# ---------------------------------------------------------------------------


def test_dispatch_sanitizes_path_traversal_name(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()
    content = (
        "```SKILL.md\n---\nname: ../../etc/evil Name!!\ndescription: bad\n---\nbody\n```\n\n"
        "```executor.py\ndef execute(params, ctx):\n    pass\n```\n"
    )

    def transport(url, payload, headers, timeout):
        return _chat_response(content)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=transport)

    forge.dispatch("wave", {}).join(timeout=5)
    event_type, payload = publisher.wait_for("forge/staged")

    staged_path = payload["path"]
    from pathlib import Path

    resolved = Path(staged_path).resolve()
    assert resolved.parent == tmp_path.resolve()
    assert ".." not in resolved.name
    assert "/" not in payload["name"]
    # only [a-z0-9-] survive sanitization
    import re

    assert re.fullmatch(r"[a-z0-9-]+", payload["name"])


def test_dispatch_rejects_when_name_sanitizes_to_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()
    content = (
        "```SKILL.md\n---\nname: !!!///___\ndescription: bad\n---\nbody\n```\n\n"
        "```executor.py\ndef execute(params, ctx):\n    pass\n```\n"
    )

    def transport(url, payload, headers, timeout):
        return _chat_response(content)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=transport)

    forge.dispatch("wave", {}).join(timeout=5)
    event_type, payload = publisher.wait_for("forge/rejected")
    assert "name" in payload["reason"].lower()


# ---------------------------------------------------------------------------
# validator wiring
# ---------------------------------------------------------------------------


def test_validator_rejection_moves_folder_to_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    def transport(url, payload, headers, timeout):
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(
        publish=publisher,
        validator=_rejecting_validator(["imports os", "calls subprocess"]),
        staging_root=tmp_path,
        transport=transport,
    )

    forge.dispatch("wave", {}).join(timeout=5)
    event_type, payload = publisher.wait_for("forge/rejected")
    assert "imports os" in payload["reason"]
    assert "calls subprocess" in payload["reason"]

    assert not (tmp_path / "wave-hello").exists()
    rejected_dir = tmp_path / ".rejected" / "wave-hello"
    assert rejected_dir.is_dir()
    assert (rejected_dir / "SKILL.md").exists()
    assert (rejected_dir / "executor.py").exists()


def test_validator_receiving_the_staged_dir(tmp_path, monkeypatch):
    """The validator is called with a Path pointing at the actually-staged dir."""
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()
    received = []

    def spy_validator(skill_dir):
        received.append(skill_dir)
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "executor.py").exists()
        return True, []

    def transport(url, payload, headers, timeout):
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=publisher, validator=spy_validator, staging_root=tmp_path, transport=transport)
    forge.dispatch("wave", {}).join(timeout=5)
    publisher.wait_for("forge/staged")

    assert len(received) == 1
    assert received[0] == tmp_path / "wave-hello"


def test_validator_raising_is_treated_as_rejection(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    def exploding_validator(skill_dir):
        raise RuntimeError("boom")

    def transport(url, payload, headers, timeout):
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=publisher, validator=exploding_validator, staging_root=tmp_path, transport=transport)

    with caplog.at_level(logging.WARNING):
        forge.dispatch("wave", {}).join(timeout=5)

    event_type, payload = publisher.wait_for("forge/rejected")
    assert "boom" in payload["reason"]


def test_no_validator_provided_and_no_forge_validator_module_fails_closed(tmp_path, monkeypatch):
    """validator=None with no injected stub -> lazy-imports reachy_nova.forge_validator.

    That sibling module is built by a different task and does not exist in
    this worktree, so the ImportError must fail CLOSED: forge/rejected with
    reason 'validator unavailable' — never forge/staged.
    """
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    def transport(url, payload, headers, timeout):
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=publisher, validator=None, staging_root=tmp_path, transport=transport)

    forge.dispatch("wave", {}).join(timeout=5)
    event_type, payload = publisher.wait_for("forge/rejected")
    assert payload["reason"] == "validator unavailable"
    assert not any(call[0] == "forge/staged" for call in publisher.calls)


# ---------------------------------------------------------------------------
# mark_activated
# ---------------------------------------------------------------------------


def test_mark_activated_emits_forge_activated_without_touching_disk(tmp_path):
    publisher = _RecordingPublisher()
    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=lambda *a, **k: {})

    forge.mark_activated("wave-hello")

    event_type, payload = publisher.wait_for("forge/activated", timeout=1)
    assert payload["name"] == "wave-hello"
    assert list(tmp_path.iterdir()) == []  # mark_activated never writes/moves files


def test_mark_activated_does_not_auto_fire_on_staged(tmp_path, monkeypatch):
    """forge/activated is ONLY emitted by an explicit mark_activated() call."""
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    def transport(url, payload, headers, timeout):
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=transport)
    forge.dispatch("wave", {}).join(timeout=5)
    publisher.wait_for("forge/staged")

    assert not any(call[0] == "forge/activated" for call in publisher.calls)


# ---------------------------------------------------------------------------
# publish callback never crashes the worker thread
# ---------------------------------------------------------------------------


def test_broken_publish_callback_never_crashes_dispatch_thread(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")

    def exploding_publish(event_type, payload):
        raise RuntimeError("publish is down")

    def transport(url, payload, headers, timeout):
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=exploding_publish, validator=_ok_validator, staging_root=tmp_path, transport=transport)

    with caplog.at_level(logging.WARNING):
        thread = forge.dispatch("wave", {})
        thread.join(timeout=5)

    assert not thread.is_alive()
    # the file still got staged even though the publish callback blew up
    assert (tmp_path / "wave-hello").is_dir()


def test_unexpected_internal_exception_still_ends_in_rejected(tmp_path, monkeypatch):
    """Any surprise bug in the dispatch body must still resolve to forge/rejected,
    never a silently-dead worker thread with no event at all."""
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()

    # AttributeError here stands in for "unexpected internal bug" — any
    # non-network exception raised deep in the dispatch body.
    def buggy_transport(url, payload, headers, timeout):
        raise AttributeError("simulated internal bug")

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=buggy_transport)
    forge.dispatch("wave", {}).join(timeout=5)
    publisher.wait_for("forge/rejected")


# ---------------------------------------------------------------------------
# improve / context plumbing
# ---------------------------------------------------------------------------


def test_dispatch_includes_improve_and_context_in_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE_URL", "http://forge.local/v1")
    publisher = _RecordingPublisher()
    seen_payloads = []

    def transport(url, payload, headers, timeout):
        seen_payloads.append(payload)
        return _chat_response(GOOD_REPLY_CONTENT)

    forge = SkillForge(publish=publisher, validator=_ok_validator, staging_root=tmp_path, transport=transport)
    forge.dispatch(
        "improve the wave skill",
        {"last_feedback": "too slow"},
        improve="def execute(params, ctx):\n    ctx.gesture('wave')\n",
    ).join(timeout=5)

    messages = seen_payloads[0]["messages"]
    user_content = "\n".join(m["content"] for m in messages if m["role"] == "user")
    assert "improve the wave skill" in user_content
    assert "too slow" in user_content
    assert "ctx.gesture('wave')" in user_content


# ---------------------------------------------------------------------------
# default staging root
# ---------------------------------------------------------------------------


def test_default_staging_root_is_under_home_reachy_nova(monkeypatch):
    from pathlib import Path

    monkeypatch.delenv("FORGE_BASE_URL", raising=False)
    forge = SkillForge(publish=lambda *a, **k: None, validator=_ok_validator)
    assert forge._staging_root == skill_forge.DEFAULT_STAGING_ROOT
    assert forge._staging_root == Path.home() / ".reachy_nova" / "skills-forged"


# ---------------------------------------------------------------------------
# real default transport wiring (mock urllib, not the network)
# ---------------------------------------------------------------------------


def test_default_transport_posts_openai_shaped_request(monkeypatch):
    import urllib.request

    captured = {}

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        captured["data"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(json.dumps(_chat_response(GOOD_REPLY_CONTENT)).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = skill_forge._default_transport(
        "http://forge.local/v1/chat/completions",
        {"model": "qwen3", "messages": [{"role": "user", "content": "hi"}]},
        {"Content-Type": "application/json", "Authorization": "Bearer tok"},
        5.0,
    )

    assert captured["method"] == "POST"
    assert captured["timeout"] == 5.0
    assert captured["data"]["model"] == "qwen3"
    assert result["choices"][0]["message"]["content"] == GOOD_REPLY_CONTENT


def test_default_transport_propagates_urlopen_errors(monkeypatch):
    import urllib.request

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError):
        skill_forge._default_transport(
            "http://forge.local/v1/chat/completions",
            {"model": "qwen3", "messages": []},
            {},
            5.0,
        )
