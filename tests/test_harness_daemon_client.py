"""The shared daemon HTTP client (task t10).

``DaemonClient`` is the ONE daemon HTTP client the harness has: speaking.py's
upload/play/stop calls and the volume tools (tools.py) both go through it.
Every test here injects fake ``post``/``get`` callables — no network is ever
touched — mirroring the seam ``speaking.py`` already used for its poster.
"""

from __future__ import annotations

import json

import pytest

from reachy_nova.harness import statedir
from reachy_nova.harness.daemon_client import (
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    DaemonClient,
    restore_volume,
)


class FakeTransport:
    """Records every call; answers with canned/queued responses."""

    def __init__(self):
        self.gets: list[str] = []
        self.posts: list[tuple[str, bytes, str]] = []
        self.deletes: list[str] = []
        self.get_response: dict = {"volume": 42}
        self.post_responses: dict[str, dict] = {}
        self.delete_responses: dict[str, dict] = {}
        self.delete_error: Exception | None = None

    def get(self, url: str, timeout: float) -> dict:
        self.gets.append(url)
        return self.get_response

    def post(self, url: str, data: bytes, content_type: str, timeout: float) -> dict:
        self.posts.append((url, data, content_type))
        for suffix, resp in self.post_responses.items():
            if url.endswith(suffix):
                return resp
        return {}

    def delete(self, url: str, timeout: float) -> dict:
        self.deletes.append(url)
        if self.delete_error is not None:
            raise self.delete_error
        for suffix, resp in self.delete_responses.items():
            if url.endswith(suffix):
                return resp
        return {}


@pytest.fixture
def transport():
    return FakeTransport()


@pytest.fixture
def client(transport):
    return DaemonClient(
        base_url="http://daemon.local:8000",
        post=transport.post,
        get=transport.get,
        delete=transport.delete,
    )


# --------------------------------------------------------------------------- #
# base_url resolution                                                         #
# --------------------------------------------------------------------------- #


def test_default_base_url_from_env(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://elsewhere:9000")
    client = DaemonClient(post=lambda *a: {}, get=lambda *a: {})
    assert client.base_url == "http://elsewhere:9000"


def test_default_base_url_fallback(monkeypatch):
    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    client = DaemonClient(post=lambda *a: {}, get=lambda *a: {})
    assert client.base_url == DEFAULT_BASE_URL


def test_base_url_trailing_slash_stripped():
    client = DaemonClient(base_url="http://x:8000/", post=lambda *a: {}, get=lambda *a: {})
    assert client.base_url == "http://x:8000"


# --------------------------------------------------------------------------- #
# volume                                                                      #
# --------------------------------------------------------------------------- #


def test_get_volume_reads_current(client, transport):
    transport.get_response = {"volume": 55, "muted": False}
    assert client.get_volume() == 55
    assert transport.gets == ["http://daemon.local:8000/api/volume/current"]


def test_set_volume_posts_json_and_returns_applied(client, transport):
    transport.post_responses["/api/volume/set"] = {"volume": 70}
    result = client.set_volume(70)
    assert result == 70
    (url, data, ctype) = transport.posts[0]
    assert url == "http://daemon.local:8000/api/volume/set"
    assert json.loads(data) == {"volume": 70}
    assert ctype == "application/json"


def test_set_volume_falls_back_to_requested_when_response_omits_it(client, transport):
    transport.post_responses["/api/volume/set"] = {}
    assert client.set_volume(33) == 33


# --------------------------------------------------------------------------- #
# speech playback (moved in from speaking.py)                                 #
# --------------------------------------------------------------------------- #


def test_upload_sound_posts_multipart_and_returns_saved_path(client, transport):
    transport.post_responses["/api/media/sounds/upload"] = {"path": "saved.wav"}
    path = client.upload_sound(b"RIFF....", "tts_synth.wav")
    assert path == "saved.wav"
    (url, data, ctype) = transport.posts[0]
    assert url == "http://daemon.local:8000/api/media/sounds/upload"
    assert ctype.startswith("multipart/form-data")
    assert b"RIFF...." in data


def test_upload_sound_falls_back_to_filename(client, transport):
    transport.post_responses["/api/media/sounds/upload"] = {}
    assert client.upload_sound(b"x", "fallback.wav") == "fallback.wav"


def test_play_sound_posts_file_path(client, transport):
    client.play_sound("saved.wav")
    (url, data, ctype) = transport.posts[0]
    assert url == "http://daemon.local:8000/api/media/play_sound"
    assert json.loads(data) == {"file": "saved.wav"}


def test_stop_sound_posts_empty_body(client, transport):
    client.stop_sound()
    (url, data, ctype) = transport.posts[0]
    assert url == "http://daemon.local:8000/api/media/stop_sound"
    assert json.loads(data) == {}


# --------------------------------------------------------------------------- #
# sounds — list_sounds / delete_sound (task t1)                               #
# --------------------------------------------------------------------------- #


def test_list_sounds_returns_list_when_response_is_a_bare_list(client, transport):
    transport.get_response = ["a.wav", "b.wav"]
    assert client.list_sounds() == ["a.wav", "b.wav"]
    assert transport.gets == ["http://daemon.local:8000/api/media/sounds"]


def test_list_sounds_extracts_list_field_from_object_response(client, transport):
    transport.get_response = {"files": ["a.wav", "b.wav"], "count": 2}
    assert client.list_sounds() == ["a.wav", "b.wav"]


def test_list_sounds_returns_empty_list_when_object_has_no_list_field(client, transport):
    transport.get_response = {"count": 0}
    assert client.list_sounds() == []


def test_delete_sound_issues_delete_and_returns_parsed_json(client, transport):
    transport.delete_responses["/api/media/sounds/x.wav"] = {"deleted": "x.wav"}
    result = client.delete_sound("x.wav")
    assert result == {"deleted": "x.wav"}
    assert transport.deletes == ["http://daemon.local:8000/api/media/sounds/x.wav"]


def test_delete_sound_url_encodes_filename():
    transport = FakeTransport()
    transport.delete_responses["/api/media/sounds/nova%20cue.wav"] = {"deleted": "nova cue.wav"}
    client = DaemonClient(
        base_url="http://daemon.local:8000",
        post=transport.post,
        get=transport.get,
        delete=transport.delete,
    )
    client.delete_sound("nova cue.wav")
    assert transport.deletes == ["http://daemon.local:8000/api/media/sounds/nova%20cue.wav"]


def test_delete_sound_http_failure_raises(client, transport):
    transport.delete_error = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        client.delete_sound("x.wav")
    assert transport.deletes == ["http://daemon.local:8000/api/media/sounds/x.wav"]


# --------------------------------------------------------------------------- #
# restore_volume                                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    return tmp_path


def test_restore_volume_reapplies_when_daemon_differs(state_dir, client, transport):
    path = statedir.volume_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"volume": 65}), encoding="utf-8")
    transport.get_response = {"volume": 40}
    transport.post_responses["/api/volume/set"] = {"volume": 65}

    result = restore_volume(path, client)

    assert result == 65
    assert any(url.endswith("/api/volume/set") for url, _, _ in transport.posts)


def test_restore_volume_skips_set_when_already_equal(state_dir, client, transport):
    path = statedir.volume_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"volume": 40}), encoding="utf-8")
    transport.get_response = {"volume": 40}

    result = restore_volume(path, client)

    assert result is None
    assert transport.posts == []  # no set call, no confirmation sound


def test_restore_volume_returns_none_when_no_persisted_file(state_dir, client, transport):
    path = statedir.volume_state_path()
    assert not path.exists()
    assert restore_volume(path, client) is None
    assert transport.gets == []
    assert transport.posts == []


def test_restore_volume_returns_none_on_corrupt_file(state_dir, client, transport):
    path = statedir.volume_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert restore_volume(path, client) is None
    assert transport.posts == []
