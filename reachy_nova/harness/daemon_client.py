"""The ONE shared daemon HTTP client (task t10).

Before this module, ``speaking.py`` carried its own standalone urllib
multipart transport — the only daemon HTTP client in the harness. That was
fine while speech playback was the only thing talking to the daemon; it stops
being fine once a second concern (voice volume, task t10) needs the same
``http://localhost:8000``-shaped conversation. This module is the shared
seam both live behind: :class:`DaemonClient` wraps GET/POST against the
daemon's base URL, and ``speaking.py``'s ``default_poster``/``default_stopper``
now delegate to it instead of rolling their own request.

Endpoints used here:

- ``GET  /api/volume/current``  -> ``{"volume": N, ...}``
- ``POST /api/volume/set``      body ``{"volume": 0..100}`` -> ``{"volume": N, ...}``
  (the daemon plays a short confirmation sound on every set — accepted, not
  suppressible, which is exactly why :func:`restore_volume` skips the call
  when the persisted level already matches the daemon's).
- ``POST /api/media/sounds/upload`` — multipart/form-data upload of a WAV.
- ``POST /api/media/play_sound``    body ``{"file": "<path>"}``.
- ``POST /api/media/stop_sound``    body ``{}`` (barge-in cut).

stdlib + injectable ``post``/``get`` callables only — the test seam mirrors
``speaking.py``'s existing ``poster``/``stopper`` pattern, and no network is
ever touched in tests.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

from reachy_nova import sensory_log

DEFAULT_BASE_URL = "http://localhost:8000"
BASE_URL_ENV = "NOVA_DAEMON_URL"

_VOLUME_GET_PATH = "/api/volume/current"
_VOLUME_SET_PATH = "/api/volume/set"
_UPLOAD_PATH = "/api/media/sounds/upload"
_PLAY_PATH = "/api/media/play_sound"
_STOP_PATH = "/api/media/stop_sound"

_HTTP_TIMEOUT_S = 10.0

STAGE = "act"
SOURCE = "nova"


# --------------------------------------------------------------------------- #
# Default transport — stdlib urllib (no third-party deps).                    #
# --------------------------------------------------------------------------- #


def _default_get(url: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url, method="GET", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read())


def _default_post(url: str, data: bytes, content_type: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read())


def _multipart_encode(filename: str, wav_bytes: bytes) -> tuple[bytes, str]:
    """Encode a single-file multipart/form-data body; return (body, content-type)."""
    boundary = "----ReachyNovaSpeakBoundary"
    ctype = f"multipart/form-data; boundary={boundary}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n"
        f"\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + wav_bytes + tail, ctype


class DaemonClient:
    """A tiny HTTP client for the daemon's volume + media endpoints.

    ``post``/``get`` are injectable so tests never touch the network:
    ``post(url, data, content_type, timeout) -> dict`` and
    ``get(url, timeout) -> dict``, mirroring ``speaking.py``'s existing
    poster/stopper seam.
    """

    def __init__(
        self,
        base_url: str | None = None,
        post: Callable[[str, bytes, str, float], dict] | None = None,
        get: Callable[[str, float], dict] | None = None,
        timeout: float = _HTTP_TIMEOUT_S,
    ) -> None:
        base = base_url or os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
        self.base_url = base.rstrip("/")
        self._post = post or _default_post
        self._get = get or _default_get
        self.timeout = timeout

    # -- volume -------------------------------------------------------------

    def get_volume(self) -> int:
        """``GET /api/volume/current`` -> the current volume, 0..100."""
        resp = self._get(f"{self.base_url}{_VOLUME_GET_PATH}", self.timeout)
        return int(resp["volume"])

    def set_volume(self, volume: int) -> int:
        """``POST /api/volume/set`` -> the volume the daemon actually applied.

        Plays a short confirmation sound on the daemon side — accepted, not
        suppressible; callers that want to avoid it (see
        :func:`restore_volume`) must not call this when the level already
        matches.
        """
        body = json.dumps({"volume": int(volume)}).encode("utf-8")
        resp = self._post(
            f"{self.base_url}{_VOLUME_SET_PATH}", body, "application/json", self.timeout
        )
        return int(resp.get("volume", volume))

    # -- speech playback (moved in from speaking.py's standalone transport) -

    def upload_sound(self, wav_bytes: bytes, filename: str) -> str:
        """Upload a complete WAV; return the daemon's saved path."""
        body, ctype = _multipart_encode(filename, wav_bytes)
        resp = self._post(f"{self.base_url}{_UPLOAD_PATH}", body, ctype, self.timeout)
        return resp.get("path", filename)

    def play_sound(self, path: str) -> None:
        body = json.dumps({"file": path}).encode("utf-8")
        self._post(f"{self.base_url}{_PLAY_PATH}", body, "application/json", self.timeout)

    def stop_sound(self) -> None:
        self._post(f"{self.base_url}{_STOP_PATH}", b"{}", "application/json", self.timeout)


# --------------------------------------------------------------------------- #
# Volume persistence — restore on harness start                               #
# --------------------------------------------------------------------------- #


def restore_volume(path: Path, client: DaemonClient) -> int | None:
    """On harness start: re-apply a persisted level if the daemon disagrees.

    Reads *path* (``statedir.volume_state_path()``); if it holds a valid
    ``{"volume": N}`` AND the daemon's current level differs, calls
    ``client.set_volume(N)`` (the one re-apply) and returns ``N``. If the
    daemon already reports the same level, the set call is skipped entirely
    — the daemon's confirmation sound is not suppressible, so matching means
    "do nothing". Returns ``None`` when there is nothing to restore (no file,
    unreadable, or already in sync) — never raises; the caller wraps this in
    a component-absent-style try/except per the module's own convention.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        persisted = int(data["volume"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    current = client.get_volume()
    if current == persisted:
        return None
    client.set_volume(persisted)
    sensory_log.stage(
        STAGE, SOURCE, "volume", f"restored old={current} new={persisted} (persisted)"
    )
    return persisted
