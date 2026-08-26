"""State-dir resolution + peer-liveness checks, mirroring reachy-mini-cli.

The wire contract is filesystem paths under one state dir:

- ``$REACHY_STATE_DIR`` -> ``$XDG_STATE_HOME/reachy`` -> ``~/.local/state/reachy``
- intents spool:  ``<state>/behavior/intents/{commands,results}/``
- reload spool:   ``<state>/behavior/reload/{commands,results}/``
- rules overlay:  ``<state>/behavior/rules.toml``
- engine heartbeat: ``<state>/behavior/state.json`` ``updated`` field
- audio tee:      ``<state>/audio_tee.sock`` (or ``$REACHY_AUDIO_TEE_SOCKET``)
- network change: ``<state>/network-change`` (JSON ``{ssid, ip, ts}``, written
  atomically by the NetworkManager dispatcher hook — see
  :mod:`reachy_nova.harness.network`)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Engine heartbeat contract (reachy-mini-cli reachy/behavior/liveness.py)
ENGINE_HEARTBEAT_TTL_S = 2.0
ENGINE_HEARTBEAT_SKEW_S = 1.0


def state_dir() -> Path:
    explicit = os.environ.get("REACHY_STATE_DIR")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "reachy"
    return Path.home() / ".local" / "state" / "reachy"


def behavior_dir() -> Path:
    return state_dir() / "behavior"


def intents_commands_dir() -> Path:
    return behavior_dir() / "intents" / "commands"


def intents_results_dir() -> Path:
    return behavior_dir() / "intents" / "results"


def reload_commands_dir() -> Path:
    return behavior_dir() / "reload" / "commands"


def reload_results_dir() -> Path:
    return behavior_dir() / "reload" / "results"


def rules_overlay_path() -> Path:
    return behavior_dir() / "rules.toml"


def state_json_path() -> Path:
    return behavior_dir() / "state.json"


def audio_tee_socket() -> Path:
    explicit = os.environ.get("REACHY_AUDIO_TEE_SOCKET")
    if explicit:
        return Path(explicit)
    return state_dir() / "audio_tee.sock"


def network_change_path() -> Path:
    """The dispatcher hook's drop file — see :mod:`reachy_nova.harness.network`."""
    return state_dir() / "network-change"


def volume_state_path() -> Path:
    """Persisted last-set voice volume — ``<state>/nova-volume.json``."""
    return state_dir() / "nova-volume.json"


def quiet_state_path() -> Path:
    """Persisted timed-quiet deadline — ``<state>/nova-quiet.json``.

    See :mod:`reachy_nova.harness.quiet`: a deadline, so a restart inside a
    quiet window comes back quiet instead of reintroducing itself out loud.
    """
    return state_dir() / "nova-quiet.json"


def harness_pid_path() -> Path:
    return state_dir() / "nova-harness.pid"


def embody_pid_path() -> Path:
    return state_dir() / "embody.pid"


def engine_is_live(now: float | None = None) -> bool:
    """Is the behavior engine ticking? Reads the self-expiring heartbeat.

    ``state.json``'s ``updated`` must be within TTL in the past and no more
    than the skew allowance in the future. Never trust PID files or systemctl
    for this — the heartbeat survives every launch path and SIGKILL.
    """
    try:
        payload = json.loads(state_json_path().read_text())
        updated = float(payload["updated"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    # The engine stamps ``updated`` from time.monotonic() (verified on-device:
    # state.json carries ~200023 while epoch is ~1.79e9) — compare on the same
    # clock or a live engine reads as permanently absent.
    now = time.monotonic() if now is None else now
    age = now - updated
    return -ENGINE_HEARTBEAT_SKEW_S <= age <= ENGINE_HEARTBEAT_TTL_S


def _pid_argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def embody_is_live() -> bool:
    """Is the ``agent embody`` layer running? PID file + exact argv identity.

    Identity is verified with exact argv tokens (``reachy`` and ``embody`` as
    separate elements), never a substring scan — PID reuse otherwise makes us
    treat a stranger as the peer.
    """
    try:
        pid = int(embody_pid_path().read_text().strip())
    except (OSError, ValueError):
        return False
    argv = _pid_argv(pid)
    return "reachy" in argv and "embody" in argv
