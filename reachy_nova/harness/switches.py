"""Feature switches: one fail-open env resolution point for the round's new
behaviours (t5).

Rolling back the whole release is possible (reinstall the previous PyPI
version), but a single misbehaving piece of this round — clicking chunk
boundaries, a slow Lite tier, a memory ledger that will not stop growing —
should be switchable off in ``.env`` without losing the rest. Same shape as
:func:`reachy_nova.harness.gate.resolve_policy`: fails OPEN. An unset or empty
value means on (today's new default); ``0``/``false``/``off``/``no``
(case-insensitive, stripped) means off; anything else — a typo, an unrelated
string — means on plus ONE named warning line, never a raised exception and
never a feature silently turned off by a mistake.

Four switches:

* ``NOVA_CHUNKED_PLAYBACK`` — chunked (vs. whole-utterance) speaker playback.
* ``NOVA_LITE_REACTIONS`` — the Nova 2 Lite reaction tier (vs. template-only).
* ``NOVA_MEMORY`` — the conversation ledger + compactor (vs. none).
* ``NOVA_PERSONA_PATH`` — not a boolean: an optional override path for the
  persona file; unset/empty means "use the built-in default resolution".

:func:`resolve` reads all four once from a given mapping (``os.environ`` by
default); :func:`describe` renders the resolved set as one line;
:func:`log` emits that line once via :func:`reachy_nova.sensory_log.stage`.
Wiring these into ``app.py`` — actually gating the ledger, the reactor and the
speaker's mode on the resolved values — happens in a later task; this module
only resolves and reports.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from .. import sensory_log

#: Env var names, exactly as the spec names them (app.py wiring depends on these).
CHUNKED_PLAYBACK_ENV = "NOVA_CHUNKED_PLAYBACK"
LITE_REACTIONS_ENV = "NOVA_LITE_REACTIONS"
MEMORY_ENV = "NOVA_MEMORY"
PERSONA_PATH_ENV = "NOVA_PERSONA_PATH"

#: Values (after stripping + casefolding) that resolve a switch to off/on.
_OFF_VALUES = frozenset({"0", "false", "off", "no"})
_ON_VALUES = frozenset({"1", "true", "on", "yes"})

_STAGE = "supervise"
_SOURCE = "nova"
_EVENT = "switches"


@dataclass(frozen=True)
class Switches:
    """The resolved set of feature switches for one process lifetime."""

    chunked_playback: bool = True
    lite_reactions: bool = True
    memory: bool = True
    persona_path: str | None = None


def _resolve_bool(env: Mapping[str, str], name: str) -> bool:
    """Resolve one boolean switch from *env[name]*, failing OPEN to on."""
    raw = env.get(name)
    if raw is None:
        return True
    candidate = raw.strip().lower()
    if candidate == "":
        return True
    if candidate in _OFF_VALUES:
        return False
    if candidate in _ON_VALUES:
        return True
    sensory_log.stage(_STAGE, _SOURCE, _EVENT, f"unrecognised {name}={raw} — using on")
    return True


def resolve(env: Mapping[str, str] | None = None) -> Switches:
    """Resolve all four switches once from *env* (default ``os.environ``).

    Fails OPEN: an unset or empty value means on; a value this module does
    not recognise ALSO means on, plus one named warning line — a typo must
    never silently disable chunked playback, the Lite tier or memory.
    """
    source: Mapping[str, str] = env if env is not None else os.environ
    persona_path = source.get(PERSONA_PATH_ENV) or None
    return Switches(
        chunked_playback=_resolve_bool(source, CHUNKED_PLAYBACK_ENV),
        lite_reactions=_resolve_bool(source, LITE_REACTIONS_ENV),
        memory=_resolve_bool(source, MEMORY_ENV),
        persona_path=persona_path,
    )


def _on_off(value: bool) -> str:
    return "on" if value else "off"


def describe(switches: Switches) -> str:
    """Render *switches* as one grep-able line naming every resolved value."""
    persona = f"file:{switches.persona_path}" if switches.persona_path else "default"
    return (
        f"switches chunked_playback={_on_off(switches.chunked_playback)} "
        f"lite_reactions={_on_off(switches.lite_reactions)} "
        f"memory={_on_off(switches.memory)} persona={persona}"
    )


def log(switches: Switches) -> None:
    """Emit :func:`describe`'s line once via ``sensory_log.stage``."""
    sensory_log.stage(_STAGE, _SOURCE, _EVENT, describe(switches))
