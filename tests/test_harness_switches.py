"""Feature switches — ``reachy_nova/harness/switches.py`` (task t5).

One resolution point for the four env-selectable behaviours this round adds:
``NOVA_CHUNKED_PLAYBACK``, ``NOVA_LITE_REACTIONS``, ``NOVA_MEMORY`` (all
default ON) and ``NOVA_PERSONA_PATH`` (default unset/embedded persona). Same
shape as ``gate.resolve_policy`` — fail OPEN: a bad value never turns a
switch off and never raises, it resolves to on plus one named warning line.
This module does not wire the switches into ``app.py``; that is a later task.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from reachy_nova.harness import switches as switches_mod
from reachy_nova.harness.switches import Switches, describe, log, resolve


# --------------------------------------------------------------------------- #
# 1. Switches is a frozen dataclass, defaulting all three booleans to on.     #
# --------------------------------------------------------------------------- #


def test_switches_is_frozen() -> None:
    s = Switches()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.chunked_playback = False  # type: ignore[misc]


def test_switches_defaults_are_all_on_and_no_persona_path() -> None:
    s = Switches()
    assert s.chunked_playback is True
    assert s.lite_reactions is True
    assert s.memory is True
    assert s.persona_path is None


# --------------------------------------------------------------------------- #
# 2. resolve() reads each of the four env vars exactly once from the given   #
#    mapping (never os.environ directly unless the mapping is omitted).      #
# --------------------------------------------------------------------------- #


def test_resolve_returns_a_frozen_switches_instance() -> None:
    result = resolve({})
    assert isinstance(result, Switches)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.memory = False  # type: ignore[misc]


def test_unset_env_means_every_switch_on_and_no_persona_path() -> None:
    s = resolve({})
    assert s.chunked_playback is True
    assert s.lite_reactions is True
    assert s.memory is True
    assert s.persona_path is None


@pytest.mark.parametrize("raw", ["0", "false", "off", "no"])
def test_off_values_turn_a_switch_off_case_and_space_insensitive(raw: str) -> None:
    for variant in (raw, raw.upper(), f"  {raw}  ", f" {raw.upper()}\n"):
        assert resolve({"NOVA_CHUNKED_PLAYBACK": variant}).chunked_playback is False, variant
        assert resolve({"NOVA_LITE_REACTIONS": variant}).lite_reactions is False, variant
        assert resolve({"NOVA_MEMORY": variant}).memory is False, variant


@pytest.mark.parametrize("raw", ["1", "true", "on", "yes"])
def test_on_values_keep_a_switch_on_with_no_warning(raw: str, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        s = resolve({"NOVA_CHUNKED_PLAYBACK": raw, "NOVA_LITE_REACTIONS": raw, "NOVA_MEMORY": raw})
    assert s.chunked_playback is True
    assert s.lite_reactions is True
    assert s.memory is True
    assert not any("unrecognised" in r.getMessage() for r in caplog.records)


def test_empty_string_means_on_not_a_warning(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        s = resolve({"NOVA_MEMORY": ""})
    assert s.memory is True
    assert not any("unrecognised" in r.getMessage() for r in caplog.records)


def test_unrecognised_value_resolves_to_on_plus_one_named_warning(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        s = resolve({"NOVA_LITE_REACTIONS": "maybe"})
    assert s.lite_reactions is True, "fail OPEN: a bad value must never turn a switch off"
    lines = [r.getMessage() for r in caplog.records]
    assert any(
        "[SENSE stage=supervise source=nova event=switches]" in line
        and "unrecognised NOVA_LITE_REACTIONS=maybe" in line
        and "using on" in line
        for line in lines
    ), lines


def test_unrecognised_value_never_raises() -> None:
    # Fail open must hold even for pathological input, not just "maybe".
    resolve({"NOVA_MEMORY": "\x00binary\x00garbage"})
    resolve({"NOVA_CHUNKED_PLAYBACK": "TRUE-ish"})


def test_each_bad_switch_warns_independently(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        resolve(
            {
                "NOVA_CHUNKED_PLAYBACK": "nope",
                "NOVA_LITE_REACTIONS": "nope",
                "NOVA_MEMORY": "nope",
            }
        )
    lines = [r.getMessage() for r in caplog.records]
    for name in ("NOVA_CHUNKED_PLAYBACK", "NOVA_LITE_REACTIONS", "NOVA_MEMORY"):
        assert any(f"unrecognised {name}=nope" in line for line in lines), lines
    assert len(lines) == 3


def test_persona_path_is_none_unset_and_the_literal_value_when_set() -> None:
    assert resolve({}).persona_path is None
    assert resolve({"NOVA_PERSONA_PATH": ""}).persona_path is None
    assert resolve({"NOVA_PERSONA_PATH": "/etc/nova/persona.md"}).persona_path == "/etc/nova/persona.md"


def test_resolve_defaults_to_os_environ(monkeypatch) -> None:
    monkeypatch.setenv("NOVA_MEMORY", "off")
    monkeypatch.setenv("NOVA_PERSONA_PATH", "/tmp/persona.md")
    s = resolve()
    assert s.memory is False
    assert s.persona_path == "/tmp/persona.md"


# --------------------------------------------------------------------------- #
# 3. describe() renders one line naming every switch's resolved value.       #
# --------------------------------------------------------------------------- #


def test_describe_all_on_default_persona() -> None:
    line = describe(Switches())
    assert line == (
        "switches chunked_playback=on lite_reactions=on memory=on persona=default"
    )


def test_describe_names_each_switch_off() -> None:
    assert "chunked_playback=off" in describe(Switches(chunked_playback=False))
    assert "lite_reactions=off" in describe(Switches(lite_reactions=False))
    assert "memory=off" in describe(Switches(memory=False))


def test_describe_shows_the_persona_source_when_set() -> None:
    line = describe(Switches(persona_path="/opt/nova/persona.md"))
    assert "persona=file:/opt/nova/persona.md" in line


def test_describe_is_one_line() -> None:
    line = describe(Switches(chunked_playback=False, lite_reactions=False, memory=False, persona_path="/x.md"))
    assert "\n" not in line
    assert line == (
        "switches chunked_playback=off lite_reactions=off memory=off persona=file:/x.md"
    )


# --------------------------------------------------------------------------- #
# 4. log() emits describe()'s line exactly once via sensory_log.stage.       #
# --------------------------------------------------------------------------- #


def test_log_emits_the_describe_line_once(caplog) -> None:
    s = Switches(memory=False)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        log(s)
    lines = [r.getMessage() for r in caplog.records]
    expected = f"[SENSE stage=supervise source=nova event=switches] {describe(s)}"
    assert lines.count(expected) == 1, lines


def test_log_does_not_warn_on_a_clean_switches_instance(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        log(Switches())
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1
    assert "unrecognised" not in lines[0]


# --------------------------------------------------------------------------- #
# 5. Env var name constants match the spec exactly (app.py wiring depends).  #
# --------------------------------------------------------------------------- #


def test_env_var_name_constants() -> None:
    assert switches_mod.CHUNKED_PLAYBACK_ENV == "NOVA_CHUNKED_PLAYBACK"
    assert switches_mod.LITE_REACTIONS_ENV == "NOVA_LITE_REACTIONS"
    assert switches_mod.MEMORY_ENV == "NOVA_MEMORY"
    assert switches_mod.PERSONA_PATH_ENV == "NOVA_PERSONA_PATH"
