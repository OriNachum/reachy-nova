"""Persona file and loader (task t3) — ``reachy_nova/harness/persona.py``.

Nova's character is text, not code: ``config/persona/nova.md`` is read at
startup and becomes the system prompt Sonic carries, so editing that file and
restarting changes who Nova is with no code change at all (spec c8).

Three things are asserted here, and the third is the one that bites in the
field:

1. **Resolution** mirrors ``bus.load_rules`` exactly — an explicit argument
   wins, then ``NOVA_PERSONA_PATH``, then the repo file — and every failure
   (absent, unreadable, empty, a directory) falls back to the embedded
   :data:`~reachy_nova.harness.persona.DEFAULT_PERSONA` with exactly ONE
   named senselog line, never a crash and never a silent personality swap.
2. **The register** (decision c18, boundary c35): the text is dry, teasing,
   sincere when it matters, never cruel, warm to a guest — and it names no
   character, no series and no author, offers no help, and says "assistant"
   nowhere. Both the file AND the embedded default are held to this: a wheel
   install hears the embedded one every day.
3. **The wheel case** (boundary c34): a wheel ships nothing from the repo-root
   ``config/`` tree, so a subprocess with the path pointing nowhere must still
   report a personality — ``source()`` says ``embedded``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from reachy_nova.harness import persona

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Names, places and works the persona must never carry (boundary c35): the
#: register is borrowed in original words, the character is not role-played.
FORBIDDEN_SOURCE_WORDS = (
    "hoid",
    "stormlight",
    "roshar",
    "sanderson",
    "cosmere",
    "kaladin",
    "shallan",
    "dalinar",
    "king's wit",
)

#: Nova is company, not a help desk (spec c8) — and never calls itself one.
FORBIDDEN_HELPER_WORDS = ("help you", "how can i help", "assistant")

#: Tool mechanics live in the tool descriptions, not in the persona: the
#: integration task appends its own tool paragraph (spec c8, "Tool mechanics
#: leave the persona").
TOOL_MECHANIC_WORDS = (
    "run_behavior",
    "declare_goal",
    "set_mode",
    "set_inhibition",
    "lock_face",
    "release_face",
    "recall_senses",
    "create_rule",
    "stay_silent",
    "toolconfiguration",
)


@pytest.fixture(autouse=True)
def _no_persona_env(monkeypatch):
    """Every test states its own path — never inherit the operator's env."""
    monkeypatch.delenv(persona.PERSONA_PATH_ENV, raising=False)


def sense_lines(caplog) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if "[SENSE" in rec.getMessage()]


def persona_lines(caplog) -> list[str]:
    return [line for line in sense_lines(caplog) if "event=persona]" in line]


def both_texts() -> list[tuple[str, str]]:
    """(label, text) for the file and the embedded default."""
    return [
        ("config/persona/nova.md", persona.DEFAULT_PERSONA_PATH.read_text(encoding="utf-8")),
        ("DEFAULT_PERSONA", persona.DEFAULT_PERSONA),
    ]


# --------------------------------------------------------------------------- #
# 1. Path resolution — the same shape bus.load_rules uses                      #
# --------------------------------------------------------------------------- #


def test_default_path_is_the_repo_config_file():
    assert persona.DEFAULT_PERSONA_PATH == REPO_ROOT / "config" / "persona" / "nova.md"
    assert persona.DEFAULT_PERSONA_PATH.is_file()


def test_env_var_is_the_documented_spelling():
    assert persona.PERSONA_PATH_ENV == "NOVA_PERSONA_PATH"


def test_load_returns_the_repo_file_when_nothing_overrides_it(caplog):
    with caplog.at_level("INFO", logger="nova.sensory"):
        text = persona.load()
    assert text == persona.DEFAULT_PERSONA_PATH.read_text(encoding="utf-8")
    assert text != persona.DEFAULT_PERSONA
    assert persona_lines(caplog) == []


def test_env_override_wins_over_the_repo_file(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere.md"
    override.write_text("You are Nova, briefly.\n", encoding="utf-8")
    monkeypatch.setenv(persona.PERSONA_PATH_ENV, str(override))
    assert persona.load() == "You are Nova, briefly.\n"


def test_explicit_argument_wins_over_the_env(tmp_path, monkeypatch):
    monkeypatch.setenv(persona.PERSONA_PATH_ENV, str(tmp_path / "never-read.md"))
    chosen = tmp_path / "chosen.md"
    chosen.write_text("Chosen persona.\n", encoding="utf-8")
    assert persona.load(chosen) == "Chosen persona.\n"


# --------------------------------------------------------------------------- #
# 2. Fallback — embedded default plus exactly one named senselog line          #
# --------------------------------------------------------------------------- #


def test_missing_file_yields_the_embedded_default_and_one_named_line(tmp_path, monkeypatch, caplog):
    missing = tmp_path / "no-such-persona.md"
    monkeypatch.setenv(persona.PERSONA_PATH_ENV, str(missing))
    with caplog.at_level("INFO", logger="nova.sensory"):
        text = persona.load()
    assert text == persona.DEFAULT_PERSONA
    lines = persona_lines(caplog)
    assert len(lines) == 1, lines
    assert str(missing) in lines[0]
    assert "stage=supervise" in lines[0]
    assert "source=nova" in lines[0]


def test_a_directory_at_the_persona_path_falls_back_with_one_line(tmp_path, caplog):
    with caplog.at_level("INFO", logger="nova.sensory"):
        text = persona.load(tmp_path)
    assert text == persona.DEFAULT_PERSONA
    lines = persona_lines(caplog)
    assert len(lines) == 1, lines
    assert str(tmp_path) in lines[0]


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads a chmod-000 file regardless of its mode",
)
def test_an_unreadable_file_falls_back_with_one_line(tmp_path, caplog):
    unreadable = tmp_path / "locked.md"
    unreadable.write_text("secret persona", encoding="utf-8")
    unreadable.chmod(0o000)
    try:
        with caplog.at_level("INFO", logger="nova.sensory"):
            text = persona.load(unreadable)
    finally:
        unreadable.chmod(0o600)
    assert text == persona.DEFAULT_PERSONA
    assert len(persona_lines(caplog)) == 1


def test_an_empty_file_falls_back_rather_than_sending_an_empty_prompt(tmp_path, caplog):
    empty = tmp_path / "empty.md"
    empty.write_text("   \n\n", encoding="utf-8")
    with caplog.at_level("INFO", logger="nova.sensory"):
        text = persona.load(empty)
    assert text == persona.DEFAULT_PERSONA
    assert len(persona_lines(caplog)) == 1


def test_the_embedded_default_is_a_real_persona_not_a_stub():
    assert len(persona.DEFAULT_PERSONA.strip()) > 300


# --------------------------------------------------------------------------- #
# 3. Provenance — the caller logs where the personality came from              #
# --------------------------------------------------------------------------- #


def test_source_names_the_file_it_read(tmp_path):
    written = tmp_path / "nova.md"
    written.write_text("Nova, from a file.\n", encoding="utf-8")
    assert persona.source(written) == f"file:{written}"


def test_source_says_embedded_when_the_file_is_absent(tmp_path):
    assert persona.source(tmp_path / "gone.md") == "embedded"


def test_read_returns_text_and_provenance_together(tmp_path):
    written = tmp_path / "nova.md"
    written.write_text("Nova, from a file.\n", encoding="utf-8")
    loaded = persona.read(written)
    assert loaded.text == "Nova, from a file.\n"
    assert loaded.source == f"file:{written}"
    assert loaded.path == written

    fallen_back = persona.read(tmp_path / "gone.md")
    assert fallen_back.text == persona.DEFAULT_PERSONA
    assert fallen_back.source == "embedded"


# --------------------------------------------------------------------------- #
# 4. The register (c18) and the boundaries (c35, c8) — file AND embedded       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label,text", both_texts())
@pytest.mark.parametrize("word", FORBIDDEN_SOURCE_WORDS)
def test_persona_names_no_character_and_no_book(label, text, word):
    assert word not in text.lower(), f"{label} names {word!r} — borrow the register, not the man"


@pytest.mark.parametrize("label,text", both_texts())
@pytest.mark.parametrize("word", FORBIDDEN_HELPER_WORDS)
def test_persona_is_not_a_helper(label, text, word):
    assert word not in text.lower(), f"{label} contains {word!r} — Nova is company, not a service"


@pytest.mark.parametrize("label,text", both_texts())
@pytest.mark.parametrize("word", TOOL_MECHANIC_WORDS)
def test_persona_carries_no_tool_mechanics(label, text, word):
    assert word not in text.lower(), f"{label} names the tool {word!r} — that paragraph is app.py's"


@pytest.mark.parametrize("label,text", both_texts())
def test_persona_states_the_register(label, text):
    lowered = text.lower()
    assert "tease" in lowered or "teasing" in lowered, label
    assert "dry" in lowered, label
    assert "sincer" in lowered, label
    assert "cruel" in lowered, label
    assert "guest" in lowered or "first meeting" in lowered, label


@pytest.mark.parametrize("label,text", both_texts())
def test_persona_asks_for_variety_and_body_cue_restraint(label, text):
    lowered = text.lower()
    assert "parenthes" in lowered, f"{label} never says how body cues arrive"
    assert "reuse" in lowered or "repeat" in lowered, f"{label} never asks for variety"


@pytest.mark.parametrize("label,text", both_texts())
def test_persona_has_no_markdown_lists(label, text):
    """Sonic reads this aloud-shaped; a bulleted list invites a spoken list."""
    for line in text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("- ", "* ", "1. ")), f"{label}: list line {line!r}"


def test_persona_file_shows_the_tone_with_exactly_two_one_shot_exchanges():
    text = persona.DEFAULT_PERSONA_PATH.read_text(encoding="utf-8")
    users = [line for line in text.splitlines() if line.startswith("User:")]
    novas = [line for line in text.splitlines() if line.startswith("Nova:")]
    assert len(users) == 2, users
    assert len(novas) == 2, novas


def test_persona_file_stays_small_enough_for_a_live_system_prompt():
    text = persona.DEFAULT_PERSONA_PATH.read_text(encoding="utf-8")
    assert len(text) <= 1800, f"{len(text)} characters — trim it"


# --------------------------------------------------------------------------- #
# 4b. Both names (t4): Reachy is mistranscribed as Richie/Reach in the field   #
# --------------------------------------------------------------------------- #


NAME_PARAGRAPH = (
    "People call you Nova or Reachy; both are you, and you answer to either. "
    "Reachy often reaches you as Richie or Reach, so treat those as your name too."
)


@pytest.mark.parametrize("label,text", both_texts())
def test_persona_answers_to_both_names_and_their_mishearings(label, text):
    assert "Nova" in text, label
    assert "Reachy" in text, label
    assert "Richie" in text, label


def test_the_names_paragraph_is_identical_in_file_and_embedded():
    for label, text in both_texts():
        assert NAME_PARAGRAPH in text, f"{label} is missing the exact names paragraph"


@pytest.mark.parametrize("label,text", both_texts())
def test_persona_stays_under_two_thousand_characters(label, text):
    assert len(text) < 2000, f"{label}: {len(text)} characters — trim it"


# --------------------------------------------------------------------------- #
# 5. The wheel case (c34): no repo config/ tree, still a personality           #
# --------------------------------------------------------------------------- #


def test_a_process_with_no_persona_file_reports_the_embedded_default(tmp_path):
    """Stand-in for the wheel install: the path points nowhere, cwd is empty.

    A wheel ships ``reachy_nova/`` only, so the repo-root ``config/persona/``
    tree is simply absent there. The harness must still boot with a character.
    """
    env = dict(os.environ)
    env[persona.PERSONA_PATH_ENV] = str(tmp_path / "absent" / "nova.md")
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") or str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", "import reachy_nova.harness.persona as p; print(p.source())"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "embedded", result.stdout


def test_a_process_with_a_persona_file_reports_that_file(tmp_path):
    written = tmp_path / "nova.md"
    written.write_text("Nova, from a file.\n", encoding="utf-8")
    env = dict(os.environ)
    env[persona.PERSONA_PATH_ENV] = str(written)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") or str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", "import reachy_nova.harness.persona as p; print(p.source())"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"file:{written}"
