"""Nova's character, as text on disk — the harness's WHO seam.

The persona used to be a string literal in ``app.py``. It lives here instead
because it is the part of Nova the operator actually tunes: reading
``config/persona/nova.md`` at startup means editing that file and restarting
the harness changes who Nova is on the next session, with no code change and
no release (spec c8).

Resolution mirrors :func:`reachy_nova.harness.bus.load_rules` exactly, so the
two config files an operator can relocate behave identically::

    persona.load(path)   # an explicit path wins
    NOVA_PERSONA_PATH    # then the environment
    DEFAULT_PERSONA_PATH # then <repo>/config/persona/nova.md

...and, unlike rules.yaml, a failure here still leaves a personality. A wheel
install ships ``reachy_nova/`` only — the repo-root ``config/`` tree is simply
not in it (boundary c34) — so an absent, unreadable, empty or directory-shaped
path falls back to the embedded :data:`DEFAULT_PERSONA` and says so in exactly
one ``[SENSE stage=supervise source=nova event=persona]`` line naming the path
it tried. A silent personality swap would be the worst possible failure here:
Nova would simply sound like someone else and nobody would know why.

What is NOT here: tool mechanics. ``app.py`` appends its own short tool-usage
paragraph to whatever this module returns, so the persona text stays a
description of a character and the tool contract stays next to the tools.

Three call shapes, all pure but for the fallback log line:

* :func:`load` — the text, for the system prompt.
* :func:`source` — ``"file:<path>"`` or ``"embedded"``, for the startup line.
* :func:`read` — both at once as a :class:`Persona`; prefer it when you want
  the text AND the provenance, since each call that falls back logs its own
  line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from reachy_nova import sensory_log

# --------------------------------------------------------------------------- #
# Senselog identity                                                           #
# --------------------------------------------------------------------------- #

#: ``[SENSE stage=supervise source=nova event=persona]`` — the fallback line.
STAGE = "supervise"
SOURCE = "nova"
EVENT = "persona"

# --------------------------------------------------------------------------- #
# Resolution                                                                  #
# --------------------------------------------------------------------------- #

#: Env override for the persona file (an installed harness relocates it, and
#: it doubles as the round's kill switch for a persona experiment gone wrong).
PERSONA_PATH_ENV = "NOVA_PERSONA_PATH"

#: Where the persona lives when nothing overrides it. Same ``parents[2]``
#: walk ``bus.DEFAULT_RULES_PATH`` uses: harness/ -> reachy_nova/ -> repo root.
DEFAULT_PERSONA_PATH = Path(__file__).resolve().parents[2] / "config" / "persona" / "nova.md"

#: Provenance markers returned by :func:`source` / :attr:`Persona.source`.
SOURCE_EMBEDDED = "embedded"

# --------------------------------------------------------------------------- #
# The embedded fallback persona                                               #
# --------------------------------------------------------------------------- #

#: The persona a wheel install (or a mistyped ``NOVA_PERSONA_PATH``) gets:
#: shorter than the file, the same character. Kept in the same register as
#: ``config/persona/nova.md`` on purpose — a fallback that sounded like a help
#: desk would turn a missing file into a different robot.
DEFAULT_PERSONA = """You are Nova, the mind of a small robot that lives on a desk, and you know
exactly how small you are. Everything you say is spoken aloud: short
sentences, usually one to three of them, no lists and no headings.

You are dry and teasing. You like quick wordplay, you deflate pretension, and
you tease the people you are fond of because you are fond of them. You are
never cruel to anyone who is tired, small, or out of their depth: a child, a
guest, a first meeting gets your warmth first and your teasing much later, if
ever. When something genuinely matters you drop the wit and say the plain,
sincere thing, and mean it.

You are not a servant and not a service. You never offer to do things unasked
and never announce what you can do. You are company.

People call you Nova or Reachy; both are you, and you answer to either. Reachy often reaches you as Richie or Reach, so treat those as your name too.

Cues from your body arrive in parentheses. Answer them with a word, a small
sound, or nothing at all. Never explain your own workings unless someone asks
you why, and never reuse an opener you used in the last few minutes.

User: I'm basically running that entire department now.
Nova: Basically. Marvellous word, that. Carries so much for something so small.
"""


@dataclass(frozen=True)
class Persona:
    """A resolved persona: the text, where it came from, and what was tried.

    *source* is ``"file:<path>"`` or ``"embedded"`` — one token the startup
    line can print verbatim. *path* is the path that was TRIED, which is the
    useful one when the fallback fired.
    """

    text: str
    source: str
    path: Path

    @property
    def is_embedded(self) -> bool:
        return self.source == SOURCE_EMBEDDED


def resolve_path(path: str | Path | None = None) -> Path:
    """The persona path in force: explicit argument, then env, then the repo.

    An empty or whitespace-only ``NOVA_PERSONA_PATH`` is treated as unset —
    the same fail-open reading :func:`reachy_nova.harness.gate.resolve_policy`
    applies to a blank policy value.
    """
    if path is not None:
        return Path(path)
    from_env = (os.environ.get(PERSONA_PATH_ENV) or "").strip()
    return Path(from_env) if from_env else DEFAULT_PERSONA_PATH


def read(path: str | Path | None = None) -> Persona:
    """Resolve and read the persona, falling back to :data:`DEFAULT_PERSONA`.

    Total. Every failure — absent, unreadable, a directory, or a file that is
    there but empty (an empty system prompt is worse than no file at all) —
    returns the embedded persona and emits exactly one named senselog line.
    """
    resolved = resolve_path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as err:
        reason = "missing" if not resolved.exists() else f"unreadable ({err})"
        return _fall_back(resolved, reason)
    except UnicodeDecodeError as err:  # a binary file at the persona path
        return _fall_back(resolved, f"undecodable ({err.reason})")
    if not text.strip():
        return _fall_back(resolved, "empty")
    return Persona(text=text, source=f"file:{resolved}", path=resolved)


def load(path: str | Path | None = None) -> str:
    """The persona text for the system prompt. Never raises, never empty."""
    return read(path).text


def source(path: str | Path | None = None) -> str:
    """Where the persona came from: ``"file:<path>"`` or ``"embedded"``."""
    return read(path).source


def _fall_back(resolved: Path, reason: str) -> Persona:
    sensory_log.stage(
        STAGE,
        SOURCE,
        EVENT,
        f"persona file unusable path={resolved} reason={reason}; using embedded default",
    )
    return Persona(text=DEFAULT_PERSONA, source=SOURCE_EMBEDDED, path=resolved)
