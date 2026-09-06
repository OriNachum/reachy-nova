"""Attention — the cold/warm window opened by the robot's own name (t5).

A robot that answers every sentence it hears is not listening, it is
interrupting. ``AttentionState`` is the one small clock that decides whether
Nova is being *addressed*:

* The robot is **COLD** until a USER transcript **names** it — "nova",
  "reachy", or one of the mishearings a speech-to-text model reliably produces
  for them ("richie", "reach", "noah"). Cold means the mouth stays shut.
* A name **opens** a warm window of :data:`DEFAULT_WINDOW_S` seconds (env
  ``NOVA_ATTENTION_WINDOW_S``). Warm means Nova is in a conversation and may
  answer the *next* thing said without having to be named again — because
  nobody says "nova" before every sentence of a conversation they are already
  having.
* Everything that happens inside the window **renews** it: a further transcript
  (named or not), an utterance Nova itself produced, or an inject that reached
  the model. A conversation stays alive as long as it is actually happening.
* Nothing for a whole window and it goes **cold** again, silently and on its
  own — the failure mode of a forgotten warm window is a robot that stops
  answering strangers a minute later, not one that never stops talking.

**Two reads off one clock.** :attr:`warm` governs the VOICE; it is true only
after a NAME opened the window. :attr:`conversation_live` governs the GAZE; it
is true after ANY USER transcript, named or not, for the same window length —
and Nova's own utterances only ever RENEW it, never open it, because a robot
cannot start a conversation by talking to itself. They deliberately disagree
for a nameless transcript from cold: someone
in the room is plainly talking, so the eyes follow them, but nobody addressed
the robot, so the mouth stays shut. Wiring both to the same boolean is what
made the old robot either blind or a chatterbox.

**Quiet wins.** Given a duck-typed ``quiet`` object (:class:`~reachy_nova.
harness.quiet.QuietState`), an active quiet reads as cold on both properties
and blocks a name from opening a new window. A person who asked for silence
does not get an exception for saying the robot's name.

Pure state: no threads, no network, no I/O beyond ONE senselog line when the
window opens and one when it closes. There is no timer behind it — expiry is
computed lazily from the clock on every read, so the close line is emitted by
the first read that observes it.

stdlib only; never imports ``reachy_mini``
(``tests/test_harness_boundary.py``).
"""

from __future__ import annotations

import difflib
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence

from .. import sensory_log

STAGE_ATTENTION = "attention"
SOURCE = "nova"
EVENT_WINDOW = "window"

#: How long a named window stays warm, in seconds.
DEFAULT_WINDOW_S = 45.0

#: Env override for :data:`DEFAULT_WINDOW_S`, in seconds.
WINDOW_ENV = "NOVA_ATTENTION_WINDOW_S"

#: The names that address this robot: the two real ones plus the three
#: mishearings the on-device ASR actually produces for them. "reach" and
#: "richie" are what "reachy" comes back as; "noah" is what "nova" comes back
#: as. They are listed as first-class names rather than left to the fuzzy
#: matcher because a truncation ("reach") is exactly what the prefix guard
#: below is built to reject — if we want it, we must say so.
DEFAULT_NAMES: tuple[str, ...] = ("nova", "reachy", "richie", "reach", "noah")

#: Minimum combined-similarity score for a FUZZY name match. Restated (not
#: imported) from the runtime's ``reachy/speech/name_match.py``: the harness
#: must not grow a dependency on the runtime's Python package, and the two
#: matchers answer different questions over different name sets.
DEFAULT_THRESHOLD = 0.50

#: A fuzzy match needs at least this many letters. "nova" and "noah" are short
#: four-letter words whose neighbourhood is full of ordinary English — "nah"
#: scores 0.64 against "noah" — so anything shorter than the names themselves
#: is never *heard as* one of them, it merely looks like one. Exact matches are
#: unaffected: a real "nova"/"noah" still matches.
MIN_FUZZY_LEN = 4

# Same tokenisation the runtime's matcher uses.
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

#: Soundex consonant classes; letters absent here (vowels plus h/w/y) carry no
#: code of their own. See :func:`_phonetic_code`.
_SOUNDEX_CODES: dict[str, str] = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}
_SOUNDEX_VOWELS = frozenset("aeiouy")
_SOUNDEX_TRANSPARENT = frozenset("hw")
_SOUNDEX_DIGITS = 3


def default_window_s() -> float:
    """The attention window, from ``NOVA_ATTENTION_WINDOW_S`` or the default.

    Parsed defensively, like :func:`reachy_nova.harness.lock_state.
    default_drop_grace_s`: unset, empty, unparseable, NaN or negative resolves
    to :data:`DEFAULT_WINDOW_S`. A typo in an env var must never be the reason
    the robot went permanently deaf or permanently chatty.

    ``0`` IS honoured, and it means "always cold unless just named": the window
    closes the same instant it opens, so every single utterance has to name the
    robot. Someone asking for that explicitly means it.
    """
    raw = os.environ.get(WINDOW_ENV)
    if raw is None:
        return DEFAULT_WINDOW_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_WINDOW_S
    if value < 0.0 or math.isnan(value):
        return DEFAULT_WINDOW_S
    return value


# --------------------------------------------------------------------------- #
# the name matcher                                                             #
# --------------------------------------------------------------------------- #


def _phonetic_code(word: str) -> str:
    """The word's Soundex code — its leading letter plus a consonant skeleton.

    Two words share a code when they share a pronunciation *shape*, which is
    what an ASR mishearing preserves and an orthographic look-alike does not::

        _phonetic_code("nova")     == "n100"
        _phonetic_code("novel")    == "n140"   # different word, not a mishearing
        _phonetic_code("november") == "n151"
        _phonetic_code("nowhere")  == "n600"

    Returns ``""`` for a word with no letters, which never equals a real name's
    code. Never raises.
    """
    letters = [ch for ch in word if ch.isalpha()]
    if not letters:
        return ""
    first = letters[0]
    digits: list[str] = []
    previous = _SOUNDEX_CODES.get(first, "")
    for char in letters[1:]:
        code = _SOUNDEX_CODES.get(char, "")
        if code:
            if code != previous:
                digits.append(code)
                if len(digits) == _SOUNDEX_DIGITS:
                    break
            previous = code
        elif char in _SOUNDEX_VOWELS:
            previous = ""  # a vowel breaks the run; h/w leave `previous` alone
    return (first + "".join(digits)).ljust(1 + _SOUNDEX_DIGITS, "0")


def _combined_score(word: str, name: str) -> float:
    """difflib ratio × length ratio — the length term punishes fragments."""
    seq_ratio = difflib.SequenceMatcher(None, word, name).ratio()
    len_ratio = min(len(word), len(name)) / max(len(word), len(name))
    return seq_ratio * len_ratio


def _word_matches_name(word: str, name: str, threshold: float) -> bool:
    """Whether one tokenised *word* addresses the robot as *name*.

    An exact match always accepts. A fuzzy match must clear, in order:

    1. *Length guard* — at least :data:`MIN_FUZZY_LEN` letters. This is what
       keeps the n-family ("now", "no", "nah", "not") out: they are shorter
       than the names they collide with, so they are never mishearings.
    2. *Prefix guard* — a strict prefix of the name is a truncation, not a
       mishearing. (``"reach"`` is a listed NAME, so it never reaches here.)
    3. *Superstring guard* — a name literally inside a longer word
       (``"nova"`` ⊂ ``"novacaine"``) is a different word.
    4. *Initial guard* — a fuzzy match shares the name's first letter
       ("know" ≠ "nova").
    5. *Phonetic guard* — a fuzzy match must SOUND like the name, not merely
       look like it. This is the guard that rejects "novel" (n140),
       "november" (n151) and "nowhere" (n600) against "nova" (n100): all three
       clear the similarity score comfortably and none of them is a
       mis-transcription of the robot's name.

    Only then does the combined similarity score decide.
    """
    if word == name:
        return True
    if len(word) < MIN_FUZZY_LEN:
        return False
    if name.startswith(word):
        return False
    if name in word:
        return False
    if not word.startswith(name[:1]):
        return False
    if _phonetic_code(word) != _phonetic_code(name):
        return False
    return _combined_score(word, name) >= threshold


def matched_name(
    text: str,
    names: Iterable[str] = DEFAULT_NAMES,
    threshold: float = DEFAULT_THRESHOLD,
) -> str | None:
    """The first word of *text* that names the robot, or ``None``.

    Same decision as :func:`is_name_match`, but it hands back the word it
    matched on so the caller can say *what* it heard — the open log line names
    it, which is the difference between a debuggable false positive and a
    mystery.
    """
    if not text:
        return None
    name_list = [n.lower() for n in names]
    for word in _WORD_RE.findall(text.lower()):
        for name in name_list:
            if _word_matches_name(word, name, threshold):
                return word
    return None


def is_name_match(
    text: str,
    names: Iterable[str] = DEFAULT_NAMES,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Return ``True`` when *text* contains a word that names this robot.

    Restates — deliberately does not import — the idea in the runtime's
    ``reachy/speech/name_match.py``: tokenise, then accept a word that either
    equals a name or clears ``difflib_ratio × length_ratio`` behind the
    structural guards documented on :func:`_word_matches_name`.
    """
    return matched_name(text, names, threshold) is not None


# --------------------------------------------------------------------------- #
# the window                                                                   #
# --------------------------------------------------------------------------- #


class AttentionState:
    """The cold/warm attention window. Thread-safe, timer-free, pure state.

    Parameters
    ----------
    clock:
        Monotonic seconds source; injectable for tests. Monotonic, not wall
        time, because this window is never persisted — a restart is a cold
        robot, which is the safe direction.
    window_s:
        Warm-window length; ``None`` resolves :func:`default_window_s` at
        construction, so the env override applies without every caller
        knowing about it.
    names:
        Name override; defaults to :data:`DEFAULT_NAMES`.
    quiet:
        Optional duck-typed object with an ``active() -> bool`` method
        (:class:`~reachy_nova.harness.quiet.QuietState`). While it reports
        active, this reads cold and refuses to open.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        window_s: float | None = None,
        names: Sequence[str] | None = None,
        quiet: object | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self.window_s = default_window_s() if window_s is None else float(window_s)
        self.names: tuple[str, ...] = (
            DEFAULT_NAMES if names is None else tuple(names)
        )
        self._quiet = quiet

        # The warm (named) window.
        self._opened_at: float | None = None
        self._warm_until: float = 0.0
        # The conversation (gaze) window — any transcript or utterance.
        self._live_until: float = 0.0

        #: When the last USER transcript arrived (monotonic), or ``None``.
        self.last_transcript_at: float | None = None
        #: Whether that transcript named the robot.
        self.last_transcript_named: bool = False
        #: When Nova last spoke (monotonic), or ``None``.
        self.last_utterance_at: float | None = None
        #: When an inject last reached the model (monotonic), or ``None``.
        self.last_inject_at: float | None = None

    # -- reads ---------------------------------------------------------------

    @property
    def warm(self) -> bool:
        """Is the NAME-opened window still open? Governs the voice."""
        self.settle()
        with self._lock:
            return self._opened_at is not None

    @property
    def conversation_live(self) -> bool:
        """Has anyone spoken — either side — inside the window? Governs the gaze.

        True after ANY USER transcript, named or not, and independent of
        :attr:`warm`: a nameless transcript from cold makes this True while
        :attr:`warm` stays False. Nova's own utterances renew it while it is
        live and can never open it (see :meth:`note_utterance`).
        """
        self.settle()
        with self._lock:
            if self._is_quiet():
                return False
            return self._clock() < self._live_until

    def remaining_s(self) -> float:
        """Seconds left on the warm window; ``0.0`` when cold."""
        self.settle()
        with self._lock:
            if self._opened_at is None:
                return 0.0
            return max(0.0, self._warm_until - self._clock())

    # -- notes ---------------------------------------------------------------

    def note_transcript(self, text: str) -> str:
        """Record a USER transcript. Returns ``opened``/``renewed``/``ignored``.

        ``opened`` — *text* named the robot and the window was cold.
        ``renewed`` — the window was already warm; anything said inside a live
        conversation keeps it alive, named or not.
        ``ignored`` — cold and nameless (or quiet): the gaze clock still runs,
        the mouth does not open.
        """
        self.settle()
        named = is_name_match(text or "", self.names)
        with self._lock:
            now = self._clock()
            self.last_transcript_at = now
            self.last_transcript_named = named
            if self._is_quiet():
                return "ignored"
            self._live_until = now + self.window_s
            if self._opened_at is not None:
                self._warm_until = now + self.window_s
                return "renewed"
            if not named:
                return "ignored"
            self._opened_at = now
            self._warm_until = now + self.window_s
        sensory_log.stage(
            STAGE_ATTENTION,
            SOURCE,
            EVENT_WINDOW,
            f"opened by={matched_name(text or '', self.names)}",
        )
        return "opened"

    def note_utterance(self) -> None:
        """Nova spoke. Renews an open window; never opens a closed one.

        Her own voice keeps a conversation alive but can never start one —
        otherwise a single unprompted line would warm the robot up forever by
        talking to itself. That is true of BOTH clocks: :attr:`warm` is renewed
        only while it is already warm, and :attr:`conversation_live` only while
        it is already live.

        The gaze clock used to be renewed unconditionally here, and the live
        robot showed what that costs (2026-09-06, "It feels rigid now. No
        liveness."): Nova's own reactions to body cues — and her opening line
        at every session start — opened a conversation nobody was having, the
        conversation layer took a face lock, the lock inhibited ``feel-alive``
        and ``orient-to-sound``, and she renewed the whole thing by speaking
        into it. A robot cannot start a conversation with itself.
        """
        self.settle()
        with self._lock:
            now = self._clock()
            self.last_utterance_at = now
            if self._is_quiet():
                return
            if now < self._live_until:
                self._live_until = now + self.window_s
            if self._opened_at is not None:
                self._warm_until = now + self.window_s

    def note_inject(self) -> None:
        """A body cue or tool result reached the model. Renews a warm window.

        Renews :attr:`warm` only, not :attr:`conversation_live`: an inject is
        the robot's own nervous system talking to its own mind, and nobody in
        the room said anything, so it is no reason for the eyes to keep
        holding a person.
        """
        self.settle()
        with self._lock:
            now = self._clock()
            self.last_inject_at = now
            if self._is_quiet():
                return
            if self._opened_at is not None:
                self._warm_until = now + self.window_s

    def on_session_rotated(self) -> None:
        """Sonic rotated its session. Deliberately changes nothing.

        The window is the HARNESS's clock, not Sonic's. A session rotation is
        plumbing — the model's context was replayed into a fresh stream — and
        the person standing in front of the robot mid-sentence neither knows
        nor cares that it happened. Resetting attention here would make the
        robot go cold in the middle of a conversation every few minutes; that
        is exactly the bug this no-op exists to name.
        """
        return None

    # -- expiry --------------------------------------------------------------

    def settle(self) -> bool:
        """Close the window if the clock (or quiet) says it is over.

        Idempotent and cheap — there is no timer thread, so this runs from
        every read and every note, and the ONE close line is emitted by
        whichever call first observes the expiry. Returns whether this call was
        the one that closed the window.
        """
        with self._lock:
            quiet = self._is_quiet()
            if quiet:
                self._live_until = 0.0
            if self._opened_at is None:
                return False
            now = self._clock()
            if not quiet and now < self._warm_until:
                return False
            elapsed = now - self._opened_at
            self._opened_at = None
            self._warm_until = 0.0
            reason = "quiet" if quiet else "expired"
        sensory_log.stage(
            STAGE_ATTENTION,
            SOURCE,
            EVENT_WINDOW,
            f"closed after={elapsed:.1f}s reason={reason}",
        )
        return True

    # -- internals -----------------------------------------------------------

    def _is_quiet(self) -> bool:
        """Whether the optional quiet object says the mouth is closed.

        Duck-typed and forgiving: anything without a usable ``active()`` simply
        is not quiet. Attention must never be the thing that crashes because a
        collaborator changed shape.
        """
        quiet = self._quiet
        if quiet is None:
            return False
        try:
            return bool(quiet.active())
        except Exception:  # pragma: no cover - defensive
            return False
