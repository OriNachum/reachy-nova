"""Attention window (task t5) — ``reachy_nova/harness/attention.py``.

``AttentionState`` is the harness's cold/warm clock: the robot is COLD until a
user transcript NAMES it, warm for a window after that, and cold again once the
window runs out. Two reads come off the same clock — ``warm`` (governs the
VOICE) and ``conversation_live`` (governs the GAZE) — and they deliberately
disagree for a nameless transcript from cold: someone is clearly talking, so the
eyes follow, but nobody said the robot's name, so the mouth stays shut.

Every test here drives an injected monotonic clock — no sleeping, no wall time.
"""

from __future__ import annotations

import logging

import pytest

from reachy_nova.harness.attention import (
    DEFAULT_NAMES,
    DEFAULT_WINDOW_S,
    AttentionState,
    default_window_s,
    is_name_match,
)


class FakeClock:
    """Injectable monotonic clock."""

    def __init__(self, t: float = 1_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeQuiet:
    """Duck-typed stand-in for :class:`~reachy_nova.harness.quiet.QuietState`."""

    def __init__(self, active: bool = False):
        self._active = active

    def active(self) -> bool:
        return self._active


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def attention(clock):
    return AttentionState(clock=clock, window_s=45.0)


def sense_lines(caplog) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if "[SENSE" in rec.getMessage()]


def window_lines(caplog) -> list[str]:
    return [line for line in sense_lines(caplog) if "event=window]" in line]


# --------------------------------------------------------------------------- #
# 1. the name matcher                                                          #
# --------------------------------------------------------------------------- #

OPENERS = [
    "nova, how are you",
    "hello richie",
    "reach, come here",
    "Noah are you there",
    "hey reachy",
]

#: The n-family. Every one of these is an ordinary word that an over-eager
#: fuzzy matcher hears as "nova"/"noah"; none of them addresses the robot.
NOT_NAMES = [
    "now",
    "no",
    "know",
    "nah",
    "not",
    "not now",
    "novel",
    "november",
    "nowhere",
]


@pytest.mark.parametrize("text", OPENERS)
def test_is_name_match_accepts_the_robots_names(text):
    assert is_name_match(text) is True


@pytest.mark.parametrize("text", NOT_NAMES)
def test_is_name_match_rejects_the_n_family(text):
    assert is_name_match(text) is False


def test_is_name_match_on_empty_and_nameless_text():
    assert is_name_match("") is False
    assert is_name_match("what time is it") is False


def test_default_names_cover_the_mishearings():
    assert set(DEFAULT_NAMES) == {"nova", "reachy", "richie", "reach", "noah"}


def test_is_name_match_accepts_a_names_override():
    assert is_name_match("hello robot", names=("robot",)) is True
    assert is_name_match("hello robot") is False


# --------------------------------------------------------------------------- #
# 2. cold until named                                                          #
# --------------------------------------------------------------------------- #


def test_starts_cold(attention):
    assert attention.warm is False
    assert attention.conversation_live is False


def test_nameless_transcript_from_cold_stays_cold(attention):
    assert attention.note_transcript("what time is it") == "ignored"
    assert attention.warm is False
    # ... but somebody is plainly talking, so the gaze clock runs.
    assert attention.conversation_live is True


@pytest.mark.parametrize("text", OPENERS)
def test_a_name_opens_the_window(clock, text):
    attention = AttentionState(clock=clock, window_s=45.0)
    assert attention.note_transcript(text) == "opened"
    assert attention.warm is True
    assert attention.conversation_live is True


@pytest.mark.parametrize("text", NOT_NAMES)
def test_the_n_family_does_not_open_the_window(clock, text):
    attention = AttentionState(clock=clock, window_s=45.0)
    assert attention.note_transcript(text) == "ignored"
    assert attention.warm is False


def test_records_the_transcript_timestamps(attention, clock):
    assert attention.last_transcript_at is None
    assert attention.last_transcript_named is False
    attention.note_transcript("what time is it")
    assert attention.last_transcript_at == clock.t
    assert attention.last_transcript_named is False
    clock.advance(1.0)
    attention.note_transcript("nova hi")
    assert attention.last_transcript_at == clock.t
    assert attention.last_transcript_named is True


# --------------------------------------------------------------------------- #
# 3. renewal                                                                   #
# --------------------------------------------------------------------------- #


def test_nameless_transcript_while_warm_renews_and_pushes_expiry(attention, clock):
    assert attention.note_transcript("nova hi") == "opened"
    clock.advance(40.0)
    assert attention.note_transcript("what time is it") == "renewed"
    clock.advance(10.0)  # 50 s after the open — expired without the renewal
    assert attention.warm is True
    clock.advance(36.0)  # 46 s after the renewal
    assert attention.warm is False


def test_utterance_and_inject_renew_while_warm(attention, clock):
    attention.note_transcript("nova hi")
    clock.advance(40.0)
    attention.note_utterance()
    clock.advance(40.0)
    assert attention.warm is True
    attention.note_inject()
    clock.advance(40.0)
    assert attention.warm is True
    clock.advance(6.0)
    assert attention.warm is False


def test_utterance_and_inject_do_not_open_from_cold(attention, clock):
    attention.note_utterance()
    assert attention.warm is False
    attention.note_inject()
    assert attention.warm is False
    assert attention.last_utterance_at is not None
    assert attention.last_inject_at is not None


def test_inject_does_not_by_itself_keep_the_conversation_live(attention, clock):
    attention.note_inject()
    assert attention.conversation_live is False


# --------------------------------------------------------------------------- #
# 4. expiry                                                                    #
# --------------------------------------------------------------------------- #


def test_window_expires_after_45_seconds(clock):
    attention = AttentionState(clock=clock)  # default window
    attention.note_transcript("nova hi")
    clock.advance(44.0)
    assert attention.warm is True
    clock.advance(2.0)
    assert attention.warm is False
    assert attention.conversation_live is False


def test_conversation_live_expires_on_the_same_window_without_a_name(attention, clock):
    attention.note_transcript("what time is it")
    assert attention.conversation_live is True
    clock.advance(44.0)
    assert attention.conversation_live is True
    clock.advance(2.0)
    assert attention.conversation_live is False


def test_conversation_live_after_an_utterance_alone(attention, clock):
    attention.note_utterance()
    assert attention.conversation_live is True
    assert attention.warm is False
    clock.advance(46.0)
    assert attention.conversation_live is False


# --------------------------------------------------------------------------- #
# 5. quiet forces cold                                                         #
# --------------------------------------------------------------------------- #


def test_quiet_forces_cold(clock):
    quiet = FakeQuiet()
    attention = AttentionState(clock=clock, window_s=45.0, quiet=quiet)
    attention.note_transcript("nova hi")
    assert attention.warm is True
    quiet._active = True
    assert attention.warm is False
    assert attention.conversation_live is False


def test_quiet_blocks_opening(clock):
    quiet = FakeQuiet(active=True)
    attention = AttentionState(clock=clock, window_s=45.0, quiet=quiet)
    assert attention.note_transcript("nova hi") == "ignored"
    assert attention.warm is False
    quiet._active = False
    # Quiet ended; the window it blocked does not come back on its own.
    assert attention.warm is False
    assert attention.note_transcript("nova hi") == "opened"
    assert attention.warm is True


# --------------------------------------------------------------------------- #
# 6. session rotation is not our clock                                         #
# --------------------------------------------------------------------------- #


def test_on_session_rotated_changes_nothing(attention, clock, caplog):
    attention.note_transcript("nova hi")
    clock.advance(10.0)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        attention.on_session_rotated()
    assert attention.warm is True
    assert attention.conversation_live is True
    assert window_lines(caplog) == []
    clock.advance(36.0)
    assert attention.warm is False


def test_on_session_rotated_does_not_open_a_cold_window(attention):
    attention.on_session_rotated()
    assert attention.warm is False
    assert attention.conversation_live is False


# --------------------------------------------------------------------------- #
# 7. exactly one line per open, one per close, none per renewal                #
# --------------------------------------------------------------------------- #


def test_one_log_line_per_open_and_close_none_per_renewal(attention, clock, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        attention.note_transcript("nova hi")
        assert len(window_lines(caplog)) == 1
        assert "opened by=" in window_lines(caplog)[0]
        assert "stage=attention" in window_lines(caplog)[0]
        assert "source=nova" in window_lines(caplog)[0]

        clock.advance(10.0)
        attention.note_transcript("what time is it")
        attention.note_utterance()
        attention.note_inject()
        assert attention.warm is True
        assert len(window_lines(caplog)) == 1  # no line per renewal

        clock.advance(46.0)
        assert attention.warm is False
        lines = window_lines(caplog)
        assert len(lines) == 2
        assert "closed after=" in lines[1]
        assert "reason=expired" in lines[1]

        # The close is latched: further reads never repeat it.
        assert attention.warm is False
        assert attention.conversation_live is False
        assert len(window_lines(caplog)) == 2


def test_quiet_close_is_named_quiet(clock, caplog):
    quiet = FakeQuiet()
    attention = AttentionState(clock=clock, window_s=45.0, quiet=quiet)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        attention.note_transcript("nova hi")
        quiet._active = True
        assert attention.warm is False
        lines = window_lines(caplog)
        assert len(lines) == 2
        assert "reason=quiet" in lines[1]


def test_a_second_open_after_a_close_logs_again(attention, clock, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        attention.note_transcript("nova hi")
        clock.advance(46.0)
        assert attention.warm is False
        assert attention.note_transcript("nova again") == "opened"
        lines = window_lines(caplog)
        assert len(lines) == 3
        assert "opened by=" in lines[2]


def test_an_ignored_transcript_logs_nothing(attention, caplog):
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        attention.note_transcript("what time is it")
        assert window_lines(caplog) == []


# --------------------------------------------------------------------------- #
# 8. env parsing                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, DEFAULT_WINDOW_S),
        ("", DEFAULT_WINDOW_S),
        ("abc", DEFAULT_WINDOW_S),
        ("-1", DEFAULT_WINDOW_S),
        ("nan", DEFAULT_WINDOW_S),
        ("10", 10.0),
        ("0", 0.0),
    ],
)
def test_default_window_s_env_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("NOVA_ATTENTION_WINDOW_S", raising=False)
    else:
        monkeypatch.setenv("NOVA_ATTENTION_WINDOW_S", raw)
    assert default_window_s() == expected
    assert DEFAULT_WINDOW_S == 45.0


def test_env_window_is_resolved_at_construction(monkeypatch, clock):
    monkeypatch.setenv("NOVA_ATTENTION_WINDOW_S", "10")
    attention = AttentionState(clock=clock)
    assert attention.window_s == 10.0
    attention.note_transcript("nova hi")
    clock.advance(9.0)
    assert attention.warm is True
    clock.advance(2.0)
    assert attention.warm is False


def test_zero_window_is_always_cold(clock):
    attention = AttentionState(clock=clock, window_s=0.0)
    assert attention.note_transcript("nova hi") == "opened"
    assert attention.warm is False


def test_names_override(clock):
    attention = AttentionState(clock=clock, names=("hal",))
    assert attention.note_transcript("nova hi") == "ignored"
    assert attention.note_transcript("hal, open the door") == "opened"
