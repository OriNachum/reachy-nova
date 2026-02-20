"""Vague human-like time formatting for Reachy Nova.

All functions use UTC internally. Model-facing expressions are deliberately
imprecise — Nova should experience time the way a conscious being would,
not as a clock but as a felt sense of "how long ago" something happened.
"""

from datetime import datetime, timezone

# Day-of-week and time-of-day labels
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_TIME_BANDS = [
    (5, "early in the morning"),
    (9, "in the morning"),
    (12, "around midday"),
    (14, "in the early afternoon"),
    (17, "in the afternoon"),
    (20, "in the evening"),
    (22, "late in the evening"),
    (24, "late at night"),
]


def _time_of_day(hour: int) -> str:
    for threshold, label in _TIME_BANDS:
        if hour < threshold:
            return label
    return "late at night"


def utc_now_vague() -> str:
    """Vague human-like time-of-day string, e.g. 'It's a Tuesday afternoon.'"""
    now = datetime.now(timezone.utc)
    day = _DAYS[now.weekday()]
    band = _time_of_day(now.hour)
    return f"It's {day}, {band}."


def utc_now_precise() -> str:
    """Precise UTC string for the one absolute anchor at startup."""
    now = datetime.now(timezone.utc)
    return now.strftime("%A, %b %d, %Y, %-I:%M %p UTC")


def relative_vague(timestamp: float) -> str:
    """Vague human-like age from a Unix timestamp to now."""
    now = datetime.now(timezone.utc).timestamp()
    delta = max(0.0, now - timestamp)

    if delta < 30:
        return "just now"
    if delta < 300:
        return "a moment ago"
    if delta < 1800:
        return "a little while ago"
    if delta < 7200:
        return "a while ago"
    if delta < 21600:
        return "a few hours ago"
    if delta < 86400:
        return "earlier today"
    if delta < 172800:
        return "yesterday"
    if delta < 604800:
        return "a few days ago"
    if delta < 1209600:
        return "last week"
    if delta < 2592000:
        return "a couple of weeks ago"
    return "a long time ago"


def session_vague(session_start: float) -> str:
    """Vague session-elapsed marker from session start time."""
    now = datetime.now(timezone.utc).timestamp()
    elapsed = max(0.0, now - session_start)

    if elapsed < 30:
        return "just now"
    if elapsed < 300:
        return "a moment ago"
    if elapsed < 1800:
        return "a little while into the conversation"
    if elapsed < 3600:
        return "some time into the conversation"
    return "well into the conversation"


def format_event(text: str, session_start: float) -> str:
    """Wrap event text with a vague temporal prefix."""
    marker = session_vague(session_start)
    return f"[{marker}] {text}"


def format_elapsed_vague(elapsed: float) -> str:
    """Format a raw elapsed-seconds value as vague duration.

    Used for restart context ('you were away for a little while').
    """
    if elapsed < 30:
        return "a brief moment"
    if elapsed < 300:
        return "a little while"
    if elapsed < 1800:
        return "a while"
    if elapsed < 7200:
        return "a couple of hours"
    if elapsed < 21600:
        return "a few hours"
    if elapsed < 86400:
        return "most of the day"
    if elapsed < 172800:
        return "about a day"
    if elapsed < 604800:
        return "a few days"
    return "a long time"
