"""Tests for the rate-limited ``audio_direction`` event.

``TrackingManager.update_doa`` (reachy_nova/tracking.py) fires an
``audio_direction`` event via ``_fire_event`` whenever it processes DoA
speech, carrying ``{bearing_deg, label, speech_active}``. Coverage here
targets the two acceptance criteria for this task:

1. bearing-to-label mapping, per the XMOS convention baked into
   ``update_doa`` (``yaw_rad = (pi/2) - angle_rad``: angle 0 = left,
   pi/2 = front, pi = right; the label thresholds are +-20deg off
   dead-ahead).
2. rate limiting: at most one event per 2s window, unless the bearing
   moves 15+ degrees between calls, in which case the window resets.

Head-tracking behavior (``doa_yaw_target``) must keep updating on every
call regardless of whether the event itself is rate-limited away — the
event is purely additive.
"""

import math
import types

import pytest

from reachy_nova import tracking
from reachy_nova.tracking import TrackingManager


class _FakeClock:
    """A settable fake clock, swapped in for ``tracking.time.time``."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    """Replace ``time`` inside the tracking module with a controllable fake.

    Scoped to ``reachy_nova.tracking`` only (monkeypatch rebinds the name
    in that module's namespace) so it never touches the real global
    ``time`` module used elsewhere.
    """
    clock = _FakeClock()
    monkeypatch.setattr(tracking, "time", types.SimpleNamespace(time=clock.time))
    return clock


@pytest.fixture
def recorded_events():
    events: list[tuple[str, dict]] = []

    def on_event(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    return events, on_event


@pytest.fixture
def manager(recorded_events):
    _, on_event = recorded_events
    return TrackingManager(on_event=on_event)


def _audio_direction_events(events):
    return [(etype, data) for etype, data in events if etype == "audio_direction"]


class TestBearingToLabelMapping:
    """XMOS: angle 0 = left, pi/2 = front, pi = right (via update_doa)."""

    def test_angle_zero_maps_to_left(self, manager, recorded_events):
        events, _ = recorded_events
        manager.update_doa((0.0, True))

        direction_events = _audio_direction_events(events)
        assert len(direction_events) == 1
        _, data = direction_events[0]
        assert data["label"] == "left"
        # angle=0 -> yaw=90deg, clamped to MAX_YAW (45deg) by update_doa's
        # existing head-tracking clip -- bearing_deg reports the same
        # post-clip value the head actually targets.
        assert data["bearing_deg"] == pytest.approx(45.0)

    def test_angle_pi_over_two_maps_to_front(self, manager, recorded_events):
        events, _ = recorded_events
        manager.update_doa((math.pi / 2, True))

        _, data = _audio_direction_events(events)[0]
        assert data["label"] == "front"
        assert data["bearing_deg"] == pytest.approx(0.0)

    def test_angle_pi_maps_to_right(self, manager, recorded_events):
        events, _ = recorded_events
        manager.update_doa((math.pi, True))

        _, data = _audio_direction_events(events)[0]
        assert data["label"] == "right"
        # angle=pi -> yaw=-90deg, clamped to -MAX_YAW (-45deg), same reasoning
        # as the angle=0 case above.
        assert data["bearing_deg"] == pytest.approx(-45.0)

    def test_just_inside_front_threshold_stays_front(self, manager, recorded_events):
        # 19deg off dead-ahead is within the +-20deg front band.
        events, _ = recorded_events
        angle = (math.pi / 2) - math.radians(19.0)
        manager.update_doa((angle, True))

        _, data = _audio_direction_events(events)[0]
        assert data["label"] == "front"

    def test_just_outside_front_threshold_is_left(self, manager, recorded_events):
        # 21deg off dead-ahead (toward angle=0/left) crosses the +20deg band.
        events, _ = recorded_events
        angle = (math.pi / 2) - math.radians(21.0)
        manager.update_doa((angle, True))

        _, data = _audio_direction_events(events)[0]
        assert data["label"] == "left"

    def test_speech_active_flag_is_true_while_speaking(self, manager, recorded_events):
        events, _ = recorded_events
        manager.update_doa((math.pi / 2, True))

        _, data = _audio_direction_events(events)[0]
        assert data["speech_active"] is True

    def test_no_event_when_doa_result_is_none(self, manager, recorded_events):
        events, _ = recorded_events
        manager.update_doa(None)

        assert _audio_direction_events(events) == []

    def test_no_event_on_first_call_without_speech(self, manager, recorded_events):
        events, _ = recorded_events
        manager.update_doa((math.pi / 2, False))

        assert _audio_direction_events(events) == []


class TestAudioDirectionLabelHelper:
    """Direct coverage of the label-mapping helper, independent of update_doa."""

    def test_boundary_values(self):
        assert tracking._audio_direction_label(20.01) == "left"
        assert tracking._audio_direction_label(20.0) == "front"
        assert tracking._audio_direction_label(-20.0) == "front"
        assert tracking._audio_direction_label(-20.01) == "right"
        assert tracking._audio_direction_label(0.0) == "front"


class TestRateLimit:
    def test_continuous_speech_same_bearing_yields_exactly_one_event(
        self, manager, recorded_events, fake_clock
    ):
        events, _ = recorded_events

        for _ in range(20):
            manager.update_doa((math.pi / 2, True))
            fake_clock.advance(0.05)  # 20 * 0.05s = 1s total, all inside the 2s window

        assert len(_audio_direction_events(events)) == 1

    def test_window_expiry_allows_a_new_event(self, manager, recorded_events, fake_clock):
        events, _ = recorded_events

        manager.update_doa((math.pi / 2, True))
        fake_clock.advance(2.1)
        manager.update_doa((math.pi / 2, True))

        assert len(_audio_direction_events(events)) == 2

    def test_window_boundary_at_exactly_two_seconds_allows_new_event(
        self, manager, recorded_events, fake_clock
    ):
        events, _ = recorded_events

        manager.update_doa((math.pi / 2, True))
        fake_clock.advance(2.0)
        manager.update_doa((math.pi / 2, True))

        assert len(_audio_direction_events(events)) == 2

    def test_large_bearing_jump_bypasses_the_rate_limit(self, manager, recorded_events, fake_clock):
        events, _ = recorded_events

        manager.update_doa((math.pi / 2, True))  # front, bearing_deg=0
        fake_clock.advance(0.1)
        manager.update_doa((0.0, True))  # left, bearing_deg=90 -- a 90deg jump

        assert len(_audio_direction_events(events)) == 2

    def test_small_bearing_move_within_window_stays_suppressed(
        self, manager, recorded_events, fake_clock
    ):
        events, _ = recorded_events

        manager.update_doa((math.pi / 2, True))  # bearing_deg=0
        fake_clock.advance(0.1)
        # 10deg move: under the 15deg jump threshold, still inside the 2s window.
        angle = (math.pi / 2) - math.radians(10.0)
        manager.update_doa((angle, True))

        assert len(_audio_direction_events(events)) == 1

    def test_bearing_jump_of_exactly_fifteen_degrees_fires(self, manager, recorded_events, fake_clock):
        events, _ = recorded_events

        manager.update_doa((math.pi / 2, True))  # bearing_deg=0
        fake_clock.advance(0.1)
        angle = (math.pi / 2) - math.radians(15.0)
        manager.update_doa((angle, True))  # bearing_deg=15, exactly at the threshold

        assert len(_audio_direction_events(events)) == 2

    def test_head_tracking_state_keeps_updating_when_event_is_suppressed(
        self, manager, recorded_events, fake_clock
    ):
        """Rate limiting only affects the event -- doa_yaw_target must still
        track every call, so head-tracking behavior is unchanged."""
        events, _ = recorded_events

        manager.update_doa((math.pi / 2, True))
        fake_clock.advance(0.1)
        angle = (math.pi / 2) - math.radians(10.0)
        manager.update_doa((angle, True))

        assert len(_audio_direction_events(events)) == 1
        assert manager.doa_yaw_target == pytest.approx(10.0)
