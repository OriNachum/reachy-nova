"""Integration seams added by t9 — speech lane wiring in main.py.

Covers: the Sonic feed is byte-identical with the lane on or off (h8/c8),
speech events publish through MQTT and inject with direction correlation,
and the per-stage sensory log trace (c1/c3 wiring, h10 discipline).
"""

import logging
import time

import numpy as np
import pytest

from reachy_nova.main import (
    DIRECTION_CORRELATION_WINDOW,
    forward_mic_audio,
    make_on_speech,
)


class SpySonic:
    def __init__(self):
        self.fed = []
        self.injected = []

    def feed_audio(self, audio):
        self.fed.append(np.copy(audio))

    def inject_text(self, text, force=False):
        self.injected.append(text)


class SpyLane:
    def __init__(self, fail=False):
        self.fed = []
        self.fail = fail

    def feed(self, audio):
        if self.fail:
            raise RuntimeError("lane exploded")
        self.fed.append(np.copy(audio))


class SpyMQTT:
    def __init__(self):
        self.events = []

    def publish_event(self, source, event_type, payload):
        self.events.append((source, event_type, payload))


def _chunks():
    rng = np.random.default_rng(42)
    return [rng.standard_normal(160).astype(np.float32) for _ in range(20)]


class TestSonicFeedByteIdentical:
    def test_identical_with_lane_on_vs_off(self):
        chunks = _chunks()
        without, with_lane = SpySonic(), SpySonic()
        lane = SpyLane()
        for c in chunks:
            forward_mic_audio(c, without, None)
        for c in chunks:
            forward_mic_audio(c, with_lane, lane)
        assert len(without.fed) == len(with_lane.fed) == len(chunks)
        for a, b in zip(without.fed, with_lane.fed):
            assert a.dtype == b.dtype
            assert np.array_equal(a, b)

    def test_lane_receives_the_same_chunks(self):
        chunks = _chunks()
        sonic, lane = SpySonic(), SpyLane()
        for c in chunks:
            forward_mic_audio(c, sonic, lane)
        assert len(lane.fed) == len(chunks)
        for a, b in zip(sonic.fed, lane.fed):
            assert np.array_equal(a, b)

    def test_lane_failure_never_affects_sonic(self):
        chunks = _chunks()
        sonic = SpySonic()
        lane = SpyLane(fail=True)
        for c in chunks:
            forward_mic_audio(c, sonic, lane)
        assert len(sonic.fed) == len(chunks)

    def test_sonic_is_fed_before_the_lane(self):
        order = []

        class OrderSonic(SpySonic):
            def feed_audio(self, audio):
                order.append("sonic")

        class OrderLane(SpyLane):
            def feed(self, audio):
                order.append("lane")

        forward_mic_audio(np.zeros(160, dtype=np.float32), OrderSonic(), OrderLane())
        assert order == ["sonic", "lane"]


PAYLOAD = {
    "clip_path": "/tmp/clips/000001.wav",
    "transcript": "hello nova",
    "duration_seconds": 3.2,
    "onset_ts": 1234.567,
}


def _handler(mqtt, sonic, last_direction, awake=True):
    return make_on_speech(
        mqtt, sonic, t0=0.0, last_direction=last_direction,
        is_awake=lambda: awake,
    )


class TestOnSpeech:
    def test_publishes_speech_detected_via_mqtt_only(self):
        mqtt, sonic = SpyMQTT(), SpySonic()
        _handler(mqtt, sonic, {"time": 0.0, "label": ""})(dict(PAYLOAD))
        assert len(mqtt.events) == 1
        source, event_type, payload = mqtt.events[0]
        assert (source, event_type) == ("speech", "speech_detected")
        assert payload["clip_path"] == PAYLOAD["clip_path"]
        assert payload["transcript"] == "hello nova"
        assert payload["duration_seconds"] == pytest.approx(3.2)
        assert payload["onset_ts"] == pytest.approx(1234.567)

    def test_fresh_direction_is_carried_and_spoken(self):
        mqtt, sonic = SpyMQTT(), SpySonic()
        last = {"time": time.time(), "label": "left"}
        _handler(mqtt, sonic, last)(dict(PAYLOAD))
        assert mqtt.events[0][2]["direction"] == "left"
        assert len(sonic.injected) == 1
        assert "from your left" in sonic.injected[0]

    def test_stale_direction_is_dropped(self):
        mqtt, sonic = SpyMQTT(), SpySonic()
        last = {"time": time.time() - DIRECTION_CORRELATION_WINDOW - 1.0, "label": "left"}
        _handler(mqtt, sonic, last)(dict(PAYLOAD))
        assert mqtt.events[0][2]["direction"] is None
        assert len(sonic.injected) == 1
        assert "left" not in sonic.injected[0]

    def test_asleep_suppresses_inject_but_still_publishes(self, caplog):
        mqtt, sonic = SpyMQTT(), SpySonic()
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            _handler(mqtt, sonic, {"time": 0.0, "label": ""}, awake=False)(dict(PAYLOAD))
        assert len(mqtt.events) == 1
        assert sonic.injected == []
        assert any("suppressed reason=sleeping" in r.message for r in caplog.records)

    def test_all_four_stages_logged_in_one_pass(self, caplog):
        """The simulated pass: capture -> vad -> event -> inject, one event id."""
        mqtt, sonic = SpyMQTT(), SpySonic()
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            _handler(mqtt, sonic, {"time": 0.0, "label": ""})(dict(PAYLOAD))
        stages = []
        event_ids = set()
        for record in caplog.records:
            msg = record.message
            if "[SENSE stage=" in msg and "source=speech" in msg:
                stages.append(msg.split("stage=")[1].split(" ")[0])
                event_ids.add(msg.split("event=")[1].split("]")[0])
        assert stages == ["capture", "vad", "event", "inject"]
        assert len(event_ids) == 1
