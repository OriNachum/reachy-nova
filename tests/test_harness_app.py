"""The composition root wires the graph the harness actually runs."""

import queue

import numpy as np

from reachy_nova.harness import app
from reachy_nova.harness.gate import EchoGate
from reachy_nova.harness.hearing import TeeHearing
from reachy_nova.harness.speaking import SonicSpeaker
from reachy_nova.harness.tools import TOOL_SPECS


def _build():
    return app.build_app()


def test_build_app_returns_sonic_speaker_hearing_in_start_order():
    components = _build()
    names = [type(c).__name__ for c in components]
    assert names[:3] == ["NovaSonic", "SonicSpeaker", "TeeHearing"]


def test_speak_leg_wiring():
    sonic, speaker, _hearing = _build()[:3]
    assert sonic.on_audio_output == speaker.on_audio_chunk
    assert sonic.on_interruption == speaker.preempt


def test_hear_leg_feeds_sonic_and_shares_the_gate():
    sonic, speaker, hearing = _build()[:3]
    assert isinstance(hearing, TeeHearing)
    assert hearing.feed == sonic.feed_audio
    assert hearing.gate is speaker.gate
    assert isinstance(speaker.gate, EchoGate)


def test_sonic_carries_the_intent_tool_specs():
    sonic = _build()[0]
    assert sonic.tools == TOOL_SPECS


def test_components_expose_the_supervisor_lifecycle():
    for component in _build():
        assert hasattr(component, "start")
        assert hasattr(component, "stop")


def test_speaker_preempt_clears_buffer_queue_and_gate():
    gate = EchoGate()
    speaker = SonicSpeaker(gate=gate, poster=lambda *a, **k: None)
    speaker.on_state_change("speaking")
    speaker.on_audio_chunk(np.zeros(2400, dtype=np.float32))
    speaker._queue.put_nowait(np.zeros(2400, dtype=np.float32))
    gate.arm_for(5.0)

    speaker.preempt()

    assert speaker._buffer_samples == 0
    with_queue_empty = False
    try:
        speaker._queue.get_nowait()
    except queue.Empty:
        with_queue_empty = True
    assert with_queue_empty
    assert not gate.active
