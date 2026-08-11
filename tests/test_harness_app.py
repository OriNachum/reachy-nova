"""The composition root wires the graph the harness actually runs (t10).

Every test runs against an ISOLATED ``REACHY_STATE_DIR`` (the autouse fixture
below): ``build_app()``'s one side effect — ensuring the standing
``nova-face-noticed`` overlay rule — must never touch the developer's real
``~/.local/state/reachy``. The default fixture state dir does not exist, so
the ensure step degrades to its named component-absent line; the tests that
exercise the install path create the behavior dir themselves.
"""

import queue
import threading

import pytest

from reachy_nova.harness import app, rules_overlay, statedir
from reachy_nova.harness.bus import NovaBus
from reachy_nova.harness.gate import EchoGate
from reachy_nova.harness.hearing import TeeHearing
from reachy_nova.harness.speaking import SonicSpeaker
from reachy_nova.harness.tools import TOOL_SPECS
from reachy_nova.harness.vision_leg import VisionLeg
from reachy_nova.nova_browser import NovaBrowser
from reachy_nova.nova_omni import NovaOmni

import numpy as np


@pytest.fixture(autouse=True)
def _isolated_composition(monkeypatch, tmp_path):
    """Isolate the state dir and neutralise ambient feature flags."""
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path / "reachy-state"))
    monkeypatch.delenv("NOVA_ACT_ENABLED", raising=False)
    monkeypatch.delenv("NOVA_OMNI_MODEL_ID", raising=False)
    yield


def _build():
    return app.build_app()


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


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


# --------------------------------------------------------------------------- #
# Echo-gate policy is wired explicitly (t2)                                    #
# --------------------------------------------------------------------------- #


def test_echo_gate_policy_flows_through_resolve_policy_explicitly(monkeypatch):
    """build_app passes resolve_policy() itself — not TeeHearing's own env read."""
    monkeypatch.setattr(app, "resolve_policy", lambda explicit=None: "half-duplex")
    hearing = _build()[2]
    assert hearing.echo_gate_policy == "half-duplex"
    assert hearing.suppresses_while_speaking


def test_echo_gate_policy_defaults_off():
    hearing = _build()[2]
    assert hearing.echo_gate_policy == "off"
    assert not hearing.suppresses_while_speaking


# --------------------------------------------------------------------------- #
# Flag-gated legs: browser (NOVA_ACT_ENABLED) + vision (NOVA_OMNI_MODEL_ID)   #
# --------------------------------------------------------------------------- #


def test_default_build_is_four_components_with_named_absences(caplog):
    with caplog.at_level("INFO", logger="nova.sensory"):
        components = _build()
    names = [type(c).__name__ for c in components]
    assert names == ["NovaSonic", "SonicSpeaker", "TeeHearing", "NovaBus"]
    lines = _messages(caplog)
    assert any("component absent name=browser reason=act-disabled" in m for m in lines)
    assert any("component absent name=vision reason=omni-model-unset" in m for m in lines)


def test_act_enabled_adds_a_supervised_browser_wired_for_progress(monkeypatch):
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    components = _build()
    sonic = components[0]
    adapters = [c for c in components if isinstance(c, app.BrowserComponent)]
    assert len(adapters) == 1
    adapter = adapters[0]
    assert isinstance(adapter.browser, NovaBrowser)
    # IntentTools wired the browser handle AND its progress narration.
    assert adapter.browser.on_progress == sonic.inject_text
    assert hasattr(adapter, "start") and hasattr(adapter, "stop")
    assert len(components) == 5


def test_browser_component_start_and_stop_never_raise_when_disabled():
    """The adapter's lifecycle is total even around a flag-off NovaBrowser."""
    adapter = app.BrowserComponent(NovaBrowser())
    adapter.start(threading.Event())  # flag off: NovaBrowser.start is a no-op
    adapter.stop()


def test_omni_model_set_adds_the_vision_leg_wired_to_bus_and_sonic(monkeypatch):
    monkeypatch.setenv("NOVA_OMNI_MODEL_ID", "us.amazon.nova-2-omni-v1:0")
    components = _build()
    sonic = components[0]
    bus_component = next(c for c in components if isinstance(c, NovaBus))
    legs = [c for c in components if isinstance(c, VisionLeg)]
    assert len(legs) == 1
    leg = legs[0]
    # clip state comes from the bus's retained-topic cache…
    assert leg._get_clip_state.__self__ is bus_component
    assert leg._get_clip_state.__func__ is NovaBus.clip_state
    # …the understanding is NovaOmni's, on the configured model…
    assert isinstance(leg._understand.__self__, NovaOmni)
    assert leg._understand.__self__.omni_model_id == "us.amazon.nova-2-omni-v1:0"
    # …and the one answer goes to Sonic's guarded inject.
    assert leg._on_answer == sonic.inject_text
    assert len(components) == 5


def test_vision_leg_degrades_to_absent_when_the_bus_cannot_build(monkeypatch, caplog):
    """No bus means no clip state: the vision leg is absent BY NAME, not broken."""
    import reachy_nova.harness.bus as bus_module

    monkeypatch.setenv("NOVA_OMNI_MODEL_ID", "us.amazon.nova-2-omni-v1:0")

    def _boom(*_a, **_k):
        raise RuntimeError("no paho on this box")

    monkeypatch.setattr(bus_module, "NovaBus", _boom)
    with caplog.at_level("INFO", logger="nova.sensory"):
        components = _build()
    names = [type(c).__name__ for c in components]
    assert names == ["NovaSonic", "SonicSpeaker", "TeeHearing"]
    lines = _messages(caplog)
    assert any("component absent name=bus" in m for m in lines)
    assert any("component absent name=vision reason=no-bus" in m for m in lines)


# --------------------------------------------------------------------------- #
# The standing face rule (t10 — the face cue crosses the bus only through it)  #
# --------------------------------------------------------------------------- #


def test_face_rule_matches_the_engine_grammar():
    """The rule this harness installs must be one the engine's schema accepts."""
    entry = rules_overlay.validate_rule(app.FACE_RULE, kind="react", require_prefix=True)
    assert entry["id"] == "nova-face-noticed"
    assert entry["when"] == {"field": "face", "op": "is_true"}
    assert entry["run"] == "nod"
    # duration_s is load-bearing: the engine refuses a looping behavior with no
    # duration (seen live 2026-08-11 — the whole overlay reload was rejected).
    assert entry["duration_s"] == 2.0
    assert entry["cooldown_s"] == 30.0  # the runtime's own face re-announce cooldown


def test_build_app_runs_the_ensure_face_rule_step(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "ensure_face_rule", lambda **kw: calls.append(kw) or True)
    _build()
    assert len(calls) == 1


def test_ensure_face_rule_degrades_absent_without_a_state_dir(caplog):
    with caplog.at_level("INFO", logger="nova.sensory"):
        assert app.ensure_face_rule(reload_timeout=0.05) is False
    assert any(
        "component absent name=face-rule reason=statedir-absent" in m
        for m in _messages(caplog)
    )
    assert not statedir.rules_overlay_path().exists()  # no litter on a runtime-less box


def test_ensure_face_rule_installs_the_standing_rule_and_spools_a_reload(caplog):
    statedir.behavior_dir().mkdir(parents=True)
    with caplog.at_level("INFO", logger="nova.sensory"):
        assert app.ensure_face_rule(reload_timeout=0.05) is True

    assert "nova-face-noticed" in rules_overlay.list_rules()
    overlay = statedir.rules_overlay_path().read_text(encoding="utf-8")
    assert 'field = "face"' in overlay
    assert 'op = "is_true"' in overlay
    assert "cooldown_s = 30.0" in overlay
    # The engine is asked to reload the changed overlay.
    assert list(statedir.reload_commands_dir().glob("*.json"))
    assert any(
        "standing rule ensured id=nova-face-noticed changed=True" in m
        for m in _messages(caplog)
    )


def test_ensure_face_rule_is_idempotent_and_reloads_nothing_the_second_time(caplog):
    statedir.behavior_dir().mkdir(parents=True)
    assert app.ensure_face_rule(reload_timeout=0.05) is True
    spooled = list(statedir.reload_commands_dir().glob("*.json"))
    with caplog.at_level("INFO", logger="nova.sensory"):
        assert app.ensure_face_rule(reload_timeout=0.05) is True
    assert list(statedir.reload_commands_dir().glob("*.json")) == spooled
    assert any("changed=False" in m for m in _messages(caplog))


def test_ensure_face_rule_preserves_an_operator_overlay(caplog):
    """The upsert merges into the nova block; operator bytes are untouched."""
    statedir.behavior_dir().mkdir(parents=True)
    operator = '[[react]]\nid = "my-rule"\nwhen = { field = "pat", op = "is_true" }\nrun = "nod"\n'
    statedir.rules_overlay_path().write_text(operator, encoding="utf-8")
    assert app.ensure_face_rule(reload_timeout=0.05) is True
    overlay = statedir.rules_overlay_path().read_text(encoding="utf-8")
    assert overlay.startswith(operator.rstrip("\n"))
    assert 'id = "my-rule"' in overlay
    assert 'id = "nova-face-noticed"' in overlay


def test_ensure_face_rule_never_raises_on_an_unwritable_overlay(monkeypatch, caplog):
    statedir.behavior_dir().mkdir(parents=True)

    def _refuse(*_a, **_k):
        raise rules_overlay.RuleRefused("synthetic refusal")

    monkeypatch.setattr(app, "upsert_rule", _refuse)
    with caplog.at_level("INFO", logger="nova.sensory"):
        assert app.ensure_face_rule(reload_timeout=0.05) is False
    assert any(
        "component absent name=face-rule reason=synthetic refusal" in m
        for m in _messages(caplog)
    )


# --------------------------------------------------------------------------- #
# Playback-aware barge-in (user speech while the gate window is armed)        #
# --------------------------------------------------------------------------- #


def test_user_transcript_during_playback_preempts_the_speaker():
    """Talking over the robot's audible voice cuts it — not just its queue."""
    components = app.build_app()
    sonic = components[0]
    speaker = components[1]
    stops = []
    speaker.stopper = stops.append
    speaker.gate.arm_for(10.0)  # a playback window is running
    sonic.on_transcript("USER", "stop talking")
    assert stops == [speaker.base_url]
    assert not speaker.gate.active


def test_user_transcript_with_no_playback_window_does_not_preempt():
    components = app.build_app()
    sonic = components[0]
    speaker = components[1]
    stops = []
    speaker.stopper = stops.append
    sonic.on_transcript("USER", "hello there")
    assert stops == []


def test_browser_result_callback_reaches_the_conversation(monkeypatch):
    """The answer from a finished browse must inject — the tool result itself
    is only the 'queued' acknowledgment."""
    monkeypatch.setenv("NOVA_ACT_ENABLED", "1")
    components = app.build_app()
    sonic = components[0]
    injected = []
    sonic.inject_text = injected.append
    browser = next(c for c in components if type(c).__name__ == "BrowserComponent")
    inner = getattr(browser, "browser", None) or getattr(browser, "_browser")
    assert inner.on_result is not None
    inner.on_result("The answer is 42.")
    assert len(injected) == 1 and "The answer is 42." in injected[0]
