"""The composition root wires the graph the harness actually runs (t10).

Every test runs against an ISOLATED ``REACHY_STATE_DIR`` (the autouse fixture
below): ``build_app()``'s one side effect — ensuring the standing
``nova-face-noticed`` overlay rule — must never touch the developer's real
``~/.local/state/reachy``. The default fixture state dir does not exist, so
the ensure step degrades to its named component-absent line; the tests that
exercise the install path create the behavior dir themselves.
"""

import json
import queue
import threading
from types import SimpleNamespace

import pytest

from reachy_nova.harness import app, rules_overlay, statedir
from reachy_nova.harness.bus import NovaBus
from reachy_nova.harness.gate import EchoGate
from reachy_nova.harness.hearing import TeeHearing
from reachy_nova.harness.sense_history import SenseHistory
from reachy_nova.harness.speaking import SonicSpeaker
from reachy_nova.harness.tools import TOOL_SPECS, IntentTools
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


def _fake_msg(topic: str, payload: dict) -> SimpleNamespace:
    """A stand-in for paho's MQTTMessage, matching test_harness_bus.py's."""
    return SimpleNamespace(topic=topic, payload=json.dumps(payload).encode())


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


def test_default_build_is_the_core_components_with_named_absences(caplog):
    """Sonic/speaker/hearing, the network leg (t5), and the bus.

    The network leg is ALWAYS present — unlike the optional legs it is cheap
    and local (a /proc read, an ``ip addr`` call, a state-dir file), so there
    is nothing to gate it on and nothing to degrade over.
    """
    with caplog.at_level("INFO", logger="nova.sensory"):
        components = _build()
    names = [type(c).__name__ for c in components]
    assert names == [
        "NovaSonic",
        "SonicSpeaker",
        "TeeHearing",
        # IntentTools rides the component list only for its quiet-expiry tick
        # (t12) — it restores the runtime's 'speak' inhibition when a timed
        # quiet runs out rather than being ended by hand.
        "IntentTools",
        "NetworkReactor",
        "NetworkUnit",
        "NovaBus",
    ]
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
    assert len(components) == 8  # core 3 + tools + network leg (2) + bus + browser


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
    assert len(components) == 8  # core 3 + tools + network leg (2) + bus + vision


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
    assert names == [
        "NovaSonic",
        "SonicSpeaker",
        "TeeHearing",
        "IntentTools",
        "NetworkReactor",
        "NetworkUnit",
    ]
    lines = _messages(caplog)
    assert any("component absent name=bus" in m for m in lines)
    assert any("component absent name=vision reason=no-bus" in m for m in lines)


# --------------------------------------------------------------------------- #
# recall_senses / SenseHistory wiring (t8)                                    #
# --------------------------------------------------------------------------- #


def test_system_prompt_tells_nova_to_recall_senses():
    assert "recall_senses" in app.HARNESS_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# System-prompt rewrite for body cues (t9)                                    #
# --------------------------------------------------------------------------- #


def test_system_prompt_never_narrates_body_mechanics():
    assert "never describe" in app.HARNESS_SYSTEM_PROMPT
    assert "mention what you did" not in app.HARNESS_SYSTEM_PROMPT


def test_system_prompt_covers_unknown_kind_errors():
    assert "unknown-kind" in app.HARNESS_SYSTEM_PROMPT


def test_system_prompt_names_the_gaze_lock_tools():
    assert "lock_face" in app.HARNESS_SYSTEM_PROMPT
    assert "release_face" in app.HARNESS_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Daemon-client wiring (t9) — the shared DaemonClient + restore_volume()      #
# --------------------------------------------------------------------------- #


def test_intent_tools_receive_a_daemon_client_on_the_same_base_url(monkeypatch):
    from reachy_nova.harness.daemon_client import DaemonClient

    captured: dict = {}
    real_intent_tools = IntentTools

    def _spy(*args, **kwargs):
        captured["daemon_client"] = kwargs.get("daemon_client")
        return real_intent_tools(*args, **kwargs)

    monkeypatch.setattr(app, "IntentTools", _spy)

    components = _build()
    speaker = components[1]

    assert isinstance(captured["daemon_client"], DaemonClient)
    assert captured["daemon_client"].base_url == speaker.base_url


def test_build_app_calls_restore_volume_on_the_persisted_path_and_client(monkeypatch):
    from reachy_nova.harness.daemon_client import DaemonClient

    calls = []
    monkeypatch.setattr(app, "restore_volume", lambda path, client: calls.append((path, client)))
    _build()
    assert len(calls) == 1
    path, client = calls[0]
    assert path == statedir.volume_state_path()
    assert isinstance(client, DaemonClient)


def test_build_app_never_raises_when_restore_volume_fails(monkeypatch, caplog):
    def _boom(*_a, **_k):
        raise RuntimeError("daemon unreachable")

    monkeypatch.setattr(app, "restore_volume", _boom)
    with caplog.at_level("INFO", logger="nova.sensory"):
        _build()  # must not raise
    assert any(
        "component absent name=volume reason=daemon unreachable" in m
        for m in _messages(caplog)
    )


def test_bus_and_intents_share_the_same_sense_history(monkeypatch):
    captured: dict = {}
    real_intent_tools = IntentTools

    def _spy(*args, **kwargs):
        captured["history"] = kwargs.get("history")
        return real_intent_tools(*args, **kwargs)

    monkeypatch.setattr(app, "IntentTools", _spy)

    components = _build()
    bus_component = next(c for c in components if isinstance(c, NovaBus))

    assert isinstance(captured["history"], SenseHistory)
    assert bus_component.history is captured["history"]


def test_bus_and_intents_share_the_same_lock_state(monkeypatch):
    """t13: the bus's on_event tap and IntentTools' lock_face/release_face
    tools update the SAME LockState — otherwise a lock-released event the
    runtime published would correct a belief nothing else reads."""
    captured: dict = {}
    real_intent_tools = IntentTools

    def _spy(*args, **kwargs):
        captured["lock_state"] = kwargs.get("lock_state")
        return real_intent_tools(*args, **kwargs)

    monkeypatch.setattr(app, "IntentTools", _spy)

    components = _build()
    bus_component = next(c for c in components if isinstance(c, NovaBus))

    lock_state = captured["lock_state"]
    assert lock_state is not None

    bus_component.on_message(
        None,
        None,
        _fake_msg(
            "reachy/events/motion/lock-released",
            {"t": "motion", "id": "p1", "reason": "max-hold"},
        ),
    )
    assert lock_state.locked is False


def test_bus_history_is_wired_even_when_bus_construction_is_absent(monkeypatch):
    """The history is cheap/local and always built, regardless of the bus."""
    import reachy_nova.harness.bus as bus_module

    def _boom(*_a, **_k):
        raise RuntimeError("no paho on this box")

    monkeypatch.setattr(bus_module, "NovaBus", _boom)

    captured: dict = {}
    real_intent_tools = IntentTools

    def _spy(*args, **kwargs):
        captured["history"] = kwargs.get("history")
        return real_intent_tools(*args, **kwargs)

    monkeypatch.setattr(app, "IntentTools", _spy)

    _build()
    assert isinstance(captured["history"], SenseHistory)


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


# --------------------------------------------------------------------------- #
# FORGE_WRITER normalization — one resolver, one place (qodo comment 3812045200) #
# --------------------------------------------------------------------------- #


def test_app_gates_the_kiro_writer_through_resolve_writer():
    """build_app() must call the shared resolve_writer(), not its own inline
    env read — this is what keeps its gate agreeing with SkillForge's own
    dispatch on what a writer value means."""
    import inspect

    source = inspect.getsource(app)
    assert "resolve_writer" in source
    assert 'os.environ.get("FORGE_WRITER"' not in source


@pytest.mark.parametrize("writer_value", ["kiro", "KIRO", "Kiro", " kiro ", "KIRO "])
def test_kiro_writer_gate_accepts_every_normalized_spelling(monkeypatch, writer_value):
    """Every spelling resolve_writer() normalizes to "kiro" enables the same
    kiro-writer components build_app() wires under the exact "kiro" value —
    this is the composition-level half of qodo comment 3812045200: before the
    fix, "KIRO" exposed these components while SkillForge rejected every
    request through them as an unknown writer."""
    monkeypatch.setenv("FORGE_WRITER", writer_value)
    components = _build()
    names = [type(c).__name__ for c in components]
    assert "KiroSessionUnit" in names


@pytest.mark.parametrize("writer_value", ["http", "HTTP", "", "carrier-pigeon"])
def test_non_kiro_writer_values_leave_the_kiro_writer_absent(monkeypatch, writer_value, caplog):
    if writer_value:
        monkeypatch.setenv("FORGE_WRITER", writer_value)
    else:
        monkeypatch.delenv("FORGE_WRITER", raising=False)
    with caplog.at_level("INFO", logger="nova.sensory"):
        components = _build()
    names = [type(c).__name__ for c in components]
    assert "KiroSessionUnit" not in names
    assert any(
        "component absent name=kiro-writer reason=writer-http" in m for m in _messages(caplog)
    )


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


def test_l5_the_prompt_tells_nova_when_to_release_face_and_recall_senses():
    """Live finding L5: on the robot Nova never called release_face on "you
    can look away" nor recall_senses on "why did you do that?"."""
    prompt = app.HARNESS_SYSTEM_PROMPT
    assert "stop following or look away, call release_face" in prompt
    assert "call recall_senses before answering" in prompt
