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

from reachy_nova.harness import app, mood as mood_module, persona, rules_overlay, statedir
from reachy_nova.harness.bus import NovaBus
from reachy_nova.harness.gate import EchoGate
from reachy_nova.harness.hearing import TeeHearing
from reachy_nova.harness.lite_reactor import LiteReactor
from reachy_nova.harness.memory_compactor import MemoryCompactor
from reachy_nova.harness.sense_history import SenseHistory
from reachy_nova.harness.speaking import SonicSpeaker
from reachy_nova.harness.tools import TOOL_SPECS, IntentTools
from reachy_nova.harness.vision_leg import VisionLeg
from reachy_nova.nova_browser import NovaBrowser
from reachy_nova.nova_omni import NovaOmni

import numpy as np


@pytest.fixture(autouse=True)
def _isolated_composition(monkeypatch, tmp_path):
    """Isolate the state dir and neutralise ambient feature flags.

    The four t14 switches are cleared too: they fail OPEN, so an ambient
    ``NOVA_MEMORY=0`` in the developer's shell would silently turn the
    default-build tests into off-path tests.
    """
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path / "reachy-state"))
    monkeypatch.delenv("NOVA_ACT_ENABLED", raising=False)
    monkeypatch.delenv("NOVA_OMNI_MODEL_ID", raising=False)
    monkeypatch.delenv("NOVA_CHUNKED_PLAYBACK", raising=False)
    monkeypatch.delenv("NOVA_LITE_REACTIONS", raising=False)
    monkeypatch.delenv("NOVA_MEMORY", raising=False)
    monkeypatch.delenv("NOVA_PERSONA_PATH", raising=False)
    yield


def _build():
    return app.build_app()


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def _sense_lines(caplog):
    """Only the ``nova.sensory`` records, in order — the journal as shipped."""
    return [r.getMessage() for r in caplog.records if r.name == "nova.sensory"]


def _ledger_records():
    """Every NDJSON line the composed ledger wrote, parsed, in file order."""
    path = statedir.ledger_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _boom(*_a, **_k):
    raise RuntimeError("synthetic construction failure")


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
        # The Lite reaction tier starts BEFORE the bus that feeds it cues…
        "LiteReactor",
        "NovaBus",
        # …and the memory compactor after it: its periodic Lite call is slow
        # and nothing waits on it, so it must never sit between cue and mouth.
        "MemoryCompactor",
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
    assert hasattr(adapter, "start")
    assert hasattr(adapter, "stop")
    # core 3 + tools + network leg (2) + lite reactor + bus + compactor + browser
    assert len(components) == 10


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
    # The leg's answer reaches Sonic through the brief-cue wrapper, not bare:
    # a raw 400-850 char Omni description became a 30 s monologue (2026-09-06).
    seen: list[tuple] = []
    sonic.inject_text = lambda text, force=False, sense_class=None: seen.append((text, sense_class))
    leg._on_answer("A person sits at a desk. " * 30)
    assert len(seen) == 1
    text, sense_class = seen[0]
    assert text.startswith("(you glance around: ")
    assert text.endswith(") (react briefly if at all)")
    assert sense_class == "vision"
    assert len(text) < 320
    # core 3 + tools + network leg (2) + lite reactor + bus + compactor + vision
    assert len(components) == 10


def test_vision_leg_degrades_to_absent_when_the_bus_cannot_build(monkeypatch, caplog):
    """No bus means no clip state: the vision leg is absent BY NAME, not broken."""
    import reachy_nova.harness.bus as bus_module

    monkeypatch.setenv("NOVA_OMNI_MODEL_ID", "us.amazon.nova-2-omni-v1:0")

    def _no_paho(*_a, **_k):
        raise RuntimeError("no paho on this box")

    monkeypatch.setattr(bus_module, "NovaBus", _no_paho)
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
        # The reactor and the compactor are independent of the bus: neither
        # needs a broker, so an absent bus costs them nothing.
        "LiteReactor",
        "MemoryCompactor",
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
    """t14 moved this rule from the prompt string into the persona file/default
    ("Never explain your own workings — 'reflex', 'rule', ... — unless someone
    asks you why"), so the ASSERTION moved with it. The property is unchanged:
    the composed prompt still forbids narrating the mechanism."""
    assert "Never explain your own workings" in app.HARNESS_SYSTEM_PROMPT
    assert "mention what you did" not in app.HARNESS_SYSTEM_PROMPT
    # …and it is the PERSONA half that carries it, never the tool guide.
    assert "Never explain your own workings" not in app.TOOL_GUIDE


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
    assert len(injected) == 1
    assert "The answer is 42." in injected[0]


def test_l5_the_prompt_tells_nova_when_to_release_face_and_recall_senses():
    """Live finding L5: on the robot Nova never called release_face on "you
    can look away" nor recall_senses on "why did you do that?"."""
    prompt = app.HARNESS_SYSTEM_PROMPT
    assert "stop following or look away, call release_face" in prompt
    assert "call recall_senses before answering" in prompt


# --------------------------------------------------------------------------- #
# t14 — persona + amy voice (c8/h5)                                            #
# --------------------------------------------------------------------------- #

#: The tool surface EXACTLY as it stood before t14 rewired the composition
#: root. h5 promises Sonic's toolConfiguration is byte-identical to today's,
#: and the persona move is the one change most likely to disturb it by
#: accident — so the names and the count are pinned here as a snapshot rather
#: than derived from the very list under test.
PRE_T14_TOOL_NAMES = [
    "run_behavior",
    "declare_goal",
    "set_mode",
    "set_inhibition",
    "goto",
    "create_rule",
    "browse",
    "enroll_face",
    "lock_face",
    "release_face",
    "look_at_face",
    "look_at_sound",
    "think",
    "forge",
    "use_skill",
    "author_rule",
    "raise_voice",
    "lower_voice",
    "set_voice_level",
    "recall_senses",
    "stay_silent",
    "end_silence",
]


def test_sonic_system_prompt_is_the_persona_text_plus_the_tool_guide():
    sonic = _build()[0]
    assert sonic.system_prompt == persona.read().text + "\n\n" + app.TOOL_GUIDE
    assert sonic.system_prompt.startswith(persona.read().text)
    assert sonic.system_prompt.endswith(app.TOOL_GUIDE)


def test_harness_system_prompt_default_is_the_embedded_persona_plus_the_guide():
    """The module-level default keeps working for importers (and for the
    prompt tests above), built from the EMBEDDED persona — build_app() builds
    the same shape from whatever persona.read() actually resolved."""
    assert app.HARNESS_SYSTEM_PROMPT == persona.DEFAULT_PERSONA + "\n\n" + app.TOOL_GUIDE
    assert app.build_system_prompt("WHO") == "WHO\n\n" + app.TOOL_GUIDE


def test_the_persona_file_in_force_is_what_sonic_is_given(monkeypatch, tmp_path):
    """Editing the persona file and restarting changes the next session's
    system prompt with no code change (c8) — including via NOVA_PERSONA_PATH,
    which build_app() passes through switches.persona_path."""
    custom = tmp_path / "another-nova.md"
    custom.write_text("You are Nova. You say very little, and mean all of it.\n", encoding="utf-8")
    monkeypatch.setenv("NOVA_PERSONA_PATH", str(custom))

    sonic = _build()[0]

    assert sonic.system_prompt.startswith("You are Nova. You say very little")
    assert sonic.system_prompt.endswith(app.TOOL_GUIDE)
    assert persona.DEFAULT_PERSONA not in sonic.system_prompt


def test_nothing_in_the_prompt_calls_nova_an_assistant():
    """The exact register this round removes — AWS's own baseline voice prompt
    is 'a warm, professional, and helpful AI assistant' (s9)."""
    sonic = _build()[0]
    assert "assistant" not in sonic.system_prompt.lower()
    assert "assistant" not in app.HARNESS_SYSTEM_PROMPT.lower()
    assert "assistant" not in app.TOOL_GUIDE.lower()


def test_the_tool_guide_is_mechanics_only():
    """Character text lives in the persona file; the guide is the contract."""
    guide = app.TOOL_GUIDE
    for tool in (
        "run_behavior",
        "goto",
        "declare_goal",
        "set_mode",
        "set_inhibition",
        "lock_face",
        "release_face",
        "create_rule",
        "recall_senses",
    ):
        assert tool in guide
    assert "did not confirm" in guide
    assert "unknown-kind" in guide
    for character_word in ("warm", "curious", "companion", "playful", "teasing", "dry", "sincere"):
        assert character_word not in guide.lower()


def test_sonic_speaks_with_the_amy_voice():
    assert app.SONIC_VOICE_ID == "amy"
    assert _build()[0].voice_id == "amy"


def test_the_tool_specs_are_unchanged_by_the_persona_move():
    names = [spec["toolSpec"]["name"] for spec in TOOL_SPECS]
    assert names == PRE_T14_TOOL_NAMES
    assert len(TOOL_SPECS) == len(PRE_T14_TOOL_NAMES)
    assert _build()[0].tools is TOOL_SPECS


# --------------------------------------------------------------------------- #
# t14 — the switches lead the journal and gate the legs (c33/h25)              #
# --------------------------------------------------------------------------- #


def test_the_journal_opens_with_every_switch_then_the_persona_source(caplog):
    with caplog.at_level("INFO", logger="nova.sensory"):
        _build()
    lines = _sense_lines(caplog)
    assert lines[0].endswith(
        "switches chunked_playback=on lite_reactions=on memory=on persona=default"
    )
    assert "event=switches" in lines[0]
    assert "event=persona" in lines[1]
    assert "persona source=file:" in lines[1]


def test_the_switch_line_names_every_resolved_value_when_they_are_off(monkeypatch, caplog):
    monkeypatch.setenv("NOVA_CHUNKED_PLAYBACK", "0")
    monkeypatch.setenv("NOVA_LITE_REACTIONS", "0")
    monkeypatch.setenv("NOVA_MEMORY", "0")
    with caplog.at_level("INFO", logger="nova.sensory"):
        _build()
    assert _sense_lines(caplog)[0].endswith(
        "switches chunked_playback=off lite_reactions=off memory=off persona=default"
    )


def test_chunked_playback_is_on_by_default():
    assert _build()[1].chunked is True


def test_chunked_playback_off_constructs_a_whole_utterance_speaker(monkeypatch):
    monkeypatch.setenv("NOVA_CHUNKED_PLAYBACK", "0")
    speaker = _build()[1]
    assert isinstance(speaker, SonicSpeaker)
    assert speaker.chunked is False


def test_memory_off_builds_no_ledger_and_no_compactor(monkeypatch, caplog):
    monkeypatch.setenv("NOVA_MEMORY", "0")
    with caplog.at_level("INFO", logger="nova.sensory"):
        components = _build()

    sonic = components[0]
    assert not any(isinstance(c, MemoryCompactor) for c in components)
    assert sonic._history_provider is None
    assert any(
        "component absent name=memory-ledger reason=switch-off" in m for m in _sense_lines(caplog)
    )
    # …and no transcript writes anything at all.
    sonic.on_transcript("USER", "hi")
    sonic.on_transcript("ASSISTANT", "hello")
    assert _ledger_records() == []


def test_memory_off_still_wires_the_speaker_idle_check(monkeypatch):
    """A rotation must not cut a chunk in half even with nothing to replay."""
    monkeypatch.setenv("NOVA_MEMORY", "0")
    sonic, speaker = _build()[:2]
    assert sonic._speaker_idle is not None
    assert sonic._speaker_idle() is speaker.idle


def test_lite_reactions_off_builds_the_bus_with_no_reactor(monkeypatch, caplog):
    monkeypatch.setenv("NOVA_LITE_REACTIONS", "0")
    with caplog.at_level("INFO", logger="nova.sensory"):
        components = _build()

    bus_component = next(c for c in components if isinstance(c, NovaBus))
    assert bus_component._reactor is None
    assert not any(isinstance(c, LiteReactor) for c in components)
    assert any(
        "component absent name=lite-reactor reason=switch-off" in m for m in _sense_lines(caplog)
    )


def test_memory_construction_failure_degrades_to_a_named_absent_line(monkeypatch, caplog):
    import reachy_nova.harness.memory_compactor as memory_compactor_module

    monkeypatch.setattr(memory_compactor_module, "MemoryCompactor", _boom)
    with caplog.at_level("INFO", logger="nova.sensory"):
        components = _build()  # must not raise

    sonic = components[0]
    assert not any(isinstance(c, MemoryCompactor) for c in components)
    assert sonic._history_provider is None
    assert any(
        "component absent name=memory-ledger reason=synthetic construction failure" in m
        for m in _sense_lines(caplog)
    )
    # The pair is all-or-nothing: no compactor means no orphan ledger either.
    sonic.on_transcript("USER", "hi")
    assert _ledger_records() == []


def test_lite_reactor_construction_failure_degrades_to_a_named_absent_line(monkeypatch, caplog):
    import reachy_nova.harness.lite_reactor as lite_reactor_module

    monkeypatch.setattr(lite_reactor_module, "LiteReactor", _boom)
    with caplog.at_level("INFO", logger="nova.sensory"):
        components = _build()  # must not raise

    bus_component = next(c for c in components if isinstance(c, NovaBus))
    assert bus_component._reactor is None
    assert not any(isinstance(c, LiteReactor) for c in components)
    assert any(
        "component absent name=lite-reactor reason=synthetic construction failure" in m
        for m in _sense_lines(caplog)
    )


# --------------------------------------------------------------------------- #
# t14 — the ledger, the compactor and the history replay (c13/h11, c12)        #
# --------------------------------------------------------------------------- #


def test_transcripts_are_appended_to_the_ledger():
    sonic = _build()[0]

    sonic.on_transcript("USER", "hi")
    sonic.on_transcript("ASSISTANT", "hello")

    records = _ledger_records()
    assert [r["kind"] for r in records] == ["USER", "ASSISTANT"]
    assert [r["text"] for r in records] == ["hi", "hello"]


def test_a_bus_inject_reaches_sonic_with_its_class_and_lands_in_the_ledger():
    components = _build()
    sonic = components[0]
    bus_component = next(c for c in components if isinstance(c, NovaBus))

    injected = []
    sonic.inject_text = lambda text, sense_class=None: (injected.append((text, sense_class)), "sent")[1]

    # The bus introspects on_inject ONCE at construction; it must have found a
    # real ``sense_class`` keyword on the wrapper, or the class never rides.
    assert bus_component._on_inject_accepts_sense_class is True
    bus_component._on_inject("(someone is petting you)", sense_class="pat")

    assert injected == [("(someone is petting you)", "pat")]
    records = _ledger_records()
    assert len(records) == 1
    assert records[0]["kind"] == "sense"
    assert records[0]["text"] == "(someone is petting you)"
    assert records[0]["sense_class"] == "pat"


def test_sonic_replays_the_compactors_history():
    components = _build()
    sonic = components[0]
    compactor = next(c for c in components if isinstance(c, MemoryCompactor))

    provider = sonic._history_provider
    assert provider is not None
    # Bound methods compare by (__self__, __func__), never by identity.
    assert provider.__self__ is compactor
    assert provider.__func__ is MemoryCompactor.history


def test_the_compactor_reads_the_same_ledger_the_transcripts_write():
    components = _build()
    sonic = components[0]
    compactor = next(c for c in components if isinstance(c, MemoryCompactor))

    sonic.on_transcript("USER", "is the tap still dripping")
    sonic.on_transcript("ASSISTANT", "constantly")

    blocks = compactor.history()
    # The history opens with the USER context block (a history that opens with
    # the assistant kills the Bedrock stream — live incident 2026-09-06) and
    # the first USER line merges into it; then the roles alternate.
    assert [b["role"] for b in blocks] == ["USER", "ASSISTANT"]
    assert blocks[0]["text"].startswith("(earlier today")
    assert blocks[0]["text"].endswith("is the tap still dripping")
    assert blocks[1]["text"] == "constantly"


def test_sonic_sees_the_speakers_idle_state():
    components = _build()
    sonic, speaker = components[0], components[1]
    speaker.poster = lambda *a, **k: None  # never touch the daemon from a test

    assert speaker.idle is True
    assert sonic._speaker_idle() is True

    speaker.on_audio_chunk(np.zeros(2400, dtype=np.float32))  # a sub-chunk tail

    assert speaker.idle is False
    assert sonic._speaker_idle() is False


def test_the_compactor_and_reactor_are_supervisor_components_started_in_order():
    components = _build()
    names = [type(c).__name__ for c in components]

    assert names.index("LiteReactor") < names.index("NovaBus") < names.index("MemoryCompactor")
    for component in (
        next(c for c in components if isinstance(c, LiteReactor)),
        next(c for c in components if isinstance(c, MemoryCompactor)),
    ):
        assert callable(component.start)
        assert callable(component.stop)
        # Construction is wiring only: no thread of theirs is running yet.
        assert component.is_alive() is False


# --------------------------------------------------------------------------- #
# t14 — mood and the Lite reactor's context (c9, c28)                          #
# --------------------------------------------------------------------------- #


def test_a_pat_inject_moves_the_mood_the_reactor_reports():
    components = _build()
    bus_component = next(c for c in components if isinstance(c, NovaBus))
    reactor = next(c for c in components if isinstance(c, LiteReactor))

    assert reactor._context_provider()["mood"] == mood_module.NEUTRAL_SENTENCE
    bus_component._on_inject("(someone is petting you)", sense_class="pat")
    assert reactor._context_provider()["mood"] == mood_module.CHEEKY_SENTENCE


def test_an_unclassed_inject_leaves_the_mood_alone():
    components = _build()
    bus_component = next(c for c in components if isinstance(c, NovaBus))
    reactor = next(c for c in components if isinstance(c, LiteReactor))

    bus_component._on_inject("(a door closed somewhere)", sense_class="sound")

    assert reactor._context_provider()["mood"] == mood_module.NEUTRAL_SENTENCE


def test_the_reaction_context_carries_senses_memory_mood_and_exchanges():
    components = _build()
    sonic = components[0]
    bus_component = next(c for c in components if isinstance(c, NovaBus))
    reactor = next(c for c in components if isinstance(c, LiteReactor))

    memory_path = statedir.memory_path()
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        json.dumps(
            {
                "topics": [{"text": "the leaking tap", "ts": 1.0}],
                "items": [{"text": "stop humming at night", "kind": "stop", "ts": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    bus_component.history.record("touch", "pat", "r1", "(someone is petting you)", "pat", None)
    sonic.on_transcript("USER", "what is that noise")
    sonic.on_transcript("ASSISTANT", "the tap again")

    context = reactor._context_provider()

    assert context["senses"] == ["(someone is petting you)"]
    assert "the leaking tap" in context["memory"]
    assert "stop humming at night" in context["memory"]
    assert context["mood"]
    assert context["exchanges"] == [
        {"role": "USER", "text": "what is that noise"},
        {"role": "ASSISTANT", "text": "the tap again"},
    ]


def test_the_reaction_context_is_empty_but_shaped_without_memory(monkeypatch):
    monkeypatch.setenv("NOVA_MEMORY", "0")
    components = _build()
    reactor = next(c for c in components if isinstance(c, LiteReactor))

    context = reactor._context_provider()

    assert context["memory"] == ""
    assert context["exchanges"] == []
    assert context["senses"] == []
    assert context["mood"]


def test_render_memory_paragraph_is_one_short_paragraph_or_nothing():
    assert app.render_memory_paragraph({"topics": [], "items": []}) == ""
    assert app.render_memory_paragraph({}) == ""
    rendered = app.render_memory_paragraph(
        {
            "topics": [{"text": "the tap"}, {"text": "Tuesday"}],
            "items": [{"text": "stop humming"}],
        }
    )
    assert rendered == "talked about the tap, Tuesday. worth remembering: stop humming"


def test_a_lite_planned_gesture_goes_through_the_intents_spool():
    components = _build()
    intents = components[3]
    reactor = next(c for c in components if isinstance(c, LiteReactor))
    assert isinstance(intents, IntentTools)

    calls = []
    intents.execute = lambda name, params: calls.append((name, params)) or "{}"
    reactor._on_gesture("nod")

    assert calls == [("run_behavior", {"name": "nod", "duration": 2.0})]


def test_a_failing_lite_gesture_is_a_named_line_not_a_crash(caplog):
    components = _build()
    intents = components[3]
    reactor = next(c for c in components if isinstance(c, LiteReactor))
    intents.execute = _boom

    with caplog.at_level("INFO", logger="nova.sensory"):
        reactor._on_gesture("nod")  # must not raise

    assert any(
        "dropped reason=lite-gesture-failed name=nod" in m for m in _sense_lines(caplog)
    )


def test_a_real_bus_message_lands_in_the_ledger_through_the_composed_wrapper():
    """End to end through the REAL rules path: on_message -> rules.yaml ->
    the composed inject wrapper -> sonic + ledger. The generic rule/fire entry
    carries no ``react: lite``, so it renders and delivers inline."""
    components = _build()
    sonic = components[0]
    bus_component = next(c for c in components if isinstance(c, NovaBus))

    injected = []
    sonic.inject_text = lambda text, sense_class=None: (injected.append((text, sense_class)), "sent")[1]

    bus_component.on_message(
        None,
        None,
        _fake_msg(
            "reachy/events/rule/fire",
            {
                "t": "rule",
                "ts": 1718362800.3,
                "tick": 15,
                "action": "fire",
                "rule": "hear",
                "kind": "react",
                "field": "speech",
                "op": "is_true",
                "reason": "fired",
                "behavior": "nod",
                "disable": [],
            },
        ),
    )

    assert len(injected) == 1
    records = _ledger_records()
    assert [r["kind"] for r in records] == ["sense"]
    assert records[0]["text"] == injected[0][0]


def test_vision_descriptions_reach_sonic_as_a_brief_capped_cue():
    """Robot 2026-09-06: raw 400-850 char Omni descriptions became 30 s monologues."""
    from reachy_nova.harness.app import VISION_CUE_MAX_CHARS, render_vision_cue

    long = ("A person sits at a desk with a laptop, a cup and a lamp. " * 20).strip()
    cue = render_vision_cue(long)
    assert cue.startswith("(you glance around: ")
    assert cue.endswith(") (react briefly if at all)")
    inner = cue[len("(you glance around: ") : -len(") (react briefly if at all)")]
    assert len(inner) <= VISION_CUE_MAX_CHARS
    assert inner.endswith("."), "cut at a sentence end"
    assert render_vision_cue("  a cat\n on the desk ") == "(you glance around: a cat on the desk) (react briefly if at all)"



# --------------------------------------------------------------------------- #
# PR #24 review fixes                                                          #
# --------------------------------------------------------------------------- #


def test_the_ledger_records_only_delivered_senses():
    """Review thread 5: a throttled/inactive cue never reached the model."""
    components = _build()
    sonic = components[0]
    bus_component = next(c for c in components if isinstance(c, NovaBus))
    sonic.inject_text = lambda text, sense_class=None: "dropped-throttled"
    bus_component._on_inject("(someone is petting you)", sense_class="pat")
    assert _ledger_records() == []
    sonic.inject_text = lambda text, sense_class=None: "deferred"
    bus_component._on_inject("(someone is petting you)", sense_class="pat")
    assert _ledger_records() == [], "a deferred cue is ledgered when the drain sends it, not before"
    # ...and the drain's hook is what appends it
    assert sonic.on_deferred_delivered is not None
    sonic.on_deferred_delivered("(just now, while you were talking: someone petted you)", "pat")
    records = _ledger_records()
    assert len(records) == 1
    assert records[0]["sense_class"] == "pat"


def test_vision_cues_go_through_the_ledger_wrapper():
    """Review thread 7: a glance is a delivered sense like any other."""
    components = _build()
    sonic = components[0]
    seen = []
    sonic.inject_text = lambda text, force=False, sense_class=None: (seen.append((text, sense_class)), "sent")[1]
    app_vision = [c for c in components if type(c).__name__ == "VisionLeg"]
    if not app_vision:
        pytest.skip("vision leg not built in this environment")
    app_vision[0]._on_answer("A cat on the desk.")
    assert seen
    assert seen[0][1] == "vision"
    records = _ledger_records()
    assert records
    assert records[-1]["sense_class"] == "vision"


def test_lite_vocalizations_play_through_the_speaker():
    """Review thread 4: a purr is synthesised and rides the speaker like a chunk."""
    from reachy_nova.harness.lite_reactor import LiteReactor

    components = _build()
    speaker = components[1]
    reactor = next(c for c in components if isinstance(c, LiteReactor))
    assert reactor._on_vocalize is not None
    fed = []
    speaker.on_audio_chunk = lambda samples: fed.append(samples)
    reactor._on_vocalize("purr")
    assert len(fed) == 1
    assert fed[0].dtype.name == "float32"
    assert len(fed[0]) > 2400
