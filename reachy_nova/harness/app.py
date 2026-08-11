"""Composition root: wire Nova Sonic to the runtime's seams.

``build_app()`` assembles the full harness graph and returns the component
list the supervisor runs. Nothing here starts threads or touches the network —
construction is wiring only, so it is fully testable off-robot. The ONE
deliberate side effect is :func:`ensure_face_rule`: the standing
``nova-face-noticed`` overlay rule is (re)written into the runtime's rules
overlay at startup, because without it the face cue never crosses the bus at
all (see the function's docstring); it degrades to a named component-absent
line when no runtime state dir exists on this box.

The graph::

    audio tee ──TeeHearing──► sonic.feed_audio          (16 kHz, echo-gated)
    sonic.on_audio_output ──► SonicSpeaker ──► daemon HTTP play  (arms the gate)
    sonic.on_interruption ──► speaker.preempt()          (barge-in / mouth loss)
    sonic.on_tool_use ─thread─► IntentTools.execute ──► sonic.send_tool_result
    bus reachy/events/# ──rules.yaml──► sonic.inject_text (throttled inside)
    bus reachy/state/clip ──VisionLeg──NovaOmni──► sonic.inject_text
    NovaBrowser.on_progress ──► sonic.inject_text        (browse tool, flag-gated)
    memory (qq) ──MemoryLeg──rules.yaml──► sonic.inject_text
    sonic.on_transcript(ASSISTANT) ──► CognitionFeed.message  (NDJSON, stdout)

Optional legs degrade, never crash: the browser exists only when
``NOVA_ACT_ENABLED`` is on, the vision leg only when ``NOVA_OMNI_MODEL_ID`` is
set (the Omni preview is model-config-gated) AND the bus built (it supplies the
retained clip state). Every absent leg emits the standard
``component absent name=<leg> reason=<why>`` senselog line, so "we started
without seeing" is visible rather than inferred.
"""

from __future__ import annotations

import threading

from .. import config
from ..nova_browser import act_enabled
from ..sensory_log import stage as _stage
from . import statedir
from .cognition_feed import CognitionFeed
from .gate import EchoGate, resolve_policy
from .hearing import TeeHearing
from .rules_overlay import upsert_rule
from .speaking import SonicSpeaker
from .tools import TOOL_SPECS, IntentTools

HARNESS_SYSTEM_PROMPT = (
    "You are Nova, the mind of a small Reachy Mini robot. You hear the room "
    "through your microphones and speak aloud through your speaker. Keep your "
    "words short, warm, and natural — you are a curious household companion, "
    "not an assistant reading documentation. "
    "You act through your body with tools: run_behavior plays a named gesture, "
    "goto moves your head and antennas, declare_goal and set_mode steer your "
    "standing behavior, set_inhibition holds behaviors back, and create_rule "
    "writes a lasting reflex that keeps working even while you sleep. Use them "
    "when someone asks you to move or react, and mention what you did in a "
    "few words. If a tool answers that the engine did not confirm, say your "
    "body did not respond."
)

# --------------------------------------------------------------------------- #
# The standing face rule (t10)                                                 #
# --------------------------------------------------------------------------- #

#: LIVE FINDING (on-device, 2026-08-12): discrete ``reachy/events/face/*``
#: topics do not exist in the runtime — sense-block events collapse into the
#: (never-subscribed) ``sense/snapshot`` stream, and only RULE FIRES cross the
#: bus as discrete events. The runtime ships no default rule on the face cue,
#: so without this standing overlay rule a recognised face never reaches Nova.
FACE_RULE_ID = "nova-face-noticed"

#: The rule itself, in the engine's own grammar (reachy-mini-cli
#: ``reachy/behavior/rules.py``): ``face`` latches the recognised name for one
#: tick (``is_true`` = "a named face was recognised this tick"), and the
#: runtime's own per-name re-announce cooldown is 30 s
#: (``face_sense.DEFAULT_REANNOUNCE_COOLDOWN``) — ``cooldown_s`` matches it so
#: the rule can never fire faster than the cue that feeds it. ``nod`` is the
#: engine's own library acknowledgement gesture.
#: ``duration_s`` is load-bearing: ``nod`` is a looping behavior with no
#: default duration, and the engine REFUSES a rule that would let a looping
#: behavior hold its channel forever (seen live 2026-08-11: the reload was
#: rejected with "carries no duration_s" and the whole overlay kept last-good).
FACE_RULE: dict = {
    "id": FACE_RULE_ID,
    "when": {"field": "face", "op": "is_true"},
    "run": "nod",
    "duration_s": 2.0,
    "cooldown_s": 30.0,
}

#: Seconds :func:`ensure_face_rule` waits for the engine's reload verdict when
#: the overlay actually changed (an unchanged overlay submits no reload).
FACE_RULE_RELOAD_TIMEOUT = 1.0


def ensure_face_rule(*, reload_timeout: float | None = None) -> bool:
    """Ensure the standing ``nova-face-noticed`` rule is in the rules overlay.

    Runs at composition time, once per startup. Total — every failure resolves
    to the standard ``component absent name=face-rule`` line, never a crash:

    * no runtime state dir on this box (a dev machine, a partial install) —
      the overlay is not even created, because a rules file with no engine to
      read it is litter;
    * a refused rule or unwritable overlay — named with the error verbatim.

    On success the (idempotent) upsert's verdict is logged: a second startup
    finds the rule already present, writes nothing and submits no reload.
    """
    timeout = FACE_RULE_RELOAD_TIMEOUT if reload_timeout is None else reload_timeout
    try:
        behavior_dir = statedir.behavior_dir()
        if not behavior_dir.is_dir():
            _stage(
                "supervise",
                "nova",
                "component",
                f"component absent name=face-rule reason=statedir-absent dir={behavior_dir}",
            )
            return False
        changed, verdict = upsert_rule(dict(FACE_RULE), reload_timeout=timeout)
    except Exception as err:  # noqa: BLE001 - a rules problem must not stop the voice
        _stage("supervise", "nova", "component", f"component absent name=face-rule reason={err}")
        return False
    _stage(
        "supervise",
        "nova",
        "face-rule",
        f"standing rule ensured id={FACE_RULE_ID} changed={changed} verdict={verdict}",
    )
    return True


# --------------------------------------------------------------------------- #
# Browser lifecycle adapter                                                    #
# --------------------------------------------------------------------------- #


class BrowserComponent:
    """Adapts :class:`~reachy_nova.nova_browser.NovaBrowser` to the supervisor.

    ``NovaBrowser.start(stop_event)`` spins its worker thread and that thread
    already watches the supervisor's ``stop_event``; the supervisor's ``stop()``
    contract is therefore a bounded join, not a second signal. The underlying
    browser stays reachable as ``.browser`` (the handle ``IntentTools`` drives).
    """

    name = "browser"

    def __init__(self, browser: object) -> None:
        self.browser = browser

    def start(self, stop_event: threading.Event) -> None:
        self.browser.start(stop_event)  # type: ignore[attr-defined]

    def stop(self, timeout: float = 2.0) -> None:
        thread = getattr(self.browser, "_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)


# --------------------------------------------------------------------------- #
# The composition root                                                         #
# --------------------------------------------------------------------------- #


def build_app() -> list[object]:
    """Construct and wire every harness component; return them in start order."""
    from ..nova_sonic import NovaSonic  # heavy AWS SDK import kept out of module import

    gate = EchoGate()
    feed = CognitionFeed()

    speaker = SonicSpeaker(gate=gate)

    sonic = NovaSonic(
        system_prompt=HARNESS_SYSTEM_PROMPT,
        tools=TOOL_SPECS,
        on_audio_output=None,  # bound below once speaker exists
        region=config.region(),
        model_id=config.sonic_model_id(),
    )

    # speak leg
    sonic.on_audio_output = speaker.on_audio_chunk
    sonic.on_interruption = speaker.preempt

    def _on_state_change(state: str) -> None:
        speaker.on_state_change(state)

    sonic.on_state_change = _on_state_change

    # act leg (browse) — a real browser exists only when Nova Act is enabled;
    # its progress narration goes through inject_text like every other sense.
    browser = None
    if act_enabled():
        try:
            from ..nova_browser import NovaBrowser

            browser = NovaBrowser()
        except Exception as err:  # noqa: BLE001
            _stage("supervise", "nova", "component", f"component absent name=browser reason={err}")
    else:
        _stage(
            "supervise", "nova", "component", "component absent name=browser reason=act-disabled"
        )

    intents = IntentTools(
        browser=browser,
        on_browse_progress=sonic.inject_text if browser is not None else None,
    )

    # act leg — tool calls run off Sonic's response thread, result posted back
    def _on_tool_use(tool_name: str, tool_use_id: str, params: dict) -> None:
        def _work() -> None:
            try:
                result = intents.execute(tool_name, params)
            except Exception as err:  # noqa: BLE001 - a tool bug must not kill the voice
                result = f'{{"ok": false, "error": "tool crashed: {err}"}}'
                _stage("act", "nova", tool_use_id, f"dropped reason=tool-crashed detail={err}")
            sonic.send_tool_result(tool_use_id, result)

        threading.Thread(target=_work, name=f"nova-tool-{tool_name}", daemon=True).start()

    sonic.on_tool_use = _on_tool_use

    # cognition feed — what was said aloud; USER lines named for the journal
    def _on_transcript(role: str, text: str) -> None:
        if role == "ASSISTANT" and text.strip():
            feed.message(text)
        elif role == "USER" and text.strip():
            _stage("hear", "nova", "transcript", f"heard {text[:120]!r}")
            # Playback-aware barge-in: Sonic's own interruption path only runs
            # while it is GENERATING, but the audible playback happens after
            # (upload + play on the daemon). A user transcript while the gate
            # window is armed means they are talking over the robot's actual
            # voice — cut it (stop_sound + purge) rather than talking on.
            if gate.active:
                _stage("speak", "nova", "barge-in", "user spoke over playback — preempting")
                speaker.preempt()

    sonic.on_transcript = _on_transcript

    # hear leg — the echo-gate policy is wired EXPLICITLY (resolve_policy reads
    # $NOVA_ECHO_GATE, default off) so the wiring is assertable, not ambient.
    hearing = TeeHearing(
        feed=sonic.feed_audio, gate=gate, echo_gate_policy=resolve_policy()
    )

    components: list[object] = [sonic, speaker, hearing]

    # read leg — bus is optional-degraded: no broker means named drops, not death
    bus_component = None
    try:
        from .bus import NovaBus

        bus_component = NovaBus(on_inject=sonic.inject_text)
        components.append(bus_component)
    except Exception as err:  # noqa: BLE001
        _stage("supervise", "nova", "component", f"component absent name=bus reason={err}")

    if browser is not None:
        components.append(BrowserComponent(browser))

    # vision leg — Omni is model-config-gated (empty id = preview not enabled)
    # and the bus supplies the retained reachy/state/clip payload it reads.
    if not config.omni_model_id():
        _stage(
            "supervise", "nova", "component", "component absent name=vision reason=omni-model-unset"
        )
    elif bus_component is None:
        _stage("supervise", "nova", "component", "component absent name=vision reason=no-bus")
    else:
        try:
            from ..nova_omni import NovaOmni
            from .vision_leg import VisionLeg

            components.append(
                VisionLeg(
                    get_clip_state=bus_component.clip_state,
                    understand=NovaOmni(),
                    on_answer=sonic.inject_text,
                )
            )
        except Exception as err:  # noqa: BLE001
            _stage("supervise", "nova", "component", f"component absent name=vision reason={err}")

    # memory leg — optional: qq backends may not exist on this box
    try:
        from ..nova_memory import NovaMemory
        from .memory_leg import MemoryLeg

        MemoryLeg(NovaMemory(), on_inject=sonic.inject_text).attach()
    except Exception as err:  # noqa: BLE001
        _stage("supervise", "nova", "component", f"component absent name=memory reason={err}")

    # standing reflexes — the face cue crosses the bus only through this rule
    ensure_face_rule()

    return components
