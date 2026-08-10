"""Composition root: wire Nova Sonic to the runtime's seams.

``build_app()`` assembles the full harness graph and returns the component
list the supervisor runs. Nothing here starts threads or touches the network —
construction is wiring only, so it is fully testable off-robot.

The graph::

    audio tee ──TeeHearing──► sonic.feed_audio          (16 kHz, echo-gated)
    sonic.on_audio_output ──► SonicSpeaker ──► daemon HTTP play  (arms the gate)
    sonic.on_interruption ──► speaker.preempt()          (barge-in / mouth loss)
    sonic.on_tool_use ─thread─► IntentTools.execute ──► sonic.send_tool_result
    bus reachy/events/# ──rules.yaml──► sonic.inject_text (throttled inside)
    memory (qq) ──MemoryLeg──rules.yaml──► sonic.inject_text
    sonic.on_transcript(ASSISTANT) ──► CognitionFeed.message  (NDJSON, stdout)
"""

from __future__ import annotations

import threading

from .. import config
from ..sensory_log import stage as _stage
from .cognition_feed import CognitionFeed
from .gate import EchoGate
from .hearing import TeeHearing
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


def build_app() -> list[object]:
    """Construct and wire every harness component; return them in start order."""
    from ..nova_sonic import NovaSonic  # heavy AWS SDK import kept out of module import

    gate = EchoGate()
    feed = CognitionFeed()
    intents = IntentTools()

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

    # cognition feed — what was said aloud
    def _on_transcript(role: str, text: str) -> None:
        if role == "ASSISTANT" and text.strip():
            feed.message(text)

    sonic.on_transcript = _on_transcript

    # hear leg
    hearing = TeeHearing(feed=sonic.feed_audio, gate=gate)

    components: list[object] = [sonic, speaker, hearing]

    # read leg — bus is optional-degraded: no broker means named drops, not death
    try:
        from .bus import NovaBus

        components.append(NovaBus(on_inject=sonic.inject_text))
    except Exception as err:  # noqa: BLE001
        _stage("supervise", "nova", "component", f"component absent name=bus reason={err}")

    # memory leg — optional: qq backends may not exist on this box
    try:
        from ..nova_memory import NovaMemory
        from .memory_leg import MemoryLeg

        MemoryLeg(NovaMemory(), on_inject=sonic.inject_text).attach()
    except Exception as err:  # noqa: BLE001
        _stage("supervise", "nova", "component", f"component absent name=memory reason={err}")

    return components
