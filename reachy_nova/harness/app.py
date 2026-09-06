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
    bus reachy/events/# ──rules.yaml──► _inject ──► sonic.inject_text
                                              └────► ledger.append("sense")
                                              └────► mood.note(pat|face)
    bus react: lite entries ──► LiteReactor ──► _inject / intents.run_behavior
    bus reachy/state/clip ──VisionLeg──NovaOmni──► sonic.inject_text
    NovaBrowser.on_progress ──► sonic.inject_text        (browse tool, flag-gated)
    memory (qq) ──MemoryLeg──rules.yaml──► sonic.inject_text
    sonic.on_transcript(ASSISTANT) ──► CognitionFeed.message  (NDJSON, stdout)
    sonic.on_transcript(USER|ASSISTANT) ──► ledger.append ──► MemoryCompactor
                                       └──► mood.note(user_turn|assistant_turn)
    MemoryCompactor.history ──► sonic history replay (at every session start)
    NetworkUnit joined/moved ──NetworkReactor──► sonic.request_immediate_restart
                                              └► kiro_unit.request_restart

Optional legs degrade, never crash: the browser exists only when
``NOVA_ACT_ENABLED`` is on, the vision leg only when ``NOVA_OMNI_MODEL_ID`` is
set (the Omni preview is model-config-gated) AND the bus built (it supplies the
retained clip state), the ledger/compactor pair only when ``NOVA_MEMORY`` is on,
and the Lite reaction tier only when ``NOVA_LITE_REACTIONS`` is on. Every absent
leg emits the standard ``component absent name=<leg> reason=<why>`` senselog
line, so "we started without seeing" — or without remembering — is visible
rather than inferred.

The switches themselves (:mod:`reachy_nova.harness.switches`) are resolved and
logged FIRST, before anything is constructed, so the journal's opening lines
say what this process is going to be before it is anything at all.
"""

from __future__ import annotations

import threading

import numpy as np
from pathlib import Path

from .. import config
from ..nova_browser import act_enabled
from ..sensory_log import stage as _stage
from ..skill_forge import resolve_writer
from . import eyes as eyes_module
from . import statedir
from .attention import AttentionState
from .cognition_feed import CognitionFeed
from .daemon_client import DaemonClient, restore_volume
from .gate import EchoGate, resolve_policy
from .gaze_stack import GazeStack
from .hearing import TeeHearing
from .ledger import Ledger
from .lock_state import LockState
from .mood import Mood
from .network import NetworkUnit
from .persona import DEFAULT_PERSONA
from .persona import read as read_persona
from .quiet import QuietState
from .rules_overlay import retire_rule, upsert_rule
from .sense_history import SenseHistory
from .speaking import SonicSpeaker
from .switches import Switches
from .switches import log as log_switches
from .switches import resolve as resolve_switches
from .tools import TOOL_SPECS, IntentTools

# --------------------------------------------------------------------------- #
# The system prompt: WHO (persona.py, on disk) + HOW (the tool guide, here)     #
# --------------------------------------------------------------------------- #

#: The Nova 2 Sonic voice Nova speaks with (en-GB). The system prompt steers
#: lexical style but not accent or pitch, so the voice id is the only knob on
#: the SOUND of the personality (AWS nova2-userguide/sonic-language-support).
SONIC_VOICE_ID = "amy"

#: The tool half of the system prompt: which tool moves what, and what to say
#: when the engine will not or cannot do it. Deliberately mechanics ONLY —
#: every sentence about who Nova is lives in ``config/persona/nova.md`` (or
#: :data:`~reachy_nova.harness.persona.DEFAULT_PERSONA` when that file is not
#: on this box), so an operator can rewrite the character without touching the
#: tool contract and vice versa. The word "assistant" appears nowhere in
#: either half on purpose: it is the exact register this round removed.
TOOL_GUIDE = (
    "You act through your body with tools: run_behavior plays a named gesture "
    "(including the gaze one-shots look-at-sound and look-at-face), goto "
    "moves your head and antennas, declare_goal and set_mode steer your "
    "standing behavior, set_inhibition holds behaviors back, lock_face keeps "
    "you looking at the person in front of you until release_face lets you "
    "look away, and create_rule writes a lasting reflex that keeps working "
    "even while you sleep. Use them when someone asks you to move or react. "
    "If a tool answers that the engine did not confirm, say your body did "
    "not respond; if it answers with an unknown-kind error, say your body "
    "does not know that move yet. "
    "When someone tells you to stop following or look away, call "
    "release_face; when they ask why you did something, what you felt, or "
    "what just happened, call recall_senses before answering."
)


#: Longest scene description Nova is handed as a glance; Omni answers run to
#: 850 characters and Sonic narrated every one of them (robot, 2026-09-06).
VISION_CUE_MAX_CHARS = 240


def render_vision_cue(text: str) -> str:
    """Shape a scene description as a brief body cue rather than a report.

    Parenthesised like every other cue, cut at the last sentence end inside
    :data:`VISION_CUE_MAX_CHARS`, and carrying the same brief marker the
    nervous-system rules append — so the model treats a glance as something
    to react to with a word or nothing, not something to read out.
    """
    body = " ".join(str(text or "").split())
    if len(body) > VISION_CUE_MAX_CHARS:
        cut = body[:VISION_CUE_MAX_CHARS]
        end = max(cut.rfind(". "), cut.rfind("; "))
        body = (cut[: end + 1] if end > 60 else cut.rstrip()) 
    return f"(you glance around: {body}) (react briefly if at all)"


#: Lite's vocalize vocabulary -> the legacy synthesiser's kinds.
VOCALIZE_KINDS = {"chirp": "chirp_up", "trill": "trill", "purr": "purr_tone"}


def build_system_prompt(persona_text: str) -> str:
    """The full system prompt: the persona, a blank line, then the tool guide."""
    return f"{persona_text}\n\n{TOOL_GUIDE}"


#: Module-level default, kept for importers that want "the prompt Nova ships
#: with" without resolving the persona file: the EMBEDDED persona plus the tool
#: guide. ``build_app()`` does not use it — it builds the same shape from
#: whatever :func:`reachy_nova.harness.persona.read` resolved at startup.
HARNESS_SYSTEM_PROMPT = build_system_prompt(DEFAULT_PERSONA)

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


def retire_face_nod_rule() -> None:
    """Tombstone :data:`FACE_RULE_ID` — the face-nod reflex the hold replaces.

    ``NOVA_FACE_HOLD`` on means the gaze stack takes a standing face lock the
    moment a conversation is live, and a runtime reflex that nods the head at
    every recognised face fights that hold for the same channel by recency
    (the same competition task t10 found for ``orient-to-sound``). So the rule
    this module itself installs for the no-hold world is retired when the hold
    is on.

    Total, like :func:`ensure_face_rule`: a state dir that does not exist on
    this box (a dev machine), an unwritable overlay or a refused reload all
    degrade to one ``component absent name=face-nod-retire reason=<why>``
    line. It never raises.
    """
    try:
        result = retire_rule(FACE_RULE_ID)
    except Exception as err:  # noqa: BLE001 - a rules problem must not stop the voice
        _stage(
            "supervise",
            "nova",
            "component",
            f"component absent name=face-nod-retire reason={err}",
        )
        return
    _stage(
        "supervise",
        "nova",
        "component",
        f"face-nod retired id={FACE_RULE_ID} changed={result.get('changed')} "
        f"verdict={result.get('verdict')}",
    )


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


class NetworkReactor:
    """Turns a network transition into an immediate restart of the cloud legs.

    The consumer half of :class:`~reachy_nova.harness.network.NetworkUnit`
    (task t5). Three rules, and the asymmetry between them is the whole design:

    * **joined / moved** — the machine has a NEW address. Every open cloud
      connection is bound to the OLD one and is now a zombie that no amount of
      waiting fixes, so both cloud legs are restarted at once: Sonic's stream
      (its liveness watchdog defaults to 180 s, which alone cannot meet the
      60 s "the mind is back" bound) and the Kiro session (whose kiro-cli
      child holds its own auth/HTTP state).
    * **dropped** — log only. Nothing is torn down on a drop: both legs
      already have their own watchdogs, and killing a session while offline
      only guarantees a failed respawn into a dead network. The units come
      back on the JOIN, which is the transition that carries new information.
    * **the initial observation** (``info["initial"]``) — log only. The unit's
      first poll always reports a transition, because "what the network
      already was" has to reach the journal (spec h13). But it is a BASELINE,
      not a change: both cloud legs were constructed seconds earlier against
      exactly this network, so restarting them here would throw away a healthy
      connecting stream and cost a reconnect cycle at every single boot.

    Not a thread of its own: it is a component solely so that ``start()``
    hands it the supervisor's ``stop_event``, which the Sonic FALLBACK restart
    path needs. Callbacks run on the NetworkUnit's poll thread and never
    raise — a failure in one leg must not cost the other one its restart.
    """

    name = "network_reactor"

    def __init__(self, sonic: object, kiro_unit: object | None = None) -> None:
        self._sonic = sonic
        self._kiro_unit = kiro_unit
        self._stop_event: threading.Event | None = None

    def start(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event

    def stop(self, timeout: float = 1.0) -> None:  # noqa: ARG002 - protocol shape
        return None

    def is_alive(self) -> bool:
        return True

    # -- the callback NetworkUnit.on_change registers -------------------------

    def on_network_change(self, event: str, info: dict) -> None:
        """``callback(event, info)`` for :meth:`NetworkUnit.on_change`."""
        if info.get("initial"):
            _stage(
                "supervise",
                "nova",
                "network",
                f"network baseline {event} ssid={info.get('ssid') or 'unknown'} "
                f"ip={info.get('ip')} (no restart — the legs started against this network)",
            )
            return
        if event == "dropped":
            _stage(
                "supervise",
                "nova",
                "network",
                "network dropped — cloud legs left to their own watchdogs "
                "(nothing torn down; restart happens on the join)",
            )
            return
        where = f"{event} ssid={info.get('ssid') or 'unknown'} ip={info.get('ip')}"
        self._restart_sonic(where)
        self._restart_kiro(where)

    # -- the two legs ----------------------------------------------------------

    def _restart_sonic(self, reason: str) -> None:
        """Restart the Sonic stream now; NEVER raises into the poll loop.

        Prefers the explicit ``request_immediate_restart(reason)`` seam (added
        by task t7: thread-safe, resets the backoff, restarts even a healthy
        stream). The ``getattr`` fallback is the stop+restart pair that existed
        before it, kept so this leg still works against an older/stub Sonic —
        and the path taken is NAMED in the log, so "which mechanism actually
        ran" is never a guess.
        """
        request = getattr(self._sonic, "request_immediate_restart", None)
        try:
            if callable(request):
                request(reason)
                _stage(
                    "supervise",
                    "nova",
                    "network",
                    f"sonic restart requested path=request_immediate_restart reason={reason}",
                )
                return
            stop_event = self._stop_event
            if stop_event is None:
                _stage(
                    "supervise",
                    "nova",
                    "network",
                    f"sonic restart skipped reason=not-started-yet trigger={reason}",
                )
                return
            self._sonic.stop()  # type: ignore[attr-defined]
            self._sonic.restart(stop_event)  # type: ignore[attr-defined]
            _stage(
                "supervise",
                "nova",
                "network",
                f"sonic restart requested path=stop+restart reason={reason}",
            )
        except Exception as err:  # noqa: BLE001 - one leg must not cost the other
            _stage(
                "supervise", "nova", "network", f"sonic restart failed reason={reason} detail={err}"
            )

    def _restart_kiro(self, reason: str) -> None:
        """Ask the Kiro session unit to respawn now; NEVER raises."""
        if self._kiro_unit is None:
            return
        try:
            self._kiro_unit.request_restart(reason)  # type: ignore[attr-defined]
            _stage(
                "supervise", "nova", "network", f"kiro session restart requested reason={reason}"
            )
        except Exception as err:  # noqa: BLE001
            _stage(
                "supervise", "nova", "network", f"kiro restart failed reason={reason} detail={err}"
            )


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
# Memory + reaction-context helpers                                            #
# --------------------------------------------------------------------------- #

#: Sense classes that move the mood. Every other class (``sound``, ``vision``,
#: ``None``) is a mood no-op — :meth:`Mood.note` would ignore it anyway, but
#: naming the two here keeps "what changes how Nova feels" readable.
MOOD_SENSE_CLASSES = ("pat", "face")

#: How many recent senses ride along in the Lite reactor's context.
REACTION_SENSES = 5

#: How many trailing USER/ASSISTANT exchanges ride along with them.
REACTION_EXCHANGES = 4

#: Bounded duration submitted with a Lite-planned gesture. The engine refuses
#: a looping behavior with no duration (see :data:`FACE_RULE`'s note), and
#: "about 2 seconds" is what the ``run_behavior`` tool description advertises.
REACTION_GESTURE_DURATION_S = 2.0


def build_memory(switches: Switches, quiet: QuietState) -> tuple[Ledger | None, object | None]:
    """The ledger/compactor pair, or ``(None, None)`` with a named absence.

    Total, like every other optional leg in this module: the switch being off,
    a missing state dir, or an import that cannot resolve boto3 all degrade to
    one ``component absent name=memory-ledger reason=<why>`` line. The pair is
    all-or-nothing — a ledger nothing ever distils is just a growing file, so
    a compactor that will not construct takes the ledger down with it.

    (The separate ``name=memory`` line further down ``build_app`` is the qq
    knowledge leg — a different, older leg that happens to share the word.)
    """
    if not switches.memory:
        _stage(
            "supervise",
            "nova",
            "component",
            "component absent name=memory-ledger reason=switch-off",
        )
        return None, None
    try:
        from .memory_compactor import MemoryCompactor

        ledger = Ledger(quiet=quiet)
        return ledger, MemoryCompactor(ledger)
    except Exception as err:  # noqa: BLE001 - no memory must never mean no voice
        _stage(
            "supervise",
            "nova",
            "component",
            f"component absent name=memory-ledger reason={err}",
        )
        return None, None


def render_memory_paragraph(memory: dict) -> str:
    """The day's memory as ONE short paragraph for the Lite reactor's context.

    ``memory`` is :meth:`MemoryCompactor.memory`'s shape — ``{"topics": [...],
    "items": [...]}``, each entry a dict with a ``text``. Returns ``""`` when
    there is nothing remembered yet, which the reactor renders as "(none)"
    rather than as an empty label.
    """

    def texts(key: str) -> list[str]:
        entries = memory.get(key) or []
        out = []
        for entry in entries:
            text = str(entry.get("text", "")).strip() if isinstance(entry, dict) else ""
            if text:
                out.append(text)
        return out

    parts = []
    topics = texts("topics")
    if topics:
        parts.append("talked about " + ", ".join(topics))
    items = texts("items")
    if items:
        parts.append("worth remembering: " + "; ".join(items))
    return ". ".join(parts)


def recent_exchanges(ledger: Ledger | None, limit: int = REACTION_EXCHANGES) -> list[dict]:
    """The last *limit* USER/ASSISTANT ledger lines as ``{"role", "text"}``.

    ``[]`` when there is no ledger (``NOVA_MEMORY=0``) — the reactor's context
    simply carries no exchanges, exactly as it did before this round.
    """
    if ledger is None:
        return []
    records = ledger.read()
    exchanges = [
        {"role": str(record.get("kind")), "text": str(record.get("text", ""))}
        for record in records
        if record.get("kind") in ("USER", "ASSISTANT") and str(record.get("text", "")).strip()
    ]
    return exchanges[-limit:]


# --------------------------------------------------------------------------- #
# The composition root                                                         #
# --------------------------------------------------------------------------- #


def build_app() -> list[object]:
    """Construct and wire every harness component; return them in start order."""
    from ..nova_sonic import NovaSonic  # heavy AWS SDK import kept out of module import

    # switches FIRST (t14/c33) — before anything is built, so the journal's
    # opening lines name every resolved value. Fails open: an unrecognised
    # value is the new default plus a warning, never a silently-off feature.
    switches = resolve_switches()
    log_switches(switches)

    # persona (t3/c8) — WHO Nova is, read from disk once. One call, because
    # each call that falls back to the embedded default logs its own line.
    persona = read_persona(switches.persona_path)
    _stage(
        "supervise",
        "nova",
        "persona",
        f"persona source={persona.source} chars={len(persona.text)}",
    )

    gate = EchoGate()
    feed = CognitionFeed()

    # timed quiet (t11/t12) — ONE object, four readers now: the speaker gates
    # playback on it, the bus marks every inject with it, the
    # stay_silent/end_silence tools arm and release it, and the ledger writes
    # nothing at all while it is armed. Constructed here rather than inside
    # any of them so they can never disagree about whether the robot is
    # currently supposed to be quiet. It reloads a still-future deadline off
    # disk, so a restart inside a quiet window comes back quiet instead of
    # loudly reintroducing itself.
    quiet = QuietState()

    # mood (t6/c9) — ALWAYS constructed: four floats and a lock, no I/O and no
    # failure mode worth degrading over (same reasoning as the sense history
    # and the network leg). Fed from the inject wrapper (pat/face) and from
    # transcripts (user/assistant turns), read by the Lite reactor's context.
    mood = Mood()

    # attention (t6/t12) — the cold/warm window the robot's own name opens.
    # ONE object, three readers: the speaker gates an utterance on it, the
    # tools refuse effectful moves taken off ambient nameless speech, and the
    # gaze stack reads its conversation liveness. Off (NOVA_ATTENTION_GATE=0)
    # means None everywhere, which is exactly the answer-everything robot of
    # every previous round — every reader already treats None as "no gate".
    attention: AttentionState | None = None
    if switches.attention_gate:
        attention = AttentionState(quiet=quiet)
    else:
        _stage(
            "supervise", "nova", "component", "component absent name=attention reason=switch-off"
        )

    speaker = SonicSpeaker(
        gate=gate,
        quiet=quiet,
        chunked=switches.chunked_playback,
        attention=attention,
    )

    # memory (t4/t10, c13) — the raw ledger and the Lite compactor that
    # distils it. Built BEFORE Sonic because Sonic takes the compactor's
    # history as a constructor argument: the replay hook is what makes a
    # session rotation keep the topic (c12) instead of starting blank.
    ledger, compactor = build_memory(switches, quiet)

    sonic = NovaSonic(
        system_prompt=build_system_prompt(persona.text),
        tools=TOOL_SPECS,
        on_audio_output=None,  # bound below once speaker exists
        region=config.region(),
        model_id=config.sonic_model_id(),
        voice_id=SONIC_VOICE_ID,
        history_provider=None if compactor is None else compactor.history,
        # Rotation waits for an idle moment, and playback lags generation by at
        # least a chunk — so "is the speaker still saying the last reply out
        # loud" is part of idle. Wired even with memory off: there is nothing
        # to replay then, but a rotation must still not cut a chunk in half.
        speaker_idle=lambda: speaker.idle,
    )

    # speak leg
    sonic.on_audio_output = speaker.on_audio_chunk
    sonic.on_interruption = speaker.preempt

    # ``gaze`` is bound further down (it needs ``intents``), but this tap is
    # only ever CALLED at runtime, long after build_app returned — so the
    # closure reads whatever the stack ended up being, or None.
    gaze: GazeStack | None = None

    def _on_state_change(state: str) -> None:
        speaker.on_state_change(state)
        # Nova starting to speak is a conversation tick for the CONVERSATION
        # layer only, so it is wired behind that layer's own switch.
        if gaze is not None and switches.face_hold:
            gaze.on_sonic_state(state)

    sonic.on_state_change = _on_state_change

    # daemon client (t9/t10) — the ONE shared HTTP client, resolved against the
    # exact same base URL speaking.py's own SonicSpeaker already resolved
    # (env NOVA_DAEMON_URL, default http://localhost:8000), so the volume
    # tools and the playback poster/stopper are always talking to the same
    # daemon. raise_voice/lower_voice/set_voice_level drive it via IntentTools;
    # restore_volume() below re-applies a persisted level on this same client.
    daemon_client = DaemonClient(base_url=speaker.base_url)

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

    # forge leg — the kiro writer is opt-in (FORGE_WRITER=kiro, deviation d1):
    # a standing watchdogged ACP session plus the forge/use_skill tool surface.
    # Anything failing here degrades to a named absent component, never a crash.
    # resolve_writer() is the SAME normalization SkillForge._run_inner uses
    # (qodo review comment 3812045200) — reading FORGE_WRITER inline here with
    # different normalization is exactly what let "KIRO" expose these
    # components while every dispatch through them was rejected.
    forge_leg = None
    kiro_unit = None
    if resolve_writer() == "kiro":
        try:
            from ..kiro_acp import KiroAcpSession
            from .forge_leg import ForgeLeg
            from .kiro_session import KiroSessionUnit

            kiro_unit = KiroSessionUnit(KiroAcpSession, cwd=str(Path.home()))
            forge_leg = ForgeLeg(sonic, kiro_unit)
        except Exception as err:  # noqa: BLE001
            kiro_unit = None
            forge_leg = None
            _stage(
                "supervise", "nova", "component", f"component absent name=kiro-writer reason={err}"
            )
    else:
        _stage(
            "supervise", "nova", "component", "component absent name=kiro-writer reason=writer-http"
        )

    # sense history (t8) — a small, cheap, in-process ring buffer, so it is
    # ALWAYS constructed (no failure mode worth degrading over, same
    # reasoning as the network leg) and shared between the bus, which
    # populates it, and IntentTools, whose recall_senses tool reads it.
    history = SenseHistory()

    # lock awareness (t13) — the harness's own belief about the runtime's
    # gaze lock. lock_face/release_face mirror a CONFIRMED verdict into it
    # below; the bus mirrors the runtime's own motion/lock-released events
    # into it (wired once bus_component exists, further down); the
    # supervisor clears it on an engine-heartbeat drop (see
    # supervisor._find_lock_state / run()'s lock_state kwarg) so a stale
    # local belief can never outlive the engine process that actually held
    # the lock.
    lock_state = LockState()

    intents = IntentTools(
        browser=browser,
        on_browse_progress=sonic.inject_text if browser is not None else None,
        forge_leg=forge_leg,
        daemon_client=daemon_client,
        history=history,
        quiet=quiet,
        lock_state=lock_state,
        attention=attention,
    )

    # gaze stack (t8/t9/t10) — the harness's single-writer posture layer. It
    # exists when EITHER layer is wanted: face_hold owns the CONVERSATION
    # layer, think_posture the BROWSING one. The producers below are wired
    # per-switch, so "the stack exists" never means "both layers move the
    # head" — with only one switch on the other layer simply never has a
    # producer to raise it.
    # ``conversation_enabled`` is the face_hold switch itself, not just the
    # wiring: the stack carries the shared attention (and has its own fallback
    # liveness clock), so leaving the layer merely unwired would still let it
    # enter conversation and issue look_at_sound/lock_face with the hold off
    # (PR #26 review).
    if switches.face_hold or switches.think_posture:
        gaze = GazeStack(
            intents,
            attention=attention,
            lock_state=lock_state,
            conversation_enabled=switches.face_hold,
        )
    else:
        _stage("supervise", "nova", "component", "component absent name=gaze reason=switch-off")

    # volume restore (t10) — re-apply a persisted level if the daemon disagrees.
    # Never raises: no persisted file, an unreachable daemon, or a bad payload
    # all degrade to the standard component-absent line, same as every other
    # optional leg in this function.
    try:
        restore_volume(statedir.volume_state_path(), daemon_client)
    except Exception as err:  # noqa: BLE001 - a volume hiccup must not stop the voice
        _stage(
            "supervise", "nova", "component", f"component absent name=volume reason={err}"
        )
    if browser is not None:
        # The browse tool's own result is just the "queued" acknowledgment —
        # the ANSWER arrives minutes later on the worker thread, and reaches
        # the conversation only through this callback.
        def _on_browse_result(text: str) -> None:
            # Order is the point: the head comes OUT of the thinking pose
            # (synchronously, on this thread) before Nova starts talking
            # about what it found — otherwise it delivers the answer staring
            # up and away. ``must_deliver`` because this is an ANSWER the
            # user asked for minutes ago, not an ambient cue: it must survive
            # a throttle, a cold attention window and an inactive stream.
            if gaze is not None:
                gaze.clear_for_result()
            sonic.inject_text(
                f"Your web browsing finished. Tell the user what you found: {text}",
                must_deliver=True,
                sense_class="browse",
            )
            if attention is not None:
                attention.note_inject()

        browser.on_result = _on_browse_result
        if switches.think_posture and gaze is not None:
            browser.on_state_change = gaze.on_browser_state

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
            if ledger is not None:
                # A reply the attention gate refused to play was never heard
                # by anyone, so it is not part of the conversation and must
                # not be distilled into memory (Ledger.append's `dropped`).
                # The cognition feed and the mood are deliberately NOT gated
                # the same way: the feed is the journal of what Nova produced
                # (a dropped reply is exactly what an operator wants to see),
                # and the mood is about having taken a turn at all.
                ledger.append(
                    "ASSISTANT", text, dropped=(speaker.attention_verdict() == "not-addressed")
                )
            mood.note("assistant_turn")
        elif role == "USER" and text.strip():
            _stage("hear", "nova", "transcript", f"heard {text[:120]!r}")
            # Attention FIRST: note_transcript is what opens (or renews) the
            # warm window, and recheck_attention immediately re-reads the
            # verdict for an utterance already in flight — a name arriving
            # mid-sentence un-mutes the rest of it.
            if attention is not None:
                attention.note_transcript(text)
                speaker.recheck_attention()
            if gaze is not None and switches.face_hold:
                gaze.on_transcript(role, text)
            if ledger is not None:
                ledger.append("USER", text)
            mood.note("user_turn")
            # Playback-aware barge-in: Sonic's own interruption path only runs
            # while it is GENERATING, but the audible playback happens after
            # (upload + play on the daemon). A user transcript while the gate
            # window is armed means they are talking over the robot's actual
            # voice — cut it (stop_sound + purge) rather than talking on.
            if gate.active:
                _stage("speak", "nova", "barge-in", "user spoke over playback — preempting")
                speaker.preempt()

    sonic.on_transcript = _on_transcript

    # inject wrapper (t14) — the ONE seam every bus cue passes through on its
    # way to the conversation. The bus used to hand its cues straight to
    # ``sonic.inject_text``; three things now hang off the same call, and they
    # hang off it HERE rather than inside the bus so the bus keeps knowing
    # nothing about ledgers or moods. ``sense_class`` is a real keyword, which
    # is what makes NovaBus pass it (it introspects the callable once).
    def _inject(text: str, sense_class: str | None = None) -> None:
        # The ledger records DELIVERED senses only (PR #24 review): a cue
        # dropped as throttled/inactive never reached the model, and a
        # deferred one is appended when the drain actually sends it (see
        # sonic.on_deferred_delivered below). Mood is about what happened to
        # the body, so it is noted either way.
        status = sonic.inject_text(text, sense_class=sense_class)
        if status == "sent" and ledger is not None:
            ledger.append("sense", text, sense_class=sense_class)
        # A cue that REACHED the model (now, or parked for the drain) is the
        # conversation still happening: it renews the attention window, so a
        # body cue Nova answers keeps the robot warm for the reply.
        if attention is not None and status in ("sent", "deferred"):
            attention.note_inject()
        if sense_class in MOOD_SENSE_CLASSES:
            mood.note(sense_class)

    if ledger is not None:
        sonic.on_deferred_delivered = lambda text, sense_class: ledger.append(
            "sense", text, sense_class=sense_class
        )

    # Lite reactor context (t11) — what the fast tier gets to reason with:
    # the recent senses, the day's memory, the mood, and the last exchanges.
    # Called on the reactor's OWN worker thread, so it never raises into it:
    # a broken context is an empty context, not a lost reaction.
    def _reaction_context() -> dict:
        try:
            senses = [str(entry.get("text", "")) for entry in history.recent(REACTION_SENSES)]
        except Exception:  # noqa: BLE001
            senses = []
        try:
            memory_text = "" if compactor is None else render_memory_paragraph(compactor.memory())
        except Exception:  # noqa: BLE001
            memory_text = ""
        try:
            exchanges = recent_exchanges(ledger)
        except Exception:  # noqa: BLE001
            exchanges = []
        return {
            "senses": senses,
            "memory": memory_text,
            "mood": mood.render(),
            "exchanges": exchanges,
        }

    # A gesture the Lite plan named goes through the SAME spool every tool
    # call uses — the reactor never touches the engine itself.
    def _reaction_gesture(name: str) -> None:
        try:
            intents.execute(
                "run_behavior", {"name": name, "duration": REACTION_GESTURE_DURATION_S}
            )
        except Exception as err:  # noqa: BLE001 - a refused gesture is not a lost reaction
            _stage(
                "act",
                "nova",
                "lite-gesture",
                f"dropped reason=lite-gesture-failed name={name} detail={err}",
            )

    # hear leg — the echo-gate policy is wired EXPLICITLY (resolve_policy reads
    # $NOVA_ECHO_GATE, default off) so the wiring is assertable, not ambient.
    hearing = TeeHearing(
        feed=sonic.feed_audio, gate=gate, echo_gate_policy=resolve_policy()
    )

    # ``intents`` is in the component list for ONE reason: its tick() poll is
    # what restores the runtime's own voice when a quiet EXPIRES rather than
    # being ended by hand (see IntentTools.tick).
    components: list[object] = [sonic, speaker, hearing, intents]

    # kiro writer — the standing session restarts under the supervisor like
    # any other component; the forge/use_skill tools above already hold it.
    if kiro_unit is not None:
        components.append(kiro_unit)

    # network leg — ALWAYS constructed: it is cheap, local, and reads nothing
    # but /proc, `ip addr` and a state-dir file, so there is no failure mode
    # worth degrading over. The reactor is appended FIRST so the supervisor
    # hands it the stop_event before the poll thread can fire a transition at
    # it (the Sonic fallback restart path needs that event).
    network_reactor = NetworkReactor(sonic, kiro_unit)
    network_unit = NetworkUnit()
    network_unit.on_change(network_reactor.on_network_change)
    components.append(network_reactor)
    components.append(network_unit)

    # Lite reaction tier (t11/t13) — appended BEFORE the bus so the supervisor
    # has started its worker by the time the first cue can be handed to it.
    # Off (NOVA_LITE_REACTIONS=0) or unbuildable means reactor=None below,
    # which is exactly the template-only rendering of every previous round.
    lite_reactor: object | None = None
    if switches.lite_reactions:
        try:
            from .lite_reactor import LiteReactor

            # Lite plans may ask for a sound (chirp|trill|purr). The harness has no
            # SDK speaker, so the legacy synthesiser's samples ride the SonicSpeaker
            # like a reply chunk: 24 kHz, flushed by inactivity, gate-serialised
            # (PR #24 review: vocalizations were parsed and then silently ignored).
            def _vocalize(kind: str) -> None:
                try:
                    from ..vocalize import synthesize

                    samples = synthesize(VOCALIZE_KINDS[kind], sample_rate=24000)
                    speaker.on_audio_chunk(np.asarray(samples, dtype=np.float32))
                except Exception as err:  # noqa: BLE001 - a sound must never cost a reaction
                    _stage("react", "lite", "vocalize", f"dropped reason=vocalize-failed kind={kind} ({err})")

            lite_reactor = LiteReactor(
                context_provider=_reaction_context, on_gesture=_reaction_gesture,
                on_vocalize=_vocalize,
            )
            components.append(lite_reactor)
        except Exception as err:  # noqa: BLE001
            lite_reactor = None
            _stage(
                "supervise", "nova", "component", f"component absent name=lite-reactor reason={err}"
            )
    else:
        _stage(
            "supervise", "nova", "component", "component absent name=lite-reactor reason=switch-off"
        )

    # The bus's event tap: the lock belief ALWAYS, plus — when a gaze stack
    # exists — the stack's own clear of a hold the runtime just dropped. Two
    # readers of one event, composed here rather than inside either of them.
    def _bus_tap(event: dict) -> None:
        lock_state.on_bus_event(event)
        if gaze is None:
            return
        if event.get("source") == "motion" and event.get("type") == "lock-released":
            gaze.on_lock_released(event.get("reason"))

    # read leg — bus is optional-degraded: no broker means named drops, not death
    bus_component = None
    try:
        from .bus import NovaBus

        bus_component = NovaBus(
            on_inject=_inject,
            on_event=_bus_tap if gaze is not None else lock_state.on_bus_event,
            history=history,
            quiet=quiet,
            reactor=lite_reactor,
        )
        components.append(bus_component)
    except Exception as err:  # noqa: BLE001
        _stage("supervise", "nova", "component", f"component absent name=bus reason={err}")

    # memory compactor (t10) — after the bus: its Lite call is periodic and
    # slow, nothing waits on it, and it must never be between a cue and the
    # mouth. Its own thread does the work; Sonic only reads its history().
    if compactor is not None:
        components.append(compactor)

    # gaze stack — after the bus (its tap may reach the stack) and before the
    # browser, whose state changes raise the browsing layer.
    if gaze is not None:
        components.append(gaze)

    if browser is not None:
        components.append(BrowserComponent(browser))

    # vision leg's inject wrapper (live finding 2026-09-06 00:23/00:38/00:43):
    # the raw Omni description (400-850 chars) handed to Sonic bare produced a
    # 30 s unprompted monologue about the scene at every harness start. A
    # glance is a body cue like any other: parenthesised, capped, marked brief.
    def _vision_cue(text: str) -> None:
        _inject(render_vision_cue(text), sense_class="vision")

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
                    on_answer=_vision_cue,
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

    # eyes (t11) — its own ~1 Hz subscription to the runtime's snapshot topic,
    # so a camera that stops producing frames is NAMED once instead of being
    # inferred from silence. Construction opens no socket (start() does), but
    # it is wrapped like every other optional leg so a missing dependency is
    # an absent component rather than a harness that will not compose. Its
    # ``.eyes`` attribute is what supervisor._find_eyes_state discovers on the
    # component list run() already receives — supervisor.run takes no
    # eyes_state kwarg, so there is nothing to pass.
    try:
        components.append(eyes_module.build_component())
    except Exception as err:  # noqa: BLE001
        _stage("supervise", "nova", "component", f"component absent name=eyes reason={err}")

    # standing reflexes — with the hold OFF the face cue crosses the bus only
    # through this rule, so it is (re)installed; with the hold ON the nod that
    # rule runs would fight the hold for the head, so the rule is tombstoned
    # instead and never re-installed (installing then retiring on every boot
    # cost two overlay writes and two reloads for nothing — t12 review).
    if switches.face_hold:
        retire_face_nod_rule()
    else:
        ensure_face_rule()

    return components
