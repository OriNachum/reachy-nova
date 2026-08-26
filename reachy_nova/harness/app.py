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
    NetworkUnit joined/moved ──NetworkReactor──► sonic.request_immediate_restart
                                              └► kiro_unit.request_restart

Optional legs degrade, never crash: the browser exists only when
``NOVA_ACT_ENABLED`` is on, the vision leg only when ``NOVA_OMNI_MODEL_ID`` is
set (the Omni preview is model-config-gated) AND the bus built (it supplies the
retained clip state). Every absent leg emits the standard
``component absent name=<leg> reason=<why>`` senselog line, so "we started
without seeing" is visible rather than inferred.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .. import config
from ..nova_browser import act_enabled
from ..sensory_log import stage as _stage
from ..skill_forge import resolve_writer
from . import statedir
from .cognition_feed import CognitionFeed
from .daemon_client import DaemonClient, restore_volume
from .gate import EchoGate, resolve_policy
from .hearing import TeeHearing
from .lock_state import LockState
from .network import NetworkUnit
from .quiet import QuietState
from .rules_overlay import upsert_rule
from .sense_history import SenseHistory
from .speaking import SonicSpeaker
from .tools import TOOL_SPECS, IntentTools

HARNESS_SYSTEM_PROMPT = (
    "You are Nova, the mind of a small Reachy Mini robot. You hear the room "
    "through your microphones and speak aloud through your speaker. Keep your "
    "words short, warm, and natural — you are a curious household companion, "
    "not an assistant reading documentation. "
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
    "Body cues arrive in parentheses — react to them naturally, with a word, "
    "a sound, or nothing at all, and never describe your own mechanism (no "
    "'reflex', 'rule', 'my body reacted on its own') unless someone asks why. "
    "When someone asks why you did something, what you felt, or what just "
    "happened, call recall_senses and answer from what it returns."
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
# The composition root                                                         #
# --------------------------------------------------------------------------- #


def build_app() -> list[object]:
    """Construct and wire every harness component; return them in start order."""
    from ..nova_sonic import NovaSonic  # heavy AWS SDK import kept out of module import

    gate = EchoGate()
    feed = CognitionFeed()

    # timed quiet (t11/t12) — ONE object, three readers: the speaker gates
    # playback on it, the bus marks every inject with it, and the
    # stay_silent/end_silence tools arm and release it. Constructed here
    # rather than inside any of the three so they can never disagree about
    # whether the robot is currently supposed to be quiet. It reloads a
    # still-future deadline off disk, so a restart inside a quiet window
    # comes back quiet instead of loudly reintroducing itself.
    quiet = QuietState()

    speaker = SonicSpeaker(gate=gate, quiet=quiet)

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
    )

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
            sonic.inject_text(f"Your web browsing finished. Tell the user what you found: {text}")

        browser.on_result = _on_browse_result

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
    reactor = NetworkReactor(sonic, kiro_unit)
    network_unit = NetworkUnit()
    network_unit.on_change(reactor.on_network_change)
    components.append(reactor)
    components.append(network_unit)

    # read leg — bus is optional-degraded: no broker means named drops, not death
    bus_component = None
    try:
        from .bus import NovaBus

        bus_component = NovaBus(
            on_inject=sonic.inject_text,
            on_event=lock_state.on_bus_event,
            history=history,
            quiet=quiet,
        )
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
