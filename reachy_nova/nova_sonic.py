"""Nova Sonic - Bidirectional speech-to-speech via Amazon Bedrock."""

import asyncio
from collections import deque
import base64
import json
import logging
import random
import threading
import uuid
import time
from collections.abc import Callable

import numpy as np

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from . import config
from .harness.history_blocks import normalise_history
from .harness import deferred_cues
from .harness.deferred_cues import DeferredCues
from .sensory_log import stage as sensory_stage

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHUNK_DURATION_MS = 100
INPUT_CHUNK_SIZE = int(INPUT_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# Resilience watchdogs (see _check_clock_step / _check_response_liveness).
# The robot has no RTC: it can boot with a stale clock and have NTP step time
# forward by hours while a Bedrock stream is already open.  A stepped clock
# turns that stream into a zombie — sends keep succeeding, no response event
# ever comes back, and nothing raises — so it must be detected, not waited on.
CLOCK_STEP_THRESHOLD_S = 60.0
# 15 minutes, not 3 (2026-09-06, robot): the proactive session rotation
# (NOVA_SONIC_ROTATE_S, ~7 min, hard deadline 470 s) already replaces any
# session — a zombie included — so this watchdog is a conservative second net,
# not the first line of defence. At 180 s it restarted a quiet room every three
# minutes on a desk where chairs, typing and a person moving about produce
# speech-level bursts that Bedrock (rightly) never answers.
DEFAULT_LIVENESS_S = 900.0
# History-replay circuit breaker: a session that dies this soon after a start
# that replayed history counts as a replay death; two in a row suspend replay.
REPLAY_DEATH_WINDOW_S = 10.0
REPLAY_DEATHS_TO_SUSPEND = 2


def _liveness_window() -> float:
    """Seconds of response silence (with input flowing) that means "zombie".

    Read at call time so ``load_dotenv()`` order never matters; a missing,
    unparseable or non-positive ``NOVA_SONIC_LIVENESS_S`` falls back to the
    default rather than disabling the watchdog.
    """
    import os

    try:
        value = float(os.environ.get("NOVA_SONIC_LIVENESS_S", ""))
    except (TypeError, ValueError):
        return DEFAULT_LIVENESS_S
    return value if value > 0 else DEFAULT_LIVENESS_S


# Speech floor for the response-liveness watchdog (see feed_audio).
#
# The harness feeds the microphone to Sonic ten times a second whether or not
# anybody is talking, so "input flowing, no response for 180s" used to mean
# "nobody spoke for three minutes": the robot's journal for 2026-09-05 carried
# six "Response liveness stall forced a restart" lines exactly 180s apart in a
# quiet room, each one a clean session that wiped the conversation. Only a
# chunk loud enough to be speech counts as input now. The default sits between
# the measured quiet-room floor (~0.002 RMS) and human speech (~0.09 RMS) on
# this microphone — see harness/hearing.py's echo-gate measurement.
DEFAULT_SPEECH_FLOOR = 0.05
# A single loud chunk is a cough, a door, a dropped spoon — not a person
# talking to the robot. Only a BURST of speech-level chunks counts as input
# for the liveness watchdog: at least SPEECH_BURST_CHUNKS above the floor
# within the last SPEECH_BURST_WINDOW chunks (2 s at 100 ms/chunk). Measured
# on the robot 2026-09-06 00:15 (quiet house): mic RMS median 0.005, one
# chunk in ten above 0.02, so a single-chunk trigger restarted the stream
# every 3 min of silence even after the floor was introduced — and at 0.02
# with a burst rule a person moving around the desk still did (00:27, 00:30).
# 0.05 is conversational speech near the robot (measured ~0.09 close-talk).
SPEECH_BURST_CHUNKS = 5
SPEECH_BURST_WINDOW = 20

# Turn detection (see _endpointing_sensitivity).
#
# Nova 2 Sonic's sessionStart accepts turnDetectionConfiguration.
# endpointingSensitivity: HIGH "detects pauses quickly, enabling faster
# responses but may cut off slower speakers", LOW waits longer. The harness
# used to send neither, leaving the service default to decide how long a
# pause has to be before Nova may answer.
DEFAULT_ENDPOINTING = "HIGH"
ENDPOINTING_VALUES = ("HIGH", "MEDIUM", "LOW")


def _speech_floor() -> float:
    """RMS above which a microphone chunk counts as input (``NOVA_SONIC_SPEECH_FLOOR``).

    Read at call time like :func:`_liveness_window`, and with the same
    fallback rule: a missing, unparseable or non-positive value means the
    default rather than a gate that lets everything (or nothing) through.
    """
    import os

    try:
        value = float(os.environ.get("NOVA_SONIC_SPEECH_FLOOR", ""))
    except (TypeError, ValueError):
        return DEFAULT_SPEECH_FLOOR
    return value if value > 0 else DEFAULT_SPEECH_FLOOR


def _endpointing_sensitivity() -> str:
    """Turn-detection sensitivity for ``sessionStart`` (``NOVA_SONIC_ENDPOINTING``).

    Read at call time so ``load_dotenv()`` order never matters, and
    case-insensitive because this is a knob typed into a ``.env`` by hand. An
    unrecognised value is a NAMED warning plus the default — a typo must not
    cost the session, and must not be silent either.
    """
    import os

    raw = os.environ.get("NOVA_SONIC_ENDPOINTING", "").strip()
    if not raw:
        return DEFAULT_ENDPOINTING
    value = raw.upper()
    if value not in ENDPOINTING_VALUES:
        logger.warning(
            f"Unrecognised NOVA_SONIC_ENDPOINTING={raw!r} — expected one of "
            f"{'/'.join(ENDPOINTING_VALUES)}; using {DEFAULT_ENDPOINTING}"
        )
        return DEFAULT_ENDPOINTING
    return value


# Restart backoff (see NovaSonic._compute_restart_delay).
#
# With no network, a stream open attempt fails fast (AWS_IO_DNS_QUERY_FAILED)
# and the OLD fixed-3s retry reopened a Bedrock stream every ~6s for as long
# as the robot stayed offline. That is noisy (log spam, thread churn) and
# gains nothing: the network is not coming back in the next 3 seconds. An
# exponential backoff — reset once a session proves itself healthy for a
# while — keeps a flaky network quiet without slowing down recovery once
# the network is actually back (the very next restart uses the base delay).
DEFAULT_RESTART_BASE_S = 3.0
DEFAULT_RESTART_MAX_S = 60.0
# How long a session must run cleanly (armed, at least one sign of life)
# before a subsequent death is treated as a fresh problem rather than a
# continuation of the same outage, resetting the backoff to the base delay.
RESTART_HEALTHY_RESET_S = 60.0
# Upper bound on the random jitter added on top of the backoff delay.
RESTART_JITTER_FRACTION = 0.10


# Proactive session rotation (see NovaSonic._rotation_due).
#
# Measured on the robot 2026-09-05: the one session that outlived the liveness
# window started 21:51:29 and died at 21:59:30 — 480.5 s — with Bedrock's
# "Model has timed out in processing the request". The Nova 2 Sonic connection
# limit on this account really is 8 minutes, and hitting it costs the whole
# conversation plus the base backoff. So the harness replaces the session
# itself, a little early and at a moment when nothing is in flight.
DEFAULT_ROTATE_S = 420.0
# Bedrock drops a Nova 2 Sonic stream that carries no INTERACTIVE content
# (speech or text) for 295 s — silent audio bytes do not count ("Please ensure
# gaps between audio bytes and interactive content are less than 295 seconds",
# robot 2026-09-06, three drops at exactly 296 s of quiet). A quiet session is
# therefore rotated cleanly a little before that, at an idle moment.
DEFAULT_IDLE_ROTATE_S = 270.0
#: The substring of Bedrock's own idle-cutoff message.
IDLE_CUTOFF_MARKER = "gaps between audio bytes and interactive content"
# ...and if nothing is ever idle, rotate anyway rather than let Bedrock do it
# for us at 480 s. This is the last quiet exit before the ceiling.
DEFAULT_ROTATE_DEADLINE_S = 470.0
# How many conversation-history blocks a fresh session replays at most.
DEFAULT_HISTORY_MAX_BLOCKS = 8
# Roles the Nova 2 input-events page accepts for replayed history content.
HISTORY_ROLES = ("USER", "ASSISTANT")


def _idle_rotate_s() -> float:
    """Seconds without interactive content before an idle rotation (``NOVA_SONIC_IDLE_ROTATE_S``).

    Parsed like ``_rotate_interval_s``; ``0`` or negative disables it.
    """
    import os

    raw = os.environ.get("NOVA_SONIC_IDLE_ROTATE_S", "")
    if raw.strip() == "":
        return DEFAULT_IDLE_ROTATE_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_IDLE_ROTATE_S
    return value if value > 0 else 0.0


def _rotate_interval_s() -> float:
    """Session age at which a rotation becomes *possible* (``NOVA_SONIC_ROTATE_S``).

    Read at call time like :func:`_liveness_window`, with the same fallback
    for a missing or unparseable value — but here ``0`` (or negative) is a
    meaningful setting rather than nonsense: it turns proactive rotation off
    entirely and leaves the session to die at the service ceiling as before.
    """
    import os

    try:
        value = float(os.environ.get("NOVA_SONIC_ROTATE_S", ""))
    except (TypeError, ValueError):
        return DEFAULT_ROTATE_S
    return value if value > 0 else 0.0


def _rotate_deadline_s() -> float:
    """Session age past which a rotation happens regardless of what is in flight.

    (``NOVA_SONIC_ROTATE_DEADLINE_S``, default 470 — ten seconds of margin on
    the measured 480.5 s ceiling.) Unlike the interval this has no "off"
    value: a deadline of zero would mean "always past it", so a non-positive
    setting falls back to the default. Rotation as a whole is switched off
    with ``NOVA_SONIC_ROTATE_S=0``.
    """
    import os

    try:
        value = float(os.environ.get("NOVA_SONIC_ROTATE_DEADLINE_S", ""))
    except (TypeError, ValueError):
        return DEFAULT_ROTATE_DEADLINE_S
    return value if value > 0 else DEFAULT_ROTATE_DEADLINE_S


def _restart_base_s() -> float:
    """Base restart delay in seconds (``NOVA_SONIC_RESTART_BASE_S``, default 3)."""
    import os

    try:
        value = float(os.environ.get("NOVA_SONIC_RESTART_BASE_S", ""))
    except (TypeError, ValueError):
        return DEFAULT_RESTART_BASE_S
    return value if value > 0 else DEFAULT_RESTART_BASE_S


def _restart_max_s() -> float:
    """Restart delay cap in seconds (``NOVA_SONIC_RESTART_MAX_S``, default 60)."""
    import os

    try:
        value = float(os.environ.get("NOVA_SONIC_RESTART_MAX_S", ""))
    except (TypeError, ValueError):
        return DEFAULT_RESTART_MAX_S
    return value if value > 0 else DEFAULT_RESTART_MAX_S


class NovaSonic:
    """Manages a bidirectional voice conversation with Nova Sonic."""

    def __init__(
        self,
        region: str | None = None,
        model_id: str | None = None,
        voice_id: str = "matthew",
        system_prompt: str = (
            "You are Nova, a small curious robot. "
            "You see, hear, and feel the world around you. "
            "Keep your words short and natural. "
            "You're warm, playful, and endlessly curious."
        ),
        on_transcript: Callable[[str, str], None] | None = None,
        on_audio_output: Callable[[np.ndarray], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
        tools: list[dict] | None = None,
        on_tool_use: Callable[[str, str, dict], None] | None = None,
        on_interruption: Callable[[], None] | None = None,
        restart_rng: random.Random | None = None,
        deferred_ttl_s: float = deferred_cues.DEFAULT_TTL_S,
        history_provider: Callable[[], list[dict]] | None = None,
        history_max_blocks: int = DEFAULT_HISTORY_MAX_BLOCKS,
        speaker_idle: Callable[[], bool] | None = None,
    ):
        self.region = region or config.region()
        self.model_id = model_id or config.sonic_model_id()
        self.voice_id = voice_id
        self.system_prompt = system_prompt
        self.on_transcript = on_transcript
        self.on_audio_output = on_audio_output
        self.on_state_change = on_state_change
        self.tools = tools
        self.on_tool_use = on_tool_use
        self.on_interruption = on_interruption
        self._decision_client = None  # lazy boto3 client for barge-in decisions

        self._client: BedrockRuntimeClient | None = None
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._active = False
        self._speaking = False
        self._sonic_stop = threading.Event()  # Independent stop for sleep mode

        self._prompt_name = str(uuid.uuid4())
        self._system_content = str(uuid.uuid4())
        self._audio_content = str(uuid.uuid4())

        # Serialize inject/tool-result sends so they don't overlap on the stream
        self._inject_lock: asyncio.Lock | None = None
        # Session generation — incremented on restart to discard stale coroutines
        self._session_gen = 0
        # Throttle inject_text to prevent flooding the stream
        self._last_inject_time = 0.0
        self._inject_min_interval = 3.0  # seconds between inject_text calls

        # Body cues that arrived mid-utterance. The speaking guard below can
        # not send them (injecting into a generating stream can hang it), but
        # it no longer throws them away either: they are parked here and
        # delivered, with their age in the text, the moment the utterance
        # ends. See reachy_nova/harness/deferred_cues.py.
        self._deferred = DeferredCues(ttl_s=deferred_ttl_s)

        # Conversation history replayed into every fresh session — the one
        # documented place it may go ("only once, after the system prompt and
        # before audio streaming begins"). Any zero-argument callable
        # returning ``[{"role": "USER"|"ASSISTANT", "text": ...}, ...]`` will
        # do; the harness passes the memory compactor's ``history``.
        self._history_provider = history_provider
        self._history_max_blocks = history_max_blocks

        # Is the *speaker* done? A rotation must not cut a chunk that is
        # still playing, and playback lags generation by at least one chunk,
        # so our own ``_speaking`` flag is not enough. Missing = idle.
        self._speaker_idle = speaker_idle
        # Session age (seconds) of a rotation that has been decided but whose
        # new session has not started yet — carried across so the journal line
        # can name the age AND the replay count in one grep-able line.
        self._pending_rotation: float | None = None
        # History-replay circuit breaker (live incident 2026-09-06): a stream
        # that dies within REPLAY_DEATH_WINDOW_S of a start that replayed
        # history, twice in a row, means the history itself is what Bedrock
        # refuses — replay is suspended until a session lives long enough.
        self._replay_blocks_last = 0
        self._replay_deaths = 0
        self._replay_suspended = False
        #: Called with (text, sense_class) after a DEFERRED cue is actually
        #: sent (the drain), so the ledger records delivered senses only
        #: (PR #24 review): the app wires it to ledger.append.
        self.on_deferred_delivered: Callable[[str, str | None], None] | None = None
        self._rotation_age_s: float | None = None
        # Monotonic time of the last INTERACTIVE content we sent (a speech
        # burst, a text inject, a tool result, the history replay).
        self._last_interactive_mono: float | None = None
        # Why the last stream death happened, as Bedrock phrased it.
        self._last_stream_error: str = ""

        # Tool use tracking
        self._current_tool_use: dict | None = None

        # Speech-burst window for the liveness watchdog (see feed_audio).
        self._energy_recent: deque[bool] = deque(maxlen=SPEECH_BURST_WINDOW)

        # Watchdog bookkeeping: a generation that stalls mid-utterance (stream
        # hang, lost contentEnd) must not pin the speaking guard forever.
        self._last_audio_time = 0.0

        # Resilience watchdog bookkeeping — armed once per session start by
        # _arm_watchdogs(), so a forced restart always re-arms both of them.
        self._session_clock_offset: float | None = None
        self._clock_step_seen = False
        self._last_response_mono: float | None = None
        self._input_since_response = False
        self._liveness_stall_seen = False
        # Why the current restart was forced (None = the stream simply died).
        self._forced_restart_reason: str | None = None

        # Restart backoff bookkeeping (see _compute_restart_delay).
        self._restart_attempt = 0
        self._restart_rng = restart_rng or random.Random()
        self._session_start_mono: float | None = None
        self._session_had_response = False
        # Set by request_immediate_restart(); honoured by _run_loop's wait
        # loop even when the current session looks perfectly healthy — the
        # caller (e.g. a network-change signal) knows better than we do.
        self._restart_now_event = threading.Event()
        self._immediate_restart_reason: str | None = None

        self.state = "idle"  # idle, listening, thinking, speaking
        self.last_user_text = ""
        self.last_assistant_text = ""

    def _set_state(self, state: str) -> None:
        self.state = state
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Resilience watchdogs
    # ------------------------------------------------------------------

    def _arm_watchdogs(self, wall: float, mono: float) -> None:
        """Capture the baselines both resilience watchdogs compare against.

        Called once per session start (including every restart), so each fresh
        session gets a fresh clock baseline and a fresh liveness deadline, and
        a cause that already fired can fire again on a later session.
        """
        self._session_clock_offset = wall - mono
        self._clock_step_seen = False
        self._last_response_mono = mono
        self._input_since_response = False
        self._liveness_stall_seen = False
        # Restart-backoff bookkeeping: a fresh session starts unproven.
        self._session_start_mono = mono
        self._session_had_response = False
        # A speech burst must be earned inside THIS session (PR #24 review).
        self._energy_recent.clear()
        self._last_interactive_mono = mono

    def _note_input_sent(self) -> None:
        """Record that we pushed something (audio or text) into the stream."""
        self._input_since_response = True
        self._last_interactive_mono = time.monotonic()

    def _note_interactive_sent(self) -> None:
        """Interactive content for Bedrock's idle clock, NOT liveness input."""
        self._last_interactive_mono = time.monotonic()

    def _note_response_event(self, mono: float | None = None) -> None:
        """Record a sign of life from Bedrock — resets the liveness deadline."""
        self._last_response_mono = time.monotonic() if mono is None else mono
        self._input_since_response = False
        self._session_had_response = True

    def _check_clock_step(self, wall: float, mono: float) -> bool:
        """True (once) when the wall clock was stepped under a live session.

        ``time.time() - time.monotonic()`` is constant while time merely
        passes; it only moves when something *sets* the wall clock (NTP on a
        Pi with no RTC).  The open Bedrock stream does not survive that, so
        the caller must restart the session.
        """
        if self._clock_step_seen or self._session_clock_offset is None:
            return False
        delta = (wall - mono) - self._session_clock_offset
        if abs(delta) <= CLOCK_STEP_THRESHOLD_S:
            return False
        self._clock_step_seen = True
        logger.warning(
            f"Clock step detected: wall clock moved {delta:+.0f}s relative to "
            f"monotonic since session start (threshold {CLOCK_STEP_THRESHOLD_S:.0f}s) "
            "— the open Bedrock stream is likely a zombie, forcing a restart"
        )
        return True

    def _check_response_liveness(self, mono: float) -> bool:
        """True (once) when input keeps flowing but Bedrock has gone silent.

        Silence alone is normal — a quiet room sends nothing and gets nothing
        back.  Silence *while we are still sending* is the zombie signature
        from the clock-step incident: injects accepted, zero response events,
        no error ever raised.
        """
        if self._liveness_stall_seen or not self._input_since_response:
            return False
        if self._last_response_mono is None:
            return False
        silent_for = mono - self._last_response_mono
        window = _liveness_window()
        if silent_for <= window:
            return False
        self._liveness_stall_seen = True
        logger.warning(
            f"Response liveness watchdog: input sent but no Bedrock response event "
            f"for {silent_for:.0f}s (limit {window:.0f}s) — forcing a session restart"
        )
        return True

    # ------------------------------------------------------------------
    # Proactive rotation
    # ------------------------------------------------------------------

    def _session_is_idle(self) -> bool:
        """True when replacing the session right now cuts nothing off.

        Four things can be in flight, and the restart path drops all four:
        Sonic generating a reply, the speaker still playing one (playback
        lags generation by at least a chunk), a tool call whose result would
        be discarded on the session-generation change, and the "thinking"
        gap between a heard transcript and the first audio. A missing
        ``speaker_idle`` callable counts as idle — an absent leg must not
        wedge the rotation shut — but a *raising* one counts as busy, since
        we then genuinely do not know; the hard deadline covers that case.
        """
        if self.state != "listening":
            return False
        if self._current_tool_use is not None:
            return False
        if self._speaking:
            return False
        if self._speaker_idle is None:
            return True
        try:
            return bool(self._speaker_idle())
        except Exception as e:  # pragma: no cover - defensive, logged not raised
            logger.debug(f"speaker_idle check failed: {e} — treating as busy")
            return False

    def _rotation_due(self, mono: float) -> float | None:
        """Session age at which to rotate *now*, or ``None`` to keep waiting.

        Called on every tick of the response-wait loop, so the common case —
        a session younger than the interval — costs one env lookup and a
        subtraction. Past the interval the session is replaced at the first
        idle moment; past the hard deadline it is replaced regardless,
        because Bedrock's own timeout at ~480 s is strictly worse (it takes
        the conversation with it and charges the backoff on the way out).
        """
        interval = _rotate_interval_s()
        if interval <= 0 or self._session_start_mono is None:
            return None
        age = mono - self._session_start_mono
        # Idle rotation: nothing interactive has been sent for a while and
        # Bedrock will cut the stream at 295 s anyway — swap it cleanly now.
        idle_s = _idle_rotate_s()
        if (
            idle_s > 0
            and self._last_interactive_mono is not None
            and mono - self._last_interactive_mono >= idle_s
            and self._session_is_idle()
        ):
            return age
        if age < interval:
            return None
        if age >= _rotate_deadline_s():
            return age
        return age if self._session_is_idle() else None

    # ------------------------------------------------------------------
    # Restart backoff
    # ------------------------------------------------------------------

    def _maybe_reset_backoff_for_healthy_session(self, death_mono: float) -> None:
        """Reset the backoff to the base delay if the session just proved itself.

        "Proved itself" means it was armed, actually heard back from Bedrock
        at least once (a real sign of life, not merely an open socket), and
        stayed up for at least ``RESTART_HEALTHY_RESET_S`` before dying. A
        session that never got a response, or died quickly, is treated as a
        continuation of the same outage and keeps escalating.
        """
        if self._session_start_mono is None or not self._session_had_response:
            return
        if death_mono - self._session_start_mono >= RESTART_HEALTHY_RESET_S:
            self._restart_attempt = 0

    def _note_session_death(self, death_mono: float) -> None:
        """Feed the history-replay circuit breaker with how this session ended.

        Called on the restart path with the monotonic time of death. A death
        within :data:`REPLAY_DEATH_WINDOW_S` of a start that replayed at least
        one history block is a *replay death*; :data:`REPLAY_DEATHS_TO_SUSPEND`
        of them in a row suspend replay (the next start sends none, with a
        warning). A session that lived past the window resets the count and
        lifts the suspension, so a transient refusal costs one clean session,
        not the memory forever.
        """
        age = None if self._session_start_mono is None else death_mono - self._session_start_mono
        if age is not None and age < REPLAY_DEATH_WINDOW_S and self._replay_blocks_last > 0:
            self._replay_deaths += 1
            if self._replay_deaths >= REPLAY_DEATHS_TO_SUSPEND and not self._replay_suspended:
                self._replay_suspended = True
                logger.warning(
                    f"history replay: {self._replay_deaths} sessions died within "
                    f"{REPLAY_DEATH_WINDOW_S:.0f}s of replaying history — suspending replay"
                )
        elif age is not None and age >= REPLAY_DEATH_WINDOW_S:
            if self._replay_suspended:
                logger.info("history replay: a session lived — replay re-enabled")
            self._replay_deaths = 0
            self._replay_suspended = False

    def _compute_restart_delay(self) -> tuple[float, int]:
        """Return ``(delay_seconds, attempt_number)`` for the next restart.

        ``attempt_number`` is 1 on the first consecutive failure, 2 on the
        second, and so on — reset to 1 whenever the backoff itself resets
        (a healthy session, or ``request_immediate_restart``). The delay
        doubles each attempt from the base, capped, then gets up to
        ``RESTART_JITTER_FRACTION`` of extra jitter on top (also capped, so
        the jitter can never push the delay past the ceiling).
        """
        attempt_number = self._restart_attempt + 1
        base = _restart_base_s()
        cap = _restart_max_s()
        delay = min(base * (2 ** self._restart_attempt), cap)
        jitter = delay * self._restart_rng.uniform(0.0, RESTART_JITTER_FRACTION)
        delay = min(delay + jitter, cap)
        self._restart_attempt += 1
        return delay, attempt_number

    def request_immediate_restart(self, reason: str) -> None:
        """Ask the sonic loop to restart the stream now, bypassing backoff.

        Safe to call from another thread (e.g. a network-change callback):
        it only sets an event and resets bookkeeping — the actual restart
        happens on the sonic loop's own thread. Restarts even a currently
        healthy session, because the caller (who observed the network
        change) knows something the liveness/clock watchdogs cannot: the
        open stream is bound to an address that no longer exists. Calling
        this twice in quick succession is harmless — the second call just
        overwrites the reason and re-sets an already-set event.
        """
        self._immediate_restart_reason = reason
        self._restart_attempt = 0
        self._restart_now_event.set()
        logger.info(f"Immediate restart requested: {reason}")

    def _should_interrupt(self, user_text: str, assistant_text: str) -> bool:
        """Ask Nova 2 Lite whether the user's speech warrants interrupting the robot."""
        if not self._decision_client:
            import boto3
            self._decision_client = boto3.client("bedrock-runtime", region_name=self.region)

        prompt = (
            f"The robot assistant was saying: \"{assistant_text[-200:]}\"\n"
            f"The user just said: \"{user_text}\"\n\n"
            "Is the user trying to interrupt, ask a question, change topic, or stop the robot? "
            "Or is it just a filler sound, acknowledgment, or background noise?\n"
            "Answer only: INTERRUPT or CONTINUE"
        )
        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 10, "temperature": 0.1, "topP": 0.9},
        }
        response = self._decision_client.invoke_model(
            modelId=config.lite_model_id(),
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        answer = result["output"]["message"]["content"][0]["text"].strip().upper()
        return "INTERRUPT" in answer

    async def _handle_barge_in(self, user_text: str, assistant_text: str):
        """Decide whether to interrupt playback when user speaks during robot speech."""
        try:
            should_interrupt = await asyncio.to_thread(
                self._should_interrupt, user_text, assistant_text
            )
            if should_interrupt:
                logger.info(f"Barge-in: interrupting (user said: {user_text!r})")
                self._speaking = False
                if self.on_interruption:
                    self.on_interruption()
            else:
                logger.info(f"Barge-in: continuing playback (user said: {user_text!r})")
        except Exception as e:
            logger.error(f"Barge-in decision failed: {e}")
            # On failure, default to interrupting (safer UX)
            self._speaking = False
            if self.on_interruption:
                self.on_interruption()

    def _init_client(self) -> None:
        import os
        endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com"
        key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
        logger.info(f"Init client: region={self.region}, endpoint={endpoint}")
        logger.info(f"  AWS_ACCESS_KEY_ID={key_id[:8]}..., session_token={'yes' if os.environ.get('AWS_SESSION_TOKEN') else 'no'}")
        config = Config(
            endpoint_uri=endpoint,
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self._client = BedrockRuntimeClient(config=config)
        logger.info("Client created OK")

    async def _send(self, event: dict) -> None:
        event_type = next(iter(event.keys()), "unknown")
        logger.debug(f"SEND → {event_type}")
        payload = json.dumps({"event": event}).encode("utf-8")
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=payload)
        )
        await self._stream.input_stream.send(chunk)
        logger.debug(f"SEND → {event_type} OK")

    async def _replay_history(self) -> int:
        """Send the conversation history as TEXT blocks; return how many crossed.

        Every restart goes through here — proactive rotation, the liveness and
        clock-step watchdogs, the network-change path and an ordinary stream
        death — so "the robot forgets everything on every restart" is fixed in
        one place rather than four.

        The provider is somebody else's code (the harness's memory compactor,
        reading a file off a disk that has been 90 % full), so it never takes
        the session down with it: a raising provider is one warning and an
        empty replay. Blocks past ``history_max_blocks`` are dropped from the
        end (the provider returns oldest-first, with its context summary at
        the front, so the front is the part worth keeping), a role the service
        does not accept is skipped with a NAMED warning rather than sent and
        rejected, and an empty block is skipped silently — an empty
        ``textInput`` is not something to spend a content block on.
        """
        blocks: list[dict] = []
        if self._history_provider is not None:
            try:
                blocks = list(self._history_provider() or [])
            except Exception as e:
                logger.warning(
                    f"history provider failed: {e} — starting the session with no replay"
                )
                blocks = []

        if self._replay_suspended and blocks:
            logger.warning(
                f"history replay suspended after {self._replay_deaths} immediate stream "
                f"deaths — starting clean ({len(blocks)} block(s) withheld)"
            )
            blocks = []

        for block in blocks:
            role = str(block.get("role", "") or "").strip().upper() if isinstance(block, dict) else ""
            if role and role not in HISTORY_ROLES:
                logger.warning(
                    f"history block skipped: role={block.get('role')!r} is not one of "
                    f"{'/'.join(HISTORY_ROLES)}"
                )
        # Defensive shaping at the SENDER: USER first, roles alternating, no
        # trailing USER — Bedrock kills the whole stream otherwise (live
        # incident 2026-09-06, see harness/history_blocks.py).
        blocks = normalise_history(blocks)

        sent = 0
        for block in blocks:
            if sent >= self._history_max_blocks:
                break
            role = block["role"]
            text = block["text"]
            content_name = str(uuid.uuid4())
            await self._send({
                "contentStart": {
                    "promptName": self._prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "interactive": False,
                    "role": role,
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            })
            await self._send({
                "textInput": {
                    "promptName": self._prompt_name,
                    "contentName": content_name,
                    "content": text,
                }
            })
            await self._send({
                "contentEnd": {
                    "promptName": self._prompt_name,
                    "contentName": content_name,
                }
            })
            sent += 1

        logger.info(f"history replayed blocks={sent}")
        self._replay_blocks_last = sent
        return sent

    async def _start_session(self) -> None:
        if not self._client:
            self._init_client()

        logger.info(f"Opening bidirectional stream for model={self.model_id}")
        self._stream = await self._client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        logger.info("Stream opened OK")

        # Session start — the endpointing sensitivity is resolved per session,
        # so a .env edit takes effect on the next restart without a redeploy.
        endpointing = _endpointing_sensitivity()
        logger.info(f"Sending sessionStart (endpointing={endpointing})")
        await self._send({
            "sessionStart": {
                "inferenceConfiguration": {
                    "maxTokens": 1024,
                    "topP": 0.9,
                    "temperature": 0.7,
                },
                "turnDetectionConfiguration": {
                    "endpointingSensitivity": endpointing,
                },
            }
        })

        # Prompt start with audio output config (and optional tool config)
        logger.info(f"Sending promptStart (voice={self.voice_id})")
        prompt_start = {
            "promptName": self._prompt_name,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": self.voice_id,
                "encoding": "base64",
                "audioType": "SPEECH",
            },
        }
        if self.tools:
            prompt_start["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
            prompt_start["toolConfiguration"] = {"tools": self.tools}
            logger.info(f"Tool configuration: {len(self.tools)} tools registered")
        await self._send({"promptStart": prompt_start})

        # System prompt
        logger.info(f"Sending system prompt ({len(self.system_prompt)} chars)")
        await self._send({
            "contentStart": {
                "promptName": self._prompt_name,
                "contentName": self._system_content,
                "type": "TEXT",
                "interactive": True,
                "role": "SYSTEM",
                "textInputConfiguration": {"mediaType": "text/plain"},
            }
        })
        await self._send({
            "textInput": {
                "promptName": self._prompt_name,
                "contentName": self._system_content,
                "content": self.system_prompt,
            }
        })
        await self._send({
            "contentEnd": {
                "promptName": self._prompt_name,
                "contentName": self._system_content,
            }
        })

        # Conversation history — the ONLY window the service gives us for it:
        # "after the system prompt and before audio streaming begins". Nothing
        # else goes out in between; in particular no assistant-initiating
        # text, because Sonic can speak unprompted on a fresh session and a
        # rotation every seven minutes must not become a re-greeting (c31).
        replayed = await self._replay_history()
        if self._rotation_age_s is not None:
            logger.info(
                f"rotation delay=0 replay={replayed} age={self._rotation_age_s:.0f}s"
            )
            self._rotation_age_s = None

        # Start audio input stream
        logger.info("Sending audio input contentStart")
        await self._send({
            "contentStart": {
                "promptName": self._prompt_name,
                "contentName": self._audio_content,
                "type": "AUDIO",
                "interactive": True,
                "role": "USER",
                "audioInputConfiguration": {
                    "mediaType": "audio/lpcm",
                    "sampleRateHertz": INPUT_SAMPLE_RATE,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                    "audioType": "SPEECH",
                    "encoding": "base64",
                },
            }
        })

        # Brief pause to let Bedrock fully process the session setup events
        # before we start sending audio/inject traffic.  Without this, audio
        # chunks hit the stream before the server is ready, causing
        # "Invalid input request" on sessions created after sleep/wake.
        await asyncio.sleep(0.5)

        # Throttle injects — new session needs a quiet-start period.
        self._last_inject_time = time.time()

        # Arm the clock-step / response-liveness watchdogs against THIS session.
        self._arm_watchdogs(time.time(), time.monotonic())

        # Session fully configured — now accept audio/inject traffic
        self._active = True
        self._set_state("listening")
        logger.info("Nova Sonic session started - listening")

    async def _process_responses(self) -> None:
        assistant_text_parts = []
        consecutive_errors = 0
        try:
            while self._active:
                try:
                    output = await self._stream.await_output()
                    result = await output[1].receive()
                    if not result.value or not result.value.bytes_:
                        consecutive_errors = 0
                        continue
                    consecutive_errors = 0
                    self._note_response_event()  # sign of life — liveness OK

                    data = json.loads(result.value.bytes_.decode("utf-8"))
                    event = data.get("event", {})
                    event_type = next(iter(event.keys()), "unknown")
                    if event_type != "audioOutput":  # don't spam audio chunks
                        logger.debug(f"RECV ← {event_type}: {json.dumps(event.get(event_type, {}))[:200]}")

                    if "contentStart" in event:
                        cs = event["contentStart"]
                        if cs.get("type") == "TOOL":
                            self._current_tool_use = {
                                "toolName": "",
                                "toolUseId": cs.get("toolUseId", ""),
                                "content": "",
                            }
                            logger.info(f"Tool use started: id={cs.get('toolUseId', '')}")

                    elif "toolUse" in event:
                        tu = event["toolUse"]
                        if self._current_tool_use is not None:
                            self._current_tool_use["toolName"] = tu.get("toolName", "")
                            self._current_tool_use["toolUseId"] = tu.get("toolUseId", self._current_tool_use["toolUseId"])
                            self._current_tool_use["content"] += tu.get("content", "")

                    elif "textOutput" in event:
                        text = event["textOutput"].get("content", "")
                        role = event["textOutput"].get("role", "")
                        if role == "ASSISTANT" or not role:
                            assistant_text_parts.append(text)
                            self.last_assistant_text = "".join(assistant_text_parts)
                        elif role == "USER":
                            if self._speaking:
                                assistant_context = "".join(assistant_text_parts)
                                asyncio.create_task(self._handle_barge_in(text, assistant_context))
                            self.last_user_text = text
                            self._set_state("thinking")
                        if self.on_transcript:
                            self.on_transcript(role or "ASSISTANT", text)

                    elif "audioOutput" in event:
                        self._last_audio_time = time.time()
                        if not self._speaking:
                            self._speaking = True
                            self._set_state("speaking")
                            assistant_text_parts = []
                            logger.info("Utterance audio started")
                        audio_b64 = event["audioOutput"].get("content", "")
                        if audio_b64:
                            pcm_bytes = base64.b64decode(audio_b64)
                            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
                            float_samples = samples.astype(np.float32) / 32768.0
                            if self.on_audio_output:
                                self.on_audio_output(float_samples)

                    elif "contentEnd" in event:
                        ce = event["contentEnd"]
                        content_type = ce.get("type", "")
                        role = ce.get("role", "")
                        stop_reason = ce.get("stopReason", "")

                        # Name every contentEnd, whatever we then do with it.
                        # On the robot (2026-09-05) the ASSISTANT contentEnd
                        # never ended the speaking state — all five utterances
                        # of that boot were flushed by the 4s speaking watchdog
                        # instead — and the logs could not say whether the event
                        # never arrived, arrived with another role/type, or
                        # carried a stopReason this branch ignores. These three
                        # fields are the answer, so they are INFO, not DEBUG.
                        logger.info(
                            f"contentEnd type={content_type or '?'} "
                            f"role={role or '?'} stopReason={stop_reason or '?'}"
                        )

                        if content_type == "TOOL" and self._current_tool_use:
                            # Tool use complete — fire callback
                            tu = self._current_tool_use
                            self._current_tool_use = None
                            tool_name = tu["toolName"]
                            tool_use_id = tu["toolUseId"]
                            try:
                                params = json.loads(tu["content"]) if tu["content"] else {}
                            except json.JSONDecodeError:
                                params = {}
                            logger.info(f"Tool use complete: {tool_name}({params})")
                            if self.on_tool_use:
                                try:
                                    self.on_tool_use(tool_name, tool_use_id, params)
                                except Exception as e:
                                    logger.error(f"on_tool_use callback error: {e}")
                        elif role == "ASSISTANT" or (
                            content_type == "AUDIO" and stop_reason == "END_TURN"
                        ):
                            # LIVE FINDING (robot, 2026-09-06 00:08): Nova 2
                            # Sonic's contentEnd carries NO role field at all —
                            # the journal shows ``contentEnd type=AUDIO role=?
                            # stopReason=END_TURN`` — so the role check alone
                            # never matched and every utterance ended on the
                            # 4 s speaking watchdog. An AUDIO END_TURN is the
                            # end of the spoken turn; a PARTIAL_TURN is not.
                            if self._speaking:
                                logger.info("Utterance ended — back to listening")
                            self._speaking = False
                            self._set_state("listening")
                            # Schedules a task; never awaited here, so a slow
                            # send cannot stall the response reader.
                            self._on_speaking_ended()

                except StopAsyncIteration:
                    break
                except Exception as e:
                    if self._active:
                        consecutive_errors += 1
                        err_str = str(e)
                        # "Invalid event bytes" is a transient SDK framing error —
                        # log a warning and keep the session alive rather than
                        # restarting and cutting audio output.
                        if "Invalid event bytes" in err_str and consecutive_errors <= 5:
                            logger.warning(f"Transient stream error (#{consecutive_errors}): {e}")
                            await asyncio.sleep(0.05)
                            continue
                        self._last_stream_error = str(e)
                        logger.error(f"Response processing error: {e}")
                    break
        except Exception as e:
            logger.error(f"Response loop error: {e}")

    async def _send_audio_chunk(self, audio_bytes: bytes) -> None:
        if not self._active:
            return
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        await self._send({
            "audioInput": {
                "promptName": self._prompt_name,
                "contentName": self._audio_content,
                "content": b64,
            }
        })

    async def _close_stream(self) -> None:
        """Best-effort cleanup of the current stream (bounded to 3s)."""
        try:
            async with asyncio.timeout(3.0):
                await self._send({
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content,
                    }
                })
                await self._send({"promptEnd": {"promptName": self._prompt_name}})
                await self._send({"sessionEnd": {}})
        except Exception:
            logger.debug("Stream close: send phase timed out or failed")
        try:
            async with asyncio.timeout(2.0):
                await self._stream.input_stream.close()
        except Exception:
            logger.debug("Stream close: input_stream.close timed out or failed")

    def _should_stop(self, stop_event: threading.Event) -> bool:
        """Check if either the global or sonic-local stop has been requested."""
        return stop_event.is_set() or self._sonic_stop.is_set()

    async def _interruptible_wait(self, stop_event: threading.Event, delay: float) -> None:
        """Sleep up to ``delay`` seconds, waking early on stop or an immediate restart.

        Both backoff sleeps below (a failed ``_start_session()`` and the
        delay before reopening a dead stream) call this instead of a bare
        ``asyncio.sleep(delay)``. A plain sleep can't observe anything until
        it returns, so a loop already parked inside a long backoff (up to
        the 60s cap) was deaf to both ``request_immediate_restart()`` — an
        "immediate" restart could still take up to 60s (PR #12 finding 2) —
        and to shutdown — stopping Sonic mid-backoff could block loop
        termination for up to 60s (finding 3). Polling every 0.1s mirrors
        the watchdog tick in the response-wait loop below.

        When woken by the immediate-restart event (rather than by elapsing
        or by stop), the event is consumed here — cleared — so the caller
        can proceed straight to a zero-additional-delay restart without the
        request lingering to fire a second, redundant break once the new
        session's response-wait loop starts.
        """
        deadline = time.monotonic() + max(0.0, delay)
        while time.monotonic() < deadline:
            if self._should_stop(stop_event):
                return
            if self._restart_now_event.is_set():
                self._restart_now_event.clear()
                logger.info(
                    "Immediate restart requested during backoff wait: "
                    f"{self._immediate_restart_reason} — skipping remaining delay"
                )
                return
            await asyncio.sleep(min(0.1, deadline - time.monotonic()))

    async def _run_loop(self, stop_event: threading.Event) -> None:
        self._inject_lock = asyncio.Lock()

        while not self._should_stop(stop_event):
            # (Re)start a fresh session
            try:
                await self._start_session()
            except Exception as e:
                delay, attempt = self._compute_restart_delay()
                logger.error(
                    f"Session start failed: {e} — retrying in {delay:.0f}s (attempt {attempt})"
                )
                await self._interruptible_wait(stop_event, delay)
                continue

            response_task = asyncio.create_task(self._process_responses())
            try:
                # Wait until stop requested OR response loop dies
                while not self._should_stop(stop_event) and not response_task.done():
                    # An external caller (e.g. a network-change signal) knows
                    # the current stream is doomed even though it still looks
                    # perfectly healthy from in here — honour it immediately,
                    # bypassing every other check below.
                    if self._restart_now_event.is_set():
                        self._forced_restart_reason = (
                            f"Immediate restart requested: {self._immediate_restart_reason}"
                        )
                        break

                    # Resilience watchdogs: a stepped wall clock or a stream
                    # that swallows input without ever answering both leave a
                    # perfectly healthy-looking task behind, so neither shows
                    # up as response_task.done(). Break out and let the normal
                    # restart path below rebuild the session from scratch.
                    if self._check_clock_step(time.time(), time.monotonic()):
                        self._forced_restart_reason = "Clock step forced a restart"
                        break
                    if self._check_response_liveness(time.monotonic()):
                        self._forced_restart_reason = (
                            "Response liveness stall forced a restart"
                        )
                        break

                    # Proactive rotation: replace the session ourselves, at an
                    # idle moment, before Bedrock's ~8-minute ceiling does it
                    # for us mid-sentence and without a recap.
                    rotation_age = self._rotation_due(time.monotonic())
                    if rotation_age is not None:
                        self._pending_rotation = rotation_age
                        break

                    # Speaking watchdog: a stalled generation (no audio for 10s
                    # while _speaking) would otherwise pin the inject guard
                    # forever and hold the speaker's utterance buffer hostage.
                    if (
                        self._speaking
                        and self._last_audio_time
                        and time.time() - self._last_audio_time > 4.0
                    ):
                        logger.warning(
                            "Speaking watchdog: no audio for 4s — clearing stuck speaking state"
                        )
                        self._speaking = False
                        self._set_state("listening")
                        # A cue parked during an utterance the service never
                        # closed properly is still owed a delivery.
                        self._on_speaking_ended()
                    await asyncio.sleep(0.1)
            finally:
                # Mark inactive FIRST to stop all incoming traffic
                self._active = False
                self._speaking = False
                self._current_tool_use = None

                # Clean up CURRENT stream with CURRENT UUIDs (before generating new ones)
                await self._close_stream()
                response_task.cancel()

            if self._should_stop(stop_event):
                break

            # Stream died, a watchdog forced it, or the rotation timer decided
            # it was time — prepare for restart. Either way the restart is a
            # fresh client with fresh UUIDs and the system prompt, plus
            # whatever ``history_provider`` remembers (see _replay_history).
            self._session_gen += 1  # invalidate any queued coroutines
            # Parked cues belong to the conversation that just died; the fresh
            # session never heard the utterance they interrupted, so delivering
            # them into it would be a reaction to nothing.
            self._deferred.clear()
            self._set_state("idle")

            immediate = self._restart_now_event.is_set()
            self._restart_now_event.clear()
            reason = self._forced_restart_reason or "Bedrock stream died"
            self._forced_restart_reason = None

            rotation_age = self._pending_rotation
            self._pending_rotation = None

            if rotation_age is not None and not immediate:
                # A planned swap of a healthy session, taken at an idle
                # moment: no backoff, and no "restarting session" warning
                # either — the single journal line for a rotation is emitted
                # by the fresh session's replay, so it can name delay, replay
                # count and age together.
                delay = 0.0
                self._restart_attempt = 0
                self._rotation_age_s = rotation_age
            elif immediate:
                # request_immediate_restart() already reset the backoff —
                # skip the delay entirely, this restart is urgent.
                delay = 0.0
                logger.warning(f"{reason} — restarting session now")
            elif IDLE_CUTOFF_MARKER in self._last_stream_error:
                # Bedrock's own idle cutoff: the stream was healthy, the room
                # was quiet. A fresh stream works at once — no backoff.
                delay = 0.0
                self._restart_attempt = 0
                self._last_stream_error = ""
                logger.warning("Bedrock idle cutoff (no interactive content for 295 s) — restarting now")
            else:
                self._note_session_death(time.monotonic())
                self._maybe_reset_backoff_for_healthy_session(time.monotonic())
                delay, attempt = self._compute_restart_delay()
                logger.warning(
                    f"{reason} — restarting session in {delay:.0f}s (attempt {attempt})"
                )

            # Force fresh client — old client may hold stale connection state
            self._client = None
            self._stream = None

            # Generate fresh UUIDs for the new session
            self._prompt_name = str(uuid.uuid4())
            self._system_content = str(uuid.uuid4())
            self._audio_content = str(uuid.uuid4())

            await self._interruptible_wait(stop_event, delay)
            # Reset inject throttle so new session gets a quiet start
            self._last_inject_time = time.time()

        self._set_state("idle")

    def start(self, stop_event: threading.Event) -> None:
        """Start the Nova Sonic session in a background thread.

        NEVER raises. The Bedrock connection itself is made inside
        ``_run_loop`` on the new thread, which already retries with backoff
        forever, so a network-less start is simply a stream that has not
        connected YET — not a failed component. Even the thread spawn is
        guarded: the harness supervisor treats a raising ``start()`` as
        ``start failed name=...`` and never retries it (the exact shape that
        left the Kiro writer absent for hours on 2026-08-26), so a failure
        here degrades to a named line instead.
        """
        import traceback

        def _run():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_loop(stop_event))
            except Exception as e:
                logger.error(f"Nova Sonic loop error: {e}")
                logger.error(traceback.format_exc())
            finally:
                loop.close()

        try:
            self._thread = threading.Thread(target=_run, name="nova-sonic", daemon=True)
            self._thread.start()
        except Exception as e:  # noqa: BLE001 - a degraded voice beats a failed component
            self._thread = None
            sensory_stage(
                "supervise", "nova", "component", f"component degraded name=sonic reason={e}"
            )
            logger.error(f"Nova Sonic thread failed to start: {e}")
            return
        logger.info("Nova Sonic thread started")

    def stop(self) -> None:
        """Stop Sonic independently (for sleep mode). Does not affect the global stop_event."""
        self._sonic_stop.set()
        self._active = False
        logger.info("Nova Sonic stopped (sleep mode)")

    def restart(self, stop_event: threading.Event) -> None:
        """Restart Sonic after an independent stop (for wake from sleep)."""
        old_thread = self._thread
        if old_thread is not None and old_thread.is_alive():
            logger.info("Waiting for old sonic thread to exit...")
            old_thread.join(timeout=8.0)
            if old_thread.is_alive():
                logger.warning("Old sonic thread still alive after 8s — proceeding anyway")

        self._sonic_stop.clear()
        self._client = None
        self._stream = None
        self._prompt_name = str(uuid.uuid4())
        self._system_content = str(uuid.uuid4())
        self._audio_content = str(uuid.uuid4())
        self._session_gen += 1
        self._last_inject_time = time.time()
        self.start(stop_event)
        logger.info("Nova Sonic restarted")

    def feed_audio(self, samples: np.ndarray) -> None:
        """Feed audio samples from the robot's microphone.

        EVERY chunk is sent — the gate below decides only what the
        response-liveness watchdog is told, never what Bedrock hears. A quiet
        room must keep streaming (Sonic's own turn detection lives on that
        stream) while still counting as "we sent nothing worth answering".

        Args:
            samples: float32 audio samples from reachy_mini at 16kHz.
                     Can be mono (N,) or stereo (N, 2).
        """
        if not self._active or not self._loop:
            return

        # Convert to mono if stereo
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        # Loudness, measured on the float32 array before the int16 conversion:
        # this runs on the hearing thread ten times a second, so it stays one
        # numpy pass over 100ms of samples and nothing more.
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0

        # Convert float32 [-1, 1] to int16 PCM bytes
        pcm = (samples * 32767).astype(np.int16).tobytes()

        try:
            asyncio.run_coroutine_threadsafe(
                self._send_audio_chunk(pcm), self._loop
            )
            self._energy_recent.append(rms > _speech_floor())
            if sum(self._energy_recent) >= SPEECH_BURST_CHUNKS:
                # liveness watchdog: somebody is actually talking (a burst,
                # not one loud chunk)
                self._note_input_sent()
        except Exception as e:
            logger.warning(f"feed_audio scheduling failed: {e}")

    def inject_text(
        self, text: str, force: bool = False, sense_class: str | None = None
    ) -> str:
        """Inject a text message into the conversation (e.g., vision description).

        Args:
            text: the text to inject as a USER message
            force: if True, skip the speaking guard (use with caution)
            sense_class: the rules entry's ``sense:`` value (``pat``, ``face``,
                ``sound``, ``vision``; ``None`` for anything unclassed). Used
                only when the cue has to be deferred: it names the latest-wins
                slot the cue is parked in, so a burst of pats during one reply
                collapses to one delivery while a pat and a face stay two
                independent facts.
        """
        if not self._active or not self._loop:
            return "dropped-inactive"

        # Don't inject while the model is actively generating audio — this can
        # destabilize the Bedrock bidirectional stream and cause it to hang.
        # The cue is parked rather than lost: _on_speaking_ended() delivers it
        # (with its age in the text) the moment the utterance finishes.
        if self._speaking and not force:
            cue = self._deferred.put(sense_class, text)
            sensory_stage(
                "inject", "speech", str(uuid.uuid4()),
                f"deferred class={cue.sense_class} text={text[:60]!r}",
            )
            return "deferred"

        # Throttle: skip if too soon after last inject to avoid flooding Bedrock
        now = time.time()
        if now - self._last_inject_time < self._inject_min_interval:
            sensory_stage(
                "inject", "speech", str(uuid.uuid4()),
                f"dropped reason=throttled interval={now - self._last_inject_time:.1f}s "
                f"text={text[:60]!r}",
            )
            return "dropped-throttled"
        self._last_inject_time = now

        gen = self._session_gen  # capture at scheduling time

        try:
            asyncio.run_coroutine_threadsafe(self._send_user_text(text, gen), self._loop)
            self._note_interactive_sent()
            # NOT liveness input: a quiet body cue or a tool result legitimately
            # gets no answer from the model (robot, 2026-09-06). Only sustained
            # speech-level mic audio counts — see feed_audio.
        except Exception as e:
            logger.warning(f"inject_text scheduling failed: {e}")
            return "dropped-scheduling"
        return "sent"

    async def _send_user_text(self, text: str, gen: int) -> None:
        """The one wire path a USER text message takes: start / text / end.

        Shared by ``inject_text`` and the deferred-cue drain so a parked cue
        reaches Bedrock through *exactly* the same events, under the same
        ``_inject_lock`` serialisation and the same stale-session check — the
        deferral changes when a cue is sent, never how.
        """
        lock = self._inject_lock
        if lock is None:
            # Only reachable before _run_loop armed the lock (tests, a very
            # early inject); one lock is still enough to serialise, because
            # every send below runs on the Sonic loop thread.
            lock = self._inject_lock = asyncio.Lock()
        content_name = str(uuid.uuid4())
        async with lock:
            if not self._active or self._session_gen != gen:
                return  # session restarted — discard stale inject
            try:
                await self._send({
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "USER",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                })
                await self._send({
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                        "content": text,
                    }
                })
            except Exception as e:
                logger.warning(f"inject_text send failed: {e}")
            finally:
                try:
                    await self._send({
                        "contentEnd": {
                            "promptName": self._prompt_name,
                            "contentName": content_name,
                        }
                    })
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Deferred cues (see harness/deferred_cues.py)
    # ------------------------------------------------------------------

    def _on_speaking_ended(self) -> None:
        """Schedule delivery of any cue parked while the model was generating.

        Called from BOTH places the speaking state ends — the ASSISTANT
        ``contentEnd`` branch in ``_process_responses`` and the 4 s speaking
        watchdog in ``_run_loop`` — because a cue parked during an utterance
        the service never closed properly must still be delivered.

        The drain is *scheduled*, never awaited here: ``_process_responses``
        is the only reader of the response stream, and a send that blocks
        (a slow stream, a busy ``_inject_lock``) must not stop it reading.

        A cue can still land in the microsecond window between ``_speaking``
        going False and the ``pending()`` check below — ``inject_text`` runs
        on other threads and reads the guard before it parks. Such a cue
        waits for the *next* transition, by which time the TTL has almost
        certainly retired it as ``dropped reason=deferred-expired``. That is
        the intended, named outcome: the moment really has passed.
        """
        if not self._deferred.pending():
            return
        gen = self._session_gen  # capture at scheduling time
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                loop.create_task(self._drain_deferred(gen))
            elif self._loop is not None:
                asyncio.run_coroutine_threadsafe(self._drain_deferred(gen), self._loop)
        except Exception as e:  # noqa: BLE001 - a lost cue must not kill the loop
            logger.warning(f"deferred drain scheduling failed: {e}")

    async def _drain_deferred(self, gen: int) -> None:
        """Deliver the parked cues, newest text per sense class, oldest first."""
        if not self._active or self._session_gen != gen:
            return  # session restarted — the cues belong to a conversation that is gone
        cues = self._deferred.drain()
        if not cues:
            return

        now = self._deferred.now()  # the slot's clock, not this module's
        delivered = 0
        for cue in cues:
            if delivered >= deferred_cues.MAX_DRAIN_PER_TRANSITION:
                self._deferred.log_overflow(cue, cue.age(now))
                continue
            text = self._deferred.render(cue, now)
            sensory_stage(
                "inject", "speech", str(uuid.uuid4()),
                f"drained class={cue.sense_class} age={cue.age(now):.1f}s "
                f"text={text[:60]!r}",
            )
            # The 3s throttle is deliberately NOT consulted: the cue already
            # waited out a whole utterance, which is what the throttle exists
            # to enforce. It is re-armed below so the injects that follow the
            # drain are spaced normally again.
            await self._send_user_text(text, gen)
            if self.on_deferred_delivered is not None:
                try:
                    self.on_deferred_delivered(text, cue.sense_class)
                except Exception as e:  # noqa: BLE001 - a ledger hiccup must not stop the drain
                    logger.warning(f"on_deferred_delivered raised: {e}")
            delivered += 1

        if delivered:
            self._last_inject_time = time.time()
            self._note_interactive_sent()
            # NOT liveness input: a quiet body cue or a tool result legitimately
            # gets no answer from the model (robot, 2026-09-06). Only sustained
            # speech-level mic audio counts — see feed_audio.

    def send_tool_result(self, tool_use_id: str, result: str) -> None:
        """Send a tool result back to the Nova Sonic conversation."""
        if not self._active or not self._loop:
            return

        content_name = str(uuid.uuid4())
        gen = self._session_gen  # capture at scheduling time

        async def _send_result():
            async with self._inject_lock:
                if not self._active or self._session_gen != gen:
                    return  # session restarted — discard stale tool result
                try:
                    await self._send({
                        "contentStart": {
                            "promptName": self._prompt_name,
                            "contentName": content_name,
                            "interactive": False,
                            "type": "TOOL",
                            "role": "TOOL",
                            "toolResultInputConfiguration": {
                                "toolUseId": tool_use_id,
                                "type": "TEXT",
                                "textInputConfiguration": {"mediaType": "text/plain"},
                            },
                        }
                    })
                    await self._send({
                        "toolResult": {
                            "promptName": self._prompt_name,
                            "contentName": content_name,
                            "content": json.dumps({"result": result}),
                        }
                    })
                    logger.info(f"Tool result sent for {tool_use_id}: {result[:100]}...")
                except Exception as e:
                    logger.warning(f"send_tool_result send failed: {e}")
                finally:
                    try:
                        await self._send({
                            "contentEnd": {
                                "promptName": self._prompt_name,
                                "contentName": content_name,
                            }
                        })
                    except Exception:
                        pass

        try:
            asyncio.run_coroutine_threadsafe(_send_result(), self._loop)
            self._note_interactive_sent()
            # NOT liveness input: a quiet body cue or a tool result legitimately
            # gets no answer from the model (robot, 2026-09-06). Only sustained
            # speech-level mic audio counts — see feed_audio.
        except Exception as e:
            logger.warning(f"send_tool_result scheduling failed: {e}")
