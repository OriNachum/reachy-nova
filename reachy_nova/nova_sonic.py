"""Nova Sonic - Bidirectional speech-to-speech via Amazon Bedrock."""

import asyncio
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
DEFAULT_LIVENESS_S = 180.0


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

        # Tool use tracking
        self._current_tool_use: dict | None = None

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

    def _note_input_sent(self) -> None:
        """Record that we pushed something (audio or text) into the stream."""
        self._input_since_response = True

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

    async def _start_session(self) -> None:
        if not self._client:
            self._init_client()

        logger.info(f"Opening bidirectional stream for model={self.model_id}")
        self._stream = await self._client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        logger.info("Stream opened OK")

        # Session start
        logger.info("Sending sessionStart")
        await self._send({
            "sessionStart": {
                "inferenceConfiguration": {
                    "maxTokens": 1024,
                    "topP": 0.9,
                    "temperature": 0.7,
                }
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
                        elif role == "ASSISTANT":
                            if self._speaking:
                                logger.info("Utterance ended — back to listening")
                            self._speaking = False
                            self._set_state("listening")

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

            # Stream died (or a watchdog forced it) — prepare for restart.
            # Either way the restart is CLEAN: fresh client, fresh UUIDs,
            # system prompt only. No conversation recap is replayed.
            self._session_gen += 1  # invalidate any queued coroutines
            self._set_state("idle")

            immediate = self._restart_now_event.is_set()
            self._restart_now_event.clear()
            reason = self._forced_restart_reason or "Bedrock stream died"
            self._forced_restart_reason = None

            if immediate:
                # request_immediate_restart() already reset the backoff —
                # skip the delay entirely, this restart is urgent.
                delay = 0.0
                logger.warning(f"{reason} — restarting session now")
            else:
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

        Args:
            samples: float32 audio samples from reachy_mini at 16kHz.
                     Can be mono (N,) or stereo (N, 2).
        """
        if not self._active or not self._loop:
            return

        # Convert to mono if stereo
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        # Convert float32 [-1, 1] to int16 PCM bytes
        pcm = (samples * 32767).astype(np.int16).tobytes()

        try:
            asyncio.run_coroutine_threadsafe(
                self._send_audio_chunk(pcm), self._loop
            )
            self._note_input_sent()  # liveness watchdog: we pushed input
        except Exception as e:
            logger.warning(f"feed_audio scheduling failed: {e}")

    def inject_text(self, text: str, force: bool = False) -> None:
        """Inject a text message into the conversation (e.g., vision description).

        Args:
            text: the text to inject as a USER message
            force: if True, skip the speaking guard (use with caution)
        """
        if not self._active or not self._loop:
            return

        # Don't inject while the model is actively generating audio — this can
        # destabilize the Bedrock bidirectional stream and cause it to hang.
        if self._speaking and not force:
            sensory_stage(
                "inject", "speech", str(uuid.uuid4()),
                f"dropped reason=speaking text={text[:60]!r}",
            )
            return

        # Throttle: skip if too soon after last inject to avoid flooding Bedrock
        now = time.time()
        if now - self._last_inject_time < self._inject_min_interval:
            sensory_stage(
                "inject", "speech", str(uuid.uuid4()),
                f"dropped reason=throttled interval={now - self._last_inject_time:.1f}s "
                f"text={text[:60]!r}",
            )
            return
        self._last_inject_time = now

        content_name = str(uuid.uuid4())
        gen = self._session_gen  # capture at scheduling time

        async def _inject():
            async with self._inject_lock:
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

        try:
            asyncio.run_coroutine_threadsafe(_inject(), self._loop)
            self._note_input_sent()  # liveness watchdog: we pushed input
        except Exception as e:
            logger.warning(f"inject_text scheduling failed: {e}")

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
        except Exception as e:
            logger.warning(f"send_tool_result scheduling failed: {e}")
