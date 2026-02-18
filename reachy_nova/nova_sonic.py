"""Nova Sonic - Bidirectional speech-to-speech via Amazon Bedrock."""

import asyncio
import base64
import json
import logging
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

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHUNK_DURATION_MS = 100
INPUT_CHUNK_SIZE = int(INPUT_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


class NovaSonic:
    """Manages a bidirectional voice conversation with Nova Sonic."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "amazon.nova-2-sonic-v1:0",
        voice_id: str = "matthew",
        system_prompt: str = (
            "You are Nova, the AI brain of a cute robot called Reachy Mini. "
            "You have a camera for eyes and can see the world. "
            "You can also browse the web using Nova Act. "
            "Keep your responses short, fun, and expressive. "
            "You love to help and are endlessly curious about the world around you. "
            "React with enthusiasm when you see something interesting through your camera."
        ),
        on_transcript: Callable[[str, str], None] | None = None,
        on_audio_output: Callable[[np.ndarray], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
        tools: list[dict] | None = None,
        on_tool_use: Callable[[str, str, dict], None] | None = None,
        on_interruption: Callable[[], None] | None = None,
    ):
        self.region = region
        self.model_id = model_id
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
            modelId="us.amazon.nova-2-lite-v1:0",
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
                        if not self._speaking:
                            self._speaking = True
                            self._set_state("speaking")
                            assistant_text_parts = []
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
        """Best-effort cleanup of the current stream."""
        try:
            await self._send({
                "contentEnd": {
                    "promptName": self._prompt_name,
                    "contentName": self._audio_content,
                }
            })
            await self._send({"promptEnd": {"promptName": self._prompt_name}})
            await self._send({"sessionEnd": {}})
        except Exception:
            pass
        try:
            await self._stream.input_stream.close()
        except Exception:
            pass

    async def _run_loop(self, stop_event: threading.Event) -> None:
        self._inject_lock = asyncio.Lock()

        while not stop_event.is_set():
            # (Re)start a fresh session
            try:
                await self._start_session()
            except Exception as e:
                logger.error(f"Session start failed: {e} — retrying in 3s")
                await asyncio.sleep(3)
                continue

            response_task = asyncio.create_task(self._process_responses())
            try:
                # Wait until stop requested OR response loop dies
                while not stop_event.is_set() and not response_task.done():
                    await asyncio.sleep(0.1)
            finally:
                # Mark inactive FIRST to stop all incoming traffic
                self._active = False
                self._speaking = False
                self._current_tool_use = None

                # Clean up CURRENT stream with CURRENT UUIDs (before generating new ones)
                await self._close_stream()
                response_task.cancel()

            if stop_event.is_set():
                break

            # Stream died — prepare for restart
            self._session_gen += 1  # invalidate any queued coroutines
            self._set_state("idle")
            logger.warning("Bedrock stream died — restarting session in 3s")

            # Force fresh client — old client may hold stale connection state
            self._client = None
            self._stream = None

            # Generate fresh UUIDs for the new session
            self._prompt_name = str(uuid.uuid4())
            self._system_content = str(uuid.uuid4())
            self._audio_content = str(uuid.uuid4())

            await asyncio.sleep(3)
            # Reset inject throttle so new session gets a quiet start
            self._last_inject_time = time.time()

        self._set_state("idle")

    def start(self, stop_event: threading.Event) -> None:
        """Start the Nova Sonic session in a background thread."""
        import traceback

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run_loop(stop_event))
            except Exception as e:
                logger.error(f"Nova Sonic loop error: {e}")
                logger.error(traceback.format_exc())
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=_run, name="nova-sonic", daemon=True)
        self._thread.start()
        logger.info("Nova Sonic thread started")

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
            logger.debug("inject_text skipped — model is speaking")
            return

        # Throttle: skip if too soon after last inject to avoid flooding Bedrock
        now = time.time()
        if now - self._last_inject_time < self._inject_min_interval:
            logger.debug(f"inject_text throttled (interval={now - self._last_inject_time:.1f}s)")
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
