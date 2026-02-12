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
        model_id: str = "amazon.nova-sonic-v1:0",
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
    ):
        self.region = region
        self.model_id = model_id
        self.voice_id = voice_id
        self.system_prompt = system_prompt
        self.on_transcript = on_transcript
        self.on_audio_output = on_audio_output
        self.on_state_change = on_state_change

        self._client: BedrockRuntimeClient | None = None
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._active = False
        self._speaking = False

        self._prompt_name = str(uuid.uuid4())
        self._system_content = str(uuid.uuid4())
        self._audio_content = str(uuid.uuid4())

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

    def _init_client(self) -> None:
        import os
        endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com"
        key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
        logger.info(f"Init client: region={self.region}, endpoint={endpoint}")
        logger.info(f"  AWS_ACCESS_KEY_ID={key_id[:8]}..., session_token={'yes' if os.environ.get('AWS_SESSION_TOKEN') else 'no'}")
        config = Config(
            endpoint_uri=endpoint,
            region=self.region,
            aws_access_key_id=key_id,
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        )
        self._client = BedrockRuntimeClient(config=config)
        logger.info("Client created OK")

    async def _send(self, event: dict) -> None:
        payload = json.dumps({"event": event}).encode("utf-8")
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=payload)
        )
        await self._stream.input_stream.send(chunk)

    async def _start_session(self) -> None:
        if not self._client:
            self._init_client()

        self._stream = await self._client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self._active = True
        self._set_state("listening")

        # Session start
        await self._send({
            "sessionStart": {
                "inferenceConfiguration": {
                    "maxTokens": 1024,
                    "topP": 0.9,
                    "temperature": 0.7,
                }
            }
        })

        # Prompt start with audio output config
        await self._send({
            "promptStart": {
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
        })

        # System prompt
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

        logger.info("Nova Sonic session started - listening")

    async def _process_responses(self) -> None:
        assistant_text_parts = []
        try:
            while self._active:
                try:
                    output = await self._stream.await_output()
                    result = await output[1].receive()
                    if not result.value or not result.value.bytes_:
                        continue

                    data = json.loads(result.value.bytes_.decode("utf-8"))
                    event = data.get("event", {})

                    if "textOutput" in event:
                        text = event["textOutput"].get("content", "")
                        role = event["textOutput"].get("role", "")
                        if role == "ASSISTANT" or not role:
                            assistant_text_parts.append(text)
                            self.last_assistant_text = "".join(assistant_text_parts)
                        elif role == "USER":
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
                        role = event["contentEnd"].get("role", "")
                        if role == "ASSISTANT":
                            self._speaking = False
                            self._set_state("listening")

                except StopAsyncIteration:
                    break
                except Exception as e:
                    if self._active:
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

    async def _run_loop(self, stop_event: threading.Event) -> None:
        await self._start_session()
        response_task = asyncio.create_task(self._process_responses())
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.01)
        finally:
            self._active = False
            try:
                await self._send({
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content,
                    }
                })
                await self._send({"promptEnd": {"promptName": self._prompt_name}})
                await self._send({"sessionEnd": {}})
                await self._stream.input_stream.close()
            except Exception:
                pass
            response_task.cancel()
            self._set_state("idle")

    def start(self, stop_event: threading.Event) -> None:
        """Start the Nova Sonic session in a background thread."""
        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run_loop(stop_event))
            except Exception as e:
                logger.error(f"Nova Sonic loop error: {e}")
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
        except Exception:
            pass

    def inject_text(self, text: str) -> None:
        """Inject a text message into the conversation (e.g., vision description)."""
        if not self._active or not self._loop:
            return

        content_name = str(uuid.uuid4())

        async def _inject():
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
            await self._send({
                "contentEnd": {
                    "promptName": self._prompt_name,
                    "contentName": content_name,
                }
            })

        try:
            asyncio.run_coroutine_threadsafe(_inject(), self._loop)
        except Exception:
            pass
