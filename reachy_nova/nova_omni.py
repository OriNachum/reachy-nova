"""Nova Omni - deep multimodal understanding (clip + image + text) via Bedrock.

Nova 2 Omni is a *request-response* model (unlike Sonic's live bidirectional
stream): one ``invoke_model`` call carries a rolling camera clip as video
bytes, an optional still frame, and free-form text context, and returns a
plain-text answer.

Omni is a preview model, so two things routinely go wrong and neither may
break the caller:

* ``config.omni_model_id()`` is an EMPTY STRING when the preview is not
  enabled for this account — then there is no Omni API call at all.
* the call itself fails (quota, region, unsupported clip format).

In both cases NovaOmni degrades to a Nova 2 Lite call built on the same
message schema ``nova_vision.py`` uses (the still image, or a text-only
prompt when no frame is available). Callers always get exactly one answer
and the returned string never mentions that a fallback happened — the
fallback *reason* is recorded on a ``[SENSE stage=understand source=vision
...]`` line instead, so degradation is visible in the log rather than in the
robot's speech.

The boto3 client is created lazily and can be injected via the constructor,
which is how the tests stay entirely off the network.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from pathlib import Path

import boto3

from . import config, sensory_log

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are the visual understanding system of a small desk robot. "
    "Describe what happens in 1-3 short, plain sentences. "
    "Be specific about people, objects, actions, and changes over time. "
    "Do not greet anyone or add emotional commentary. "
    "Do not say 'I see' — just state what is there."
)

DEFAULT_PROMPT = "Describe what is happening. Be brief and specific."

#: Sensory-log coordinates for every line this module emits.
_STAGE = "understand"
_SOURCE = "vision"

#: File suffix -> Bedrock video format name, for the few that differ.
_FORMAT_ALIASES = {"m4v": "mp4", "mpeg4": "mp4", "mkv": "matroska", "qt": "mov"}
_DEFAULT_VIDEO_FORMAT = "mp4"


class NovaOmni:
    """Request-response deep-understanding client with a Nova 2 Lite fallback."""

    def __init__(
        self,
        region: str | None = None,
        omni_model_id: str | None = None,
        lite_model_id: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 512,
        client=None,
    ):
        """
        Args:
            region: AWS region; defaults to ``config.region()``.
            omni_model_id: Omni model id; defaults to ``config.omni_model_id()``,
                which is an empty string when the preview is not enabled. An
                explicit ``""`` also means "disabled".
            lite_model_id: fallback model id; defaults to ``config.lite_model_id()``.
            system_prompt: system text sent with every request.
            max_tokens: inference cap for both models.
            client: an already-built bedrock-runtime client. When omitted, one
                is created lazily on the first call (never at construction).
        """
        self.region = region or config.region()
        self.omni_model_id = (
            config.omni_model_id() if omni_model_id is None else omni_model_id
        )
        self.lite_model_id = lite_model_id or config.lite_model_id()
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

        self._client = client
        self.last_result = ""
        self.last_model_used = ""

    # -- client -----------------------------------------------------------

    @property
    def client(self):
        """The bedrock-runtime client, built on first use and then cached."""
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    # -- public API -------------------------------------------------------

    def understand(
        self,
        clip_path: str | None = None,
        image_bytes: bytes | None = None,
        context: str = "",
    ) -> str:
        """Describe a clip/image/context in one request-response round trip.

        Args:
            clip_path: path to a short video clip (e.g. the rolling clip the
                camera rider writes). Sent to Omni only — Lite takes no video.
            image_bytes: JPEG-encoded still frame, sent to whichever model runs.
            context: free-form text context or question for the model.

        Returns:
            The model's text answer. On Omni failure this is the Lite answer,
            with no hint that a fallback occurred; on a total outage it is a
            bracketed error string (never a raised exception).
        """
        event = uuid.uuid4().hex[:8]

        if not self.omni_model_id:
            self._log(
                event,
                "omni not enabled (empty model id) — fallback to lite without an API call",
            )
            return self._invoke_lite(image_bytes, context, event)

        try:
            return self._invoke_omni(clip_path, image_bytes, context, event)
        except Exception as exc:  # noqa: BLE001 - degrade, never propagate
            logger.warning("Nova Omni understanding failed, falling back to Lite: %s", exc)
            self._log(event, f"omni call failed ({exc}) — fallback to lite")
            return self._invoke_lite(image_bytes, context, event)

    # -- model calls ------------------------------------------------------

    def _invoke_omni(
        self,
        clip_path: str | None,
        image_bytes: bytes | None,
        context: str,
        event: str,
    ) -> str:
        content: list[dict] = []

        if clip_path:
            clip = Path(clip_path)
            video_bytes = clip.read_bytes()
            content.append(
                {
                    "video": {
                        "format": _video_format(clip),
                        "source": {"bytes": base64.b64encode(video_bytes).decode("utf-8")},
                    }
                }
            )

        if image_bytes:
            content.append(_image_block(image_bytes))

        content.append({"text": context or DEFAULT_PROMPT})

        text = self._invoke(self.omni_model_id, content)
        self.last_result = text
        self.last_model_used = self.omni_model_id
        sensory_log.stage(
            _STAGE, _SOURCE, event, f"omni answered ({len(text)} chars)"
        )
        return text

    def _invoke_lite(self, image_bytes: bytes | None, context: str, event: str) -> str:
        """The NovaVision-style Lite call used whenever Omni is unavailable."""
        content: list[dict] = []
        if image_bytes:
            content.append(_image_block(image_bytes))
        content.append({"text": context or DEFAULT_PROMPT})

        try:
            text = self._invoke(self.lite_model_id, content)
        except Exception as exc:  # noqa: BLE001 - the caller still gets a string
            logger.error("Nova Lite fallback also failed: %s", exc)
            self._log(event, f"lite fallback also failed ({exc})")
            return f"[Understanding unavailable: {exc}]"

        self.last_result = text
        self.last_model_used = self.lite_model_id
        sensory_log.stage(
            _STAGE, _SOURCE, event, f"lite fallback answered ({len(text)} chars)"
        )
        return text

    def _invoke(self, model_id: str, content: list[dict]) -> str:
        body = {
            "schemaVersion": "messages-v1",
            "system": [{"text": self.system_prompt}],
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "topP": 0.9,
                "temperature": 0.7,
            },
        }

        response = self.client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        try:
            text = result["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected {model_id} response shape: {exc}") from exc
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty response text from {model_id}")
        return text

    # -- logging ----------------------------------------------------------

    def _log(self, event: str, detail: str) -> None:
        sensory_log.stage(_STAGE, _SOURCE, event, detail)


def _image_block(image_bytes: bytes) -> dict:
    return {
        "image": {
            "format": "jpeg",
            "source": {"bytes": base64.b64encode(image_bytes).decode("utf-8")},
        }
    }


def _video_format(clip: Path) -> str:
    """Bedrock video format name derived from the clip's file suffix."""
    suffix = clip.suffix.lstrip(".").lower()
    if not suffix:
        return _DEFAULT_VIDEO_FORMAT
    return _FORMAT_ALIASES.get(suffix, suffix)
