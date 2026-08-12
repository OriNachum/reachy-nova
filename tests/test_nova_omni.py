"""Tests for reachy_nova.nova_omni — the NovaOmni request-response client (t2).

NovaOmni is the deep-understanding client: it sends a rolling camera *clip*
(video bytes), an optional still image, and free-form text context to Nova 2
Omni in one ``invoke_model`` round trip. Omni is a preview model, so
``config.omni_model_id()`` may be an empty string ("not enabled") and the
call itself may fail — either way NovaOmni must degrade to a Nova 2 Lite
call and still hand the caller one plain answer.

Every bedrock client here is a fake, so the suite never touches the network.
"""

from __future__ import annotations

import base64
import json
import logging
import re

import pytest

from reachy_nova import nova_omni
from reachy_nova.nova_omni import NovaOmni

_SENSE_LINE_RE = re.compile(
    r"^\[SENSE stage=(?P<stage>\S+) source=(?P<source>\S+) event=(?P<event>\S+)\] (?P<detail>.*)$"
)

OMNI_ID = "amazon.nova-2-omni-v1:0"
LITE_ID = "us.amazon.nova-2-lite-v1:0"


class _FakeBody:
    """Stands in for the streaming body botocore returns."""

    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw


def _response(text: str) -> dict:
    return {"body": _FakeBody({"output": {"message": {"content": [{"text": text}]}}})}


class FakeBedrock:
    """A bedrock-runtime double that answers (or fails) per model id.

    ``calls`` records every ``(modelId, parsed_body)`` pair so tests can assert
    which model was actually reached and what schema it was handed.
    """

    def __init__(self, texts: dict[str, str] | None = None, fail: set[str] | None = None):
        self.texts = texts or {}
        self.fail = fail or set()
        self.calls: list[tuple[str, dict]] = []

    def invoke_model(self, modelId: str, body: str, **kwargs):  # noqa: N803 - boto3 casing
        self.calls.append((modelId, json.loads(body)))
        if modelId in self.fail:
            raise RuntimeError(f"forced failure for {modelId}")
        return _response(self.texts.get(modelId, f"answer from {modelId}"))

    @property
    def model_ids(self) -> list[str]:
        return [model_id for model_id, _ in self.calls]

    def body_for(self, model_id: str) -> dict:
        for called_id, body in self.calls:
            if called_id == model_id:
                return body
        raise AssertionError(f"{model_id} was never invoked; got {self.model_ids}")


def _sense_lines(caplog) -> list[re.Match]:
    matches = []
    for record in caplog.records:
        if record.name != "nova.sensory":
            continue
        match = _SENSE_LINE_RE.match(record.getMessage())
        assert match is not None, f"unparseable sensory line: {record.getMessage()!r}"
        matches.append(match)
    return matches


@pytest.fixture
def clip(tmp_path):
    """A tiny fake mp4 on disk — NovaOmni only ever reads its raw bytes."""
    path = tmp_path / "rolling.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-clip-bytes")
    return path


def _omni(client, **kwargs) -> NovaOmni:
    kwargs.setdefault("omni_model_id", OMNI_ID)
    kwargs.setdefault("lite_model_id", LITE_ID)
    return NovaOmni(client=client, **kwargs)


class TestOmniHappyPath:
    """When Omni is enabled and healthy, it answers and Lite is never touched."""

    def test_returns_omni_text_without_calling_lite(self, clip):
        client = FakeBedrock(texts={OMNI_ID: "A person waves at the robot."})
        result = _omni(client).understand(clip_path=str(clip), context="What happened?")

        assert result == "A person waves at the robot."
        assert client.model_ids == [OMNI_ID]

    def test_sends_clip_bytes_base64_in_the_nova_message_schema(self, clip):
        client = FakeBedrock()
        _omni(client).understand(clip_path=str(clip))

        body = client.body_for(OMNI_ID)
        assert body["schemaVersion"] == "messages-v1"
        content = body["messages"][0]["content"]
        video = next(block["video"] for block in content if "video" in block)
        assert video["format"] == "mp4"
        assert base64.b64decode(video["source"]["bytes"]) == clip.read_bytes()

    def test_video_format_comes_from_the_file_suffix(self, tmp_path):
        path = tmp_path / "rolling.webm"
        path.write_bytes(b"webm-bytes")
        client = FakeBedrock()
        _omni(client).understand(clip_path=str(path))

        content = client.body_for(OMNI_ID)["messages"][0]["content"]
        assert next(b["video"] for b in content if "video" in b)["format"] == "webm"

    def test_sends_clip_image_and_context_together(self, clip):
        client = FakeBedrock()
        _omni(client).understand(
            clip_path=str(clip), image_bytes=b"jpeg-bytes", context="Who is that?"
        )

        content = client.body_for(OMNI_ID)["messages"][0]["content"]
        assert any("video" in block for block in content)
        image = next(block["image"] for block in content if "image" in block)
        assert base64.b64decode(image["source"]["bytes"]) == b"jpeg-bytes"
        assert any("Who is that?" in block.get("text", "") for block in content)

    def test_image_only_call_still_reaches_omni(self):
        client = FakeBedrock(texts={OMNI_ID: "A mug on the desk."})
        result = _omni(client).understand(image_bytes=b"jpeg-bytes")

        assert result == "A mug on the desk."
        assert client.model_ids == [OMNI_ID]
        content = client.body_for(OMNI_ID)["messages"][0]["content"]
        assert not any("video" in block for block in content)


class TestOmniDisabled:
    """An empty model id means "preview not enabled" — fall back with no API call."""

    def test_empty_model_id_goes_straight_to_lite(self, clip):
        client = FakeBedrock(texts={LITE_ID: "Lite describes the scene."})
        omni = _omni(client, omni_model_id="")
        result = omni.understand(clip_path=str(clip), image_bytes=b"jpeg-bytes")

        assert result == "Lite describes the scene."
        assert client.model_ids == [LITE_ID], "Omni must not be invoked when unset"

    def test_disabled_fallback_is_named_in_a_sense_line(self, caplog, clip):
        client = FakeBedrock()
        omni = _omni(client, omni_model_id="")

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            omni.understand(clip_path=str(clip))

        lines = _sense_lines(caplog)
        assert lines, "expected a [SENSE ...] line naming the fallback"
        line = lines[0]
        assert line.group("stage") == "understand"
        assert line.group("source") == "vision"
        assert line.group("event")
        assert "fallback" in line.group("detail").lower()

    def test_config_supplies_the_empty_default(self, monkeypatch, clip):
        """With no NOVA_OMNI_MODEL_ID in the env, config's default disables Omni."""
        monkeypatch.delenv("NOVA_OMNI_MODEL_ID", raising=False)
        client = FakeBedrock(texts={LITE_ID: "config-driven lite answer"})
        omni = NovaOmni(client=client)

        assert omni.understand(clip_path=str(clip)) == "config-driven lite answer"
        assert client.model_ids == [LITE_ID]

    def test_config_env_override_enables_omni(self, monkeypatch, clip):
        monkeypatch.setenv("NOVA_OMNI_MODEL_ID", OMNI_ID)
        client = FakeBedrock(texts={OMNI_ID: "env-enabled omni answer"})
        omni = NovaOmni(client=client)

        assert omni.understand(clip_path=str(clip)) == "env-enabled omni answer"
        assert client.model_ids == [OMNI_ID]


class TestForcedOmniFailureFallsBackToLite:
    """The acceptance criterion: a forced Omni failure exercises the Lite-2 path."""

    def test_omni_error_returns_the_lite_answer(self, clip):
        client = FakeBedrock(
            texts={LITE_ID: "Lite: someone is holding a red mug."},
            fail={OMNI_ID},
        )
        result = _omni(client).understand(clip_path=str(clip), image_bytes=b"jpeg-bytes")

        assert result == "Lite: someone is holding a red mug."
        assert client.model_ids == [OMNI_ID, LITE_ID], "Omni tried first, then Lite"

    def test_fallback_lite_call_carries_the_image_not_the_video(self, clip):
        client = FakeBedrock(fail={OMNI_ID})
        _omni(client).understand(clip_path=str(clip), image_bytes=b"jpeg-bytes")

        content = client.body_for(LITE_ID)["messages"][0]["content"]
        assert not any("video" in block for block in content), "Lite 2 takes no video"
        image = next(block["image"] for block in content if "image" in block)
        assert base64.b64decode(image["source"]["bytes"]) == b"jpeg-bytes"

    def test_fallback_without_an_image_sends_a_text_only_prompt(self, clip):
        client = FakeBedrock(texts={LITE_ID: "no frame available answer"}, fail={OMNI_ID})
        result = _omni(client).understand(clip_path=str(clip), context="describe it")

        assert result == "no frame available answer"
        content = client.body_for(LITE_ID)["messages"][0]["content"]
        assert all("image" not in block and "video" not in block for block in content)
        assert any(block.get("text") for block in content)

    def test_missing_clip_file_falls_back_instead_of_raising(self, tmp_path):
        client = FakeBedrock(texts={LITE_ID: "still answered"})
        result = _omni(client).understand(
            clip_path=str(tmp_path / "nope.mp4"), image_bytes=b"jpeg-bytes"
        )

        assert result == "still answered"
        assert LITE_ID in client.model_ids

    def test_result_never_mentions_the_fallback(self, clip):
        client = FakeBedrock(texts={LITE_ID: "A calm empty room."}, fail={OMNI_ID})
        result = _omni(client).understand(clip_path=str(clip), image_bytes=b"x")

        lowered = result.lower()
        assert "fallback" not in lowered
        assert "lite" not in lowered
        assert "omni" not in lowered
        assert "error" not in lowered

    def test_failure_reason_is_named_in_a_sense_line(self, caplog, clip):
        client = FakeBedrock(fail={OMNI_ID})

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            _omni(client).understand(clip_path=str(clip), image_bytes=b"x")

        details = [line.group("detail") for line in _sense_lines(caplog)]
        assert details, "expected a [SENSE ...] line naming the fallback reason"
        joined = " ".join(details).lower()
        assert "fallback" in joined
        assert "forced failure" in joined, "the underlying exception must be named"
        for line in _sense_lines(caplog):
            assert line.group("stage") == "understand"
            assert line.group("source") == "vision"

    def test_unparseable_omni_response_falls_back(self, clip):
        class BrokenOmni(FakeBedrock):
            def invoke_model(self, modelId, body, **kwargs):  # noqa: N803
                self.calls.append((modelId, json.loads(body)))
                if modelId == OMNI_ID:
                    return {"body": _FakeBody({"unexpected": "shape"})}
                return _response("lite recovered")

        client = BrokenOmni()
        assert _omni(client).understand(clip_path=str(clip)) == "lite recovered"
        assert client.model_ids == [OMNI_ID, LITE_ID]


class TestBothPathsFail:
    """A total outage returns a string — never an exception into the main loop."""

    def test_returns_an_error_string_when_lite_also_fails(self, clip):
        client = FakeBedrock(fail={OMNI_ID, LITE_ID})
        result = _omni(client).understand(clip_path=str(clip), image_bytes=b"x")

        assert isinstance(result, str)
        assert result.startswith("[")
        assert "forced failure" in result


class TestClientIsLazyAndInjectable:
    """Tests must never construct a real boto3 client."""

    def test_constructor_does_not_build_a_client(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("boto3.client must not be called at construction time")

        monkeypatch.setattr(nova_omni.boto3, "client", explode, raising=False)
        NovaOmni()  # must not raise

    def test_client_is_built_lazily_once_and_cached(self, monkeypatch, clip):
        built: list[dict] = []
        fake = FakeBedrock()

        def fake_client(service_name, region_name=None, **kwargs):
            built.append({"service": service_name, "region": region_name})
            return fake

        monkeypatch.setattr(nova_omni.boto3, "client", fake_client, raising=False)
        omni = NovaOmni(region="eu-west-1", omni_model_id=OMNI_ID, lite_model_id=LITE_ID)
        omni.understand(clip_path=str(clip))
        omni.understand(clip_path=str(clip))

        assert len(built) == 1, "client should be created once and cached"
        assert built[0] == {"service": "bedrock-runtime", "region": "eu-west-1"}

    def test_injected_client_is_used_verbatim(self, clip):
        def explode(*args, **kwargs):
            raise AssertionError("injected client must be used")

        client = FakeBedrock()
        omni = _omni(client)
        omni.understand(clip_path=str(clip))
        assert client.calls


class TestModuleHygiene:
    """The client is harness-side: no robot SDK, no new dependencies."""

    def test_does_not_import_reachy_mini(self):
        source = (nova_omni.__file__ or "").strip()
        assert source.endswith("nova_omni.py")
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "reachy_mini" not in text

    def test_region_defaults_to_config(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
        assert NovaOmni().region == "ap-south-1"
