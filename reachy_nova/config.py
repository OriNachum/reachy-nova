"""Central model/region configuration for every Nova service (t1).

All Bedrock model IDs and the AWS region live here, env-overridable with the
previously hardcoded literals as defaults. Values are resolved at *call* time
(not import time) so ``load_dotenv()`` in an entry point takes effect no matter
the import order.
"""

from __future__ import annotations

import os

_DEFAULTS = {
    "AWS_DEFAULT_REGION": "us-east-1",
    "NOVA_SONIC_MODEL_ID": "amazon.nova-2-sonic-v1:0",
    "NOVA_LITE_MODEL_ID": "us.amazon.nova-2-lite-v1:0",
    "NOVA_OMNI_MODEL_ID": "",  # preview — empty means "not enabled, use fallback"
    "NOVA_EMBEDDING_MODEL_ID": "amazon.nova-2-multimodal-embeddings-v1:0",
}


def _get(key: str) -> str:
    return os.environ.get(key, _DEFAULTS[key])


def region() -> str:
    """AWS region for every Bedrock call."""
    return _get("AWS_DEFAULT_REGION")


def sonic_model_id() -> str:
    """Nova 2 Sonic — realtime bidirectional voice."""
    return _get("NOVA_SONIC_MODEL_ID")


def lite_model_id() -> str:
    """Nova 2 Lite — fast judgments (barge-in, vision fallback)."""
    return _get("NOVA_LITE_MODEL_ID")


def omni_model_id() -> str:
    """Nova 2 Omni — deep multimodal understanding (preview; may be empty)."""
    return _get("NOVA_OMNI_MODEL_ID")


def embedding_model_id() -> str:
    """Nova multimodal embeddings — memory retrieval."""
    return _get("NOVA_EMBEDDING_MODEL_ID")
