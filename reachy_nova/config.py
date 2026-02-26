"""Deployment configuration for Reachy Nova.

Loads a YAML config file that controls which backends and features are enabled,
allowing the same codebase to run on DGX Spark (full GPU), wireless setups,
or a Raspberry Pi CM4 with lightweight alternatives.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Project root (parent of reachy_nova/ package)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ParakeetConfig:
    model: str = "nvidia/parakeet-tdt-0.6b-v2"


@dataclass
class OpenWakeWordConfig:
    model: str = "hey_jarvis"
    threshold: float = 0.5


@dataclass
class WakeWordConfig:
    backend: str = "parakeet"  # parakeet | openwakeword | disabled
    phrase: str = "hey reachy"
    snap_fallback: bool = True
    parakeet: ParakeetConfig = field(default_factory=ParakeetConfig)
    openwakeword: OpenWakeWordConfig = field(default_factory=OpenWakeWordConfig)


@dataclass
class FeaturesConfig:
    browser: bool = True
    browser_headless: bool = False
    yolo_tracking: bool = True
    memory: bool = True


@dataclass
class NovaConfig:
    mode: str = "auto"  # auto | lite | wireless | ondevice
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)


# Mode-based defaults: applied first, then explicit YAML values override.
_MODE_DEFAULTS: dict[str, dict] = {
    "auto": {},
    "lite": {},
    "wireless": {
        "features": {"browser_headless": True},
    },
    "ondevice": {
        "wake_word": {"backend": "openwakeword"},
        "features": {
            "yolo_tracking": False,
            "memory": False,
            "browser_headless": True,
        },
    },
}


def load_config(path: str | Path | None = None) -> NovaConfig:
    """Load deployment config from YAML.

    Resolution order:
        1. Explicit *path* argument
        2. ``REACHY_NOVA_CONFIG`` environment variable
        3. ``config/deployment.yaml`` relative to project root
        4. Built-in defaults (all features enabled, parakeet backend)
    """
    if path is None:
        path = os.environ.get("REACHY_NOVA_CONFIG")

    if path is not None:
        path = Path(path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path

    if path is None:
        path = _PROJECT_ROOT / "config" / "deployment.yaml"

    raw: dict = {}
    if path.exists():
        logger.info(f"[Config] Loading deployment config from {path}")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        logger.info(f"[Config] No config at {path} — using defaults")

    return _build_config(raw)


def _build_config(raw: dict) -> NovaConfig:
    """Build a NovaConfig from raw YAML dict, applying mode defaults first."""
    mode = raw.get("mode", "auto")

    # Start with mode defaults
    defaults = _MODE_DEFAULTS.get(mode, {})
    merged = _deep_merge(defaults, raw)

    # Build nested configs
    ww_raw = merged.get("wake_word", {})
    parakeet_raw = ww_raw.get("parakeet", {})
    oww_raw = ww_raw.get("openwakeword", {})

    wake_word = WakeWordConfig(
        backend=ww_raw.get("backend", WakeWordConfig.backend),
        phrase=ww_raw.get("phrase", WakeWordConfig.phrase),
        snap_fallback=ww_raw.get("snap_fallback", WakeWordConfig.snap_fallback),
        parakeet=ParakeetConfig(
            model=parakeet_raw.get("model", ParakeetConfig.model),
        ),
        openwakeword=OpenWakeWordConfig(
            model=oww_raw.get("model", OpenWakeWordConfig.model),
            threshold=oww_raw.get("threshold", OpenWakeWordConfig.threshold),
        ),
    )

    feat_raw = merged.get("features", {})
    features = FeaturesConfig(
        browser=feat_raw.get("browser", FeaturesConfig.browser),
        browser_headless=feat_raw.get("browser_headless", FeaturesConfig.browser_headless),
        yolo_tracking=feat_raw.get("yolo_tracking", FeaturesConfig.yolo_tracking),
        memory=feat_raw.get("memory", FeaturesConfig.memory),
    )

    config = NovaConfig(mode=mode, wake_word=wake_word, features=features)
    logger.info(
        f"[Config] mode={config.mode}, wake_word={config.wake_word.backend}, "
        f"yolo={config.features.yolo_tracking}, memory={config.features.memory}, "
        f"browser={config.features.browser}, headless={config.features.browser_headless}"
    )
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
