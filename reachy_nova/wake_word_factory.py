"""Factory for creating wake word detectors based on deployment config."""

import logging

import numpy as np

from .config import WakeWordConfig
from .wake_word_base import WakeWordBase

logger = logging.getLogger(__name__)


class NullWakeWordDetector(WakeWordBase):
    """No-op detector used when wake word detection is disabled or unavailable."""

    def detect(self, audio: np.ndarray) -> bool:
        return False

    def reset(self) -> None:
        pass


def create_wake_word(config: WakeWordConfig) -> WakeWordBase:
    """Create a wake word detector based on config.

    Falls back to NullWakeWordDetector with a warning if the requested
    backend's dependencies are not installed.
    """
    backend = config.backend

    if backend == "disabled":
        logger.info("[WakeWord] Wake word detection disabled by config")
        return NullWakeWordDetector()

    if backend == "parakeet":
        try:
            from .wake_word import WakeWordDetector
            return WakeWordDetector(
                phrase=config.phrase,
                model_name=config.parakeet.model,
            )
        except ImportError:
            logger.warning(
                "[WakeWord] Parakeet requested but nemo_toolkit not installed. "
                "Install with: uv sync --extra full. Falling back to NullDetector."
            )
            return NullWakeWordDetector()
        except Exception as e:
            logger.warning(f"[WakeWord] Failed to load Parakeet: {e}. Falling back to NullDetector.")
            return NullWakeWordDetector()

    if backend == "openwakeword":
        try:
            from .wake_word_oww import OpenWakeWordDetector
            return OpenWakeWordDetector(
                model_name=config.openwakeword.model,
                threshold=config.openwakeword.threshold,
            )
        except ImportError:
            logger.warning(
                "[WakeWord] OpenWakeWord requested but not installed. "
                "Install with: uv sync --extra pi. Falling back to NullDetector."
            )
            return NullWakeWordDetector()
        except Exception as e:
            logger.warning(f"[WakeWord] Failed to load OpenWakeWord: {e}. Falling back to NullDetector.")
            return NullWakeWordDetector()

    logger.warning(f"[WakeWord] Unknown backend {backend!r}. Using NullDetector.")
    return NullWakeWordDetector()
