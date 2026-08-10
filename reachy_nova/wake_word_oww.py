"""Wake word detection using OpenWakeWord (CPU-only, Pi-compatible).

Uses the openwakeword library for lightweight, always-on wake word detection
without requiring a GPU.
"""

import logging
import os

import numpy as np

from .wake_word_base import WakeWordBase

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class OpenWakeWordDetector(WakeWordBase):
    """OpenWakeWord-based wake word detector."""

    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5):
        import openwakeword
        from openwakeword.model import Model

        # Resolve short name (e.g. "alexa") to full bundled .onnx path
        model_path = model_name
        for p in openwakeword.get_pretrained_model_paths():
            if os.path.basename(p).startswith(model_name):
                model_path = p
                break
        self._model = Model(wakeword_model_paths=[model_path])
        self._threshold = threshold
        self._model_name = model_name
        logger.info(f"[WakeWord-OWW] Loaded model {model_name!r}, threshold={threshold}")

    def detect(self, audio: np.ndarray) -> bool:
        """Feed float32 audio and return True when wake word is detected."""
        # OpenWakeWord expects int16 samples
        int16_audio = (audio * 32767).astype(np.int16)
        prediction = self._model.predict(int16_audio)

        for model_name, score in prediction.items():
            if score >= self._threshold:
                logger.info(f"[WakeWord-OWW] Detected {model_name!r} (score={score:.3f})")
                self._model.reset()
                return True
        return False

    def reset(self) -> None:
        """Reset internal model state."""
        self._model.reset()
