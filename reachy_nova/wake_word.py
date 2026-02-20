"""Wake word detection using openWakeWord.

Supports built-in pretrained models (e.g. "hey_jarvis") and custom ONNX models
trained with the openWakeWord toolkit.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects a wake word in audio using openWakeWord."""

    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5):
        """
        Args:
            model: Built-in model name (e.g. "hey_jarvis") or path to custom .onnx file.
            threshold: Detection confidence threshold (0–1).
        """
        from openwakeword.model import Model

        self.threshold = threshold
        self._oww = Model(wakeword_models=[model], inference_framework="onnx")
        logger.info(f"[WakeWord] Loaded model: {model} (threshold={threshold})")

    def detect(self, audio: np.ndarray) -> bool:
        """Check if the wake word is present in audio.

        Args:
            audio: float32 samples in [-1, 1] at 16 kHz.

        Returns:
            True if any model score meets or exceeds the threshold.
        """
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        predictions = self._oww.predict(audio_int16)
        return any(score >= self.threshold for score in predictions.values())

    def reset(self) -> None:
        """Reset internal buffer state — call when entering sleep."""
        try:
            if hasattr(self._oww, "reset"):
                self._oww.reset()
            else:
                # Fallback: clear per-model buffers if they expose a reset
                for m in self._oww.models.values():
                    if hasattr(m, "reset"):
                        m.reset()
        except Exception as e:
            logger.debug(f"[WakeWord] Reset warning: {e}")
