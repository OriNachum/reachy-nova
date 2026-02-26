"""Abstract base class for wake word detectors."""

from abc import ABC, abstractmethod

import numpy as np


class WakeWordBase(ABC):
    """Interface that all wake word backends must implement."""

    @abstractmethod
    def detect(self, audio: np.ndarray) -> bool:
        """Feed an audio chunk and return True when the wake phrase is heard.

        Args:
            audio: float32 samples in [-1, 1] at 16 kHz.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state (audio buffer, pending transcription, etc.)."""
        ...
