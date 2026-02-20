"""Wake word detection using NVIDIA Parakeet TDT (ASR-based).

Buffers a rolling window of sleep-mode audio, periodically transcribes it with
Parakeet TDT via NeMo, and returns True when the configured wake phrase is found
in the transcript.  Transcription runs in a background thread so the sleep
breathing animation is never blocked.
"""

import concurrent.futures
import logging
import re
import time

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Hz — must match the existing mic pipeline


class WakeWordDetector:
    """ASR-based wake word detector backed by NVIDIA Parakeet TDT."""

    def __init__(
        self,
        phrase: str = "hey reachy",
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        transcribe_interval: float = 2.0,
        buffer_seconds: float = 4.0,
    ):
        """
        Args:
            phrase: The spoken phrase that wakes the robot (case-insensitive).
            model_name: NeMo model name or local checkpoint path.
            transcribe_interval: Seconds between transcription runs.
            buffer_seconds: Rolling audio window kept for each transcription.
        """
        import nemo.collections.asr as nemo_asr

        self._phrase = _normalize(phrase)
        self._transcribe_interval = transcribe_interval
        self._buffer_max = int(buffer_seconds * SAMPLE_RATE)

        self._buffer: list[np.ndarray] = []
        self._last_transcribe = 0.0

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="wake-word"
        )
        self._pending: concurrent.futures.Future | None = None

        logger.info(f"[WakeWord] Loading model {model_name!r} …")
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self._model.eval()
        # Suppress NeMo's per-call "Transcribing..." INFO messages
        logging.getLogger("nemo").setLevel(logging.WARNING)
        logger.info(f"[WakeWord] Ready — phrase: {self._phrase!r}, interval: {transcribe_interval}s")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, audio: np.ndarray) -> bool:
        """Feed an audio chunk and return True when the wake phrase is heard.

        Args:
            audio: float32 samples in [-1, 1] at 16 kHz.

        Returns:
            True exactly once when the phrase is found; False otherwise.
        """
        self._push(audio)

        # Check if a background transcription just finished
        if self._pending is not None and self._pending.done():
            fut, self._pending = self._pending, None
            try:
                if fut.result():
                    return True
            except Exception as e:
                logger.warning(f"[WakeWord] Transcription error: {e}")

        # Launch a new transcription if the interval has elapsed and none is running
        now = time.time()
        if self._pending is None and now - self._last_transcribe >= self._transcribe_interval:
            self._last_transcribe = now
            snapshot = np.concatenate(self._buffer).copy()
            self._pending = self._executor.submit(self._transcribe, snapshot)

        return False

    def reset(self) -> None:
        """Clear audio buffer and cancel any pending transcription.

        Call this every time the robot enters sleep so that audio captured
        while awake cannot trigger an immediate re-wake.
        """
        self._buffer.clear()
        self._last_transcribe = 0.0
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
        self._pending = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push(self, chunk: np.ndarray) -> None:
        """Append chunk to the rolling buffer, trimming to buffer_max samples."""
        self._buffer.append(chunk)
        total = sum(len(c) for c in self._buffer)
        while len(self._buffer) > 1 and total - len(self._buffer[0]) >= self._buffer_max:
            total -= len(self._buffer.pop(0))

    def _transcribe(self, audio: np.ndarray) -> bool:
        """Run inference on a float32 audio snapshot (called from thread pool)."""
        results = self._model.transcribe([audio], verbose=False)
        r = results[0]
        text = r.text if hasattr(r, "text") else str(r)
        transcript = _normalize(text)
        logger.debug(f"[WakeWord] Transcript: {transcript!r}")
        matched = self._phrase in transcript
        if matched:
            logger.info(f"[WakeWord] Phrase '{self._phrase}' matched in: {transcript!r}")
        return matched


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for robust phrase matching."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()
