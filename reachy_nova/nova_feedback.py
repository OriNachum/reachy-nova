"""Nova Feedback - RLHF feedback capture for Reachy Nova.

Records multimodal feedback packages (conversation, camera frames, audio)
for future reinforcement learning from human feedback. Each feedback event
creates a timestamped folder with metadata, messages, JPEG frames, and a
WAV audio file.
"""

import json
import logging
import struct
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Storage root
FEEDBACK_DIR = Path.home() / "reachy_nova_data" / "feedback"

# Buffer limits
MAX_MESSAGES = 50
MAX_FRAMES = 240          # 2 min at 2 Hz
FRAME_INTERVAL = 0.5      # seconds between frame captures
MAX_AUDIO_SAMPLES = 2 * 60 * 16000  # 2 minutes at 16kHz
AUDIO_SAMPLE_RATE = 16000
JPEG_QUALITY = 85


class NovaFeedback:
    """Rolling multimodal buffers with snapshot-and-save for RLHF feedback."""

    def __init__(self):
        self._lock = threading.Lock()

        # Conversation buffer: list of {role, text, timestamp}
        self._messages: deque[dict] = deque(maxlen=MAX_MESSAGES)

        # Frame buffer: list of (timestamp, jpeg_bytes)
        self._frames: deque[tuple[float, bytes]] = deque(maxlen=MAX_FRAMES)
        self._last_frame_time = 0.0

        # Audio buffer: list of float32 numpy chunks
        self._audio_chunks: deque[np.ndarray] = deque()
        self._audio_total_samples = 0

    # --- Buffer feeds (called from main loop) ---

    def update_conversation(self, role: str, text: str) -> None:
        """Add a conversation turn to the rolling buffer."""
        entry = {
            "role": role,
            "text": text,
            "timestamp": time.time(),
        }
        with self._lock:
            self._messages.append(entry)

    def update_frame(self, frame: np.ndarray) -> None:
        """Add a camera frame (throttled to 2Hz, JPEG-compressed on capture)."""
        now = time.time()
        if now - self._last_frame_time < FRAME_INTERVAL:
            return
        self._last_frame_time = now

        # JPEG encode
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        jpeg_bytes = buf.tobytes()

        with self._lock:
            self._frames.append((now, jpeg_bytes))

    def update_audio(self, chunk: np.ndarray) -> None:
        """Add an audio chunk to the rolling buffer (16kHz float32 mono)."""
        n = len(chunk)
        with self._lock:
            self._audio_chunks.append(chunk.copy())
            self._audio_total_samples += n
            # Trim oldest chunks to stay within 2-minute window
            while self._audio_total_samples > MAX_AUDIO_SAMPLES and self._audio_chunks:
                removed = self._audio_chunks.popleft()
                self._audio_total_samples -= len(removed)

    # --- Snapshot and save ---

    def record(self, sentiment: str, what: str, trigger: str = "") -> str:
        """Snapshot all buffers and save a feedback package in a background thread.

        Returns immediately with a confirmation string.
        """
        feedback_id = uuid.uuid4().hex[:6]
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        folder_name = f"{timestamp}_{sentiment}_fb_{feedback_id}"

        # Snapshot under lock
        with self._lock:
            messages = list(self._messages)
            frames = list(self._frames)
            audio_chunks = list(self._audio_chunks)

        metadata = {
            "id": feedback_id,
            "timestamp": time.time(),
            "timestamp_str": timestamp,
            "sentiment": sentiment,
            "what": what,
            "trigger": trigger,
            "num_messages": len(messages),
            "num_frames": len(frames),
            "audio_chunks": len(audio_chunks),
        }

        # Fire-and-forget background save
        threading.Thread(
            target=self._save_package,
            args=(folder_name, metadata, messages, frames, audio_chunks),
            daemon=True,
            name=f"feedback-save-{feedback_id}",
        ).start()

        logger.info(
            f"[Feedback] Recording {sentiment} feedback: {what} "
            f"({len(messages)} msgs, {len(frames)} frames, {len(audio_chunks)} audio chunks)"
        )
        return f"[Feedback recorded: {sentiment} — {what}]"

    def _save_package(
        self,
        folder_name: str,
        metadata: dict,
        messages: list[dict],
        frames: list[tuple[float, bytes]],
        audio_chunks: list[np.ndarray],
    ) -> None:
        """Write feedback package to disk (runs in background thread)."""
        try:
            folder = FEEDBACK_DIR / folder_name
            folder.mkdir(parents=True, exist_ok=True)

            # feedback.json
            with open(folder / "feedback.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # messages.json
            with open(folder / "messages.json", "w") as f:
                json.dump(messages, f, indent=2)

            # frames/
            if frames:
                frames_dir = folder / "frames"
                frames_dir.mkdir(exist_ok=True)
                for i, (ts, jpeg_bytes) in enumerate(frames):
                    (frames_dir / f"{i:06d}.jpg").write_bytes(jpeg_bytes)

            # audio.wav (16kHz mono int16 PCM)
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
                self._write_wav(folder / "audio.wav", audio)

            logger.info(f"[Feedback] Saved package: {folder}")

        except Exception as e:
            logger.error(f"[Feedback] Save error: {e}")

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray) -> None:
        """Write float32 audio as 16kHz mono int16 WAV using struct (no wave module quirks)."""
        # Convert float32 [-1, 1] to int16
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        data = pcm.tobytes()
        data_size = len(data)

        # WAV header (44 bytes)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,     # file size - 8
            b"WAVE",
            b"fmt ",
            16,                 # fmt chunk size
            1,                  # PCM format
            1,                  # mono
            AUDIO_SAMPLE_RATE,  # sample rate
            AUDIO_SAMPLE_RATE * 2,  # byte rate (16-bit mono)
            2,                  # block align
            16,                 # bits per sample
            b"data",
            data_size,
        )

        with open(path, "wb") as f:
            f.write(header)
            f.write(data)

    # --- Stats ---

    def get_stats(self) -> dict:
        """Return current buffer sizes for monitoring."""
        with self._lock:
            return {
                "messages": len(self._messages),
                "frames": len(self._frames),
                "audio_samples": self._audio_total_samples,
                "audio_seconds": round(self._audio_total_samples / AUDIO_SAMPLE_RATE, 1),
            }
