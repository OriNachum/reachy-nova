"""Nova Feedback - RLHF feedback capture for Reachy Nova.

Records multimodal feedback packages (conversation, camera frames, audio)
for future reinforcement learning from human feedback. Each feedback event
creates a timestamped folder with metadata, messages, JPEG frames, and a
WAV audio file.

Supports three storage modes via FEEDBACK_STORAGE env var:
  - "local"    — Save to disk only (default fallback if no AWS creds)
  - "local+s3" — Save to disk, then upload to S3 (default)
  - "s3"       — Upload directly from memory to S3, no local disk writes
"""

import io
import json
import logging
import os
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

    def __init__(self, session_id: str = ""):
        self._lock = threading.Lock()
        self._session_id = session_id

        # Conversation buffer: list of {role, text, timestamp}
        self._messages: deque[dict] = deque(maxlen=MAX_MESSAGES)

        # Frame buffer: list of (timestamp, jpeg_bytes)
        self._frames: deque[tuple[float, bytes]] = deque(maxlen=MAX_FRAMES)
        self._last_frame_time = 0.0

        # Audio buffer: list of float32 numpy chunks
        self._audio_chunks: deque[np.ndarray] = deque()
        self._audio_total_samples = 0

        # S3 configuration
        self._s3_bucket = os.environ.get("FEEDBACK_S3_BUCKET", "reachy-nova-feedback")
        self._storage_mode = os.environ.get("FEEDBACK_STORAGE", "local+s3")
        self._s3_client = None
        self._s3_initialized = False

        if self._storage_mode not in ("local", "local+s3", "s3"):
            logger.warning(f"[Feedback] Unknown FEEDBACK_STORAGE={self._storage_mode}, falling back to 'local'")
            self._storage_mode = "local"

        logger.info(f"[Feedback] Storage mode: {self._storage_mode}, bucket: {self._s3_bucket}")

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str):
        self._session_id = value

    # --- S3 lazy initialization ---

    def _get_s3_client(self):
        """Lazy-init boto3 S3 client on first use. Returns None if unavailable."""
        if self._s3_initialized:
            return self._s3_client
        self._s3_initialized = True
        try:
            import boto3
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            self._s3_client = boto3.client("s3", region_name=region)
            # Auto-create bucket if it doesn't exist
            try:
                if region == "us-east-1":
                    self._s3_client.create_bucket(Bucket=self._s3_bucket)
                else:
                    self._s3_client.create_bucket(
                        Bucket=self._s3_bucket,
                        CreateBucketConfiguration={"LocationConstraint": region},
                    )
                logger.info(f"[Feedback] S3 bucket created: {self._s3_bucket}")
            except self._s3_client.exceptions.BucketAlreadyOwnedByYou:
                pass
            except Exception as e:
                # Bucket may already exist (owned by someone else) or other error
                if "BucketAlreadyExists" not in str(e):
                    logger.warning(f"[Feedback] S3 bucket creation note: {e}")
            logger.info(f"[Feedback] S3 client ready (region={region})")
        except Exception as e:
            logger.warning(f"[Feedback] S3 not available: {e}")
            self._s3_client = None
            if self._storage_mode in ("local+s3", "s3"):
                logger.warning("[Feedback] Falling back to local storage (no S3)")
                self._storage_mode = "local"
        return self._s3_client

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

    # --- Message access for session persistence ---

    def get_recent_messages(self, count: int = 20) -> list[dict]:
        """Return the last N conversation messages (thread-safe)."""
        with self._lock:
            msgs = list(self._messages)
        return msgs[-count:]

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
            "session_id": self._session_id,
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
        """Save feedback package according to storage mode (runs in background thread)."""
        mode = self._storage_mode

        if mode == "local":
            self._save_local(folder_name, metadata, messages, frames, audio_chunks)

        elif mode == "local+s3":
            self._save_local(folder_name, metadata, messages, frames, audio_chunks)
            self._upload_folder_to_s3(folder_name)

        elif mode == "s3":
            self._save_s3_direct(folder_name, metadata, messages, frames, audio_chunks)

    def _save_local(
        self,
        folder_name: str,
        metadata: dict,
        messages: list[dict],
        frames: list[tuple[float, bytes]],
        audio_chunks: list[np.ndarray],
    ) -> None:
        """Write feedback package to local disk."""
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

            logger.info(f"[Feedback] Saved package locally: {folder}")

        except Exception as e:
            logger.error(f"[Feedback] Local save error: {e}")

    def _upload_folder_to_s3(self, folder_name: str) -> None:
        """Walk a local feedback folder and upload each file to S3."""
        s3 = self._get_s3_client()
        if not s3:
            return
        folder = FEEDBACK_DIR / folder_name
        if not folder.exists():
            return
        prefix = f"feedback/{self._session_id}/{folder_name}"
        try:
            for path in folder.rglob("*"):
                if path.is_file():
                    key = f"{prefix}/{path.relative_to(folder)}"
                    s3.upload_file(str(path), self._s3_bucket, key)
            logger.info(f"[Feedback] Uploaded to S3: s3://{self._s3_bucket}/{prefix}")
        except Exception as e:
            logger.error(f"[Feedback] S3 upload error: {e}")

    def _save_s3_direct(
        self,
        folder_name: str,
        metadata: dict,
        messages: list[dict],
        frames: list[tuple[float, bytes]],
        audio_chunks: list[np.ndarray],
    ) -> None:
        """Upload feedback directly from memory to S3 (no local disk writes)."""
        s3 = self._get_s3_client()
        if not s3:
            # Fallback to local if S3 is unavailable
            logger.warning("[Feedback] S3 unavailable in s3-only mode, falling back to local save")
            self._save_local(folder_name, metadata, messages, frames, audio_chunks)
            return

        prefix = f"feedback/{self._session_id}/{folder_name}"
        try:
            # feedback.json
            s3.put_object(
                Bucket=self._s3_bucket,
                Key=f"{prefix}/feedback.json",
                Body=json.dumps(metadata, indent=2).encode(),
                ContentType="application/json",
            )

            # messages.json
            s3.put_object(
                Bucket=self._s3_bucket,
                Key=f"{prefix}/messages.json",
                Body=json.dumps(messages, indent=2).encode(),
                ContentType="application/json",
            )

            # frames/
            for i, (ts, jpeg_bytes) in enumerate(frames):
                s3.put_object(
                    Bucket=self._s3_bucket,
                    Key=f"{prefix}/frames/{i:06d}.jpg",
                    Body=jpeg_bytes,
                    ContentType="image/jpeg",
                )

            # audio.wav (build in memory)
            if audio_chunks:
                wav_bytes = self._build_wav_bytes(audio_chunks)
                s3.put_object(
                    Bucket=self._s3_bucket,
                    Key=f"{prefix}/audio.wav",
                    Body=wav_bytes,
                    ContentType="audio/wav",
                )

            logger.info(f"[Feedback] Uploaded directly to S3: s3://{self._s3_bucket}/{prefix}")

        except Exception as e:
            logger.error(f"[Feedback] S3 direct upload failed: {e}")
            # Fallback: save locally so data is not lost
            logger.warning("[Feedback] Falling back to local save")
            self._save_local(folder_name, metadata, messages, frames, audio_chunks)

    @staticmethod
    def _build_wav_bytes(audio_chunks: list[np.ndarray]) -> bytes:
        """Build a complete WAV file in memory from audio chunks."""
        audio = np.concatenate(audio_chunks)
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        data = pcm.tobytes()
        data_size = len(data)

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            AUDIO_SAMPLE_RATE,
            AUDIO_SAMPLE_RATE * 2,
            2,
            16,
            b"data",
            data_size,
        )

        buf = io.BytesIO()
        buf.write(header)
        buf.write(data)
        return buf.getvalue()

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
                "storage_mode": self._storage_mode,
                "s3_bucket": self._s3_bucket if self._storage_mode != "local" else None,
            }
