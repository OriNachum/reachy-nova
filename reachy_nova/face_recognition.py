"""Face recognition engine for Reachy Nova.

Uses OpenCV's built-in YuNet (face detection) + SFace (face recognition).
Lightweight enough for Raspberry Pi CM4 — no insightface/ONNX GPU needed.

YuNet model: ~230KB, very fast face detector
SFace model: ~37MB, 128-dim face embeddings

Threading pattern mirrors YOLO in tracking.py: update_frame() called from
the main loop dispatches to a daemon thread if interval elapsed and not busy.
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

from .face_manager import FaceManager

logger = logging.getLogger(__name__)

DETECT_INTERVAL = 0.5  # 500ms between detections
REANNOUNCE_COOLDOWN = 30.0  # don't re-announce same person within 30s

# Model storage
MODELS_DIR = Path.home() / ".reachy_nova" / "models"

# OpenCV model zoo URLs
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"


def _ensure_model(filename: str, url: str) -> Path:
    """Download model if not present."""
    path = MODELS_DIR / filename
    if path.exists():
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {filename}...")
    urlretrieve(url, str(path))
    logger.info(f"Downloaded {filename} ({path.stat().st_size / 1024:.0f} KB)")
    return path


class FaceRecognition:
    """Background face recognition using OpenCV YuNet + SFace."""

    def __init__(
        self,
        face_manager: FaceManager,
        on_match: Callable[[str, str, float], None] | None = None,
        on_face_bbox: Callable[[tuple | None], None] | None = None,
    ):
        """
        Args:
            face_manager: FaceManager instance for matching
            on_match: callback(unique_id, name, score) when a face is recognized
            on_face_bbox: callback(bbox_norm) with normalized (x1,y1,x2,y2) or None
        """
        self.face_manager = face_manager
        self.on_match = on_match
        self.on_face_bbox = on_face_bbox

        # Models (lazy-loaded)
        self._detector = None
        self._recognizer = None
        self._model_loaded = False

        # Threading
        self._thread: threading.Thread | None = None
        self._busy = False
        self._stop_event: threading.Event | None = None
        self._lock = threading.Lock()

        # Timing
        self._last_detect_time = 0.0

        # State
        self._current_embedding: np.ndarray | None = None
        self._recognized_person: tuple[str, str, float] | None = None  # (id, name, score)
        self._last_announced: dict[str, float] = {}  # id -> timestamp

    def start(self, stop_event: threading.Event):
        """Register stop event. Detection runs via update_frame() from main loop."""
        self._stop_event = stop_event
        logger.info("Face recognition engine ready")

    def _load_model(self):
        """Lazy-load YuNet + SFace on first use."""
        if self._model_loaded:
            return
        try:
            yunet_path = _ensure_model(YUNET_FILE, YUNET_URL)
            sface_path = _ensure_model(SFACE_FILE, SFACE_URL)

            self._detector = cv2.FaceDetectorYN.create(
                str(yunet_path), "", (320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(sface_path), "",
            )
            self._model_loaded = True
            logger.info("Face models loaded (YuNet + SFace)")
        except Exception as e:
            logger.error(f"Failed to load face models: {e}")
            self._model_loaded = True  # don't retry
            self._detector = None
            self._recognizer = None

    def update_frame(self, frame: np.ndarray, t: float):
        """Submit a frame for face recognition (non-blocking).

        Called from main loop. Dispatches to background thread if interval
        elapsed and previous detection is done.

        Args:
            frame: BGR numpy array from camera
            t: current time (seconds since app start)
        """
        if t - self._last_detect_time < DETECT_INTERVAL:
            return
        if self._busy:
            return

        self._last_detect_time = t
        self._busy = True

        frame_copy = frame.copy()
        self._thread = threading.Thread(
            target=self._run_detection, args=(frame_copy,), daemon=True
        )
        self._thread.start()

    def _run_detection(self, frame: np.ndarray):
        """Run face detection and recognition in background thread."""
        try:
            self._load_model()
            if self._detector is None or self._recognizer is None:
                return

            h, w = frame.shape[:2]
            self._detector.setInputSize((w, h))

            _, faces = self._detector.detect(frame)
            if faces is None or len(faces) == 0:
                with self._lock:
                    self._current_embedding = None
                    self._recognized_person = None
                if self.on_face_bbox:
                    try:
                        self.on_face_bbox(None)
                    except Exception as e:
                        logger.warning(f"on_face_bbox callback error: {e}")
                return

            # Pick largest face by bounding box area
            areas = faces[:, 2] * faces[:, 3]  # width * height
            best_idx = int(np.argmax(areas))
            best_face = faces[best_idx]

            # Fire normalized face bbox callback (x, y, w, h are first 4 columns)
            bx, by, bw, bh = float(best_face[0]), float(best_face[1]), float(best_face[2]), float(best_face[3])
            bbox_norm = (bx / w, by / h, (bx + bw) / w, (by + bh) / h)
            if self.on_face_bbox:
                try:
                    self.on_face_bbox(bbox_norm)
                except Exception as e:
                    logger.warning(f"on_face_bbox callback error: {e}")

            # Align and extract embedding (128-dim)
            aligned = self._recognizer.alignCrop(frame, best_face)
            embedding = self._recognizer.feature(aligned).flatten()

            with self._lock:
                self._current_embedding = embedding

            # Match against known faces
            match = self.face_manager.match(embedding)
            if match:
                fid, name, score = match
                with self._lock:
                    self._recognized_person = match

                # Check reannounce cooldown
                now = time.time()
                last = self._last_announced.get(fid, 0.0)
                if now - last > REANNOUNCE_COOLDOWN:
                    self._last_announced[fid] = now
                    logger.info(f"Face recognized: {name} (id={fid}, score={score:.3f})")
                    if self.on_match:
                        try:
                            self.on_match(fid, name, score)
                        except Exception as e:
                            logger.warning(f"on_match callback error: {e}")
            else:
                with self._lock:
                    self._recognized_person = None

        except Exception as e:
            logger.warning(f"Face recognition error: {e}")
        finally:
            self._busy = False

    def get_current_embedding(self) -> np.ndarray | None:
        """Get the latest detected face embedding (thread-safe)."""
        with self._lock:
            return self._current_embedding.copy() if self._current_embedding is not None else None

    def get_recognized_person(self) -> tuple[str, str, float] | None:
        """Get the currently recognized person (thread-safe).

        Returns:
            (unique_id, name, score) or None
        """
        with self._lock:
            return self._recognized_person

    def is_admin_authenticated(self) -> bool:
        """Check if the currently recognized person is an admin.

        Camera-only auth: checks if face_recognition currently sees an admin face.
        Voice claims are not accepted.
        """
        person = self.get_recognized_person()
        if person is None:
            return False
        fid, _, _ = person
        return self.face_manager.is_admin(fid)
