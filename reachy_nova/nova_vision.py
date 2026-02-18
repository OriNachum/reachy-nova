"""Nova Vision - Camera frame analysis via Amazon Bedrock Nova 2 Lite."""

import base64
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable

import cv2
import numpy as np
import boto3

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Describe what is visible in this image in 1-2 short, plain sentences. "
    "Be specific about people, objects, and what's happening. "
    "Do not greet anyone or add emotional commentary. "
    "Do not say 'I see' — just state what is there."
)


class NovaVision:
    """Periodically analyzes camera frames using Nova 2 Lite."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "us.amazon.nova-2-lite-v1:0",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        analyze_interval: float = 30.0,
        on_description: Callable[[str], None] | None = None,
    ):
        self.region = region
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.analyze_interval = analyze_interval
        self.on_description = on_description

        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._thread: threading.Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._force_analyze = threading.Event()

        # Ring buffer of last 3 frames: (timestamp, frame)
        self._frame_buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=3)

        self.last_description = ""
        self.analyzing = False
        self.last_analyze_time = 0.0

    def update_frame(self, frame: np.ndarray) -> None:
        """Update the latest camera frame (BGR format from OpenCV)."""
        with self._frame_lock:
            self._latest_frame = frame
            self._frame_buffer.append((time.time(), frame.copy()))

    def trigger_analyze(self) -> None:
        """Force an immediate analysis of the current frame (event path)."""
        self._force_analyze.set()

    def reset_timer(self) -> None:
        """Reset the fallback analysis countdown (called after event-triggered analysis)."""
        self.last_analyze_time = time.time()

    def get_latest_frames(self) -> list[tuple[float, np.ndarray]]:
        """Return up to 3 most recent buffered frames as (timestamp, frame) tuples."""
        with self._frame_lock:
            return list(self._frame_buffer)

    def analyze_latest(self, prompt: str = "What do you see?") -> str:
        """Analyze the most recent buffered frame with a custom prompt (tool path).

        Returns the result directly without firing the on_description callback.
        Resets the fallback timer.
        """
        with self._frame_lock:
            if not self._frame_buffer:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                else:
                    return "[No camera frame available]"
            else:
                _, frame = self._frame_buffer[-1]
                frame = frame.copy()

        self.reset_timer()
        # Call analyze_frame but suppress the on_description callback
        saved_callback = self.on_description
        self.on_description = None
        try:
            result = self.analyze_frame(frame, prompt)
        finally:
            self.on_description = saved_callback
        return result

    def analyze_frame(self, frame: np.ndarray, prompt: str = "What do you see?") -> str:
        """Analyze a single frame and return the description."""
        self.analyzing = True
        try:
            # Resize for efficiency (max 720p)
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))

            # Encode as JPEG
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf).decode("utf-8")

            body = {
                "schemaVersion": "messages-v1",
                "system": [{"text": self.system_prompt}],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": {"format": "jpeg", "source": {"bytes": b64}}},
                            {"text": prompt},
                        ],
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 256,
                    "topP": 0.9,
                    "temperature": 0.7,
                },
            }

            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            description = result["output"]["message"]["content"][0]["text"]

            self.last_description = description
            self.last_analyze_time = time.time()

            if self.on_description:
                self.on_description(description)

            return description

        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return f"[Vision error: {e}]"
        finally:
            self.analyzing = False

    def _run_loop(self, stop_event: threading.Event) -> None:
        logger.info("Nova Vision loop started (fallback interval={self.analyze_interval}s)")
        while not stop_event.is_set():
            # Calculate remaining time until fallback fires
            elapsed = time.time() - self.last_analyze_time
            remaining = max(0.1, self.analyze_interval - elapsed)

            # Wait for forced trigger or fallback timeout
            triggered = self._force_analyze.wait(timeout=remaining)
            if stop_event.is_set():
                break
            self._force_analyze.clear()

            with self._frame_lock:
                frame = self._latest_frame
                if frame is None:
                    continue
                frame = frame.copy()

            prompt = "Describe what is visible. Be brief and specific."
            self.analyze_frame(frame, prompt)
            self.reset_timer()

    def start(self, stop_event: threading.Event) -> None:
        """Start periodic vision analysis in a background thread."""
        self._thread = threading.Thread(
            target=self._run_loop, args=(stop_event,), name="nova-vision", daemon=True
        )
        self._thread.start()
        logger.info("Nova Vision thread started")
