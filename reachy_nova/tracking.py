"""Active vision tracking for Reachy Nova.

Fuses multiple signals (DoA speaker direction, YOLO person detection,
snap/clap transient detection, pat detection) into head target angles.
Falls back to idle sinusoidal animation when no active tracking target exists.

Priority order: snap > face > speaker > idle
"""

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# Head motion limits (degrees)
MAX_YAW = 45.0
MAX_PITCH = 25.0      # max upward tilt
MIN_PITCH = -15.0     # max downward tilt (limited to avoid head-body collision)

# Idle animation parameters (match original main.py behavior)
IDLE_YAW_AMP = 25.0
IDLE_YAW_SPEED = 0.15
IDLE_PITCH_AMP = 3.0
IDLE_PITCH_SPEED = 0.08


class PatDetector:
    """Detects patting gestures on the robot head.

    Compares the commanded head pose with the actual head pose read back
    from the servos. When someone pats the head, the actual pose deviates
    from the commanded pose. Repeated impulses within a short time window
    are classified as a pat.

    Tracks both pitch (forward/down push = "scratch") and yaw (side-to-side
    nudge = "side_pat") to differentiate touch types.
    """

    def __init__(self):
        # Rolling deviation history (timestamp, deviation_degrees)
        self.deviation_history: deque[tuple[float, float]] = deque(maxlen=150)

        # Detected press impulses: (timestamp, axis) where axis is "pitch" or "yaw"
        self.press_times: deque[tuple[float, str]] = deque(maxlen=20)
        self.last_pat_time: float = 0.0

        # --- Tunable parameters (pitch) ---
        self.press_threshold: float = 0.8    # degrees: deviation to count as a "press"
        self.release_threshold: float = 0.3  # degrees: deviation below this = released
        self.min_presses: int = 2            # minimum presses for a pat
        self.pat_window: float = 3.0         # seconds: window to accumulate presses
        self.pat_cooldown: float = 2.0       # seconds: between pat events

        # --- Tunable parameters (yaw) ---
        self.yaw_press_threshold: float = 0.8   # degrees: yaw deviation to count as press
        self.yaw_release_threshold: float = 0.3  # degrees: yaw release threshold

        # Internal state (pitch)
        self._in_press: bool = False
        self._baseline_offset: float = 0.0   # rolling baseline for steady-state offset
        self._baseline_alpha: float = 0.003   # slow EMA to track servo bias

        # Internal state (yaw)
        self.yaw_deviation_history: deque[tuple[float, float]] = deque(maxlen=150)
        self._yaw_in_press: bool = False
        self._yaw_baseline_offset: float = 0.0

        # Current touch type carried from level1 to level2
        self._current_touch_type: str = "scratch"

        # --- Two-level state machine ---
        self._state: str = "idle"                   # "idle" | "level1" | "level2_cooldown"
        self._level1_time: float = 0.0              # when level 1 fired
        self._level2_threshold: float = 0.0         # random 4-8s for level 2
        self._last_press_time: float = 0.0          # most recent press (for gap detection)
        self._interaction_gap_timeout: float = 5.0  # no presses for 5s -> reset
        self._level2_cooldown: float = 5.0          # cooldown after level 2

    def _classify_touch(self) -> str:
        """Classify touch type based on recent press axis distribution.

        Returns:
            "scratch" if pitch-dominated, "side_pat" if yaw-dominated.
        """
        now = time.time()
        cutoff = now - self.pat_window
        pitch_count = sum(1 for t, axis in self.press_times if t > cutoff and axis == "pitch")
        yaw_count = sum(1 for t, axis in self.press_times if t > cutoff and axis == "yaw")
        return "side_pat" if yaw_count > pitch_count else "scratch"

    def update(
        self,
        commanded_pitch: float,
        actual_pitch: float,
        commanded_yaw: float = 0.0,
        actual_yaw: float = 0.0,
    ) -> tuple[str, str] | None:
        """Feed one sample of commanded vs actual pitch and yaw.

        Args:
            commanded_pitch: the pitch we commanded (degrees, positive=up)
            actual_pitch: the pitch read back from the servo (degrees)
            commanded_yaw: the yaw we commanded (degrees)
            actual_yaw: the yaw read back from the servo (degrees)

        Returns:
            (level, touch_type) tuple on detection — level is "level1" or "level2",
            touch_type is "scratch" or "side_pat". Returns None if no event.
        """
        now = time.time()

        # --- Pitch deviation processing ---
        raw_deviation = actual_pitch - commanded_pitch
        self._baseline_offset += self._baseline_alpha * (raw_deviation - self._baseline_offset)
        deviation = raw_deviation - self._baseline_offset
        self.deviation_history.append((now, deviation))

        # Detect pitch press: head pushed downward (negative deviation)
        if deviation < -self.press_threshold and not self._in_press:
            self._in_press = True
            self.press_times.append((now, "pitch"))
            self._last_press_time = now
            logger.debug(f"Pat pitch press: deviation={deviation:.2f}deg")
        elif deviation > -self.release_threshold:
            self._in_press = False

        # --- Yaw deviation processing ---
        raw_yaw_dev = actual_yaw - commanded_yaw
        self._yaw_baseline_offset += self._baseline_alpha * (raw_yaw_dev - self._yaw_baseline_offset)
        yaw_dev = raw_yaw_dev - self._yaw_baseline_offset
        self.yaw_deviation_history.append((now, yaw_dev))

        # Detect yaw press: head nudged sideways (absolute deviation)
        if abs(yaw_dev) > self.yaw_press_threshold and not self._yaw_in_press:
            self._yaw_in_press = True
            self.press_times.append((now, "yaw"))
            self._last_press_time = now
            logger.debug(f"Pat yaw press: deviation={yaw_dev:.2f}deg")
        elif abs(yaw_dev) < self.yaw_release_threshold:
            self._yaw_in_press = False

        # --- State machine ---
        if self._state == "idle":
            cutoff = now - self.pat_window
            recent_presses = sum(1 for t, _ in self.press_times if t > cutoff)

            if recent_presses >= self.min_presses and now - self.last_pat_time > self.pat_cooldown:
                touch_type = self._classify_touch()
                self._current_touch_type = touch_type
                self.last_pat_time = now
                self.press_times.clear()
                self._state = "level1"
                self._level1_time = now
                self._level2_threshold = random.uniform(4.0, 8.0)
                logger.info(f"Pat level 1! type={touch_type} ({recent_presses} presses, level2 threshold={self._level2_threshold:.1f}s)")
                return ("level1", touch_type)

        elif self._state == "level1":
            if self._last_press_time > 0 and now - self._last_press_time > self._interaction_gap_timeout:
                logger.info("Pat interaction gap — resetting to idle")
                self._state = "idle"
                return None

            elapsed = now - self._level1_time
            if elapsed > self._level2_threshold:
                touch_type = self._current_touch_type
                self.last_pat_time = now
                self.press_times.clear()
                self._state = "level2_cooldown"
                logger.info(f"Pat level 2! type={touch_type} (sustained {elapsed:.1f}s)")
                return ("level2", touch_type)

        elif self._state == "level2_cooldown":
            if now - self.last_pat_time > self._level2_cooldown:
                logger.info("Pat cooldown expired — ready for new detection")
                self._state = "idle"
                self.press_times.clear()

        return None

    def reset(self):
        """Reset detector state."""
        self.deviation_history.clear()
        self.yaw_deviation_history.clear()
        self.press_times.clear()
        self._in_press = False
        self._yaw_in_press = False
        self._baseline_offset = 0.0
        self._yaw_baseline_offset = 0.0
        self._current_touch_type = "scratch"
        self._state = "idle"
        self._level1_time = 0.0
        self._level2_threshold = 0.0
        self._last_press_time = 0.0


class TrackingManager:
    """Fuses DoA, vision, and snap signals into head target angles."""

    def __init__(self, on_event: Callable[[str, dict], None] | None = None):
        self.on_event = on_event
        self._prev_mode = "idle"  # for mode_changed events

        # --- YOLO face/person detection (lazy-loaded, runs in bg thread) ---
        self.model = None
        self.detect_interval = 0.2  # run detection every 200ms
        self.last_detect_time = 0.0
        self.person_bbox = None  # (x1, y1, x2, y2) normalized 0-1
        self.person_lost_time = 0.0
        self.person_hold_duration = 2.0  # keep tracking 2s after lost
        self.frame_shape = None  # (h, w) of last processed frame

        # Background YOLO thread
        self._pending_frame = None
        self._vision_lock = threading.Lock()
        self._vision_thread = None
        self._vision_busy = False

        # Face tracking gains (integrating controller)
        self.face_kp = 30.0  # proportional gain (degrees per unit error)
        self.face_yaw_accum = 0.0
        self.face_pitch_accum = 0.0

        # --- DoA speaker tracking ---
        self.last_doa_angle = 0.0  # radians from XMOS
        self.doa_speech_active = False
        self.speaker_lost_time = 0.0
        self.speaker_hold_duration = 3.0  # hold speaker direction 3s
        self.doa_yaw_target = 0.0  # computed yaw in degrees

        # --- Snap detection ---
        self.energy_history = deque(maxlen=30)
        self.snap_time = 0.0
        self.snap_target_yaw = 0.0
        self.snap_duration = 1.5  # hold snap look for 1.5s
        self.prev_chunk_low = True  # previous chunk was below threshold
        self.snap_min_rms = 0.02   # absolute floor — below this is ambient noise, not a snap

        # --- Smoothing ---
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.smooth_factor = 0.15  # EMA alpha for smooth transitions

        # --- Pat detection ---
        self.pat_detector = PatDetector()
        self.pat_level1_time: float = 0.0     # timestamp when level 1 pat fired

        # --- YuNet face bbox (fed from FaceRecognition callback) ---
        self.yunet_face_bbox: tuple | None = None   # (x1,y1,x2,y2) normalized
        self.yunet_face_time: float = 0.0
        self.yunet_face_hold: float = 1.5           # seconds to hold after last detection

        # --- Face recognition state ---
        self.recognized_person: str | None = None  # name of recognized person
        self.face_recognized_time: float = 0.0

        # --- Focus mode state ---
        self.focus_active: bool = False
        self.focus_target: str = "face"           # "face" or recognized person name
        self.focus_last_seen: float = 0.0         # last time target was visible
        self.focus_lost_threshold: float = 3.0    # seconds before entering search
        self.focus_searching: bool = False
        self.focus_search_start: float = 0.0
        self.focus_search_max: float = 60.0       # auto-abandon after this many seconds
        self.focus_scan_yaw: float = 0.0          # current scan yaw
        self.focus_scan_dir: float = 1.0          # +1 right, -1 left
        self.focus_prompted_llm: bool = False     # gate for repeated LLM prompts
        self.on_focus_event: Callable[[str, dict], None] | None = None

        # --- Current mode ---
        self.mode = "idle"

    def _load_model(self):
        """Lazy-load YOLO model on first use."""
        if self.model is not None:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            logger.info("YOLO model loaded (yolov8n.pt)")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = False  # sentinel to avoid retrying

    def _fire_event(self, event_type: str, data: dict | None = None) -> None:
        """Fire an event to the registered callback."""
        if self.on_event:
            try:
                self.on_event(event_type, data or {})
            except Exception as e:
                logger.warning(f"Event callback error ({event_type}): {e}")

    def update_doa(self, doa_result):
        """Process DoA from XMOS mic array.

        Args:
            doa_result: (angle_radians, speech_detected_bool) or None
        """
        if doa_result is None:
            return

        angle_rad, speech_detected = doa_result

        if speech_detected:
            self.last_doa_angle = angle_rad
            self.doa_speech_active = True
            self.speaker_lost_time = 0.0

            # DoA mapping: 0=left, pi/2=front, pi=right
            # Convert to yaw: front=0, left=positive, right=negative
            yaw_rad = (np.pi / 2) - angle_rad
            self.doa_yaw_target = np.degrees(yaw_rad)
            self.doa_yaw_target = np.clip(self.doa_yaw_target, -MAX_YAW, MAX_YAW)
        else:
            if self.doa_speech_active:
                # Speech just stopped, start hold timer
                if self.speaker_lost_time == 0.0:
                    self.speaker_lost_time = time.time()
                # Check if hold expired
                if time.time() - self.speaker_lost_time > self.speaker_hold_duration:
                    self.doa_speech_active = False

    def _run_detection(self, frame):
        """Run YOLO detection in background thread. Non-blocking."""
        try:
            self._load_model()
            if not self.model:
                return

            h, w = frame.shape[:2]
            results = self.model(frame, classes=[0], verbose=False, conf=0.4)

            with self._vision_lock:
                self.frame_shape = (h, w)
                had_person = self.person_bbox is not None

                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
                    best_idx = areas.argmax().item()
                    box = boxes.xyxy[best_idx].cpu().numpy()

                    self.person_bbox = (box[0] / w, box[1] / h, box[2] / w, box[3] / h)
                    self.person_lost_time = 0.0

                    # Skip YOLO body-center accumulator update if a fresh YuNet face bbox
                    # was seen recently — face update already happened in update_face_bbox()
                    yunet_fresh = (
                        self.yunet_face_bbox is not None
                        and time.time() - self.yunet_face_time < self.yunet_face_hold
                    )
                    if not yunet_fresh:
                        cx = (self.person_bbox[0] + self.person_bbox[2]) / 2.0
                        cy = (self.person_bbox[1] + self.person_bbox[3]) / 2.0
                        error_x = cx - 0.5
                        error_y = cy - 0.5

                        self.face_yaw_accum -= self.face_kp * error_x
                        self.face_pitch_accum += self.face_kp * error_y * 0.5

                        self.face_yaw_accum = np.clip(self.face_yaw_accum, -MAX_YAW, MAX_YAW)
                        self.face_pitch_accum = np.clip(self.face_pitch_accum, MIN_PITCH, MAX_PITCH)

                    # Fire person_detected when transitioning from no person to person
                    if not had_person:
                        self._fire_event("person_detected", {
                            "bbox": tuple(float(v) for v in self.person_bbox),
                        })
                else:
                    if self.person_bbox is not None and self.person_lost_time == 0.0:
                        self.person_lost_time = time.time()

                    if self.person_lost_time > 0 and time.time() - self.person_lost_time > self.person_hold_duration:
                        self.person_bbox = None
                        self.face_yaw_accum *= 0.95
                        self.face_pitch_accum *= 0.95
                        # Fire person_lost when person disappears after hold
                        if had_person:
                            self._fire_event("person_lost", {})
        except Exception as e:
            logger.warning(f"Vision tracking error: {e}")
        finally:
            self._vision_busy = False

    def update_vision(self, frame, t):
        """Submit frame for YOLO detection (non-blocking, runs in background thread).

        Args:
            frame: BGR numpy array from camera
            t: current time (seconds since start)
        """
        if t - self.last_detect_time < self.detect_interval:
            return

        # Skip if previous detection is still running
        if self._vision_busy:
            return

        self.last_detect_time = t
        self._vision_busy = True

        # Copy frame to avoid race conditions with camera buffer
        frame_copy = frame.copy()
        self._vision_thread = threading.Thread(
            target=self._run_detection, args=(frame_copy,), daemon=True
        )
        self._vision_thread.start()

    def detect_snap(self, audio_chunk):
        """Detect sharp audio transient (snap, clap, etc.).

        Args:
            audio_chunk: float32 numpy array of audio samples
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return

        rms = np.sqrt(np.mean(audio_chunk ** 2))
        self.energy_history.append(rms)

        if len(self.energy_history) < 5:
            return

        rolling_avg = np.mean(list(self.energy_history)[:-1])

        if rolling_avg < 1e-6:
            self.prev_chunk_low = True
            return

        # Snap = sudden spike: current > 5x average AND previous was quiet
        # Also require absolute minimum RMS to filter out ambient noise fluctuations
        is_spike = rms > 5.0 * rolling_avg and rms > self.snap_min_rms
        was_quiet = self.prev_chunk_low

        if is_spike and was_quiet:
            self.snap_time = time.time()
            # Use DoA if available to determine snap direction
            if self.last_doa_angle > 0:
                yaw_rad = (np.pi / 2) - self.last_doa_angle
                self.snap_target_yaw = np.clip(np.degrees(yaw_rad), -MAX_YAW, MAX_YAW)
            logger.info(f"Snap detected! RMS={rms:.4f}, avg={rolling_avg:.4f}, target_yaw={self.snap_target_yaw:.1f}")
            self._fire_event("snap_detected", {"rms": float(rms), "target_yaw": float(self.snap_target_yaw)})

        self.prev_chunk_low = rms < 2.0 * rolling_avg

    def detect_pat(self, commanded_pose: np.ndarray, actual_pose: np.ndarray):
        """Detect patting gesture by comparing commanded vs actual head pose.

        Args:
            commanded_pose: 4x4 homogeneous matrix sent to set_target()
            actual_pose: 4x4 homogeneous matrix read from get_current_head_pose()
        """
        try:
            # Extract pitch (index 1) and yaw (index 2) from both poses
            cmd_euler = Rotation.from_matrix(commanded_pose[:3, :3]).as_euler("xyz", degrees=True)
            act_euler = Rotation.from_matrix(actual_pose[:3, :3]).as_euler("xyz", degrees=True)
            commanded_pitch = cmd_euler[1]
            actual_pitch = act_euler[1]
            commanded_yaw = cmd_euler[2]
            actual_yaw = act_euler[2]

            result = self.pat_detector.update(
                commanded_pitch, actual_pitch,
                commanded_yaw, actual_yaw,
            )
            event_data = {
                "commanded_pitch": float(commanded_pitch),
                "actual_pitch": float(actual_pitch),
                "deviation": float(actual_pitch - commanded_pitch),
                "commanded_yaw": float(commanded_yaw),
                "actual_yaw": float(actual_yaw),
                "yaw_deviation": float(actual_yaw - commanded_yaw),
            }
            if result is not None:
                level, touch_type = result
                event_data["touch_type"] = touch_type
                if level == "level1":
                    self.pat_level1_time = time.time()
                    self._fire_event("pat_level1", event_data)
                elif level == "level2":
                    self._fire_event("pat_level2", event_data)
        except Exception as e:
            logger.debug(f"Pat detection error: {e}")

    def get_head_target(self, t, voice_state="idle", mood="happy"):
        """Compute head target angles based on priority: snap > face > speaker > idle.

        Args:
            t: time in seconds since app start
            voice_state: current voice state (idle, listening, speaking, thinking)
            mood: current mood string

        Returns:
            (yaw_degrees, pitch_degrees) tuple
        """
        now = time.time()
        target_yaw = 0.0
        target_pitch = 0.0
        prev_mode = self.mode

        # --- Focus mode: persistent target-locked tracking ---
        if self.focus_active:
            with self._vision_lock:
                person_visible = self.person_bbox is not None
                yunet_visible = (
                    self.yunet_face_bbox is not None
                    and now - self.yunet_face_time < self.yunet_face_hold
                )
                target_visible = person_visible or yunet_visible

            if target_visible:
                # Target in view — update last_seen and fall through to normal tracking below
                self.focus_last_seen = now
                self.focus_searching = False
                self.focus_prompted_llm = False
                # (normal priority cascade handles visual targeting)
            else:
                lost_elapsed = now - self.focus_last_seen if self.focus_last_seen > 0 else 0.0
                if lost_elapsed >= self.focus_lost_threshold:
                    # Enter / continue search mode
                    if not self.focus_searching:
                        self.focus_searching = True
                        self.focus_search_start = now
                        logger.info("[Focus] Target lost — starting search sweep")

                    search_elapsed = now - self.focus_search_start

                    # Prompt LLM once per search session
                    if not self.focus_prompted_llm and self.on_focus_event:
                        self.focus_prompted_llm = True
                        try:
                            self.on_focus_event("lost", {"elapsed": lost_elapsed})
                        except Exception as e:
                            logger.warning(f"on_focus_event callback error: {e}")

                    # Auto-abandon
                    if search_elapsed >= self.focus_search_max:
                        logger.info("[Focus] Search timed out — auto-stopping focus")
                        self.focus_active = False
                        self.focus_searching = False
                        if self.on_focus_event:
                            try:
                                self.on_focus_event("abandoned", {"elapsed": search_elapsed})
                            except Exception as e:
                                logger.warning(f"on_focus_event callback error: {e}")
                    else:
                        # Slow sinusoidal scan
                        self.mode = "face"
                        scan_yaw = MAX_YAW * np.sin(2.0 * np.pi * search_elapsed / 20.0)
                        target_yaw = scan_yaw
                        target_pitch = self.face_pitch_accum  # hold last known pitch
                        self.current_yaw += 0.05 * (target_yaw - self.current_yaw)
                        self.current_pitch += 0.05 * (target_pitch - self.current_pitch)
                        self.current_pitch = float(np.clip(self.current_pitch, MIN_PITCH, MAX_PITCH))
                        if self.mode != prev_mode:
                            self._fire_event("mode_changed", {"from": prev_mode, "to": self.mode})
                        return self.current_yaw, self.current_pitch

        # Priority 1: Snap look
        if now - self.snap_time < self.snap_duration:
            self.mode = "snap"
            target_yaw = self.snap_target_yaw
            target_pitch = 0.0
            # Fast approach for snaps
            self.current_yaw += 0.5 * (target_yaw - self.current_yaw)
            self.current_pitch += 0.5 * (target_pitch - self.current_pitch)

        # Priority 2: Face/person tracking (YOLO body or fresh YuNet face)
        elif self._has_person_or_face(now):
            self.mode = "face"
            with self._vision_lock:
                target_yaw = self.face_yaw_accum
                target_pitch = self.face_pitch_accum
            self.current_yaw += self.smooth_factor * (target_yaw - self.current_yaw)
            self.current_pitch += self.smooth_factor * (target_pitch - self.current_pitch)

        # Priority 3: Speaker (DoA)
        elif self.doa_speech_active:
            self.mode = "speaker"
            target_yaw = self.doa_yaw_target
            target_pitch = -5.0  # slight downward look at speaker
            alpha = 0.2  # moderate speed for speaker tracking
            self.current_yaw += alpha * (target_yaw - self.current_yaw)
            self.current_pitch += alpha * (target_pitch - self.current_pitch)

        # Priority 4: Idle animation (sinusoidal, state-dependent)
        else:
            self.mode = "idle"

            if voice_state == "listening":
                target_yaw = 5.0 * np.sin(2.0 * np.pi * 0.1 * t)
                target_pitch = -5.0
            elif voice_state == "speaking":
                target_yaw = 15.0 * np.sin(2.0 * np.pi * 0.3 * t)
                target_pitch = 5.0 * np.sin(2.0 * np.pi * 0.4 * t)
            elif voice_state == "thinking":
                target_yaw = 20.0 * np.sin(2.0 * np.pi * 0.05 * t)
                target_pitch = -10.0
            else:
                target_yaw = IDLE_YAW_AMP * np.sin(2.0 * np.pi * IDLE_YAW_SPEED * t)
                target_pitch = IDLE_PITCH_AMP * np.sin(2.0 * np.pi * IDLE_PITCH_SPEED * t)

            # Smooth transition from tracking back to idle
            self.current_yaw += 0.1 * (target_yaw - self.current_yaw)
            self.current_pitch += 0.1 * (target_pitch - self.current_pitch)

            # Decay accumulated face tracking angles when idle
            with self._vision_lock:
                self.face_yaw_accum *= 0.98
                self.face_pitch_accum *= 0.98

        # Final safety clamp on pitch to prevent head-body collision
        self.current_pitch = float(np.clip(self.current_pitch, MIN_PITCH, MAX_PITCH))

        # Fire mode_changed event on any transition
        if self.mode != prev_mode:
            self._fire_event("mode_changed", {"from": prev_mode, "to": self.mode})

        return self.current_yaw, self.current_pitch

    def update_face_bbox(self, bbox_norm: tuple | None):
        """Update YuNet face bbox from FaceRecognition callback (thread-safe).

        Args:
            bbox_norm: normalized (x1, y1, x2, y2) or None if no face detected
        """
        with self._vision_lock:
            self.yunet_face_bbox = bbox_norm
            if bbox_norm is not None:
                self.yunet_face_time = time.time()
                cx = (bbox_norm[0] + bbox_norm[2]) / 2.0
                cy = (bbox_norm[1] + bbox_norm[3]) / 2.0
                error_x = cx - 0.5
                error_y = cy - 0.5
                self.face_yaw_accum -= self.face_kp * error_x
                self.face_pitch_accum += self.face_kp * error_y * 0.5
                self.face_yaw_accum = np.clip(self.face_yaw_accum, -MAX_YAW, MAX_YAW)
                self.face_pitch_accum = np.clip(self.face_pitch_accum, MIN_PITCH, MAX_PITCH)

    def update_face_recognition(self, match_data: tuple[str, str, float] | None):
        """Update face recognition state from FaceRecognition engine.

        Args:
            match_data: (unique_id, name, score) or None if no match
        """
        if match_data:
            _, name, _ = match_data
            self.recognized_person = name
            self.face_recognized_time = time.time()
            self._fire_event("face_recognized", {
                "id": match_data[0],
                "name": name,
                "score": match_data[2],
            })
        else:
            self.recognized_person = None

    def start_focus(self, target: str = "face"):
        """Enable persistent focus tracking on a face or person."""
        self.focus_active = True
        self.focus_target = target
        self.focus_last_seen = time.time()
        self.focus_searching = False
        self.focus_prompted_llm = False
        logger.info(f"[Focus] Started focus on target='{target}'")

    def stop_focus(self):
        """Disable focus mode, returning to normal reactive tracking."""
        self.focus_active = False
        self.focus_searching = False
        self.focus_prompted_llm = False
        logger.info("[Focus] Focus stopped")

    def continue_focus_search(self):
        """Reset search timer to extend search by up to focus_search_max more seconds."""
        if self.focus_active:
            self.focus_search_start = time.time()
            self.focus_prompted_llm = False
            logger.info("[Focus] Search timer reset — continuing search")

    def _has_person(self) -> bool:
        """Check if a person is currently being tracked (thread-safe)."""
        with self._vision_lock:
            return self.person_bbox is not None

    def _has_person_or_face(self, now: float | None = None) -> bool:
        """Check if a person or fresh YuNet face is currently tracked (thread-safe)."""
        if now is None:
            now = time.time()
        with self._vision_lock:
            if self.person_bbox is not None:
                return True
            return (
                self.yunet_face_bbox is not None
                and now - self.yunet_face_time < self.yunet_face_hold
            )
