"""Active vision tracking for Reachy Nova.

Fuses multiple signals (DoA speaker direction, YOLO person detection,
snap/clap transient detection, pat detection) into head target angles.
Falls back to idle sinusoidal animation when no active tracking target exists.

Priority order: snap > face > speaker > idle
"""

import logging
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

    Compares the commanded head pitch with the actual head pitch read back
    from the servos. When someone pats the head from above, the actual pitch
    deviates downward from the commanded pitch. Repeated downward impulses
    within a short time window are classified as a pat.

    This distinguishes pats from accidental bumps (single event) and normal
    servo noise (below threshold).
    """

    def __init__(self):
        # Rolling deviation history (timestamp, deviation_degrees)
        self.deviation_history: deque[tuple[float, float]] = deque(maxlen=150)

        # Timestamps of detected press impulses
        self.press_times: deque[float] = deque(maxlen=20)
        self.last_pat_time: float = 0.0

        # --- Tunable parameters ---
        self.press_threshold: float = 1.5    # degrees: deviation to count as a "press"
        self.release_threshold: float = 0.6  # degrees: deviation below this = released
        self.min_presses: int = 2            # minimum presses for a pat
        self.pat_window: float = 2.5         # seconds: window to accumulate presses
        self.pat_cooldown: float = 3.0       # seconds: between pat events

        # Internal state
        self._in_press: bool = False
        self._baseline_offset: float = 0.0   # rolling baseline for steady-state offset
        self._baseline_alpha: float = 0.005   # slow EMA to track servo bias

    def update(self, commanded_pitch: float, actual_pitch: float) -> bool:
        """Feed one sample of commanded vs actual pitch.

        Args:
            commanded_pitch: the pitch we commanded (degrees, positive=up)
            actual_pitch: the pitch read back from the servo (degrees)

        Returns:
            True if a pat gesture was just detected.
        """
        now = time.time()

        # Raw deviation (negative = head pushed downward relative to command)
        raw_deviation = actual_pitch - commanded_pitch

        # Update slow baseline to compensate for constant servo offset / gravity
        self._baseline_offset += self._baseline_alpha * (raw_deviation - self._baseline_offset)

        # Corrected deviation: remove baseline drift
        deviation = raw_deviation - self._baseline_offset
        self.deviation_history.append((now, deviation))

        # Detect press events: deviation dips below threshold (head pushed down)
        if deviation < -self.press_threshold and not self._in_press:
            self._in_press = True
            self.press_times.append(now)
            logger.debug(f"Pat press detected: deviation={deviation:.2f}deg")
        elif deviation > -self.release_threshold:
            self._in_press = False

        # Count recent presses within the time window
        cutoff = now - self.pat_window
        recent_presses = sum(1 for t in self.press_times if t > cutoff)

        # Pat detected?
        if recent_presses >= self.min_presses and now - self.last_pat_time > self.pat_cooldown:
            self.last_pat_time = now
            self.press_times.clear()
            logger.info(f"Pat detected! ({recent_presses} presses in {self.pat_window}s)")
            return True

        return False

    def reset(self):
        """Reset detector state."""
        self.deviation_history.clear()
        self.press_times.clear()
        self._in_press = False
        self._baseline_offset = 0.0


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
        self.pat_reaction_time: float = 0.0   # timestamp when last pat was detected
        self.pat_reaction_duration: float = 1.5  # seconds of head nuzzle
        self.pat_nuzzle_freq: float = 2.5     # Hz — gentle side-to-side speed
        self.pat_nuzzle_amp: float = 8.0      # degrees — nuzzle amplitude

        # --- Face recognition state ---
        self.recognized_person: str | None = None  # name of recognized person
        self.face_recognized_time: float = 0.0

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
            # Extract pitch from both poses (euler "xyz" convention, index 1 = pitch)
            cmd_euler = Rotation.from_matrix(commanded_pose[:3, :3]).as_euler("xyz", degrees=True)
            act_euler = Rotation.from_matrix(actual_pose[:3, :3]).as_euler("xyz", degrees=True)
            commanded_pitch = cmd_euler[1]
            actual_pitch = act_euler[1]

            if self.pat_detector.update(commanded_pitch, actual_pitch):
                self.pat_reaction_time = time.time()
                self._fire_event("pat_detected", {
                    "commanded_pitch": float(commanded_pitch),
                    "actual_pitch": float(actual_pitch),
                    "deviation": float(actual_pitch - commanded_pitch),
                })
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

        # Priority 1: Snap look
        if now - self.snap_time < self.snap_duration:
            self.mode = "snap"
            target_yaw = self.snap_target_yaw
            target_pitch = 0.0
            # Fast approach for snaps
            self.current_yaw += 0.5 * (target_yaw - self.current_yaw)
            self.current_pitch += 0.5 * (target_pitch - self.current_pitch)

        # Priority 2: Face/person tracking (read under lock)
        elif self._has_person():
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

        # Pat nuzzle overlay: gentle decaying side-to-side after a pat
        pat_elapsed = now - self.pat_reaction_time
        if 0 < pat_elapsed < self.pat_reaction_duration:
            # Decaying envelope: starts full, fades to zero
            envelope = 1.0 - (pat_elapsed / self.pat_reaction_duration)
            nuzzle = self.pat_nuzzle_amp * envelope * np.sin(
                2.0 * np.pi * self.pat_nuzzle_freq * pat_elapsed
            )
            self.current_yaw += nuzzle

        # Final safety clamp on pitch to prevent head-body collision
        self.current_pitch = float(np.clip(self.current_pitch, MIN_PITCH, MAX_PITCH))

        # Fire mode_changed event on any transition
        if self.mode != prev_mode:
            self._fire_event("mode_changed", {"from": prev_mode, "to": self.mode})

        return self.current_yaw, self.current_pitch

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

    def _has_person(self) -> bool:
        """Check if a person is currently being tracked (thread-safe)."""
        with self._vision_lock:
            return self.person_bbox is not None
