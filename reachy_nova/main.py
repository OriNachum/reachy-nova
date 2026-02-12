"""Reachy Nova - Voice, Vision & Browser AI for Reachy Mini.

Integrates Amazon Nova Sonic (voice), Nova Pro (vision), and Nova Act (browser)
to create an interactive AI-powered robot experience.
"""

import logging
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import numpy as np
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

from .nova_sonic import NovaSonic, OUTPUT_SAMPLE_RATE
from .nova_vision import NovaVision
from .nova_browser import NovaBrowser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Robot animation parameters
IDLE_YAW_SPEED = 0.15      # Slow idle head sweep
LISTEN_YAW_SPEED = 0.0     # Stay still when listening
SPEAK_YAW_SPEED = 0.3      # Animated when speaking
ANTENNA_SPEED = 0.5         # Antenna wiggle speed
EXCITED_ANTENNA_AMP = 35.0  # Bigger wiggles when excited
CALM_ANTENNA_AMP = 15.0     # Gentle wiggles when calm


class ReachyNova(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        t0 = time.time()

        # --- Shared state ---
        state_lock = threading.Lock()
        app_state = {
            "voice_state": "idle",
            "vision_enabled": True,
            "vision_description": "",
            "vision_analyzing": False,
            "browser_state": "idle",
            "browser_task": "",
            "browser_result": "",
            "browser_screenshot": "",
            "last_user_text": "",
            "last_assistant_text": "",
            "antenna_mode": "auto",  # auto, excited, calm, off
            "mood": "happy",  # happy, curious, excited, thinking
        }

        def update_state(**kwargs):
            with state_lock:
                app_state.update(kwargs)

        # --- Audio output buffer ---
        audio_output_buffer = []
        audio_lock = threading.Lock()
        audio_playing = False

        def handle_audio_output(samples: np.ndarray):
            nonlocal audio_playing
            with audio_lock:
                audio_output_buffer.append(samples)
                if not audio_playing:
                    audio_playing = True
                    try:
                        reachy_mini.media.start_playing()
                    except Exception:
                        pass

        # --- Nova Sonic (Voice) ---
        def on_transcript(role: str, text: str):
            if role == "USER":
                update_state(last_user_text=text, mood="thinking")
                logger.info(f"[User] {text}")
                # Check for browser commands
                lower = text.lower()
                if any(kw in lower for kw in ["search for", "google", "look up", "browse", "open website", "go to"]):
                    browser.queue_task(text, url="https://www.google.com")
                    update_state(browser_task=text)
                # Check for vision commands
                if any(kw in lower for kw in ["what do you see", "look around", "describe", "what's around"]):
                    vision.trigger_analyze()
            else:
                update_state(last_assistant_text=text, mood="happy")
                logger.info(f"[Nova] {text}")

        def on_voice_state(state: str):
            update_state(voice_state=state)
            if state == "speaking":
                update_state(mood="excited")
            elif state == "listening":
                update_state(mood="curious")

        sonic = NovaSonic(
            on_transcript=on_transcript,
            on_audio_output=handle_audio_output,
            on_state_change=on_voice_state,
        )

        # --- Nova Pro (Vision) ---
        def on_vision_description(desc: str):
            update_state(vision_description=desc, vision_analyzing=False, mood="excited")
            logger.info(f"[Vision] {desc}")
            # Feed vision description to voice conversation
            sonic.inject_text(
                f"[Your camera sees: {desc}] React to what you see briefly."
            )

        vision = NovaVision(
            on_description=on_vision_description,
            analyze_interval=10.0,
        )

        # --- Nova Act (Browser) ---
        def on_browser_result(result: str):
            update_state(browser_result=result, browser_task="")
            logger.info(f"[Browser] {result}")
            sonic.inject_text(
                f"[Browser task result: {result}] Tell the user what happened."
            )

        def on_browser_screenshot(b64: str):
            update_state(browser_screenshot=b64)

        def on_browser_state(state: str):
            update_state(browser_state=state)

        browser = NovaBrowser(
            on_result=on_browser_result,
            on_screenshot=on_browser_screenshot,
            on_state_change=on_browser_state,
            headless=True,
        )

        # --- API Endpoints ---
        class VisionToggle(BaseModel):
            enabled: bool

        class BrowserTask(BaseModel):
            instruction: str
            url: str | None = None

        class AntennaMode(BaseModel):
            mode: str

        @self.settings_app.get("/api/state")
        def get_state():
            with state_lock:
                return {**app_state, "uptime": time.time() - t0}

        @self.settings_app.post("/api/vision/toggle")
        def toggle_vision(body: VisionToggle):
            update_state(vision_enabled=body.enabled)
            return {"vision_enabled": body.enabled}

        @self.settings_app.post("/api/vision/analyze")
        def trigger_vision():
            vision.trigger_analyze()
            update_state(vision_analyzing=True)
            return {"status": "analyzing"}

        @self.settings_app.post("/api/browser/task")
        def submit_browser_task(body: BrowserTask):
            browser.queue_task(body.instruction, body.url)
            update_state(browser_task=body.instruction)
            return {"status": "queued", "instruction": body.instruction}

        @self.settings_app.post("/api/antenna/mode")
        def set_antenna_mode(body: AntennaMode):
            update_state(antenna_mode=body.mode)
            return {"antenna_mode": body.mode}

        @self.settings_app.post("/api/mood")
        def set_mood(body: dict):
            mood = body.get("mood", "happy")
            update_state(mood=mood)
            return {"mood": mood}

        # --- Start services ---
        sonic.start(stop_event)
        vision.start(stop_event)
        browser.start(stop_event)

        # Start audio recording from robot mic
        try:
            reachy_mini.media.start_recording()
        except Exception as e:
            logger.warning(f"Could not start mic recording: {e}")

        logger.info("Reachy Nova is alive! All systems go.")

        # --- Main control loop ---
        while not stop_event.is_set():
            t = time.time() - t0

            # Get current mood for animation
            with state_lock:
                mood = app_state["mood"]
                voice_state = app_state["voice_state"]
                antenna_mode = app_state["antenna_mode"]
                vision_enabled = app_state["vision_enabled"]

            # --- Head animation based on state ---
            if voice_state == "listening":
                # Subtle attentive movement
                yaw_deg = 5.0 * np.sin(2.0 * np.pi * 0.1 * t)
                pitch_deg = -5.0  # Slight downward look (attentive)
            elif voice_state == "speaking":
                # Animated, expressive head movement
                yaw_deg = 15.0 * np.sin(2.0 * np.pi * SPEAK_YAW_SPEED * t)
                pitch_deg = 5.0 * np.sin(2.0 * np.pi * 0.4 * t)
            elif voice_state == "thinking":
                # Look up and to the side (thinking pose)
                yaw_deg = 20.0 * np.sin(2.0 * np.pi * 0.05 * t)
                pitch_deg = -10.0
            else:
                # Idle - gentle scanning
                yaw_deg = 25.0 * np.sin(2.0 * np.pi * IDLE_YAW_SPEED * t)
                pitch_deg = 3.0 * np.sin(2.0 * np.pi * 0.08 * t)

            head_pose = create_head_pose(yaw=yaw_deg, pitch=pitch_deg, degrees=True)

            # --- Antenna animation ---
            if antenna_mode == "off":
                antennas_deg = np.array([0.0, 0.0])
            elif antenna_mode == "excited" or mood == "excited":
                amp = EXCITED_ANTENNA_AMP
                a = amp * np.sin(2.0 * np.pi * 1.5 * t)
                antennas_deg = np.array([a, -a])
            elif antenna_mode == "calm":
                amp = CALM_ANTENNA_AMP
                a = amp * np.sin(2.0 * np.pi * 0.3 * t)
                antennas_deg = np.array([a, -a])
            elif mood == "thinking":
                # Asymmetric "thinking" antenna pose
                a1 = 20.0 * np.sin(2.0 * np.pi * 0.2 * t)
                a2 = -10.0 + 5.0 * np.sin(2.0 * np.pi * 0.3 * t)
                antennas_deg = np.array([a1, a2])
            elif mood == "curious":
                amp = 20.0
                a = amp * np.sin(2.0 * np.pi * 0.8 * t)
                antennas_deg = np.array([a, a])  # Same direction = curious
            else:
                amp = 20.0
                a = amp * np.sin(2.0 * np.pi * ANTENNA_SPEED * t)
                antennas_deg = np.array([a, -a])

            antennas_rad = np.deg2rad(antennas_deg)
            reachy_mini.set_target(head=head_pose, antennas=antennas_rad)

            # --- Feed mic audio to Nova Sonic ---
            try:
                audio = reachy_mini.media.get_audio_sample()
                if audio is not None:
                    sonic.feed_audio(audio)
            except Exception:
                pass

            # --- Feed camera frames to Nova Vision ---
            if vision_enabled:
                try:
                    frame = reachy_mini.media.get_frame()
                    if frame is not None:
                        vision.update_frame(frame)
                except Exception:
                    pass

            # --- Push buffered audio output to speaker ---
            with audio_lock:
                chunks = list(audio_output_buffer)
                audio_output_buffer.clear()

            if chunks:
                # Resample from 24kHz (Nova Sonic) to output sample rate if needed
                try:
                    output_sr = reachy_mini.media.get_output_audio_samplerate()
                except Exception:
                    output_sr = 24000

                for chunk in chunks:
                    if output_sr != OUTPUT_SAMPLE_RATE:
                        # Simple linear resample
                        ratio = output_sr / OUTPUT_SAMPLE_RATE
                        n_out = int(len(chunk) * ratio)
                        indices = np.linspace(0, len(chunk) - 1, n_out)
                        chunk = np.interp(indices, np.arange(len(chunk)), chunk).astype(np.float32)
                    try:
                        reachy_mini.media.push_audio_sample(chunk)
                    except Exception as e:
                        if "start_playing" in str(e):
                            try:
                                reachy_mini.media.start_playing()
                                reachy_mini.media.push_audio_sample(chunk)
                            except Exception:
                                pass

            time.sleep(0.02)

        # Cleanup
        try:
            reachy_mini.media.stop_recording()
        except Exception:
            pass
        try:
            reachy_mini.media.stop_playing()
        except Exception:
            pass


if __name__ == "__main__":
    app = ReachyNova()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
