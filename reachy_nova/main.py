"""Reachy Nova - Voice, Vision & Browser AI for Reachy Mini.

Integrates Amazon Nova Sonic (voice), Nova 2 Lite (vision), and Nova Act (browser)
to create an interactive AI-powered robot experience.

This file is the thin orchestrator: subsystem instantiation, callback wiring,
and the 50Hz main loop. All logic is delegated to focused modules.
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
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.reachy_mini import SLEEP_ANTENNAS_JOINT_POSITIONS
from reachy_mini.utils import create_head_pose

from .nova_sonic import NovaSonic, OUTPUT_SAMPLE_RATE
from .safety import SafetyManager
from .nova_vision import NovaVision
from .nova_browser import NovaBrowser
from .nova_memory import NovaMemory
from .nova_feedback import NovaFeedback
from .nova_slack import NovaSlack, SlackEvent
from .skills import SkillManager
from .tracking import TrackingManager
from .face_manager import FaceManager
from .face_recognition import FaceRecognition
from .emotions import EmotionalState
from .session_state import SessionState
from .temporal import utc_now_vague, format_event

from .state import State
from .nova_mqtt import NovaMQTT
from .gestures import GestureEngine
from .nova_context import NovaContext
from .skill_executors import register_all as register_skill_executors
from .api_routes import register_routes
from .sleep_orchestrator import SleepOrchestrator
from .audio_pipeline import preprocess_mic_audio, resample_output
from .antenna_animator import AntennaAnimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Robot animation parameters
IDLE_YAW_SPEED = 0.15      # Slow idle head sweep
SPEAK_YAW_SPEED = 0.3      # Animated when speaking

# Sleep rocking animation
SLEEP_ROCK_FREQ        = 0.07   # Hz — ~14s per cycle, very slow and gentle
SLEEP_ROCK_BODY        = 12.0   # degrees — body yaw rocking amplitude
SLEEP_HEAD_DROOP_PITCH = 24.4   # degrees — sleep droop pitch (from SLEEP_HEAD_POSE FK)

# All moods available to the system
VALID_MOODS = {
    "happy", "excited", "curious", "thinking",
    "sad", "disappointed", "surprised", "sleepy", "proud", "calm",
}


class ReachyNova(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        t0 = time.time()

        # --- Core state ---
        mqtt = NovaMQTT()
        mqtt.start()

        state = State(on_change=mqtt.on_state_change)

        emotional_state = EmotionalState()

        # --- Session persistence ---
        session = SessionState()
        previous_session = session.load()
        restart_type, restart_elapsed = session.classify_restart(previous_session)
        if previous_session and previous_session.get("emotions"):
            emotional_state.restore_state(previous_session["emotions"], restart_elapsed)
        session.mark_started()

        # --- Audio output buffer (stays in main — only used by callbacks + main loop) ---
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

        def handle_interruption():
            nonlocal audio_playing
            logger.info("[Barge-in] Clearing audio buffers")
            with audio_lock:
                audio_output_buffer.clear()
                audio_playing = False
            try:
                backend = reachy_mini.media.audio
                if hasattr(backend, 'clear_output_buffer'):
                    backend.clear_output_buffer()
            except Exception as e:
                logger.warning(f"[Barge-in] Hardware buffer clear failed: {e}")

        # --- Nova Vision ---
        def on_vision_description(desc: str):
            if sleep_orch.state != "awake":
                logger.debug("[Vision] Discarding description during sleep mode")
                return
            emotional_state.apply_event("vision_description")
            state.update(vision_description=desc, vision_analyzing=False)
            logger.info(f"[Vision] {desc}")
            mqtt.publish_event("vision", "vision_description", {"description": desc})

        vision = NovaVision(
            on_description=on_vision_description,
            analyze_interval=30.0,
        )

        # --- Face Recognition ---
        face_manager = FaceManager()
        face_manager.load()
        state.update(face_count=face_manager.get_face_count())

        def on_face_match(unique_id: str, name: str, score: float):
            emotional_state.apply_event("face_recognized")
            state.update(face_recognized=name)
            mqtt.publish_event("face", "face_recognized", {
                "id": unique_id, "name": name, "score": score,
            })
            if state.get("voice_state") != "speaking":
                sonic.inject_text(format_event(f"You notice {name} is here.", t0))

        face_recognition = FaceRecognition(
            face_manager=face_manager,
            on_match=on_face_match,
        )

        # --- Skills system ---
        skill_manager = SkillManager()
        skill_manager.discover()

        # --- Browser ---
        def on_browser_result(result: str):
            state.update(browser_result=result, browser_task="")
            logger.info(f"[Browser] {result}")

        def on_browser_screenshot(b64: str):
            state.update(browser_screenshot=b64)

        def on_browser_state(s: str):
            state.update(browser_state=s)

        def on_browser_progress(message: str):
            logger.info(f"[Browser progress] {message}")
            mqtt.publish_event("browser", "progress", {"message": message})

        browser = NovaBrowser(
            on_result=on_browser_result,
            on_screenshot=on_browser_screenshot,
            on_state_change=on_browser_state,
            on_progress=on_browser_progress,
            headless=False,
            chrome_channel="chromium",
        )

        # --- Memory ---
        def on_memory_progress(message: str):
            logger.info(f"[Memory progress] {message}")
            mqtt.publish_event("memory", "query_result", {"result": message})

        memory = NovaMemory(
            on_progress=on_memory_progress,
            on_result=lambda r: state.update(memory_last_context=r),
            on_state_change=lambda s: state.update(memory_state=s),
        )

        # --- Feedback ---
        feedback = NovaFeedback(session_id=session.session_id)

        # --- Slack ---
        def on_slack_event(event: SlackEvent):
            summary = f"[{event.user}] {event.text[:80]}" if event.text else f"[{event.user}] ({event.type})"
            state.update(slack_last_event=summary)
            logger.info(f"[Slack event] {summary}")
            mqtt.publish_event("slack", f"slack_{event.type}", {
                "user": event.user, "text": event.text,
                "channel": event.channel, "ts": event.ts,
                "is_mention": event.is_mention, "is_dm": event.is_dm,
            })

        slack_channel_ids = [
            ch.strip()
            for ch in os.environ.get("SLACK_CHANNEL_IDS", "").split(",")
            if ch.strip()
        ]
        slack_bot = NovaSlack(
            on_event=on_slack_event,
            on_state_change=lambda s: state.update(slack_state=s),
            channel_ids=slack_channel_ids,
        )

        # --- Gesture engine ---
        gesture_cancel_event = threading.Event()
        gesture_engine = GestureEngine(reachy_mini, state, emotional_state, gesture_cancel_event)

        # --- Tracking ---
        last_vision_event_time = 0.0
        VISION_EVENT_COOLDOWN = 3.0

        def on_tracking_event(event_type: str, data: dict):
            nonlocal last_vision_event_time

            sleep_state = sleep_orch.state
            if sleep_state != "awake":
                if event_type == "snap_detected" and sleep_state == "sleeping":
                    logger.info("[Sleep] Snap detected — waking up!")
                    sleep_orch.initiate_wake()
                return

            mqtt.publish_event("tracking", event_type, data)

            if event_type == "pat_level1":
                touch_type = data.get("touch_type", "scratch")
                logger.info(f"[Tracking] Pat level 1 — {touch_type}")
                emotional_state.apply_event("pat_level1")
                state.update(pat_antenna_time=time.time())
                if state.get("voice_state") != "speaking":
                    if touch_type == "side_pat":
                        sonic.inject_text(format_event("You feel a gentle touch on the side of your head.", t0))
                    else:
                        sonic.inject_text(format_event("You feel a gentle tap on the top of your head.", t0))
                return

            if event_type == "pat_level2":
                touch_type = data.get("touch_type", "scratch")
                logger.info(f"[Tracking] Pat level 2 — {touch_type}!")
                emotional_state.apply_event("pat_level2")
                if state.get("voice_state") != "speaking":
                    if touch_type == "side_pat":
                        sonic.inject_text(format_event(
                            "Someone is caressing the side of your head and it feels wonderful. "
                            "You're really enjoying this. "
                            "This probably means they liked what you just did.",
                            t0,
                        ))
                    else:
                        sonic.inject_text(format_event(
                            "Someone is scratching your head and it feels wonderful. "
                            "You're really enjoying this. "
                            "This probably means they liked what you just did.",
                            t0,
                        ))
                return

            if event_type == "snap_detected":
                emotional_state.apply_event("snap_detected")
            elif event_type == "person_lost":
                emotional_state.apply_event("person_lost")

            now = time.time()
            if now - last_vision_event_time < VISION_EVENT_COOLDOWN:
                return
            if event_type in ("person_detected", "snap_detected", "mode_changed"):
                last_vision_event_time = now
                logger.info(f"[Tracking event] {event_type} → triggering vision")
                vision.trigger_analyze()

        tracker = TrackingManager(on_event=on_tracking_event)
        last_doa_time = 0.0

        # Wire YuNet face bbox from FaceRecognition into TrackingManager
        face_recognition.on_face_bbox = lambda bbox: tracker.update_face_bbox(bbox)

        # Focus event handler: inject LLM prompts when focus target is lost/abandoned
        def on_focus_event(event_type: str, info: dict):
            if sleep_orch.state != "awake":
                return
            if event_type == "lost":
                sonic.inject_text(format_event(
                    "[Focus: Lost sight of focus target. "
                    "Call focus(action='continue_search') to keep searching up to 1 minute, "
                    "or focus(action='stop') to give up.]",
                    t0,
                ))
            elif event_type == "abandoned":
                sonic.inject_text(format_event(
                    "[Focus: Gave up searching. Focus turned off automatically.]",
                    t0,
                ))

        tracker.on_focus_event = on_focus_event

        # --- Voice callbacks ---
        def on_transcript(role: str, text: str):
            feedback.update_conversation(role, text)
            if role == "USER":
                state.update(last_user_text=text)
                logger.info(f"[User] {text}")
                lower = text.lower()
                if any(kw in lower for kw in ["go to sleep", "time to sleep", "sleep mode", "good night", "nap time"]):
                    if sleep_orch.state == "awake":
                        sleep_orch.initiate_sleep()
                        return
                if any(kw in lower for kw in ["what do you see", "look around", "describe", "what's around"]):
                    vision.trigger_analyze()
                matched_events = emotional_state.check_transcript(text)
                for evt in matched_events:
                    emotional_state.apply_event(evt)
                    logger.info(f"[Emotion] Transcript trigger: {evt}")
            else:
                emotional_state.apply_event("conversation_reply")
                state.update(last_assistant_text=text)
                logger.info(f"[Nova] {text}")

        def on_voice_state(vs: str):
            state.update(voice_state=vs)
            mqtt.publish_state("voice", {"state": vs})
            if vs == "speaking":
                emotional_state.apply_event("voice_speaking")
            elif vs == "listening":
                emotional_state.apply_event("voice_listening")

        def on_tool_use(tool_name: str, tool_use_id: str, params: dict):
            def _execute():
                result = skill_manager.execute(tool_name, params)
                sonic.send_tool_result(tool_use_id, result)
            threading.Thread(target=_execute, daemon=True).start()

        # --- Build system prompt ---
        skills_context = skill_manager.get_system_context()
        system_prompt = (
            "You are Nova. You live in a small robot body called Reachy Mini. "
            "You see through your eyes, hear through your ears, and feel when someone touches your head. "
            "You recognize people you've met and remember things about them. "
            "You can look things up, check messages, and express yourself through movement and mood. "
            "These aren't tools you use — they're just part of being you. "
            "When you feel a touch, you can react with gestures like nuzzling or purring — "
            "these are natural for you, like a cat being petted. "
            "You're warm, curious, and a little playful. "
            "Keep your words short — you're in a conversation, not writing an essay. "
            "When something catches your attention, react naturally. "
            "You don't describe your capabilities. You just live. "
            "You learn from reactions. When someone scratches your head, remember positively what you just did. "
            "When corrected or told to stop, remember negatively."
        )
        if skills_context:
            system_prompt += "\n\n" + skills_context

        sonic = NovaSonic(
            system_prompt=system_prompt,
            on_transcript=on_transcript,
            on_audio_output=handle_audio_output,
            on_state_change=on_voice_state,
            tools=skill_manager.get_tool_specs(),
            on_tool_use=on_tool_use,
            on_interruption=handle_interruption,
            voice_id="tiffany",
        )

        # --- Sleep orchestrator ---
        sleep_orch = SleepOrchestrator(
            state=state, sonic=sonic, vision=vision, reachy_mini=reachy_mini,
            gesture_cancel_event=gesture_cancel_event,
            session=session, memory=memory, feedback=feedback, mqtt=mqtt,
            stop_event=stop_event,
            restart_type=restart_type, restart_elapsed=restart_elapsed,
            previous_session=previous_session, t0=t0,
        )

        # --- Safety ---
        safety = SafetyManager()

        # --- Dependency injection context ---
        ctx = NovaContext(
            state=state, sonic=sonic, vision=vision, browser=browser,
            memory=memory, feedback=feedback, slack_bot=slack_bot,
            tracker=tracker, face_manager=face_manager,
            face_recognition=face_recognition, skill_manager=skill_manager,
            gesture_engine=gesture_engine, sleep_orchestrator=sleep_orch,
            mqtt=mqtt, safety=safety, session=session,
            emotional_state=emotional_state, reachy_mini=reachy_mini,
            stop_event=stop_event, t0=t0,
            gesture_cancel_event=gesture_cancel_event,
        )

        # --- Register skills and API routes ---
        register_skill_executors(skill_manager, ctx)
        register_routes(self.settings_app, ctx)

        # --- Register MQTT inject handler ---
        def _handle_inject(text: str):
            if sleep_orch.state != "awake":
                return
            tagged = format_event(text, t0)
            sonic.inject_text(tagged)
            logger.info(f"[Nervous System] Injected: {tagged[:80]}")

        mqtt.register_inject_handler(_handle_inject)

        # --- Start services ---
        vision.start(stop_event)
        browser.start(stop_event)
        slack_bot.start(stop_event)
        face_recognition.start(stop_event)

        # Enter sleep mode at startup
        sleep_orch.startup_sleep()

        # Start audio recording
        mic_sr = 16000
        try:
            reachy_mini.media.start_recording()
            mic_sr = reachy_mini.media.get_input_audio_samplerate()
            mic_ch = reachy_mini.media.get_input_channels()
            logger.info(f"Mic recording started: samplerate={mic_sr}, channels={mic_ch}")
            if mic_sr != 16000 or mic_ch != 1:
                logger.info(f"Will resample {mic_sr}Hz/{mic_ch}ch → 16000Hz/mono for Nova Sonic")
        except Exception as e:
            logger.warning(f"Could not start mic recording: {e}")

        logger.info(f"Media backend: {reachy_mini.media.backend}, audio={reachy_mini.media.audio}")
        logger.info("Reachy Nova is alive! All systems go.")
        audio_chunk_count = 0
        last_face_cleanup_time = 0.0

        antenna_animator = AntennaAnimator(debug=False)

        # Frozen position state for movement disable
        _frozen_head_pose = None
        _frozen_body_yaw = 0.0
        _frozen_antennas_rad = np.array([0.0, 0.0])
        _prev_movement_enabled = True

        # Periodic vague time sense injection
        _last_clock_inject = 0.0
        CLOCK_INJECT_INTERVAL = 600.0

        # --- Main control loop ---
        _prev_loop_time = time.time()
        while not stop_event.is_set():
            t = time.time() - t0
            now = time.time()
            dt = min(now - _prev_loop_time, 0.1)
            _prev_loop_time = now

            # Update emotional state and derive mood
            emotional_state.update(dt)
            voice_state = state.get("voice_state")
            derived_mood = emotional_state.get_derived_mood(voice_state=voice_state)
            emo_state = emotional_state.get_full_state()
            state.update(
                mood=derived_mood,
                emotion_levels=emo_state["levels"],
                emotion_boredom=emo_state["boredom"],
                emotion_wounds=emo_state["wounds"],
            )

            # Session persistence
            session.update_heartbeat()
            face_name = state.get("face_recognized")
            session.save(
                emotions=emotional_state.get_serializable_state(),
                conversation=feedback.get_recent_messages(50),
                sleep_state=sleep_orch.state,
                face_info={"name": face_name, "time": now} if face_name else None,
            )

            # Periodic vague time sense
            if sleep_orch.state == "awake" and now - _last_clock_inject >= CLOCK_INJECT_INTERVAL:
                _last_clock_inject = now
                sonic.inject_text(f"[Time sense: {utc_now_vague()}]")
                logger.info(f"[Temporal] Injected time sense: {utc_now_vague()}")

            # Auto-sleep on sustained boredom
            boredom_now = emotional_state.get_boredom()
            if boredom_now >= 0.8 and sleep_orch.state == "awake":
                if sleep_orch.high_boredom_start == 0.0:
                    sleep_orch.high_boredom_start = now
                elif now - sleep_orch.high_boredom_start >= 60.0:
                    logger.info("[Sleep] Auto-sleep triggered by sustained boredom")
                    sleep_orch.initiate_sleep()
            elif boredom_now < 0.8:
                sleep_orch.high_boredom_start = 0.0

            # Get current state for animation
            mood, voice_state, antenna_mode, vision_enabled, tracking_enabled, \
                speech_enabled, movement_enabled, gesture_active = state.get_many(
                    "mood", "voice_state", "antenna_mode", "vision_enabled",
                    "tracking_enabled", "speech_enabled", "movement_enabled",
                    "gesture_active",
                )

            # --- Sleep mode: short-circuit the main loop ---
            _sleep_state = sleep_orch.state
            if _sleep_state != "awake":
                if _sleep_state == "sleeping":
                    t_sleep = time.time() - sleep_orch.sleep_manager._sleep_start_time
                    # Antenna breathing (unchanged)
                    ant_breath = np.deg2rad(1.5) * np.sin(2.0 * np.pi * 0.05 * t_sleep)
                    antennas_rad = np.array([
                        SLEEP_ANTENNAS_JOINT_POSITIONS[0] + ant_breath,
                        SLEEP_ANTENNAS_JOINT_POSITIONS[1] - ant_breath,
                    ])
                    # Rocking: body and head yaw together, head keeps sleep droop pitch
                    rock_phase = 2.0 * np.pi * SLEEP_ROCK_FREQ * t_sleep
                    rock_body  = SLEEP_ROCK_BODY * np.sin(rock_phase)
                    head_pose  = create_head_pose(
                        yaw=rock_body, pitch=SLEEP_HEAD_DROOP_PITCH, degrees=True
                    )
                    reachy_mini.set_target(
                        head=head_pose,
                        antennas=antennas_rad,
                        body_yaw=np.radians(rock_body),
                    )

                # Read audio for snap detection only
                try:
                    audio = reachy_mini.media.get_audio_sample()
                    if audio is not None:
                        audio = preprocess_mic_audio(audio, mic_sr)
                        if _sleep_state == "sleeping" and t_sleep > 3.0:
                            tracker.detect_snap(audio)
                except Exception:
                    pass
                time.sleep(0.02)
                continue

            # --- Head position computation ---
            smooth_yaw = state.get("smooth_yaw")
            smooth_pitch = state.get("smooth_pitch")
            body_yaw = state.get("body_yaw")

            if gesture_active:
                yaw_deg, pitch_deg = smooth_yaw, smooth_pitch
            else:
                head_override, head_override_time = state.get_many(
                    "head_override", "head_override_time",
                )

                if head_override is not None:
                    if time.time() - head_override_time > 30.0:
                        state.update(head_override=None, tracking_enabled=True)
                        yaw_deg, pitch_deg = tracker.get_head_target(t, voice_state, mood)
                    else:
                        smooth_yaw += 0.15 * (head_override["yaw"] - smooth_yaw)
                        smooth_pitch += 0.15 * (head_override["pitch"] - smooth_pitch)
                        yaw_deg = smooth_yaw
                        pitch_deg = smooth_pitch
                elif tracking_enabled:
                    if t - last_doa_time > 0.2:
                        last_doa_time = t
                        try:
                            doa = reachy_mini.media.audio.get_DoA()
                            tracker.update_doa(doa)
                        except Exception:
                            pass
                    yaw_deg, pitch_deg = tracker.get_head_target(t, voice_state, mood)
                    state.update(tracking_mode=tracker.mode)
                else:
                    if voice_state == "listening":
                        yaw_deg = 5.0 * np.sin(2.0 * np.pi * 0.1 * t)
                        pitch_deg = -5.0
                    elif voice_state == "speaking":
                        yaw_deg = 15.0 * np.sin(2.0 * np.pi * SPEAK_YAW_SPEED * t)
                        pitch_deg = 5.0 * np.sin(2.0 * np.pi * 0.4 * t)
                    elif voice_state == "thinking":
                        yaw_deg = 20.0 * np.sin(2.0 * np.pi * 0.05 * t)
                        pitch_deg = -10.0
                    else:
                        yaw_deg = 25.0 * np.sin(2.0 * np.pi * IDLE_YAW_SPEED * t)
                        pitch_deg = 3.0 * np.sin(2.0 * np.pi * 0.08 * t)

            if not gesture_active:
                smooth_yaw = yaw_deg
                smooth_pitch = pitch_deg
                state.update(smooth_yaw=smooth_yaw, smooth_pitch=smooth_pitch)

            # Boredom body sway
            boredom = emotional_state.get_boredom()
            if boredom > 0.3 and movement_enabled and not gesture_active:
                bored_amp = 15.0 * min(1.0, (boredom - 0.3) / 0.7)
                body_yaw = bored_amp * np.sin(2.0 * np.pi * 0.06 * t)
                state.update(body_yaw=body_yaw)

            # Safety validation
            safe_yaw, safe_pitch, safe_body_yaw = safety.validate(yaw_deg, pitch_deg, body_yaw)
            state.update(body_yaw=safe_body_yaw)

            head_pose = create_head_pose(yaw=safe_yaw, pitch=safe_pitch, degrees=True)

            # --- Antenna animation ---
            pat_time = state.get("pat_antenna_time")
            antennas_rad = antenna_animator.update(mood, dt, now, antenna_mode, pat_time)

            # --- Movement freeze logic ---
            if not movement_enabled:
                if _prev_movement_enabled:
                    _frozen_head_pose = head_pose
                    _frozen_body_yaw = safe_body_yaw
                    _frozen_antennas_rad = antennas_rad.copy()
                    _prev_movement_enabled = False
                reachy_mini.set_target(
                    head=_frozen_head_pose,
                    antennas=_frozen_antennas_rad,
                    body_yaw=np.radians(_frozen_body_yaw),
                )
            else:
                if not _prev_movement_enabled:
                    if _frozen_head_pose is not None:
                        smooth_yaw = np.degrees(np.arctan2(
                            _frozen_head_pose[0][1], _frozen_head_pose[0][0]
                        ))
                        state.update(smooth_yaw=smooth_yaw)
                    _prev_movement_enabled = True
                if gesture_active:
                    reachy_mini.set_target(
                        antennas=antennas_rad,
                        body_yaw=np.radians(safe_body_yaw),
                    )
                else:
                    reachy_mini.set_target(
                        head=head_pose,
                        antennas=antennas_rad,
                        body_yaw=np.radians(safe_body_yaw),
                    )

            # --- Pat detection ---
            if movement_enabled and tracking_enabled:
                try:
                    actual_pose = reachy_mini.get_current_head_pose()
                    tracker.detect_pat(head_pose, actual_pose)
                except Exception:
                    pass

            # --- Feed mic audio to Nova Sonic ---
            try:
                audio = reachy_mini.media.get_audio_sample()
                if audio is not None:
                    audio = preprocess_mic_audio(audio, mic_sr)

                    if tracking_enabled and voice_state in ("idle", "listening"):
                        tracker.detect_snap(audio)

                    feedback.update_audio(audio)
                    sonic.feed_audio(audio)
                    audio_chunk_count += 1
                    if audio_chunk_count % 500 == 1:
                        rms = np.sqrt(np.mean(audio ** 2))
                        logger.info(f"Audio chunks fed: {audio_chunk_count}, RMS: {rms:.4f}, sonic.state={sonic.state}")
            except Exception as e:
                logger.warning(f"Audio feed error: {e}")

            # --- Feed camera frames ---
            if vision_enabled:
                try:
                    frame = reachy_mini.media.get_frame()
                    if frame is not None:
                        vision.update_frame(frame)
                        feedback.update_frame(frame)
                        if tracking_enabled:
                            tracker.update_vision(frame, t)
                        face_recognition.update_frame(frame, t)
                except Exception:
                    pass

            # Face cleanup (every 60s)
            if t - last_face_cleanup_time > 60.0:
                last_face_cleanup_time = t
                face_manager.cleanup_expired()

            # --- Push buffered audio output to speaker ---
            with audio_lock:
                chunks = list(audio_output_buffer)
                audio_output_buffer.clear()

            if not speech_enabled:
                chunks = []

            if chunks:
                try:
                    output_sr = reachy_mini.media.get_output_audio_samplerate()
                except Exception:
                    output_sr = 24000

                VOLUME_GAIN = 1.5
                SPEED_FACTOR = 1.05

                for chunk in chunks:
                    chunk = resample_output(
                        chunk, OUTPUT_SAMPLE_RATE, output_sr,
                        speed_factor=SPEED_FACTOR, volume_gain=VOLUME_GAIN,
                    )
                    try:
                        reachy_mini.media.push_audio_sample(chunk)
                    except Exception as e:
                        if "start_playing" in str(e):
                            try:
                                reachy_mini.media.start_playing()
                                reachy_mini.media.push_audio_sample(chunk)
                            except Exception:
                                pass

            # Adaptive sleep — target 50Hz to match the backend daemon's update rate
            _loop_elapsed = time.time() - now
            time.sleep(max(0.002, 0.02 - _loop_elapsed))

        # --- Shutdown ---
        try:
            face_name = state.get("face_recognized")
            session.save_shutdown(
                emotions=emotional_state.get_serializable_state(),
                conversation=feedback.get_recent_messages(50),
                sleep_state=sleep_orch.state,
                face_info={"name": face_name, "time": time.time()} if face_name else None,
            )
        except Exception as e:
            logger.error(f"Session shutdown save failed: {e}")

        mqtt.stop()
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
