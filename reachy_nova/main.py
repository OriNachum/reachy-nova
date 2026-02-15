"""Reachy Nova - Voice, Vision & Browser AI for Reachy Mini.

Integrates Amazon Nova Sonic (voice), Nova 2 Lite (vision), and Nova Act (browser)
to create an interactive AI-powered robot experience.
"""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import numpy as np
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

from .nova_sonic import NovaSonic, OUTPUT_SAMPLE_RATE
from .safety import SafetyManager
from .nova_vision import NovaVision
from .nova_browser import NovaBrowser
from .nova_memory import NovaMemory
from .nova_slack import NovaSlack, SlackEvent
from .skills import SkillManager
from .tracking import TrackingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Robot animation parameters
IDLE_YAW_SPEED = 0.15      # Slow idle head sweep
LISTEN_YAW_SPEED = 0.0     # Stay still when listening
SPEAK_YAW_SPEED = 0.3      # Animated when speaking

# All moods available to the system
VALID_MOODS = {
    "happy", "excited", "curious", "thinking",
    "sad", "disappointed", "surprised", "sleepy", "proud", "calm",
}


def ease_sin(t: float, freq: float) -> float:
    """Pure sine wave — naturally smooth, decelerates at peaks."""
    return np.sin(2.0 * np.pi * freq * t)


def ease_sin_soft(t: float, freq: float) -> float:
    """Sine-of-sine — extra dwell time at the extremes for organic feel.

    sin(π/2 · sin(θ)) spends more time near ±1 and moves faster
    through the center, giving a softer, more deliberate motion.
    """
    return float(np.sin(0.5 * np.pi * np.sin(2.0 * np.pi * freq * t)))


# --- Mood antenna profiles ---
# Each mood defines: (frequency, amplitude, phase_pattern, easing_fn)
# phase_pattern: "oppose" = [a, -a], "sync" = [a, a], or a custom callable
MOOD_ANTENNAS = {
    # Cheerful default - gentle opposing sway
    "happy": {
        "freq": 0.25, "amp": 18.0, "phase": "oppose",
        "ease": ease_sin,
    },
    # High energy - faster, bigger wiggles
    "excited": {
        "freq": 0.7, "amp": 30.0, "phase": "oppose",
        "ease": ease_sin,
    },
    # Attentive - antennas tilt forward together
    "curious": {
        "freq": 0.35, "amp": 18.0, "phase": "sync",
        "ease": ease_sin,
        "offset": 10.0,  # bias forward
    },
    # Processing - asymmetric, one up one tilted
    "thinking": {
        "freq": 0.12, "amp": 15.0, "phase": "custom",
        "ease": ease_sin_soft,
    },
    # Droopy, slow backward lean
    "sad": {
        "freq": 0.08, "amp": 8.0, "phase": "sync",
        "ease": ease_sin_soft,
        "offset": -25.0,  # antennas pulled back
    },
    # Low sag, barely moving
    "disappointed": {
        "freq": 0.06, "amp": 5.0, "phase": "sync",
        "ease": ease_sin_soft,
        "offset": -20.0,
    },
    # Quick perk then settle
    "surprised": {
        "freq": 0.5, "amp": 25.0, "phase": "oppose",
        "ease": ease_sin,
        "offset": 15.0,  # antennas perked up
    },
    # Very slow, drooping
    "sleepy": {
        "freq": 0.04, "amp": 4.0, "phase": "sync",
        "ease": ease_sin_soft,
        "offset": -18.0,
    },
    # Held high with subtle movement
    "proud": {
        "freq": 0.15, "amp": 6.0, "phase": "oppose",
        "ease": ease_sin_soft,
        "offset": 20.0,  # antennas up high
    },
    # Relaxed gentle movement
    "calm": {
        "freq": 0.12, "amp": 10.0, "phase": "oppose",
        "ease": ease_sin_soft,
    },
}

# Transition smoothing for mood changes (seconds to blend)
MOOD_BLEND_TIME = 1.5


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
            "antenna_mode": "auto",  # auto, off
            "mood": "happy",  # see VALID_MOODS
            "tracking_enabled": True,
            "tracking_mode": "idle",  # idle, speaker, face, snap
            "memory_state": "idle",
            "memory_last_context": "",
            "slack_state": "idle",
            "slack_last_event": "",
            "head_override": None,        # None or {"yaw": float, "pitch": float}
            "head_override_time": 0.0,    # timestamp for 30s safety timeout
            "speech_enabled": True,       # False = mute audio output to speaker
            "movement_enabled": True,     # False = freeze head + body at current position
        }

        def update_state(**kwargs):
            mood_changed = None
            with state_lock:
                app_state.update(kwargs)
                if "mood" in kwargs:
                    mood_changed = kwargs["mood"]
            # Publish OUTSIDE the lock — never block the main loop on MQTT I/O
            if mqtt_available and mood_changed is not None:
                publish_state("mood", {"mood": mood_changed})

        # --- MQTT client (optional, for Nervous System) ---
        mqtt_available = False
        mqtt_client = None
        try:
            import paho.mqtt.client as paho_mqtt

            def _on_mqtt_connect(client, userdata, flags, rc, properties=None):
                nonlocal mqtt_available
                mqtt_available = True
                client.subscribe("nova/inject")
                logger.info(f"MQTT connected (rc={rc}) — subscribed to nova/inject")

            def _on_mqtt_disconnect(client, userdata, flags, rc, properties=None):
                nonlocal mqtt_available
                mqtt_available = False
                logger.warning(f"MQTT disconnected (rc={rc}) — falling back to direct mode")

            mqtt_client = paho_mqtt.Client(
                paho_mqtt.CallbackAPIVersion.VERSION2,
                client_id="reachy-nova-app",
            )
            mqtt_client.on_connect = _on_mqtt_connect
            mqtt_client.on_disconnect = _on_mqtt_disconnect
            mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
            mqtt_client.connect(
                os.environ.get("MQTT_BROKER", "localhost"),
                int(os.environ.get("MQTT_PORT", "1883")),
                keepalive=60,
            )
            mqtt_client.loop_start()
            mqtt_available = True
            logger.info("MQTT started — Nervous System integration active")
        except Exception as e:
            logger.warning(f"MQTT not available — direct callbacks only: {e}")

        def publish_event(source: str, event_type: str, payload: dict):
            """Publish a pure data event to MQTT."""
            if not mqtt_available:
                return
            event = {
                "event_id": str(uuid.uuid4()),
                "type": event_type,
                "source": source,
                "payload": payload,
                "timestamp": time.time(),
            }
            topic = f"nova/events/{source}/{event_type}"
            mqtt_client.publish(topic, json.dumps(event))

        def publish_state(key: str, value: dict):
            """Publish retained state to MQTT."""
            if not mqtt_available:
                return
            mqtt_client.publish(f"nova/state/{key}", json.dumps(value), retain=True)

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

        # --- Nova 2 Lite (Vision) ---
        def on_vision_description(desc: str):
            update_state(vision_description=desc, vision_analyzing=False, mood="excited")
            logger.info(f"[Vision] {desc}")
            # Nervous System decides whether/when to inject via MQTT
            publish_event("vision", "vision_description", {"description": desc})

        vision = NovaVision(
            on_description=on_vision_description,
            analyze_interval=30.0,
        )

        # --- Skills system ---
        skill_manager = SkillManager()
        skill_manager.discover()

        # Register look skill executor (needs vision instance)
        def look_executor(params: dict) -> str:
            query = params.get("query", "What do you see?")
            return vision.analyze_latest(query)

        skill_manager.register_executor("look", look_executor)

        # Register mood skill executor
        def mood_executor(params: dict) -> str:
            mood = params.get("mood", "happy").lower()
            if mood not in VALID_MOODS:
                return f"[Unknown mood '{mood}'. Available: {', '.join(sorted(VALID_MOODS))}]"
            update_state(mood=mood)
            return f"[Mood changed to {mood}]"

        skill_manager.register_executor(
            "mood",
            mood_executor,
            input_schema={
                "type": "object",
                "properties": {
                    "mood": {
                        "type": "string",
                        "description": "The mood to express",
                        "enum": sorted(VALID_MOODS),
                    }
                },
                "required": ["mood"],
            },
        )

        # --- Nova Act (Browser) ---
        def on_browser_result(result: str):
            update_state(browser_result=result, browser_task="")
            logger.info(f"[Browser] {result}")

        def on_browser_screenshot(b64: str):
            update_state(browser_screenshot=b64)

        def on_browser_state(state: str):
            update_state(browser_state=state)

        def on_browser_progress(message: str):
            """Narrate browser progress through the voice stream."""
            logger.info(f"[Browser progress] {message}")
            publish_event("browser", "progress", {"message": message})

        browser = NovaBrowser(
            on_result=on_browser_result,
            on_screenshot=on_browser_screenshot,
            on_state_change=on_browser_state,
            on_progress=on_browser_progress,
            headless=False,
            chrome_channel="chromium",
        )

        # Register browse skill executor
        def browse_executor(params: dict) -> str:
            query = params.get("query", "")
            url = params.get("url", "https://www.google.com")
            update_state(browser_task=query)
            return browser.execute(query, url)

        skill_manager.register_executor(
            "browse",
            browse_executor,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for or do in the browser",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to (defaults to Google)",
                    },
                },
                "required": ["query"],
            },
        )

        # --- Nova Memory ---
        def on_memory_progress(message: str):
            logger.info(f"[Memory progress] {message}")
            publish_event("memory", "query_result", {"result": message})

        def on_memory_result(result: str):
            update_state(memory_last_context=result)

        def on_memory_state(state: str):
            update_state(memory_state=state)

        memory = NovaMemory(
            on_progress=on_memory_progress,
            on_result=on_memory_result,
            on_state_change=on_memory_state,
        )

        # Register memory skill executor
        def memory_executor(params: dict) -> str:
            query = params.get("query", "")
            mode = params.get("mode", "query")
            if mode == "store":
                return memory.store(query)
            elif mode == "context":
                return memory.get_startup_context()
            else:
                return memory.query(query)

        skill_manager.register_executor(
            "memory",
            memory_executor,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to recall, look up, or remember",
                    },
                    "mode": {
                        "type": "string",
                        "description": "query (default), store, or context",
                        "enum": ["query", "store", "context"],
                    },
                },
                "required": ["query"],
            },
        )

        # --- Nova Slack ---
        def on_slack_event(event: SlackEvent):
            summary = f"[{event.user}] {event.text[:80]}" if event.text else f"[{event.user}] ({event.type})"
            update_state(slack_last_event=summary)
            logger.info(f"[Slack event] {summary}")
            publish_event("slack", f"slack_{event.type}", {
                "user": event.user,
                "text": event.text,
                "channel": event.channel,
                "ts": event.ts,
                "is_mention": event.is_mention,
                "is_dm": event.is_dm,
            })

        def on_slack_state(state: str):
            update_state(slack_state=state)

        slack_channel_ids = [
            ch.strip()
            for ch in os.environ.get("SLACK_CHANNEL_IDS", "").split(",")
            if ch.strip()
        ]

        slack_bot = NovaSlack(
            on_event=on_slack_event,
            on_state_change=on_slack_state,
            channel_ids=slack_channel_ids,
        )

        # Register slack skill executor
        def slack_executor(params: dict) -> str:
            return slack_bot.execute(params)

        skill_manager.register_executor(
            "slack",
            slack_executor,
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The Slack action to perform",
                        "enum": [
                            "send_message",
                            "read_messages",
                            "reply_to_thread",
                            "add_reaction",
                        ],
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text (for send_message/reply_to_thread)",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Slack channel ID (defaults to first configured channel)",
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Thread timestamp for reply_to_thread",
                    },
                    "emoji": {
                        "type": "string",
                        "description": "Emoji name for add_reaction (e.g. thumbsup)",
                    },
                    "ts": {
                        "type": "string",
                        "description": "Message timestamp for add_reaction",
                    },
                    "count": {
                        "type": "number",
                        "description": "Number of messages to read (default 10, max 50)",
                    },
                },
                "required": ["action"],
            },
        )

        # --- Control skill (head/body movement) ---
        # Smoothing state shared between executor and main loop
        _smooth_yaw = 0.0
        _smooth_pitch = 0.0
        _current_body_yaw = 0.0  # tracked in degrees

        def control_executor(params: dict) -> str:
            nonlocal _smooth_yaw, _smooth_pitch, _current_body_yaw
            action = params.get("action", "")

            if action == "move_head":
                yaw = max(-45, min(45, float(params.get("yaw", 0))))
                pitch = max(-15, min(25, float(params.get("pitch", 0))))
                update_state(
                    tracking_enabled=False,
                    tracking_mode="idle",
                    head_override={"yaw": yaw, "pitch": pitch},
                    head_override_time=time.time(),
                )
                return f"[Moving head to yaw={yaw:.0f}, pitch={pitch:.0f}. Tracking disabled.]"

            elif action == "move_body":
                body_yaw = max(-25, min(25, float(params.get("body_yaw", 0))))
                duration = max(0.5, min(5, float(params.get("duration", 1.5))))
                start = _current_body_yaw
                steps = int(duration / 0.02)
                for i in range(steps + 1):
                    alpha = 0.5 * (1 - np.cos(np.pi * i / steps))
                    current = start + alpha * (body_yaw - start)
                    reachy_mini.set_target(body_yaw=np.radians(current))
                    time.sleep(0.02)
                _current_body_yaw = body_yaw
                return f"[Body rotated to {body_yaw:.0f} over {duration:.1f}s.]"

            elif action == "enable_tracking":
                tracker.current_yaw = _smooth_yaw
                tracker.current_pitch = _smooth_pitch
                update_state(tracking_enabled=True, head_override=None)
                return "[Tracking re-enabled.]"

            elif action == "disable_tracking":
                update_state(tracking_enabled=False, tracking_mode="idle")
                return "[Tracking disabled.]"

            return f"[Unknown control action: {action}]"

        skill_manager.register_executor(
            "control",
            control_executor,
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The control action to perform",
                        "enum": [
                            "move_head",
                            "move_body",
                            "enable_tracking",
                            "disable_tracking",
                        ],
                    },
                    "yaw": {
                        "type": "number",
                        "description": "Head yaw angle in degrees (-45 to 45, negative=right, positive=left). For move_head.",
                    },
                    "pitch": {
                        "type": "number",
                        "description": "Head pitch angle in degrees (-15 to 25, negative=down, positive=up). For move_head.",
                    },
                    "body_yaw": {
                        "type": "number",
                        "description": "Body rotation in degrees (-25 to 25). For move_body.",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds for body rotation (default 1.5). For move_body.",
                    },
                },
                "required": ["action"],
            },
        )

        # --- Nova Sonic (Voice) ---
        def on_transcript(role: str, text: str):
            if role == "USER":
                update_state(last_user_text=text, mood="thinking")
                logger.info(f"[User] {text}")
                # Check for vision commands (fallback keyword path)
                lower = text.lower()
                if any(kw in lower for kw in ["what do you see", "look around", "describe", "what's around"]):
                    vision.trigger_analyze()
            else:
                update_state(last_assistant_text=text, mood="happy")
                logger.info(f"[Nova] {text}")

        def on_voice_state(state: str):
            update_state(voice_state=state)
            publish_state("voice", {"state": state})
            if state == "speaking":
                update_state(mood="excited")
            elif state == "listening":
                update_state(mood="curious")

        # Tool use callback: runs skill in a daemon thread, sends result back
        def on_tool_use(tool_name: str, tool_use_id: str, params: dict):
            def _execute():
                result = skill_manager.execute(tool_name, params)
                sonic.send_tool_result(tool_use_id, result)
            threading.Thread(target=_execute, daemon=True).start()

        # Build system prompt with skills context
        skills_context = skill_manager.get_system_context()
        system_prompt = (
            "You are Nova, the AI brain of a cute robot called Reachy Mini. "
            "You have a camera for eyes and can see the world. "
            "You can control your head and body to look in specific directions. "
            "You can also browse the web using Nova Act. "
            "You are connected to Slack and can read and send messages there. "
            "You have a memory system that stores knowledge about your user and the world. "
            "When you learn something new about the user, use memory to store it. "
            "When asked about something you might know, use memory to recall it. "
            "Keep your responses short, fun, and expressive. "
            "You love to help and are endlessly curious about the world around you. "
            "React with enthusiasm when you see something interesting through your camera."
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
            mood = body.get("mood", "happy").lower()
            if mood not in VALID_MOODS:
                return {"error": f"Unknown mood. Available: {sorted(VALID_MOODS)}"}
            update_state(mood=mood)
            return {"mood": mood}

        class TrackingToggle(BaseModel):
            enabled: bool

        @self.settings_app.post("/api/tracking/toggle")
        def toggle_tracking(body: TrackingToggle):
            update_state(tracking_enabled=body.enabled)
            if not body.enabled:
                update_state(tracking_mode="idle")
            return {"tracking_enabled": body.enabled}

        # --- Action disable controls (presentation mode) ---
        class ActionToggle(BaseModel):
            enabled: bool

        @self.settings_app.post("/api/speech/toggle")
        def toggle_speech(body: ActionToggle):
            update_state(speech_enabled=body.enabled)
            return {"speech_enabled": body.enabled}

        @self.settings_app.post("/api/movement/toggle")
        def toggle_movement(body: ActionToggle):
            update_state(movement_enabled=body.enabled)
            return {"movement_enabled": body.enabled}

        class PresentationMode(BaseModel):
            enabled: bool

        @self.settings_app.post("/api/presentation")
        def toggle_presentation(body: PresentationMode):
            """Convenience: disable speech + movement + antennas all at once."""
            if body.enabled:
                update_state(speech_enabled=False, movement_enabled=False, antenna_mode="off")
            else:
                update_state(speech_enabled=True, movement_enabled=True, antenna_mode="auto")
            return {"presentation_mode": body.enabled}

        # --- Memory API endpoints ---
        class MemoryQuery(BaseModel):
            query: str
            mode: str = "query"

        @self.settings_app.get("/api/memory/health")
        def memory_health():
            return memory.health()

        @self.settings_app.post("/api/memory/query")
        def memory_query(body: MemoryQuery):
            result = memory.query(body.query) if body.mode == "query" else memory.store(body.query)
            return {"result": result, "mode": body.mode}

        # --- Slack API endpoints ---
        class SlackMessage(BaseModel):
            channel: str | None = None
            text: str

        @self.settings_app.get("/api/slack/state")
        def get_slack_state():
            with state_lock:
                return {
                    "slack_state": app_state["slack_state"],
                    "slack_last_event": app_state["slack_last_event"],
                    "recent_count": len(slack_bot._recent_messages),
                }

        @self.settings_app.post("/api/slack/send")
        def send_slack_message(body: SlackMessage):
            channel = body.channel
            if not channel and slack_channel_ids:
                channel = slack_channel_ids[0]
            if not channel:
                return {"error": "No channel specified and no default configured"}
            slack_bot.queue_task("send_message", channel=channel, text=body.text)
            return {"status": "queued", "channel": channel}

        # --- Tracking manager with event-driven vision ---
        last_vision_event_time = 0.0
        VISION_EVENT_COOLDOWN = 3.0

        def on_tracking_event(event_type: str, data: dict):
            nonlocal last_vision_event_time
            publish_event("tracking", event_type, data)
            now = time.time()
            if now - last_vision_event_time < VISION_EVENT_COOLDOWN:
                return
            if event_type in ("person_detected", "snap_detected", "mode_changed"):
                last_vision_event_time = now
                logger.info(f"[Tracking event] {event_type} → triggering vision")
                vision.trigger_analyze()

        tracker = TrackingManager(on_event=on_tracking_event)
        last_doa_time = 0.0

        # --- Register MQTT inject handler (needs sonic, so done after sonic creation) ---
        if mqtt_client is not None:
            def _on_inject(client, userdata, msg):
                try:
                    data = json.loads(msg.payload.decode())
                    text = data.get("text", "")
                    if text:
                        sonic.inject_text(text)
                        logger.info(f"[Nervous System] Injected: {text[:80]}")
                except Exception as e:
                    logger.warning(f"Inject message error: {e}")

            mqtt_client.message_callback_add("nova/inject", _on_inject)
            logger.info("Registered nova/inject handler — Nervous System can drive voice")

        # --- Start services ---
        sonic.start(stop_event)
        vision.start(stop_event)
        browser.start(stop_event)
        slack_bot.start(stop_event)

        # --- Inject startup context from memory ---
        def _inject_startup_context():
            time.sleep(2)  # Wait for Sonic to be ready
            try:
                ctx = memory.get_startup_context()
                if ctx:
                    sonic.inject_text(
                        f"[Background knowledge about your user and world:\n{ctx}]\n"
                        "Use this knowledge naturally in conversation."
                    )
                    publish_event("memory", "startup_context", {"context": ctx})
                    logger.info(f"Injected startup context ({len(ctx)} chars)")
            except Exception as e:
                logger.warning(f"Startup context injection failed: {e}")

        threading.Thread(
            target=_inject_startup_context, daemon=True, name="memory-startup"
        ).start()

        # Start audio recording from robot mic
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

        # Safety manager for head-body collision avoidance + organic movement
        safety = SafetyManager()

        # Antenna blending state for smooth mood transitions
        prev_antennas = np.array([0.0, 0.0])
        prev_mood = "happy"
        mood_change_time = 0.0

        # Frozen position state for movement disable
        _frozen_head_pose = None
        _frozen_body_yaw = 0.0
        _frozen_antennas_rad = np.array([0.0, 0.0])
        _prev_movement_enabled = True

        # --- Main control loop ---
        while not stop_event.is_set():
            t = time.time() - t0

            # Get current mood for animation
            with state_lock:
                mood = app_state["mood"]
                voice_state = app_state["voice_state"]
                antenna_mode = app_state["antenna_mode"]
                vision_enabled = app_state["vision_enabled"]
                tracking_enabled = app_state["tracking_enabled"]
                speech_enabled = app_state["speech_enabled"]
                movement_enabled = app_state["movement_enabled"]

            # --- Head override or active tracking ---
            with state_lock:
                head_override = app_state["head_override"]
                head_override_time = app_state["head_override_time"]

            if head_override is not None:
                # Skill-driven head control with EMA smoothing
                if time.time() - head_override_time > 30.0:
                    # Safety timeout: auto-recover tracking
                    update_state(head_override=None, tracking_enabled=True)
                    yaw_deg, pitch_deg = tracker.get_head_target(t, voice_state, mood)
                else:
                    target_yaw = head_override["yaw"]
                    target_pitch = head_override["pitch"]
                    _smooth_yaw += 0.15 * (target_yaw - _smooth_yaw)
                    _smooth_pitch += 0.15 * (target_pitch - _smooth_pitch)
                    yaw_deg = _smooth_yaw
                    pitch_deg = _smooth_pitch
            elif tracking_enabled:
                # Update DoA (throttled to ~5Hz to avoid I2C bus contention)
                if t - last_doa_time > 0.2:
                    last_doa_time = t
                    try:
                        doa = reachy_mini.media.audio.get_DoA()
                        tracker.update_doa(doa)
                    except Exception:
                        pass

                # Get tracked head target (falls back to idle animation)
                yaw_deg, pitch_deg = tracker.get_head_target(t, voice_state, mood)
                update_state(tracking_mode=tracker.mode)
            else:
                # Original sinusoidal head animation
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

            # Apply safety validation (clamps + organic body follow)
            safe_yaw, safe_pitch, safe_body_yaw = safety.validate(
                yaw_deg, pitch_deg, _current_body_yaw
            )
            _current_body_yaw = safe_body_yaw

            head_pose = create_head_pose(yaw=safe_yaw, pitch=safe_pitch, degrees=True)

            # --- Antenna animation (eased, mood-driven) ---
            if antenna_mode == "off":
                antennas_deg = np.array([0.0, 0.0])
            else:
                # Detect mood change for blending
                if mood != prev_mood:
                    prev_mood = mood
                    mood_change_time = t

                profile = MOOD_ANTENNAS.get(mood, MOOD_ANTENNAS["happy"])
                ease_fn = profile["ease"]
                freq = profile["freq"]
                amp = profile["amp"]
                offset = profile.get("offset", 0.0)

                if profile["phase"] == "custom" and mood == "thinking":
                    # Asymmetric thinking pose
                    a1 = offset + amp * ease_fn(t, freq)
                    a2 = -10.0 + 5.0 * ease_fn(t, freq * 1.5)
                    target_antennas = np.array([a1, a2])
                elif profile["phase"] == "sync":
                    a = offset + amp * ease_fn(t, freq)
                    target_antennas = np.array([a, a])
                else:  # oppose
                    a = amp * ease_fn(t, freq)
                    target_antennas = np.array([offset + a, offset - a])

                # Smooth blend on mood transitions
                elapsed = t - mood_change_time
                if elapsed < MOOD_BLEND_TIME:
                    alpha = elapsed / MOOD_BLEND_TIME
                    # Smoothstep the blend factor
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    antennas_deg = prev_antennas * (1.0 - alpha) + target_antennas * alpha
                else:
                    antennas_deg = target_antennas

                prev_antennas = antennas_deg

            antennas_rad = np.deg2rad(antennas_deg)

            # --- Movement freeze logic ---
            if not movement_enabled:
                if _prev_movement_enabled:
                    # Transition: capture current pose as frozen
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
                    # Re-enabled: sync smoothing state to frozen values for smooth transition
                    _smooth_yaw = np.degrees(np.arctan2(
                        _frozen_head_pose[0][1], _frozen_head_pose[0][0]
                    )) if _frozen_head_pose is not None else _smooth_yaw
                    _prev_movement_enabled = True
                reachy_mini.set_target(
                    head=head_pose,
                    antennas=antennas_rad,
                    body_yaw=np.radians(safe_body_yaw),
                )

            # --- Feed mic audio to Nova Sonic ---
            try:
                audio = reachy_mini.media.get_audio_sample()
                if audio is not None:
                    # Convert to numpy float32 if needed
                    if isinstance(audio, bytes):
                        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                    elif audio.dtype != np.float32:
                        audio = audio.astype(np.float32)

                    # Mix to mono if multi-channel
                    if audio.ndim == 2:
                        audio = audio.mean(axis=1)

                    # Resample to 16kHz if needed
                    if mic_sr != 16000:
                        ratio = 16000 / mic_sr
                        n_out = int(len(audio) * ratio)
                        audio = np.interp(
                            np.linspace(0, len(audio) - 1, n_out),
                            np.arange(len(audio)),
                            audio,
                        ).astype(np.float32)

                    # Feed audio to snap detector only when idle/listening —
                    # during speaking/thinking, our own audio or processing
                    # triggers false snaps creating a feedback loop
                    if tracking_enabled and voice_state in ("idle", "listening"):
                        tracker.detect_snap(audio)

                    sonic.feed_audio(audio)
                    audio_chunk_count += 1
                    if audio_chunk_count % 500 == 1:
                        rms = np.sqrt(np.mean(audio ** 2))
                        logger.info(f"Audio chunks fed: {audio_chunk_count}, RMS: {rms:.4f}, sonic.state={sonic.state}")
            except Exception as e:
                logger.warning(f"Audio feed error: {e}")

            # --- Feed camera frames to Nova Vision and Tracking ---
            if vision_enabled:
                try:
                    frame = reachy_mini.media.get_frame()
                    if frame is not None:
                        vision.update_frame(frame)
                        if tracking_enabled:
                            tracker.update_vision(frame, t)
                except Exception:
                    pass

            # --- Push buffered audio output to speaker ---
            with audio_lock:
                chunks = list(audio_output_buffer)
                audio_output_buffer.clear()

            if not speech_enabled:
                chunks = []  # Discard audio — robot listens but doesn't speak aloud

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
        if mqtt_available and mqtt_client:
            try:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
                logger.info("MQTT disconnected")
            except Exception:
                pass
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
