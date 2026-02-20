"""REST API endpoints for the Reachy Nova web dashboard.

Registers all HTTP endpoints that the browser UI at localhost:8042
calls to read state and send commands.
"""

import time

from pydantic import BaseModel

VALID_MOODS = {
    "happy", "excited", "curious", "thinking",
    "sad", "disappointed", "surprised", "sleepy", "proud", "calm",
}


# --- Pydantic request models ---

class VisionToggle(BaseModel):
    enabled: bool

class BrowserTask(BaseModel):
    instruction: str
    url: str | None = None

class AntennaMode(BaseModel):
    mode: str

class EmotionEvent(BaseModel):
    event: str
    intensity: float = 1.0

class TrackingToggle(BaseModel):
    enabled: bool

class ActionToggle(BaseModel):
    enabled: bool

class PresentationMode(BaseModel):
    enabled: bool

class SleepAction(BaseModel):
    action: str

class MemoryQuery(BaseModel):
    query: str
    mode: str = "query"

class SlackMessage(BaseModel):
    channel: str | None = None
    text: str


def register_routes(app, ctx) -> None:
    """Attach all API routes to the FastAPI instance."""

    @app.get("/api/state")
    def get_state():
        snapshot = ctx.state.snapshot()
        snapshot["uptime"] = time.time() - ctx.t0
        return snapshot

    @app.post("/api/vision/toggle")
    def toggle_vision(body: VisionToggle):
        ctx.state.update(vision_enabled=body.enabled)
        return {"vision_enabled": body.enabled}

    @app.post("/api/vision/analyze")
    def trigger_vision():
        ctx.vision.trigger_analyze()
        ctx.state.update(vision_analyzing=True)
        return {"status": "analyzing"}

    @app.post("/api/browser/task")
    def submit_browser_task(body: BrowserTask):
        ctx.browser.queue_task(body.instruction, body.url)
        ctx.state.update(browser_task=body.instruction)
        return {"status": "queued", "instruction": body.instruction}

    @app.post("/api/antenna/mode")
    def set_antenna_mode(body: AntennaMode):
        ctx.state.update(antenna_mode=body.mode)
        return {"antenna_mode": body.mode}

    @app.post("/api/mood")
    def set_mood(body: dict):
        mood = body.get("mood", "happy").lower()
        if mood not in VALID_MOODS:
            return {"error": f"Unknown mood. Available: {sorted(VALID_MOODS)}"}
        ctx.emotional_state.set_mood_override(mood, duration=10.0)
        return {"mood": mood}

    @app.get("/api/emotions")
    def get_emotions():
        return ctx.emotional_state.get_full_state()

    @app.post("/api/emotions/event")
    def apply_emotion_event(body: EmotionEvent):
        if body.event not in ctx.emotional_state.get_event_names():
            return {"error": f"Unknown event. Available: {ctx.emotional_state.get_event_names()}"}
        ctx.emotional_state.apply_event(body.event, body.intensity)
        return {"status": "applied", "event": body.event, "state": ctx.emotional_state.get_full_state()}

    @app.post("/api/emotions/reset")
    def reset_emotions():
        ctx.emotional_state.reload_config()
        return {"status": "config reloaded"}

    @app.post("/api/tracking/toggle")
    def toggle_tracking(body: TrackingToggle):
        ctx.state.update(tracking_enabled=body.enabled)
        if not body.enabled:
            ctx.state.update(tracking_mode="idle")
        return {"tracking_enabled": body.enabled}

    @app.post("/api/speech/toggle")
    def toggle_speech(body: ActionToggle):
        ctx.state.update(speech_enabled=body.enabled)
        return {"speech_enabled": body.enabled}

    @app.post("/api/movement/toggle")
    def toggle_movement(body: ActionToggle):
        ctx.state.update(movement_enabled=body.enabled)
        return {"movement_enabled": body.enabled}

    @app.post("/api/presentation")
    def toggle_presentation(body: PresentationMode):
        if body.enabled:
            ctx.state.update(speech_enabled=False, movement_enabled=False, antenna_mode="off")
        else:
            ctx.state.update(speech_enabled=True, movement_enabled=True, antenna_mode="auto")
        return {"presentation_mode": body.enabled}

    @app.post("/api/sleep")
    def sleep_control(body: SleepAction):
        if body.action == "sleep":
            ctx.sleep_orchestrator.initiate_sleep()
            return {"status": "sleeping", "sleep_mode": ctx.sleep_orchestrator.sleep_manager.state}
        elif body.action == "wake":
            ctx.sleep_orchestrator.initiate_wake()
            return {"status": "waking", "sleep_mode": ctx.sleep_orchestrator.sleep_manager.state}
        return {"error": f"Unknown action: {body.action}"}

    @app.get("/api/sleep/state")
    def get_sleep_state():
        return {"sleep_mode": ctx.sleep_orchestrator.sleep_manager.state}

    @app.get("/api/memory/health")
    def memory_health():
        return ctx.memory.health()

    @app.post("/api/memory/query")
    def memory_query(body: MemoryQuery):
        result = ctx.memory.query(body.query) if body.mode == "query" else ctx.memory.store(body.query)
        return {"result": result, "mode": body.mode}

    @app.get("/api/feedback/stats")
    def feedback_stats():
        return ctx.feedback.get_stats()

    @app.get("/api/session")
    def get_session():
        return ctx.session.get_session_info()

    @app.get("/api/slack/state")
    def get_slack_state():
        return {
            "slack_state": ctx.state.get("slack_state"),
            "slack_last_event": ctx.state.get("slack_last_event"),
            "recent_count": len(ctx.slack_bot._recent_messages),
        }

    @app.post("/api/slack/send")
    def send_slack_message(body: SlackMessage):
        channel = body.channel
        if not channel and ctx.slack_bot.channel_ids:
            channel = next(iter(ctx.slack_bot.channel_ids))
        if not channel:
            return {"error": "No channel specified and no default configured"}
        ctx.slack_bot.queue_task("send_message", channel=channel, text=body.text)
        return {"status": "queued", "channel": channel}
