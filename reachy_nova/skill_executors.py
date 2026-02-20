"""Tool-callable skill implementations for Reachy Nova.

Registers all 13 skill executors with the SkillManager.
Each executor is a module-level function taking (params, ctx).
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# All moods available to the system
VALID_MOODS = {
    "happy", "excited", "curious", "thinking",
    "sad", "disappointed", "surprised", "sleepy", "proud", "calm",
}


def _look_executor(params: dict, ctx) -> str:
    query = params.get("query", "What do you see?")
    return ctx.vision.analyze_latest(query)


def _mood_executor(params: dict, ctx) -> str:
    event = params.get("event")
    if event:
        if event in ctx.emotional_state.get_event_names():
            ctx.emotional_state.apply_event(event)
            return f"[Emotion event '{event}' applied]"
        return f"[Unknown event '{event}'. Available: {', '.join(ctx.emotional_state.get_event_names())}]"
    mood = params.get("mood", "happy").lower()
    if mood not in VALID_MOODS:
        return f"[Unknown mood '{mood}'. Available: {', '.join(sorted(VALID_MOODS))}]"
    ctx.emotional_state.set_mood_override(mood, duration=10.0)
    return f"[Mood override set to {mood} for 10s]"


def _browse_executor(params: dict, ctx) -> str:
    query = params.get("query", "")
    url = params.get("url", "https://www.google.com")
    ctx.state.update(browser_task=query)
    return ctx.browser.execute(query, url)


def _memory_executor(params: dict, ctx) -> str:
    query = params.get("query", "")
    mode = params.get("mode", "query")
    if mode == "store":
        return ctx.memory.store(query)
    elif mode == "context":
        return ctx.memory.get_startup_context()
    else:
        return ctx.memory.query(query)


def _remember_positively_executor(params: dict, ctx) -> str:
    what = params.get("what", "")
    trigger = params.get("trigger", "")
    return ctx.feedback.record(sentiment="positive", what=what, trigger=trigger)


def _remember_negatively_executor(params: dict, ctx) -> str:
    what = params.get("what", "")
    trigger = params.get("trigger", "")
    return ctx.feedback.record(sentiment="negative", what=what, trigger=trigger)


def _slack_executor(params: dict, ctx) -> str:
    return ctx.slack_bot.execute(params)


def _control_executor(params: dict, ctx) -> str:
    action = params.get("action", "")

    if action == "move_head":
        yaw = max(-45, min(45, float(params.get("yaw", 0))))
        pitch = max(-15, min(25, float(params.get("pitch", 0))))
        ctx.state.update(
            tracking_enabled=False,
            tracking_mode="idle",
            head_override={"yaw": yaw, "pitch": pitch},
            head_override_time=time.time(),
        )
        return f"[Moving head to yaw={yaw:.0f}, pitch={pitch:.0f}. Tracking disabled.]"

    elif action == "move_body":
        body_yaw = max(-25, min(25, float(params.get("body_yaw", 0))))
        duration = max(0.5, min(5, float(params.get("duration", 1.5))))
        start = ctx.state.get("body_yaw")
        steps = int(duration / 0.02)
        for i in range(steps + 1):
            alpha = 0.5 * (1 - np.cos(np.pi * i / steps))
            current = start + alpha * (body_yaw - start)
            ctx.reachy_mini.set_target(body_yaw=np.radians(current))
            time.sleep(0.02)
        ctx.state.update(body_yaw=body_yaw)
        return f"[Body rotated to {body_yaw:.0f} over {duration:.1f}s.]"

    elif action == "enable_tracking":
        ctx.tracker.current_yaw = ctx.state.get("smooth_yaw")
        ctx.tracker.current_pitch = ctx.state.get("smooth_pitch")
        ctx.state.update(tracking_enabled=True, head_override=None)
        return "[Tracking re-enabled.]"

    elif action == "disable_tracking":
        ctx.state.update(tracking_enabled=False, tracking_mode="idle")
        return "[Tracking disabled.]"

    return f"[Unknown control action: {action}]"


def _gesture_executor(params: dict, ctx) -> str:
    gesture = params.get("gesture", "").lower()
    return ctx.gesture_engine.execute(gesture)


def _face_executor(params: dict, ctx) -> str:
    op = params.get("operation", "")
    name = params.get("name", "")
    uid = params.get("unique_id", "")
    target_id = params.get("target_id", "")
    target_name = params.get("target_name", "")

    admin_ops = {"list", "count", "images", "whois"}
    if op in admin_ops:
        if not ctx.face_recognition.is_admin_authenticated():
            return "[Admin operation only. Admin must be visible to the camera.]"

    if op == "remember":
        embedding = ctx.face_recognition.get_current_embedding()
        if embedding is None:
            return "[No face detected in camera. Please look at me.]"
        temp_id = ctx.face_manager.remember_temporary(embedding)
        return f"[Face remembered temporarily (15 min). Temp ID: {temp_id}]"

    elif op == "consent":
        temp_id = uid
        if not temp_id and ctx.face_manager.temp_faces:
            temp_id = list(ctx.face_manager.temp_faces.keys())[-1]
        if not temp_id:
            return "[No temporary face to save. Use 'remember' first.]"
        if not name:
            return "[Full name required for permanent storage.]"
        result = ctx.face_manager.consent(temp_id, name)
        if result:
            ctx.state.update(face_count=ctx.face_manager.get_face_count())
            return f"[Face saved permanently as {name}. ID: {result}]"
        return "[Failed to save face. Temp ID may be expired or invalid.]"

    elif op == "forget":
        if not uid or not name:
            return "[Both unique_id and name required to forget a face.]"
        if ctx.face_manager.forget(uid, name):
            ctx.state.update(face_count=ctx.face_manager.get_face_count())
            return f"[Face {uid} ({name}) deleted.]"
        return "[Failed to delete. Check ID/name or admin protection.]"

    elif op == "add_angles":
        if not uid or not name:
            return "[Both unique_id and name required.]"
        embedding = ctx.face_recognition.get_current_embedding()
        if embedding is None:
            return "[No face detected in camera. Please look at me.]"
        if ctx.face_manager.add_angles(uid, name, embedding):
            return f"[Added new angle for {name}. Recognition should improve.]"
        return "[Failed to add angle. Check ID and name.]"

    elif op == "merge":
        if not uid or not name or not target_id or not target_name:
            return "[Need unique_id, name, target_id, and target_name to merge.]"
        if ctx.face_manager.merge(uid, name, target_id, target_name):
            ctx.state.update(face_count=ctx.face_manager.get_face_count())
            return f"[Merged {target_id} into {uid}. {target_name} merged with {name}.]"
        return "[Merge failed. Check IDs and names.]"

    elif op == "list":
        faces = ctx.face_manager.list_faces()
        if not faces:
            return "[No faces stored.]"
        lines = [f"- {f['id']}: {f['name']} ({f['num_embeddings']} embeddings)"
                 + (" [ADMIN]" if f['is_admin'] else "")
                 for f in faces]
        return "[Known faces:\n" + "\n".join(lines) + "]"

    elif op == "count":
        count = ctx.face_manager.get_face_count()
        return f"[{count} permanent face(s) stored.]"

    elif op == "images":
        if not uid or not name:
            return "[Both unique_id and name required.]"
        paths = ctx.face_manager.get_person_images(uid, name)
        if not paths:
            return "[No images found for that person.]"
        return f"[{len(paths)} embedding file(s) for {name}.]"

    elif op == "whois":
        if not name:
            return "[Name required for whois lookup.]"
        found_id = ctx.face_manager.get_unique_id(name)
        if found_id:
            return f"[{name} has ID: {found_id}]"
        return f"[No face found for '{name}'.]"

    return f"[Unknown face operation: {op}]"


def register_all(skill_manager, ctx) -> None:
    """Register all skill executors with their JSON schemas."""

    skill_manager.register_executor(
        "look",
        lambda params: _look_executor(params, ctx),
    )

    skill_manager.register_executor(
        "mood",
        lambda params: _mood_executor(params, ctx),
        input_schema={
            "type": "object",
            "properties": {
                "mood": {
                    "type": "string",
                    "description": "Direct mood override (temporary, 10s). Use 'event' for natural emotion changes.",
                    "enum": sorted(VALID_MOODS),
                },
                "event": {
                    "type": "string",
                    "description": "Emotion event to apply (preferred over mood override)",
                    "enum": ctx.emotional_state.get_event_names(),
                },
            },
        },
    )

    skill_manager.register_executor(
        "browse",
        lambda params: _browse_executor(params, ctx),
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

    skill_manager.register_executor(
        "memory",
        lambda params: _memory_executor(params, ctx),
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

    skill_manager.register_executor(
        "remember_positively",
        lambda params: _remember_positively_executor(params, ctx),
        input_schema={
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "description": "What you did that was liked",
                },
                "trigger": {
                    "type": "string",
                    "description": "What triggered the feedback (e.g. head scratch, verbal praise)",
                },
            },
            "required": ["what"],
        },
    )

    skill_manager.register_executor(
        "remember_negatively",
        lambda params: _remember_negatively_executor(params, ctx),
        input_schema={
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "description": "What you did that was disliked",
                },
                "trigger": {
                    "type": "string",
                    "description": "What triggered the feedback (e.g. verbal correction, told to stop)",
                },
            },
            "required": ["what"],
        },
    )

    skill_manager.register_executor(
        "slack",
        lambda params: _slack_executor(params, ctx),
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

    skill_manager.register_executor(
        "control",
        lambda params: _control_executor(params, ctx),
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

    skill_manager.register_executor(
        "gesture",
        lambda params: _gesture_executor(params, ctx),
        input_schema={
            "type": "object",
            "properties": {
                "gesture": {
                    "type": "string",
                    "description": "The gesture to perform",
                    "enum": ["yes", "no", "curious", "pondering", "boredom", "nuzzle", "purr", "enjoy"],
                },
            },
            "required": ["gesture"],
        },
    )

    skill_manager.register_executor(
        "face",
        lambda params: _face_executor(params, ctx),
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The face operation to perform",
                    "enum": [
                        "remember", "consent", "forget",
                        "add_angles", "merge",
                        "list", "count", "images", "whois",
                    ],
                },
                "name": {
                    "type": "string",
                    "description": "Full name of the person",
                },
                "unique_id": {
                    "type": "string",
                    "description": "Face ID (or temp_id for consent)",
                },
                "target_id": {
                    "type": "string",
                    "description": "Second face ID for merge",
                },
                "target_name": {
                    "type": "string",
                    "description": "Second face name for merge",
                },
            },
            "required": ["operation"],
        },
    )
