"""Nova Sonic's tool registry over the reachy-mini-cli intents spool (task t9).

The wireless harness has no in-process robot: the body is driven by the
reachy-mini-cli behavior engine running on the Wireless, and the ONLY thing
between Nova's voice model and that engine is a directory of JSON files. This
module is that seam — a published set of ``toolConfiguration`` entries whose
handlers write a command into ``<state>/behavior/intents/commands/`` and read
the engine's answer back out of ``<state>/behavior/intents/results/``.

The one invariant worth stating plainly: **no tool call silently vanishes.**
Every :meth:`IntentTools.execute` returns a JSON payload the model can read, in
exactly one of three shapes —

``{"ok": true, ...}``
    the engine applied it (its own result dict, verbatim);
``{"ok": false, "error": "..."}``
    it was refused — either here (a pre-flight refusal, in which case nothing
    was written to the spool at all) or by the engine;
``{"ok": null, "submitted": "<cmd_id>", "note": ...}``
    degraded: the command IS on disk but no engine confirmed it inside the
    await timeout. The command file deliberately STAYS spooled, so an engine
    started a second later still applies it.

That third shape is why the pre-flight validation here matters. A model that
gets ``{"ok": null, "submitted": ...}`` back concludes it worked; a refusal it
cannot see is not a refusal, it is a silence the model reads as success and
repeats. So anything we can judge locally without duplicating an engine-side
number — an unknown tool, a malformed argument, a goto duration outside the
engine's own published ``(0, 10]`` bound — is refused BEFORE the spool write.

What is deliberately NOT re-validated here
------------------------------------------
* **Behavior/mode names.** The catalog lives in the peer package, which this
  package must never import (see ``tests/test_harness_boundary.py``). A wrong
  name comes back as an engine ``{"ok": false, ...}`` naming the valid ones.
* **Per-axis goto ranges** (head ±mm/±deg, antennas, body yaw). Those bounds
  are engine constants with cited precedent; a second copy here is a second
  number to drift. The engine refuses fail-closed and we surface its refusal.
  The DURATION bound is the exception — it is re-checked because it is the one
  bound that decides how long a runaway command holds the body, and 10 s is
  published as part of this harness's own wire contract.
* **Whether there is a face to enroll.** ``enroll_face`` names whichever face
  the runtime is currently looking at; only the runtime has the camera, the
  FaceStore and the recency window, so "is an unknown face in view?" comes back
  as its own typed refusal. The NAME is checked here, because that argument is
  the model's invention and it lands in a durable identity store.

Nothing here imports ``reachy_mini`` or the ``reachy`` (reachy-mini-cli)
package: file paths are the whole contract.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from reachy_nova import sensory_log
from reachy_nova.harness import statedir
from reachy_nova.harness.rules_overlay import RuleRefused, upsert_rule
from reachy_nova.nova_browser import act_enabled

# --------------------------------------------------------------------------- #
# Names + constants                                                           #
# --------------------------------------------------------------------------- #

RUN_BEHAVIOR = "run_behavior"
DECLARE_GOAL = "declare_goal"
SET_MODE = "set_mode"
SET_INHIBITION = "set_inhibition"
GOTO = "goto"
CREATE_RULE = "create_rule"
BROWSE = "browse"
ENROLL_FACE = "enroll_face"
LOCK_FACE = "lock_face"
RELEASE_FACE = "release_face"
FORGE = "forge"
USE_SKILL = "use_skill"
AUTHOR_RULE = "author_rule"
RAISE_VOICE = "raise_voice"
LOWER_VOICE = "lower_voice"
SET_VOICE_LEVEL = "set_voice_level"
RECALL_SENSES = "recall_senses"

#: The harness's ENTIRE action set, in publication order. Each addition past
#: the original six is a deliberate widening of the blast radius, never an
#: incidental one — ``browse`` (task t4) is the first: it drives
#: :class:`~reachy_nova.nova_browser.NovaBrowser`, not the intents spool, and
#: stays refused whenever ``NOVA_ACT_ENABLED`` is off. ``enroll_face`` (task t8)
#: is the second: it is spool-backed like the original five, but writes a
#: durable identity rather than a transient pose. ``recall_senses`` (task t8)
#: is the third: like the voice-level tools it is dispatched locally rather
#: than through the intents spool, reading the in-process
#: :class:`~reachy_nova.harness.sense_history.SenseHistory` ring buffer
#: instead of talking to the runtime at all. ``lock_face``/``release_face``
#: (task t9) are the fourth and fifth: spool-backed no-argument ops, same
#: shape as the original five, riding the runtime's own gaze-lock intent
#: kinds.
ACTION_SET: tuple[str, ...] = (
    RUN_BEHAVIOR,
    DECLARE_GOAL,
    SET_MODE,
    SET_INHIBITION,
    GOTO,
    CREATE_RULE,
    BROWSE,
    ENROLL_FACE,
    LOCK_FACE,
    RELEASE_FACE,
    FORGE,
    USE_SKILL,
    AUTHOR_RULE,
    RAISE_VOICE,
    LOWER_VOICE,
    SET_VOICE_LEVEL,
    RECALL_SENSES,
)

#: Seconds a tool call waits for the engine to confirm before degrading.
DEFAULT_AWAIT_TIMEOUT = 1.0

#: How often the result poll re-reads the results dir.
RESULT_POLL_S = 0.02

#: The note a degraded (submitted-but-unconfirmed) result carries.
DEGRADED_NOTE = "engine did not confirm in time — is the behavior engine running?"

#: The head axes a goto may target (the engine's own six, mm for x/y/z and
#: degrees for roll/pitch/yaw).
HEAD_AXES: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw")

#: A goto's duration must be > 0 and at most this many seconds — the engine's
#: published ceiling, re-checked here so a runaway duration is refused where
#: the model can read the refusal.
MAX_GOTO_DURATION_S = 10.0

#: Top-level fields a goto command may carry.
_GOTO_FIELDS = frozenset({"label", "head", "antennas", "body_yaw", "duration", "interpolation"})

#: The sensory-log stage every tool call lands under — one grep finds every
#: action the voice model took.
SENSE_STAGE = "act"
SENSE_SOURCE = "nova"

#: Refusal reason when ``browse`` is called with Nova Act off (the default).
#: Named so the model can tell "disabled" apart from a bad-argument refusal.
BROWSE_DISABLED_REASON = (
    "browsing is disabled — set NOVA_ACT_ENABLED=1 to enable Nova Act browser automation"
)

#: Refusal reason when ``browse`` is called before a :class:`NovaBrowser`
#: handle has been wired in (flag on, but ``IntentTools`` built without one).
BROWSE_NOT_WIRED_REASON = "browser automation is enabled but not wired up yet"

#: Refusal reason when ``forge``/``use_skill``/``author_rule`` are called
#: without a wired ForgeLeg (the kiro writer is opt-in: FORGE_WRITER=kiro).
FORGE_NOT_WIRED_REASON = (
    "the skill forge is disabled — set FORGE_WRITER=kiro to enable the on-device kiro writer"
)

#: The intents-spool op behind the ``enroll_face`` tool. The tool name and the
#: wire op deliberately differ: the tool is named for what Nova is doing
#: (learning a face's name), the op for the runtime's own registry entry
#: requested in agentculture/reachy-mini-cli#166.
ENROLL_OP = "enroll"

#: Which face the enroll command targets. ``"current"`` means "the face the
#: runtime is looking at right now" — resolving WHICH face that is belongs to
#: the runtime's FaceStore (it owns the temporary-face buffer and the
#: recency window), never to this harness, which has no camera of its own.
ENROLL_FACE_TARGET = "current"

#: Longest name we will pass through. A name is a name; anything past this is
#: either a mis-transcription of a whole sentence or someone trying to stuff
#: the identity store, and either way the runtime should not have to judge it.
MAX_ENROLL_NAME_LEN = 64

#: Refusal when ``enroll_face`` is called with no usable name — missing, not a
#: string, or nothing but whitespace.
ENROLL_NAME_INVALID_REASON = (
    "'name' must be the person's name as a non-empty string — ask them for it and try again"
)

#: Refusal when the "name" is far too long to be one.
ENROLL_NAME_TOO_LONG_REASON = (
    f"'name' must be at most {MAX_ENROLL_NAME_LEN} characters — that is a sentence, not a name"
)

#: Refusal when the name carries control characters (newlines, NULs). A name
#: reaches a durable on-disk identity store, so the one thing worth checking
#: locally is that it is text a person could actually be called.
ENROLL_NAME_UNPRINTABLE_REASON = "'name' must contain only printable characters"

#: Voice-level bounds — clamp, never refuse: a model asking to go louder/
#: quieter than the daemon supports should still get a usable ``ok: true``
#: at the nearest edge, with a note explaining why it stopped there.
MIN_VOICE_LEVEL = 10
MAX_VOICE_LEVEL = 100
DEFAULT_VOICE_STEP = 10

#: Refusal when ``raise_voice``/``lower_voice``/``set_voice_level`` are called
#: without a wired :class:`~reachy_nova.harness.daemon_client.DaemonClient`
#: (mirrors ``FORGE_NOT_WIRED_REASON``'s component-absent shape).
VOICE_NOT_WIRED_REASON = "voice level control is not wired up yet"

#: ``recall_senses`` (task t8) — bounds on how many recent senses a call may
#: ask for. Out-of-range values are CLAMPED, never refused: a model asking for
#: too much/too little history should still get a usable answer, same
#: philosophy as the voice-level tools' clamp-not-refuse.
MIN_RECALL_SENSES_N = 1
MAX_RECALL_SENSES_N = 20
DEFAULT_RECALL_SENSES_N = 5

#: Refusal when ``recall_senses`` is called without a wired
#: :class:`~reachy_nova.harness.sense_history.SenseHistory` (mirrors
#: ``VOICE_NOT_WIRED_REASON``'s component-absent shape).
HISTORY_NOT_WIRED_REASON = "sense history not wired"


class ToolRefused(ValueError):
    """A pre-flight refusal: nothing was written to the spool."""


# --------------------------------------------------------------------------- #
# Tool specs — Nova Sonic ``toolConfiguration`` shape                         #
# --------------------------------------------------------------------------- #


def _spec(name: str, description: str, schema: dict) -> dict:
    return {
        "toolSpec": {
            "name": name,
            "description": description,
            "inputSchema": {"json": json.dumps(schema)},
        }
    }


_HEAD_SCHEMA = {
    "type": "object",
    "description": (
        "Head pose offsets. x/y/z in millimetres, roll/pitch/yaw in degrees. "
        "Give only the axes you want to move."
    ),
    "properties": {axis: {"type": "number"} for axis in HEAD_AXES},
}


TOOL_SPECS: list[dict] = [
    _spec(
        RUN_BEHAVIOR,
        "Move the body once: run a named behaviour (nod, shake, antenna-sway, "
        "'look-at-sound' to turn toward the last sound you heard, 'look-at-face' "
        "to glance at the person in front of you, ...) for a bounded time — the "
        "gaze one-shots default to about 2 seconds. One-shot — the robot does it "
        "and stops. Use this for a single reaction right now.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Behaviour to run, e.g. 'nod', 'shake', 'antenna-sway'.",
                },
                "params": {
                    "type": "object",
                    "description": "Numeric tweaks for the behaviour, e.g. {\"amp\": 8}.",
                },
                "duration": {
                    "type": "number",
                    "description": "Seconds to run. Omit for the behaviour's own default.",
                },
                "loop": {
                    "type": "boolean",
                    "description": "Repeat for the whole duration instead of once.",
                },
            },
            "required": ["name"],
        },
    ),
    _spec(
        DECLARE_GOAL,
        "Keep the body doing something until told otherwise: the robot re-starts "
        "this behaviour by itself whenever it stops, indefinitely. Call with no "
        "goal to let the body settle back to normal.",
        {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Behaviour to sustain. Omit to clear the standing goal.",
                },
                "params": {
                    "type": "object",
                    "description": "Numeric tweaks for the behaviour.",
                },
            },
            "required": [],
        },
    ),
    _spec(
        SET_MODE,
        "Switch the robot's overall behaviour mode (how lively or calm its "
        "automatic reactions are). Call with no mode to go back to the default.",
        {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Mode name to activate. Omit to clear the override.",
                }
            },
            "required": [],
        },
    ),
    _spec(
        SET_INHIBITION,
        "Hold specific movements back — the listed behaviours stop and cannot "
        "start until you clear them. REPLACES the whole blocked list; pass an "
        "empty list to let everything move again.",
        {
            "type": "object",
            "properties": {
                "behaviors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Behaviour names to block; [] unblocks everything.",
                }
            },
            "required": ["behaviors"],
        },
    ),
    _spec(
        GOTO,
        "Move to an exact pose and hold it: point the head, set the antennas, or "
        "turn the body over a given number of seconds. Name at least one of "
        "head, antennas or body_yaw. Duration must be over 0 and at most "
        f"{MAX_GOTO_DURATION_S:g} seconds.",
        {
            "type": "object",
            "properties": {
                "head": _HEAD_SCHEMA,
                "antennas": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Antenna angles in degrees as [right, left].",
                },
                "body_yaw": {
                    "type": "number",
                    "description": "Body rotation in degrees (positive turns left).",
                },
                "duration": {
                    "type": "number",
                    "description": "Seconds for the move, over 0 and up to "
                    f"{MAX_GOTO_DURATION_S:g}.",
                },
                "interpolation": {
                    "type": "string",
                    "description": "Motion profile, e.g. 'minjerk' or 'linear'.",
                },
                "label": {
                    "type": "string",
                    "description": "Short name for this move, shown in the engine's status.",
                },
            },
            "required": [],
        },
    ),
    _spec(
        CREATE_RULE,
        "Teach the robot a lasting reflex: when a sense crosses a threshold, run "
        "a behaviour (and optionally say a line). The reflex keeps working after "
        "this conversation ends. Rule ids must start with 'nova-'.",
        {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique rule id; must start with 'nova-'. Reusing an id "
                    "replaces that rule.",
                },
                "when": {
                    "type": "object",
                    "description": "The trigger: which sense, how it is compared, to what.",
                    "properties": {
                        "field": {
                            "type": "string",
                            "description": "Sense to watch: doa, speech, rms, rms_ratio, pat, "
                            "face, frame_available, transcript, self_moving.",
                        },
                        "op": {
                            "type": "string",
                            "description": "Comparison: lt, gt, ge, le, eq, ne, is_true, "
                            "is_false, absent_for.",
                        },
                        "value": {
                            "description": "What to compare against. Omit for is_true/is_false."
                        },
                    },
                    "required": ["field", "op"],
                },
                "run": {
                    "type": "string",
                    "description": "Behaviour to run when the trigger fires.",
                },
                "params": {"type": "object", "description": "Numeric tweaks for the behaviour."},
                "duration_s": {
                    "type": "number",
                    "description": "Seconds the behaviour runs each time it fires.",
                },
                "cooldown_s": {
                    "type": "number",
                    "description": "Minimum seconds between two firings.",
                },
                "hysteresis": {
                    "type": "number",
                    "description": "Margin around the threshold that stops it flapping.",
                },
                "say": {
                    "type": "string",
                    "description": "Optional line to speak when it fires (max 500 characters).",
                },
            },
            "required": ["id", "when", "run"],
        },
    ),
    _spec(
        BROWSE,
        "Browse the web: give it a natural-language task and, optionally, a page "
        "to start from, and it works a browser to do it in the background while "
        "you keep talking. Only available when browsing is enabled.",
        {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Natural-language task for the browser to carry out.",
                },
                "url": {
                    "type": "string",
                    "description": "Optional starting page. Omit to start from the default.",
                },
            },
            "required": ["instruction"],
        },
    ),
    _spec(
        ENROLL_FACE,
        "Remember whose face you are looking at: when someone tells you their "
        "name while you can see them ('I'm Ori'), call this with that name and "
        "the robot stores their face under it, so it recognises and greets them "
        "next time. Only use it for the person in front of you right now.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The person's name, exactly as they said it.",
                }
            },
            "required": ["name"],
        },
    ),
    _spec(
        LOCK_FACE,
        "Keep looking at the person you are facing until told to stop: locks "
        "your gaze onto their face so you keep tracking them as they move, "
        "even past a single glance. Call release_face to stop.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _spec(
        RELEASE_FACE,
        "Stop following a locked face and look away: releases a gaze lock "
        "started with lock_face, so you go back to normal look-around "
        "behavior.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _spec(
        FORGE,
        "Teach yourself a new skill: describe what the skill should do and a "
        "coder agent writes it in the background. You will be told when it is "
        "staged or rejected, and once activated you call it with use_skill. "
        "Only available when the kiro writer is configured.",
        {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What the new skill should do, in plain language.",
                },
                "improve": {
                    "type": "string",
                    "description": "Optional: name of an existing forged skill to improve.",
                },
            },
            "required": ["goal"],
        },
    ),
    _spec(
        USE_SKILL,
        "Run one of the skills you forged earlier, by name. If you are unsure "
        "what exists, call it with an empty or wrong name and the error lists "
        "the available skills.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The activated skill's name.",
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the skill, if it takes any.",
                },
            },
            "required": ["name"],
        },
    ),
    _spec(
        AUTHOR_RULE,
        "Ask your coder agent to write a lasting behavior rule from a "
        "plain-language goal — unlike create_rule, you describe WHAT you "
        "want and the writer designs the rule; returns the engine's "
        "verdict.",
        {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What the rule should do, in plain language.",
                },
            },
            "required": ["goal"],
        },
    ),
    _spec(
        RAISE_VOICE,
        "Speak a bit louder. Confirm briefly and naturally afterward — "
        "do not read out the volume number.",
        {
            "type": "object",
            "properties": {
                "step": {
                    "type": "number",
                    "description": f"How much louder, {MIN_VOICE_LEVEL}-{MAX_VOICE_LEVEL} "
                    f"scale. Omit for the default step ({DEFAULT_VOICE_STEP}).",
                },
            },
            "required": [],
        },
    ),
    _spec(
        LOWER_VOICE,
        "Speak a bit quieter. Confirm briefly and naturally afterward — "
        "do not read out the volume number.",
        {
            "type": "object",
            "properties": {
                "step": {
                    "type": "number",
                    "description": f"How much quieter, {MIN_VOICE_LEVEL}-{MAX_VOICE_LEVEL} "
                    f"scale. Omit for the default step ({DEFAULT_VOICE_STEP}).",
                },
            },
            "required": [],
        },
    ),
    _spec(
        SET_VOICE_LEVEL,
        "Set your speaking volume to an exact level. Confirm briefly and "
        "naturally afterward — do not read out the volume number.",
        {
            "type": "object",
            "properties": {
                "volume": {
                    "type": "number",
                    "description": f"Target volume, {MIN_VOICE_LEVEL}-{MAX_VOICE_LEVEL}.",
                },
            },
            "required": ["volume"],
        },
    ),
    _spec(
        RECALL_SENSES,
        "Recall what you actually just sensed and did: use it when someone "
        "asks why you moved, what you felt, or what just happened, and answer "
        "from the actual entries in your own words — never describe internal "
        "mechanisms like tools, injects, or rules.",
        {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": f"How many recent senses to recall, "
                    f"{MIN_RECALL_SENSES_N}-{MAX_RECALL_SENSES_N}. Omit for the "
                    f"default ({DEFAULT_RECALL_SENSES_N}).",
                },
            },
            "required": [],
        },
    ),
]


# --------------------------------------------------------------------------- #
# Argument validation helpers — every one refuses BEFORE any spool write      #
# --------------------------------------------------------------------------- #


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolRefused(f"{label} must be a number (got {value!r})")
    return float(value)


def _params(raw: object) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ToolRefused("'params' must be an object of name: number pairs")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ToolRefused(f"'params' keys must be strings (got {key!r})")
        out[key] = _number(value, f"params.{key}")
    return out


def _behavior_name(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ToolRefused(f"{label} must be a non-empty behaviour name (got {raw!r})")
    return raw


def _build_run_behavior(args: Mapping) -> dict:
    """``run_behavior`` — a one-time, bounded admission.

    The ``lifetime`` table carries ONLY the keys the caller actually gave. That
    is deliberate: the engine resolves a missing ``looping``/``duration`` key
    against the behaviour's own library default, so inventing ``looping: false``
    here would turn every unqualified call into a one-shot with no duration —
    which the engine then (correctly) refuses. The unbounded combination
    (looping with no duration) stays an engine-side refusal for the same
    reason: only the engine knows the entry's defaults.
    """
    name = _behavior_name(args.get("name"), "'name'")
    lifetime: dict[str, object] = {}
    if "loop" in args and args["loop"] is not None:
        if not isinstance(args["loop"], bool):
            raise ToolRefused(f"'loop' must be true or false (got {args['loop']!r})")
        lifetime["looping"] = args["loop"]
    if "duration" in args and args["duration"] is not None:
        duration = _number(args["duration"], "'duration'")
        if duration <= 0:
            raise ToolRefused(f"'duration' must be > 0 (got {duration!r})")
        lifetime["duration"] = duration
    return {
        "op": RUN_BEHAVIOR,
        "name": name,
        "params": _params(args.get("params")),
        "lifetime": lifetime,
    }


def _build_declare_goal(args: Mapping) -> dict:
    goal = args.get("goal")
    if goal is None:
        return {"op": DECLARE_GOAL, "goal": None, "params": {}}
    goal = _behavior_name(goal, "'goal'")
    return {"op": DECLARE_GOAL, "goal": goal, "params": _params(args.get("params"))}


def _build_set_mode(args: Mapping) -> dict:
    mode = args.get("mode")
    if mode is not None and (not isinstance(mode, str) or not mode.strip()):
        raise ToolRefused(f"'mode' must be a mode name or omitted (got {mode!r})")
    return {"op": SET_MODE, "mode": mode}


def _build_set_inhibition(args: Mapping) -> dict:
    raw = args.get("behaviors")
    if not isinstance(raw, list):
        raise ToolRefused("'behaviors' must be a list of behaviour names ([] clears them all)")
    behaviors = [_behavior_name(item, "each entry of 'behaviors'") for item in raw]
    return {"op": SET_INHIBITION, "behaviors": behaviors}


def _goto_head(raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ToolRefused(f"'head' must be an object of axis: value pairs (got {raw!r})")
    unknown = sorted(set(raw) - set(HEAD_AXES))
    if unknown:
        raise ToolRefused(f"unknown head axis/axes {unknown}; allowed: {list(HEAD_AXES)}")
    return {axis: _number(value, f"head.{axis}") for axis, value in raw.items()}


def _goto_antennas(raw: object) -> list[float]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 2:
        raise ToolRefused(f"'antennas' must be a [right, left] pair in degrees (got {raw!r})")
    return [_number(raw[0], "antennas.right"), _number(raw[1], "antennas.left")]


def _build_goto(args: Mapping) -> dict:
    unknown = sorted(set(args) - _GOTO_FIELDS)
    if unknown:
        raise ToolRefused(f"unknown goto field(s) {unknown}; allowed: {sorted(_GOTO_FIELDS)}")
    payload: dict[str, object] = {"op": GOTO}
    if args.get("label") is not None:
        if not isinstance(args["label"], str):
            raise ToolRefused(f"'label' must be a string (got {args['label']!r})")
        payload["label"] = args["label"]
    named_channel = False
    if args.get("head") is not None:
        payload["head"] = _goto_head(args["head"])
        named_channel = True
    if args.get("antennas") is not None:
        payload["antennas"] = _goto_antennas(args["antennas"])
        named_channel = True
    if args.get("body_yaw") is not None:
        payload["body_yaw"] = _number(args["body_yaw"], "'body_yaw'")
        named_channel = True
    if not named_channel:
        raise ToolRefused(
            "a goto must name at least one channel to move: head, antennas or body_yaw"
        )
    if args.get("duration") is not None:
        duration = _number(args["duration"], "'duration'")
        if duration <= 0 or duration > MAX_GOTO_DURATION_S:
            raise ToolRefused(
                f"'duration' must be over 0 and at most {MAX_GOTO_DURATION_S:g} seconds "
                f"(got {duration!r})"
            )
        payload["duration"] = duration
    if args.get("interpolation") is not None:
        if not isinstance(args["interpolation"], str):
            raise ToolRefused(f"'interpolation' must be a string (got {args['interpolation']!r})")
        payload["interpolation"] = args["interpolation"]
    return payload


def _build_enroll_face(args: Mapping) -> dict:
    """``enroll_face`` — name the face the runtime is currently looking at.

    Everything that makes this hard lives in the runtime: which face is
    "current", whether one was seen recently enough, whether the vision extra is
    even installed. Those come back as the runtime's own typed refusals
    (``no-recent-unknown-face``, ``vision-unavailable``, ...) and are surfaced
    verbatim — re-deciding any of them here would be guessing at a camera this
    process cannot see.

    What IS judged locally is the one argument the model invented: the name it
    heard. A name goes into a DURABLE identity store, so a mis-transcribed
    sentence or a control-character-laden string is refused before the spool
    write, where the model can read the refusal and simply ask again.
    """
    raw = args.get("name")
    if not isinstance(raw, str):
        raise ToolRefused(ENROLL_NAME_INVALID_REASON)
    name = raw.strip()
    if not name:
        raise ToolRefused(ENROLL_NAME_INVALID_REASON)
    if len(name) > MAX_ENROLL_NAME_LEN:
        raise ToolRefused(ENROLL_NAME_TOO_LONG_REASON)
    if not name.isprintable():
        raise ToolRefused(ENROLL_NAME_UNPRINTABLE_REASON)
    return {"op": ENROLL_OP, "name": name, "face": ENROLL_FACE_TARGET}


def _build_lock_face(args: Mapping) -> dict:  # noqa: ARG001 - no arguments to read
    """``lock_face`` — keep the gaze on whoever the runtime already knows."""
    return {"op": LOCK_FACE}


def _build_release_face(args: Mapping) -> dict:  # noqa: ARG001 - no arguments to read
    """``release_face`` — stop a standing gaze lock."""
    return {"op": RELEASE_FACE}


_BUILDERS = {
    RUN_BEHAVIOR: _build_run_behavior,
    DECLARE_GOAL: _build_declare_goal,
    SET_MODE: _build_set_mode,
    SET_INHIBITION: _build_set_inhibition,
    GOTO: _build_goto,
    ENROLL_FACE: _build_enroll_face,
    LOCK_FACE: _build_lock_face,
    RELEASE_FACE: _build_release_face,
}

#: Dispatched locally against the daemon client, never through the intents
#: spool — there is no behavior engine on the other end of a volume change.
VOICE_TOOLS: frozenset[str] = frozenset({RAISE_VOICE, LOWER_VOICE, SET_VOICE_LEVEL})


# --------------------------------------------------------------------------- #
# The registry                                                                #
# --------------------------------------------------------------------------- #


class IntentTools:
    """Submit / await / execute against the intents spool.

    ``commands_dir`` and ``results_dir`` default to
    :func:`reachy_nova.harness.statedir.intents_commands_dir` /
    :func:`~reachy_nova.harness.statedir.intents_results_dir`, resolved LAZILY
    on each call so a state dir chosen after construction (a test's
    ``REACHY_STATE_DIR``, an operator's re-export) is honoured.
    """

    def __init__(
        self,
        commands_dir: Path | str | None = None,
        results_dir: Path | str | None = None,
        await_timeout: float = DEFAULT_AWAIT_TIMEOUT,
        browser: object | None = None,
        on_browse_progress: Callable[[str], None] | None = None,
        forge_leg: object | None = None,
        daemon_client: object | None = None,
        history: object | None = None,
    ) -> None:
        # ``forge``/``use_skill`` (deviation d1) drive a ForgeLeg handle the
        # same way ``browse`` drives a NovaBrowser: injected, and refused with
        # a named reason when absent rather than half-working.
        self._forge_leg = forge_leg
        # ``raise_voice``/``lower_voice``/``set_voice_level`` (task t10) drive
        # a DaemonClient handle directly — not spool-backed, refused with a
        # named reason when absent, same shape as the browser/forge legs.
        self._daemon_client = daemon_client
        # ``recall_senses`` (task t8) reads a SenseHistory ring buffer
        # directly — same absent-component shape again, never half-working.
        self._history = history
        self._commands_dir = Path(commands_dir) if commands_dir is not None else None
        self._results_dir = Path(results_dir) if results_dir is not None else None
        self.await_timeout = float(await_timeout)
        # ``browse`` (task t4) drives a NovaBrowser handle directly rather than
        # the intents spool — that handle is injected here (tests pass a fake;
        # production wiring, when act_enabled(), happens in app.py). Wiring the
        # progress callback here — rather than on every ``browse`` call — keeps
        # it a one-time hookup, matching how NovaBrowser's other callbacks are
        # set once at construction.
        self._browser = browser
        self._on_browse_progress = on_browse_progress
        if browser is not None and on_browse_progress is not None:
            browser.on_progress = on_browse_progress

    # -- paths ------------------------------------------------------------- #

    def commands_dir(self) -> Path:
        d = self._commands_dir or statedir.intents_commands_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def results_dir(self) -> Path:
        d = self._results_dir or statedir.intents_results_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- the spool --------------------------------------------------------- #

    def submit(self, op_payload: Mapping) -> str:
        """Drop one command file into the spool; return its ``cmd_id``.

        Written to a temp name in the SAME directory and ``os.replace``-d in, so
        the engine's drain can never read a half-written command. The
        ``<time.time_ns()>-<cmd_id>.json`` name keeps a ``sorted()`` drain in
        submission order.
        """
        cmd_id = uuid.uuid4().hex
        payload = {"cmd_id": cmd_id, **dict(op_payload)}
        target = self.commands_dir() / f"{time.time_ns()}-{cmd_id}.json"
        _atomic_write(target, json.dumps(payload))
        return cmd_id

    def await_result(self, cmd_id: str, timeout: float | None = None) -> dict | None:
        """Poll for the engine's result for *cmd_id*; consume it; ``None`` on timeout."""
        deadline = time.monotonic() + (self.await_timeout if timeout is None else timeout)
        path = self.results_dir() / f"{cmd_id}.json"
        while True:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = None
            if isinstance(data, dict):
                _safe_unlink(path)
                return data
            if time.monotonic() >= deadline:
                return None
            time.sleep(RESULT_POLL_S)

    def submit_and_await(self, op_payload: Mapping) -> dict:
        """Submit, then wait — degrading to the submitted-only payload on timeout."""
        cmd_id = self.submit(op_payload)
        result = self.await_result(cmd_id)
        if result is None:
            return {"ok": None, "submitted": cmd_id, "note": DEGRADED_NOTE}
        return result

    # -- the tool entry point ---------------------------------------------- #

    def execute(self, tool_name: str, params: dict) -> str:
        """Run one tool call; return the JSON result payload for the model.

        Never raises and never returns nothing: an unknown tool, a malformed
        argument, an engine refusal, a silent engine — each has its own visible
        payload, and each emits exactly one ``[SENSE stage=act source=nova]``
        line naming the outcome.
        """
        try:
            payload = self._dispatch(tool_name, params)
        except ToolRefused as err:
            payload = {"ok": False, "error": str(err)}
        except RuleRefused as err:
            payload = {"ok": False, "error": str(err)}
        except Exception as err:  # pragma: no cover - defensive: never crash the voice loop
            payload = {"ok": False, "error": f"{type(err).__name__}: {err}"}
        # Voice tools log their own single ``event=volume`` senselog line
        # (old=N new=M) inside ``_voice_tool`` — the generic per-tool-name
        # line here would be a second line for the same call.
        if tool_name not in VOICE_TOOLS:
            _log_outcome(tool_name, payload)
        return json.dumps(payload)

    def _dispatch(self, tool_name: str, params: dict) -> dict:
        if tool_name not in ACTION_SET:
            raise ToolRefused(
                f"unknown tool {tool_name!r}; available: {', '.join(ACTION_SET)}"
            )
        if not isinstance(params, Mapping):
            raise ToolRefused(f"arguments for {tool_name!r} must be an object (got {params!r})")
        if tool_name == CREATE_RULE:
            return self._create_rule(params)
        if tool_name == BROWSE:
            return self._browse(params)
        if tool_name in (FORGE, USE_SKILL, AUTHOR_RULE):
            return self._forge_tool(tool_name, params)
        if tool_name in VOICE_TOOLS:
            return self._voice_tool(tool_name, params)
        if tool_name == RECALL_SENSES:
            return self._recall_senses(params)
        return self.submit_and_await(_BUILDERS[tool_name](params))

    def _forge_tool(self, tool_name: str, params: Mapping) -> dict:
        """``forge``/``use_skill``/``author_rule`` — delegate to the injected ForgeLeg."""
        if self._forge_leg is None:
            raise ToolRefused(FORGE_NOT_WIRED_REASON)
        if tool_name == FORGE:
            goal = params.get("goal")
            if not isinstance(goal, str) or not goal.strip():
                raise ToolRefused("'goal' must be a non-empty string")
            improve = params.get("improve")
            if improve is not None and not isinstance(improve, str):
                raise ToolRefused("'improve' must be a string when given")
            return self._forge_leg.forge(goal, improve=improve)
        if tool_name == AUTHOR_RULE:
            goal = params.get("goal")
            if not isinstance(goal, str) or not goal.strip():
                raise ToolRefused("'goal' must be a non-empty string")
            return self._forge_leg.author_rule(goal)
        name = params.get("name")
        if not isinstance(name, str):
            raise ToolRefused("'name' must be a string")
        raw_params = params.get("params")
        if raw_params is not None and not isinstance(raw_params, Mapping):
            raise ToolRefused("'params' must be an object when given")
        return self._forge_leg.use_skill(name, dict(raw_params or {}))

    def _create_rule(self, params: Mapping) -> dict:
        """Author a nova-namespaced reflex in the rules overlay, then reload.

        The overlay is NOT the intents spool: a rule is durable configuration,
        so it goes through :mod:`reachy_nova.harness.rules_overlay` (validate a
        candidate file, ``os.replace`` it in, submit a reload) and the reload
        verdict is reported here — a rules file the engine refused to reload
        means the OLD rules are still the live ones, which the model must see
        as ``ok: false``.
        """
        changed, verdict = upsert_rule(dict(params), reload_timeout=self.await_timeout)
        result = {
            "ok": True,
            "rule_id": params.get("id"),
            "changed": changed,
            "reload": verdict,
        }
        if verdict.startswith("reload rejected"):
            result["ok"] = False
            result["error"] = f"{verdict} — the previously loaded rules stay active"
        return result

    def _browse(self, params: Mapping) -> dict:
        """``browse`` — hand a natural-language task to :class:`NovaBrowser`.

        Not spool-backed: unlike the spool-backed tools this drives a component
        living in THIS process (Nova's own browser automation), not the
        Wireless behavior engine, so there is no command file to write.

        Refused — before ever touching ``self._browser`` — when
        :func:`~reachy_nova.nova_browser.act_enabled` is off (the default; see
        module docstring) or when no browser handle was wired in. Both are
        pre-flight :class:`ToolRefused`, so neither imports ``nova_act`` or
        ``playwright``: that stays entirely inside ``NovaBrowser`` itself, on
        its own enabled-only code paths.
        """
        instruction = params.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ToolRefused(f"'instruction' must be a non-empty string (got {instruction!r})")
        url = params.get("url")
        if url is not None and not isinstance(url, str):
            raise ToolRefused(f"'url' must be a string or omitted (got {url!r})")
        if not act_enabled():
            raise ToolRefused(BROWSE_DISABLED_REASON)
        if self._browser is None:
            raise ToolRefused(BROWSE_NOT_WIRED_REASON)
        self._browser.queue_task(instruction, url)
        return {"ok": True, "queued": True, "instruction": instruction, "url": url}

    # -- voice level (task t10) --------------------------------------------

    def _voice_tool(self, tool_name: str, params: Mapping) -> dict:
        """``raise_voice``/``lower_voice``/``set_voice_level`` — one senselog line.

        Unlike the spool-backed tools, this logs itself (event ``volume``,
        ``old=N new=M``) rather than going through the generic per-tool-name
        line in :func:`_log_outcome` — ``execute`` skips that call for
        members of :data:`VOICE_TOOLS` so exactly one line is ever emitted,
        success or failure.
        """
        old: int | None = None
        new: int | None = None
        try:
            payload, old, new = self._voice_apply(tool_name, params)
        except ToolRefused as err:
            payload = {"ok": False, "error": str(err)}
        if payload.get("ok") is True:
            detail = f"old={old} new={new}"
        else:
            detail = f"refused reason={payload.get('error')}"
        sensory_log.stage(SENSE_STAGE, SENSE_SOURCE, "volume", detail)
        return payload

    def _voice_apply(self, tool_name: str, params: Mapping) -> tuple[dict, int, int]:
        """Compute + apply the new level; raises :class:`ToolRefused` on any failure."""
        if self._daemon_client is None:
            raise ToolRefused(VOICE_NOT_WIRED_REASON)
        try:
            current = int(self._daemon_client.get_volume())
        except Exception as err:  # noqa: BLE001 - any client failure is a refusal
            raise ToolRefused(f"could not read current volume: {err}") from err

        if tool_name == SET_VOICE_LEVEL:
            target = _number(params.get("volume"), "'volume'")
        else:
            raw_step = params.get("step")
            step = DEFAULT_VOICE_STEP if raw_step is None else _number(raw_step, "'step'")
            target = current + step if tool_name == RAISE_VOICE else current - step

        clamped = int(round(max(MIN_VOICE_LEVEL, min(MAX_VOICE_LEVEL, target))))
        note = None
        if target > MAX_VOICE_LEVEL:
            note = "at maximum"
        elif target < MIN_VOICE_LEVEL:
            note = "at minimum"

        if clamped == current:
            applied = current
        else:
            try:
                applied = int(self._daemon_client.set_volume(clamped))
            except Exception as err:  # noqa: BLE001 - any client failure is a refusal
                raise ToolRefused(f"could not set volume: {err}") from err

        payload: dict = {"ok": True, "volume": applied}
        if note:
            payload["note"] = note
        return payload, current, applied

    # -- recall_senses (task t8) -------------------------------------------

    def _recall_senses(self, params: Mapping) -> dict:
        """``recall_senses`` — read back the actual recent sense history.

        Refused (before ever touching ``self._history``) when no
        :class:`~reachy_nova.harness.sense_history.SenseHistory` was wired at
        construction. ``n`` is CLAMPED into ``[MIN_RECALL_SENSES_N,
        MAX_RECALL_SENSES_N]`` rather than refused — same clamp-not-refuse
        philosophy as the voice-level tools, since the model asking for too
        much/too little history should still get a usable answer.
        """
        if self._history is None:
            raise ToolRefused(HISTORY_NOT_WIRED_REASON)
        raw_n = params.get("n")
        if raw_n is None:
            n = DEFAULT_RECALL_SENSES_N
        else:
            n = int(_number(raw_n, "'n'"))
        clamped = max(MIN_RECALL_SENSES_N, min(MAX_RECALL_SENSES_N, n))
        return {"ok": True, "senses": self._history.recent(clamped)}


# --------------------------------------------------------------------------- #
# Small file + logging helpers                                                #
# --------------------------------------------------------------------------- #


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _log_outcome(tool_name: str, payload: Mapping) -> None:
    ok = payload.get("ok")
    if ok is None and "submitted" in payload:
        detail = f"degraded submitted={payload['submitted']} reason={DEGRADED_NOTE}"
    elif ok is True:
        detail = f"confirmed {_compact(payload)}"
    elif ok is False:
        reason = payload.get("error") or payload.get("reason") or _compact(payload)
        detail = f"refused reason={reason}"
    else:  # pragma: no cover - an engine result with no 'ok' key at all
        detail = f"submitted result={_compact(payload)}"
    sensory_log.stage(SENSE_STAGE, SENSE_SOURCE, tool_name, detail)


def _compact(payload: Mapping) -> str:
    try:
        return json.dumps(payload, separators=(",", ":"))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(payload)
