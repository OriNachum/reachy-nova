"""Turn a natural-language goal into a landed behavior rule (task t7).

A "Kiro writer" — an on-device coding agent driven over ACP
(:mod:`reachy_nova.kiro_acp`) — can be handed a goal in English ("nod when
someone pats my head") and asked to author the ``[[react]]`` rule that
implements it. This module is the one place that turns that conversational
turn into a durable rule: it builds an authoring prompt that restates
:mod:`reachy_nova.harness.rules_overlay`'s schema, sends it to a session,
parses exactly one fenced JSON object out of the reply, and lands it through
``rules_overlay.upsert_rule`` — the ONLY function in this codebase allowed to
write the operator's ``rules.toml``.

This module never re-implements that validation or write logic; it is a thin
goal -> prompt -> parse -> ``upsert_rule`` pipeline, and every stage failure
(a session that raises, a reply with no fence, unparseable JSON, a rule the
overlay's own validator refuses, or a REJECTED reload) comes back as a
structured result — see :func:`author_rule` — never an uncaught exception.

Result contract
----------------
:func:`author_rule` always returns a dict with exactly these keys::

    {"ok": bool, "rule_id": str | None, "verdict": str | None, "reason": str | None}

``ok`` is False for every failure mode above, INCLUDING a REJECTED reload —
the reload handshake happening at all does not make authoring a success. When
``ok`` is False, ``reason`` is a human-readable explanation (and IS the
verdict string, verbatim, when the failure is a rejected reload — so a caller
never has to inspect both fields to find out why). ``verdict`` carries
whatever :func:`rules_overlay.upsert_rule` reported once the rule made it
that far (``"reload confirmed: ..."``, ``"reload rejected: ..."``, ``"reload
submitted (...) but not confirmed ..."``, or ``"unchanged — ..."``) and is
``None`` for any failure before that call (a session error, a missing fence,
bad JSON, or a rule the overlay's validator refused). ``rule_id`` is the
``"id"`` field of the parsed rule object where one was present, even on a
later failure, so a caller can still say which rule was attempted.

stdlib + :mod:`reachy_nova.harness.rules_overlay` + :mod:`reachy_nova.sensory_log`
only. Never imports ``reachy_mini`` and never references ``set_target`` — see
``tests/test_harness_boundary.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from reachy_nova import sensory_log
from reachy_nova.harness import rules_overlay

# --------------------------------------------------------------------------- #
# Senselog identity                                                           #
# --------------------------------------------------------------------------- #

#: ``[SENSE stage=rules source=kiro event=...]`` — every line this module emits.
STAGE = "rules"
SOURCE = "kiro"

#: How long to wait for the Kiro session to finish one authoring turn.
DEFAULT_PROMPT_TIMEOUT = 90.0


class PromptSession(Protocol):
    """The shape this module needs from a session: :meth:`prompt` only."""

    def prompt(self, text: str, timeout: float = ...) -> str: ...


# --------------------------------------------------------------------------- #
# The authoring prompt                                                        #
# --------------------------------------------------------------------------- #


def build_prompt(goal: str) -> str:
    """The exact text sent to the session — restates rules_overlay's schema.

    Restating (never importing the engine's own schema, since rules_overlay
    itself never imports it either — see that module's docstring) is the cost
    of staying inside the fail-closed design: whatever comes back is still
    re-validated by ``upsert_rule`` before anything is written, so a stale or
    imprecise restatement here can only ever produce a REFUSED rule, never a
    bad one landing on disk.
    """
    sense_fields = ", ".join(sorted(rules_overlay.SENSE_FIELDS))
    ordered_ops = ", ".join(sorted(rules_overlay.ORDERED_OPS))
    equality_ops = ", ".join(sorted(rules_overlay.EQUALITY_OPS))
    boolean_ops = ", ".join(sorted(rules_overlay.BOOLEAN_OPS))
    duration_ops = ", ".join(sorted(rules_overlay.DURATION_OPS))
    return (
        "You are authoring exactly ONE behavior rule for a running robot's "
        "rules engine, from this goal:\n"
        f"    {goal}\n"
        "\n"
        "Reply with EXACTLY one fenced JSON object — a single ```json ... ``` "
        "code fence — and nothing else: no prose before, between, or after "
        "it, and no second fence.\n"
        "\n"
        "The object is one [[react]] rule with these fields:\n"
        f'  - "id": string, REQUIRED, must start with {rules_overlay.RULE_ID_PREFIX!r} '
        '(e.g. "nova-greet-on-face"). This namespace is what keeps a '
        "nova-authored rule enumerable and removable as a set, and keeps it "
        "from ever colliding with an operator's own rule.\n"
        '  - "when": REQUIRED, an object {"field": ..., "op": ..., "value": ...} '
        "testing exactly ONE sense-snapshot field:\n"
        f"      field, one of: {sense_fields}\n"
        "      op, one of:\n"
        f"        numeric comparators (require a numeric 'value'): {ordered_ops}\n"
        f"        equality comparators (require a 'value', any scalar): {equality_ops}\n"
        f"        boolean-presence comparators (take NO 'value' at all): {boolean_ops}\n"
        f"        duration comparator (require a numeric 'value' in seconds): {duration_ops}\n"
        '  - "run": string, REQUIRED, the behaviour name to run.\n'
        '  - "params": OPTIONAL object of {name: number} pairs passed to the behaviour.\n'
        '  - "duration_s": number > 0 — how long the behaviour runs. OPTIONAL\n'
        "    for one-shot behaviours, but REQUIRED for looping behaviours with\n"
        "    no default duration ('speak' is one): the engine REJECTS a\n"
        "    looping rule without duration_s because it would hold its channel\n"
        "    forever. When in doubt, include a small duration_s (2-5 seconds).\n"
        '  - "cooldown_s": OPTIONAL number >= 0 — minimum time between firings.\n'
        '  - "hysteresis": OPTIONAL number >= 0.\n'
        f'  - "say": OPTIONAL string, at most {rules_overlay.MAX_SAY_CHARS} characters — '
        "a line spoken when the rule fires. It is REFUSED, never truncated, "
        "if it is longer than that.\n"
        "\n"
        "No other fields are allowed anywhere in the object. Output nothing "
        "but the single fenced JSON object — no explanation, no markdown "
        "outside the fence."
    )


# --------------------------------------------------------------------------- #
# Locating and parsing the fenced object — liberal find, strict parse         #
# --------------------------------------------------------------------------- #

#: Any fenced code block, labeled or not (mirrors skill_forge.py's fence regex).
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)\n?```", re.DOTALL)

#: Fence labels that count as an explicit claim "this is the rule".
_PREFERRED_LABELS = ("json", "rule", "rule.json")


def _extract_fenced_object(reply: str) -> str | None:
    """The text of the one fenced object in *reply*, or ``None`` if there is none.

    Liberal in locating the fence: any fenced block counts, a ``json``/``rule``
    label is preferred over an unlabeled or oddly-labeled one, and a reply
    with stray prose around the fence is fine — only the fenced body is
    returned. Strict from here on: what comes back is handed to ``json.loads``
    unmodified, so a value that merely LOOKS like an object still has to
    parse as one.
    """
    fences = _FENCE_RE.findall(reply)
    if not fences:
        return None
    for label, body in fences:
        if label.strip().lower() in _PREFERRED_LABELS:
            stripped = body.strip()
            if stripped:
                return stripped
    for _label, body in fences:
        stripped = body.strip()
        if stripped:
            return stripped
    return None


def _rule_id_of(rule: Any) -> str | None:
    if isinstance(rule, dict):
        rule_id = rule.get("id")
        if isinstance(rule_id, str):
            return rule_id
    return None


# --------------------------------------------------------------------------- #
# The result contract                                                        #
# --------------------------------------------------------------------------- #


def _result(
    ok: bool,
    *,
    rule_id: str | None = None,
    verdict: str | None = None,
    reason: str | None = None,
) -> dict:
    return {"ok": ok, "rule_id": rule_id, "verdict": verdict, "reason": reason}


def _log(event: str, detail: str) -> None:
    sensory_log.stage(STAGE, SOURCE, event, detail)


# --------------------------------------------------------------------------- #
# The public entry point                                                     #
# --------------------------------------------------------------------------- #


def author_rule(
    goal: str,
    session: PromptSession,
    overlay: Any = rules_overlay,
    *,
    path: Path | str | None = None,
    reload_timeout: float = rules_overlay.DEFAULT_RELOAD_TIMEOUT,
    prompt_timeout: float = DEFAULT_PROMPT_TIMEOUT,
) -> dict:
    """Goal -> authoring prompt -> ``session.prompt`` -> parse -> ``overlay.upsert_rule``.

    Every failure mode returns a structured result (see the module docstring
    for the exact contract) rather than raising:

    - *session* raising (a dead process, a timeout, an ACP error) — caught
      around the ``session.prompt`` call specifically.
    - no fenced object findable in the reply.
    - a fenced object that is not valid JSON, or not a JSON object.
    - *overlay* refusing the parsed rule (schema violation, ``say`` over the
      cap, an id outside the ``nova-`` namespace, ...).
    - a REJECTED reload verdict — the rule reaches the overlay and is written
      to disk, but the running engine refused to pick it up. This is still
      reported as ``ok=False``; the rule having landed on disk does not make
      the reload having failed a success.

    *overlay* defaults to :mod:`reachy_nova.harness.rules_overlay` itself and
    is only a parameter so tests can substitute a stand-in with the same
    ``upsert_rule(rule, *, path=..., reload_timeout=...) -> (changed, verdict)``
    surface — this module never re-implements or duplicates its schema or its
    write logic.
    """
    prompt_text = build_prompt(goal)
    _log("prompt", f"goal={goal!r} chars={len(prompt_text)}")

    try:
        reply = session.prompt(prompt_text, timeout=prompt_timeout)
    except Exception as err:  # noqa: BLE001 - a dead/misbehaving session must not crash us
        reason = f"kiro session.prompt() failed: {err}"
        _log("error", reason)
        return _result(False, reason=reason)

    fenced = _extract_fenced_object(reply)
    if fenced is None:
        reason = "no fenced rule object found in kiro's reply"
        _log("error", f"{reason}: reply={reply[:200]!r}")
        return _result(False, reason=reason)

    try:
        rule = json.loads(fenced)
    except json.JSONDecodeError as err:
        reason = f"fenced content is not valid JSON: {err}"
        _log("error", f"{reason}: fenced={fenced[:200]!r}")
        return _result(False, reason=reason)

    if not isinstance(rule, dict):
        reason = f"fenced content is not a JSON object (got {type(rule).__name__})"
        _log("error", reason)
        return _result(False, reason=reason)

    rule_id = _rule_id_of(rule)
    _log("parsed", f"rule_id={rule_id!r}")

    try:
        changed, verdict = overlay.upsert_rule(rule, path=path, reload_timeout=reload_timeout)
    except Exception as err:  # noqa: BLE001 - the overlay's own refusal, never re-raised
        reason = f"rules overlay refused the rule: {err}"
        _log("error", reason)
        return _result(False, rule_id=rule_id, reason=reason)

    _log("landed", f"rule_id={rule_id!r} changed={changed}")
    _log("verdict", verdict)

    if "rejected" in verdict.lower():
        return _result(False, rule_id=rule_id, verdict=verdict, reason=verdict)
    return _result(True, rule_id=rule_id, verdict=verdict)
