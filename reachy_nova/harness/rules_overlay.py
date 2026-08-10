"""The nova-managed block of the behavior rules overlay (task t9).

``create_rule`` is the one tool whose effect OUTLIVES the conversation: a rule
is durable configuration, read by the behavior engine from
``<state>/behavior/rules.toml`` — the OPERATOR's file, which the engine also
lets a human hand-edit. So this module never owns that file; it owns a
sentinel-delimited block inside it:

    # --- nova managed rules BEGIN ---
    [[react]]
    ...
    # --- nova managed rules END ---

Everything before ``BEGIN`` and after ``END`` is the operator's and is carried
through VERBATIM on every write (the only byte this module will ever add to
operator text is a single trailing newline where one is structurally required,
and that is a fixed point — a second write adds nothing). Inside the block,
rules merge BY ID, so re-authoring the same reflex replaces it instead of
stacking a duplicate.

Fail-closed, in three stages
----------------------------
1. The rule dict itself is validated against the engine's declarative schema
   (fields, sense vocabulary, comparator classes, the 500-character ``say``
   cap) BEFORE any file is touched. A bad rule therefore cannot even create an
   overlay, let alone damage one.
2. The full candidate file — operator bytes included — is written to a sibling
   temp file, parsed with :mod:`tomllib` and re-validated. Only then is it
   ``os.replace``-d into place, so a rules file the engine would reject never
   exists at the real path even momentarily, and a concurrent reader sees the
   old file or the new one, never a torn one.
3. A reload command is submitted to ``<state>/behavior/reload/commands/`` and
   its verdict awaited. A REJECTED reload means the engine kept the rules it
   already had — the new file is on disk but is not what is running — so the
   verdict is reported to the caller rather than swallowed.

The schema mirrored here is the engine's, restated (not imported — this package
must never import the peer): ``[[react]]`` rules with a single ``when``
predicate over one sense field. Restating it is the cost of the boundary; the
tests pin the vocabulary so a drift is a visible diff rather than a silent
mismatch, and any rule this validator accepts but the engine does not still
comes back as a REJECTED reload, which is reported, not assumed away.
"""

from __future__ import annotations

import json
import os
import time
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from reachy_nova.harness import statedir

# --------------------------------------------------------------------------- #
# The contract's fixed vocabulary                                             #
# --------------------------------------------------------------------------- #

#: Sentinels bracketing the block this module owns.
MANAGED_BEGIN = "# --- nova managed rules BEGIN ---"
MANAGED_END = "# --- nova managed rules END ---"

#: Required prefix on every rule id nova authors. Nova rules PERSIST after the
#: harness stops, so the prefix is what keeps them enumerable and removable as
#: a set — and what makes a collision with an operator-authored rule impossible.
RULE_ID_PREFIX = "nova-"

#: Hard ceiling on a rule's spoken line. Fail-closed: refused, never truncated.
MAX_SAY_CHARS = 500

#: The sense-snapshot fields a predicate may test.
SENSE_FIELDS: frozenset[str] = frozenset(
    {
        "doa",
        "speech",
        "rms",
        "rms_ratio",
        "pat",
        "face",
        "frame_available",
        "transcript",
        "self_moving",
    }
)

#: Ordered numeric comparators — require a numeric ``value``.
ORDERED_OPS: frozenset[str] = frozenset({"lt", "gt", "ge", "le"})
#: Equality comparators — require a ``value`` (any JSON scalar).
EQUALITY_OPS: frozenset[str] = frozenset({"eq", "ne"})
#: Boolean-presence comparators — take NO ``value``.
BOOLEAN_OPS: frozenset[str] = frozenset({"is_true", "is_false"})
#: "Absent for at least N seconds" — a duration op, numeric ``value``.
DURATION_OPS: frozenset[str] = frozenset({"absent_for"})
COMPARATORS: frozenset[str] = ORDERED_OPS | EQUALITY_OPS | BOOLEAN_OPS | DURATION_OPS

_PREDICATE_FIELDS = frozenset({"field", "op", "value"})
_REACT_FIELDS = frozenset(
    {"id", "enabled", "when", "run", "params", "cooldown_s", "hysteresis", "duration_s", "say"}
)
_REACT_REQUIRED = ("id", "when", "run")
_INHIBIT_FIELDS = frozenset({"id", "enabled", "when", "disable", "cooldown_s", "hysteresis"})
_INHIBIT_REQUIRED = ("id", "when", "disable")
_TOP_LEVEL_FIELDS = frozenset({"active_mode", "react", "inhibit", "modes"})

#: Rendered key order — the shipped rules files read in this order, so a hand
#: inspection of the managed block looks like the file it lives in.
RULE_KEY_ORDER: tuple[str, ...] = (
    "id",
    "when",
    "run",
    "params",
    "duration_s",
    "cooldown_s",
    "hysteresis",
    "say",
)

#: Seconds to wait for the engine's reload verdict before degrading.
DEFAULT_RELOAD_TIMEOUT = 1.0
RESULT_POLL_S = 0.02


class RuleRefused(ValueError):
    """A rule (or the candidate file it would produce) was refused, fail-closed."""


# --------------------------------------------------------------------------- #
# Rule validation — runs before any file is touched                           #
# --------------------------------------------------------------------------- #


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuleRefused(f"{label} must be a number (got {value!r})")
    return float(value)


def _validate_predicate(raw: object, path: str) -> dict:
    if not isinstance(raw, Mapping):
        raise RuleRefused(f"{path}.when must be an object of field/op/value (got {raw!r})")
    unknown = sorted(set(raw) - _PREDICATE_FIELDS)
    if unknown:
        raise RuleRefused(
            f"{path}.when has unexpected key(s) {unknown}; allowed: "
            f"{sorted(_PREDICATE_FIELDS)}"
        )
    field = raw.get("field")
    if not isinstance(field, str) or field not in SENSE_FIELDS:
        raise RuleRefused(
            f"{path}.when.field is unknown (got {field!r}); use one of: "
            f"{', '.join(sorted(SENSE_FIELDS))}"
        )
    op = raw.get("op")
    if not isinstance(op, str) or op not in COMPARATORS:
        raise RuleRefused(
            f"{path}.when.op is unknown (got {op!r}); use one of: "
            f"{', '.join(sorted(COMPARATORS))}"
        )
    out: dict[str, object] = {"field": field, "op": op}
    has_value = "value" in raw and raw["value"] is not None
    if op in BOOLEAN_OPS:
        if has_value:
            raise RuleRefused(f"{path}.when: op {op!r} takes no 'value'")
        return out
    if not has_value:
        raise RuleRefused(f"{path}.when: op {op!r} requires a 'value'")
    value = raw["value"]
    if op in ORDERED_OPS or op in DURATION_OPS:
        out["value"] = _number(value, f"{path}.when.value")
    elif isinstance(value, (str, bool, int, float)):
        out["value"] = value
    else:
        raise RuleRefused(f"{path}.when.value must be a string, number or boolean (got {value!r})")
    return out


def _validate_say(raw: object, path: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise RuleRefused(f"{path}.say must be a non-empty string, or be left out entirely")
    if len(raw) > MAX_SAY_CHARS:
        raise RuleRefused(
            f"{path}.say is {len(raw)} characters, over the {MAX_SAY_CHARS}-character limit "
            "— shorten it (it is never truncated for you)"
        )
    return raw


def _validate_params(raw: object, path: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise RuleRefused(f"{path}.params must be an object of name: number pairs (got {raw!r})")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise RuleRefused(f"{path}.params keys must be strings (got {key!r})")
        out[key] = _number(value, f"{path}.params.{key}")
    return out


def _validate_id(raw: object, path: str, *, require_prefix: bool) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise RuleRefused(f"{path}.id must be a non-empty string (got {raw!r})")
    if require_prefix and not raw.startswith(RULE_ID_PREFIX):
        raise RuleRefused(
            f"rule id {raw!r} must start with {RULE_ID_PREFIX!r} — nova-authored rules "
            "outlive the session, so they are namespaced to stay enumerable and to never "
            "collide with an operator's own rule"
        )
    return raw


def validate_rule(raw: object, *, kind: str = "react", require_prefix: bool = True) -> dict:
    """Validate one rule dict; return the normalized entry. Raises :class:`RuleRefused`.

    ``require_prefix`` is True for rules NOVA authors and False when re-checking
    an operator's own rules inside a candidate file — the ``nova-`` namespace is
    a rule about what this harness writes, not a rule about what the engine may
    load.
    """
    if not isinstance(raw, Mapping):
        raise RuleRefused(f"a rule must be an object (got {raw!r})")
    allowed = _REACT_FIELDS if kind == "react" else _INHIBIT_FIELDS
    required = _REACT_REQUIRED if kind == "react" else _INHIBIT_REQUIRED
    path = f"[[{kind}]]"
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise RuleRefused(
            f"{path} has unexpected field(s) {unexpected}; allowed: {sorted(allowed)}"
        )
    missing = [key for key in required if raw.get(key) is None]
    if missing:
        raise RuleRefused(f"{path} is missing required field(s) {missing}")

    entry: dict[str, object] = {"id": _validate_id(raw["id"], path, require_prefix=require_prefix)}
    entry["when"] = _validate_predicate(raw["when"], path)
    if kind == "react":
        run = raw["run"]
        if not isinstance(run, str) or not run.strip():
            raise RuleRefused(f"{path}.run must be a non-empty behaviour name (got {run!r})")
        entry["run"] = run
        if raw.get("params") is not None:
            entry["params"] = _validate_params(raw["params"], path)
        if raw.get("duration_s") is not None:
            duration = _number(raw["duration_s"], f"{path}.duration_s")
            if duration <= 0:
                raise RuleRefused(f"{path}.duration_s must be > 0 (got {duration!r})")
            entry["duration_s"] = duration
        if raw.get("say") is not None:
            entry["say"] = _validate_say(raw["say"], path)
    else:
        disable = raw["disable"]
        if not isinstance(disable, Sequence) or isinstance(disable, (str, bytes)):
            raise RuleRefused(
                f"{path}.disable must be a list of behaviour names (got {disable!r})"
            )
        for item in disable:
            if not isinstance(item, str) or not item.strip():
                raise RuleRefused(f"{path}.disable entries must be behaviour names (got {item!r})")
        entry["disable"] = list(disable)
    for key in ("cooldown_s", "hysteresis"):
        if raw.get(key) is not None:
            value = _number(raw[key], f"{path}.{key}")
            if value < 0:
                raise RuleRefused(f"{path}.{key} must be >= 0 (got {value!r})")
            entry[key] = value
    if raw.get("enabled") is not None:
        if not isinstance(raw["enabled"], bool):
            raise RuleRefused(f"{path}.enabled must be true or false (got {raw['enabled']!r})")
        entry["enabled"] = raw["enabled"]
    return entry


def validate_rules_document(data: object) -> None:
    """Validate a whole parsed rules file — the candidate gate before ``os.replace``."""
    if not isinstance(data, Mapping):
        raise RuleRefused("a rules file must be a TOML table")
    unexpected = sorted(set(data) - _TOP_LEVEL_FIELDS)
    if unexpected:
        raise RuleRefused(
            f"rules file has unexpected top-level field(s) {unexpected}; allowed: "
            f"{sorted(_TOP_LEVEL_FIELDS)}"
        )
    if data.get("active_mode") is not None and not isinstance(data["active_mode"], str):
        raise RuleRefused("rules file 'active_mode' must be a string")
    seen: set[str] = set()
    for kind in ("react", "inhibit"):
        entries = data.get(kind, [])
        if not isinstance(entries, list):
            raise RuleRefused(f"rules file '{kind}' must be an array of tables")
        for entry in entries:
            checked = validate_rule(entry, kind=kind, require_prefix=False)
            rule_id = str(checked["id"])
            if rule_id in seen:
                raise RuleRefused(f"duplicate rule id {rule_id!r} in the rules file")
            seen.add(rule_id)


# --------------------------------------------------------------------------- #
# TOML rendering — a narrow, fixed shape, so no writer dependency             #
# --------------------------------------------------------------------------- #


def _toml_scalar(value: object, path: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        # JSON string escapes are valid TOML basic-string escapes.
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(v, f"{path}[]") for v in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(_toml_pairs(value, path)) + " }"
    raise RuleRefused(f"{path}: a {type(value).__name__} cannot be written to a rules file")


def _toml_pairs(table: Mapping, path: str) -> list[str]:
    pairs: list[str] = []
    for key, value in table.items():
        if not isinstance(key, str):
            raise RuleRefused(f"{path}: keys must be strings (got {key!r})")
        if value is None:
            continue  # an absent field, not a null one
        pairs.append(f"{key} = {_toml_scalar(value, f'{path}.{key}')}")
    return pairs


def render_rule(entry: Mapping) -> str:
    ordered = [k for k in RULE_KEY_ORDER if k in entry]
    ordered += sorted(k for k in entry if k not in RULE_KEY_ORDER)
    lines = ["[[react]]"]
    for key in ordered:
        lines.extend(_toml_pairs({key: entry[key]}, "react"))
    return "\n".join(lines)


def _render_managed(entries: list[dict]) -> str:
    """The managed block, with no leading/trailing newline of its own."""
    body = "\n\n".join(render_rule(entry) for entry in entries)
    if body:
        return f"{MANAGED_BEGIN}\n{body}\n{MANAGED_END}"
    return f"{MANAGED_BEGIN}\n{MANAGED_END}"


# --------------------------------------------------------------------------- #
# Splitting / joining the operator's file                                     #
# --------------------------------------------------------------------------- #


def _split_overlay(text: str) -> tuple[str, str, str]:
    """``(operator_head, managed_body, operator_tail)`` — head/tail verbatim.

    A file with no sentinels is all head. A file with a BEGIN and no END (a
    half-written block that somehow survived) is head + managed, so the next
    write repairs it rather than nesting a second block.
    """
    if MANAGED_BEGIN not in text:
        return text, "", ""
    head, _, rest = text.partition(MANAGED_BEGIN)
    managed, sentinel, tail = rest.partition(MANAGED_END)
    if not sentinel:
        return head, managed, ""
    return head, managed, tail


def _managed_entries(managed: str) -> list[dict]:
    """The react entries currently inside the managed block, in file order.

    An unparseable block is treated as empty — the block is nova's own content
    to rebuild, and the head/tail (the operator's) are untouched either way.
    Note this is the ONLY place unparseable content is tolerated: the candidate
    gate below still parses the reassembled file, so a broken operator section
    stops the write.
    """
    if not managed.strip():
        return []
    try:
        data = tomllib.loads(managed)
    except tomllib.TOMLDecodeError:
        return []
    entries = data.get("react", [])
    return [dict(e) for e in entries if isinstance(e, Mapping)]


def _join_overlay(head: str, managed: str, tail: str) -> str:
    """Reassemble with the MINIMUM separation, so the result is a fixed point.

    A separator appended unconditionally would grow the file by one newline on
    every write — unbounded growth in a file the robot re-reads forever — so
    each is added only when the operator's own bytes do not already provide it.
    """
    if head and not head.endswith("\n"):
        head += "\n"
    if not tail:
        tail = "\n"
    elif not tail.startswith("\n"):
        tail = "\n" + tail
    return head + managed + tail


def _merge_entry(existing: list[dict], entry: dict) -> list[dict]:
    """Replace nova's own rule of the same id, else append. Never touches others."""
    rule_id = entry["id"]
    merged = [dict(e) for e in existing if e.get("id") != rule_id]
    merged.append(entry)
    return sorted(merged, key=lambda e: str(e.get("id", "")))


# --------------------------------------------------------------------------- #
# The reload spool                                                            #
# --------------------------------------------------------------------------- #


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def submit_reload() -> str:
    """Drop a reload command for the running engine; return its ``cmd_id``."""
    cmd_id = uuid.uuid4().hex
    commands = statedir.reload_commands_dir()
    commands.mkdir(parents=True, exist_ok=True)
    _atomic_write(commands / f"{time.time_ns()}-{cmd_id}.json", json.dumps({"cmd_id": cmd_id}))
    return cmd_id


def await_reload(cmd_id: str, timeout: float = DEFAULT_RELOAD_TIMEOUT) -> dict | None:
    results = statedir.reload_results_dir()
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{cmd_id}.json"
    deadline = time.monotonic() + timeout
    while True:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            _unlink(path)
            return data
        if time.monotonic() >= deadline:
            return None
        time.sleep(RESULT_POLL_S)


def _reload_verdict(timeout: float) -> str:
    cmd_id = submit_reload()
    result = await_reload(cmd_id, timeout=timeout)
    if result is None:
        return (
            f"reload submitted ({cmd_id}) but not confirmed — "
            "is the behavior engine running? The file is on disk either way."
        )
    if result.get("ok") is True:
        return f"reload confirmed: {json.dumps(result, separators=(',', ':'))}"
    error = result.get("error") or json.dumps(result, separators=(",", ":"))
    return f"reload rejected: {error}"


# --------------------------------------------------------------------------- #
# The public surface                                                          #
# --------------------------------------------------------------------------- #


def upsert_rule(
    rule: object,
    *,
    path: Path | str | None = None,
    reload_timeout: float = DEFAULT_RELOAD_TIMEOUT,
) -> tuple[bool, str]:
    """Merge *rule* into the managed block; return ``(changed, verdict)``.

    ``changed`` is False when the candidate file is byte-identical to what is
    already on disk — a re-authored, unchanged reflex writes nothing and
    submits no reload, so a model that repeats itself cannot churn the file or
    the engine.

    Raises :class:`RuleRefused` — with nothing written — for a rule outside the
    schema, an id outside the ``nova-`` namespace, a ``say`` over the cap, or a
    candidate file that does not parse/validate.
    """
    entry = validate_rule(rule, kind="react", require_prefix=True)

    target = Path(path) if path is not None else statedir.rules_overlay_path()
    try:
        current = target.read_text(encoding="utf-8")
    except OSError:
        current = ""

    head, managed, tail = _split_overlay(current)
    merged = _merge_entry(_managed_entries(managed), entry)
    candidate = _join_overlay(head, _render_managed(merged), tail)
    if candidate == current:
        return False, "unchanged — this rule is already in the overlay, so nothing was reloaded"

    _install(target, candidate)
    return True, _reload_verdict(reload_timeout)


def _install(target: Path, text: str) -> None:
    """Validate *text* on a sibling temp file, then ``os.replace`` it into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        try:
            data = tomllib.loads(tmp.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as err:
            raise RuleRefused(f"the resulting rules file is not valid TOML: {err}") from err
        validate_rules_document(data)
    except OSError as err:
        _unlink(tmp)
        raise RuleRefused(f"could not write the rules overlay: {err}") from err
    except RuleRefused:
        _unlink(tmp)
        raise
    os.replace(tmp, target)


def list_rules(path: Path | str | None = None) -> tuple[str, ...]:
    """Every ``nova-`` rule id currently in the overlay, sorted.

    Prefix-scanned over the whole parsed file rather than over the managed
    block, because the PREFIX — not the sentinel comments — is the contract:
    this is what an operator's ``grep`` finds. A missing or unreadable overlay
    is an empty tuple, never a raise.
    """
    target = Path(path) if path is not None else statedir.rules_overlay_path()
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return ()
    ids: list[str] = []
    for kind in ("react", "inhibit"):
        for entry in data.get(kind, []) or []:
            if isinstance(entry, Mapping):
                rule_id = entry.get("id")
                if isinstance(rule_id, str) and rule_id.startswith(RULE_ID_PREFIX):
                    ids.append(rule_id)
    return tuple(sorted(ids))
