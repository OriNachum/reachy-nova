"""Memory compactor — Nova 2 Lite distils the ledger into a day's memory (t10).

``Ledger`` (task t4) keeps the raw half of c11's memory story: every USER/
ASSISTANT transcript and every delivered sense, appended as NDJSON. This
module is the distilled half: a background thread that periodically asks
Nova 2 Lite what the last 24 hours were about, in exactly two shapes a
companion would be expected to remember —

* **topics** — subjects the conversation touched.
* **items** — requests, stated preferences, running jokes, and things Nova
  was told to stop doing.

— and keeps the result at ``<state>/nova-memory.json`` (see
:func:`reachy_nova.harness.statedir.memory_path`), each entry timestamped so
it can expire after :data:`DEFAULT_MAX_AGE_S` (24 h). :meth:`history`
renders the surviving memory into the shape :mod:`reachy_nova.nova_sonic`
replays as conversation history at session start (c12): one USER-role
context block, then the last few exchanges verbatim — deliberately shaped so
Nova does not treat the replay as something to comment on (c31: a rotation
must not produce a greeting).

Four properties are load-bearing here:

* **Never on a hot thread (c28).** :meth:`compact` runs Bedrock's
  ``invoke_model`` synchronously, which can take a second or more; it is
  only ever called from this module's OWN daemon thread (or directly by a
  test), never from Sonic's response loop or the MQTT thread.
* **Tolerant of a chatty model.** A live probe of Nova 2 Lite from the robot
  (2026-09-06) returned a well-formed answer followed by a trailing
  "*Reasoning:*" paragraph even though the prompt demands JSON only.
  :func:`_extract_json_object` scans for the first BALANCED ``{...}`` block
  and ignores anything the model appends after it, rather than requiring the
  whole reply to be valid JSON.
* **Fails closed on the memory file, never on the mouth.** A Lite failure or
  an unparseable reply leaves the previous ``nova-memory.json`` untouched and
  emits exactly one named senselog drop line; :meth:`compact` never raises
  out of its caller, thread or test alike (h7).
* **A boot with no RTC.** The robot has no real-time clock and can come up
  with a clock far in the past until NTP steps it forward. If the injected
  wall clock is EARLIER than the newest timestamp already on disk, that is
  the stale-boot-clock case, not "everything just expired" — expiry is
  skipped for that run (new entries still merge in) and the condition is
  logged once, not on every tick until NTP catches up.

Atomic writes (temp file + ``os.replace``) follow the same pattern as
:mod:`reachy_nova.harness.quiet` and :mod:`reachy_nova.harness.ledger`: a
``kill -9`` mid-write leaves either the old or the new memory file, never a
torn one.

stdlib + boto3 only; never imports ``reachy_mini`` (enforced over the whole
``reachy_nova/harness`` package by ``tests/test_harness_boundary.py``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import boto3

from reachy_nova import config, sensory_log
from reachy_nova.harness import statedir

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Senselog identity                                                           #
# --------------------------------------------------------------------------- #

#: ``[SENSE stage=memory source=nova event=compact]`` — every line this
#: module emits (the ledger emits its own lines under ``event=ledger`` at
#: the same stage/source).
STAGE = "memory"
SOURCE = "nova"
EVENT = "compact"

#: Latched-per-run drop reason for a failed compaction (Lite error, timeout,
#: or an unparseable reply).
REASON_COMPACTION_FAILED = "memory-compaction-failed"

# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #

#: How often the background thread runs :meth:`MemoryCompactor.compact`.
DEFAULT_INTERVAL_S = 300.0

#: The window read from the ledger, and the age past which an entry expires.
DEFAULT_MAX_AGE_S = 86400.0

#: Inference cap for the compaction call — small, since the reply is a short
#: JSON object, not prose.
DEFAULT_MAX_TOKENS = 512

#: Granularity of the interruptible wait loop.
_WAIT_SLICE_S = 0.05

SYSTEM_PROMPT = (
    "You distil a day's conversation transcript for a small desk robot's "
    "memory. Return ONLY a JSON object — no prose before or after it, no "
    "markdown code fences — in exactly this shape: "
    '{"topics": [{"text": "short topic phrase"}], '
    '"items": [{"text": "short item phrase", '
    '"kind": "request|preference|joke|stop|fact"}]}. '
    "\"topics\" are the subjects the conversation touched. \"items\" are the "
    "specific things a companion would be expected to remember: a request, "
    "a stated preference, a running joke, something Nova was told to stop "
    "doing, or another durable fact about the person. Keep every text short "
    "(well under 20 words). If nothing in the transcript is worth "
    'remembering, return {"topics": [], "items": []}.'
)

_VALID_ITEM_KINDS = frozenset({"request", "preference", "joke", "stop", "fact"})


def _log(detail: str) -> None:
    sensory_log.stage(STAGE, SOURCE, EVENT, detail)


# --------------------------------------------------------------------------- #
# JSON extraction — tolerate trailing prose after the object                  #
# --------------------------------------------------------------------------- #


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """The first BALANCED ``{...}`` object in *text*, parsed — or ``None``.

    Scans from the first ``{`` and tracks brace depth, skipping over braces
    that appear inside JSON string literals (so ``{"text": "a {brace}"}``
    still closes correctly). Everything from the matching ``}`` onward — a
    live Lite reply carried a trailing "*Reasoning:*" paragraph on
    2026-09-06 — is ignored rather than making the whole reply unparseable.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


# --------------------------------------------------------------------------- #
# Entry coercion / merge — pure helpers, trivially testable                  #
# --------------------------------------------------------------------------- #


def _normalize_text(text: str) -> str:
    """Casefolded, whitespace-collapsed text — the dedupe key."""
    return " ".join(text.strip().casefold().split())


def _coerce_entries(raw: Any, now: float, *, with_kind: bool) -> list[dict[str, Any]]:
    """Turn a Lite reply's ``topics``/``items`` value into ``{text, ts[, kind]}``.

    Tolerant on purpose: a string item, a missing/blank ``kind``, or stray
    non-dict entries in the list are skipped or defaulted rather than
    failing the whole compaction — only a reply with no parsable JSON object
    at all counts as the unparseable-reply failure case.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, str):
            text, kind = entry, ""
        elif isinstance(entry, dict):
            text = entry.get("text")
            kind = entry.get("kind", "")
        else:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        record: dict[str, Any] = {"text": text.strip(), "ts": now}
        if with_kind:
            kind = kind.strip().lower() if isinstance(kind, str) else ""
            record["kind"] = kind if kind in _VALID_ITEM_KINDS else "fact"
        out.append(record)
    return out


def _merge_entries(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Dedupe by normalised text; a repeat keeps the EARLIER ``ts``."""
    merged: dict[str, dict[str, Any]] = {}
    for entry in existing:
        key = _normalize_text(str(entry.get("text", "")))
        if key:
            merged[key] = dict(entry)
    for entry in new:
        key = _normalize_text(str(entry.get("text", "")))
        if not key:
            continue
        if key in merged:
            updated = dict(entry)
            prior_ts = merged[key].get("ts", entry["ts"])
            updated["ts"] = min(prior_ts, entry["ts"]) if isinstance(prior_ts, (int, float)) else entry["ts"]
            merged[key] = updated
        else:
            merged[key] = dict(entry)
    return list(merged.values())


def _entry_ts(entry: Any) -> float:
    ts = entry.get("ts") if isinstance(entry, dict) else None
    return float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else 0.0


def _record_ts(record: dict[str, Any]) -> float:
    ts = record.get("ts")
    return float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else 0.0


# --------------------------------------------------------------------------- #
# The compactor                                                               #
# --------------------------------------------------------------------------- #


class MemoryCompactor:
    """Background thread: Lite distils the ledger into a day's memory.

    Mirrors the harness component protocol (``network.NetworkUnit``,
    ``kiro_session.KiroSessionUnit``): ``start(stop_event)`` / ``stop()`` /
    ``is_alive()``, plus a ``name`` :func:`supervisor._component_name` picks
    up.

    Args:
        ledger: the :class:`~reachy_nova.harness.ledger.Ledger` to read from
            and truncate.
        path: memory file to read/write. Defaults to
            :func:`reachy_nova.harness.statedir.memory_path`.
        client: an already-built bedrock-runtime client; built lazily
            (region from ``region``/``config.region()``) when omitted, so
            tests never touch the network.
        model_id: Lite model id; defaults to ``config.lite_model_id()``.
        interval_s: seconds between compactions on the background thread.
        max_age_s: the read window AND the expiry age — both the ledger and
            the memory file keep only the last ``max_age_s``.
        clock: wall-clock source (epoch seconds), injectable for tests.
        monotonic: monotonic-seconds source used only to time the
            interruptible wait between compactions; injectable for tests.
        region / max_tokens: keyword-only extras beyond the plan's listed
            signature, for the same reason ``NovaOmni`` takes them — a
            region override and a request-size cap.
    """

    #: Picked up by ``supervisor._component_name`` when composed as a component.
    name = "memory_compactor"

    def __init__(
        self,
        ledger: Any,
        path: Path | str | None = None,
        client: Any = None,
        model_id: str | None = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        *,
        region: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.ledger = ledger
        self._path = statedir.memory_path() if path is None else Path(path)
        self.region = region or config.region()
        self.model_id = model_id or config.lite_model_id()
        self.interval_s = float(interval_s)
        self.max_age_s = float(max_age_s)
        self.max_tokens = int(max_tokens)
        self._clock = clock
        self._monotonic = monotonic

        self._client = client

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._external_stop: threading.Event | None = None
        self._lock = threading.Lock()

        #: Newest ledger-record ``ts`` seen at the last SUCCESSFUL compaction
        #: — how "unchanged since last time" is detected.
        self._last_ledger_ts: float | None = None
        #: Latches the stale-boot-clock warning so it logs once, not every tick.
        self._stale_clock_warned = False

        #: Successful compactions that actually wrote a new memory file.
        self.compactions = 0
        #: Lite failures or unparseable replies (previous file left intact).
        self.failures = 0
        #: Entries dropped for aging past ``max_age_s`` (across all runs).
        self.expired = 0

    # -- lifecycle -------------------------------------------------------- #

    def start(self, stop_event: threading.Event) -> None:
        """Start the ONE background thread that periodically calls :meth:`compact`."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._external_stop = stop_event
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="nova-memory-compactor", daemon=True
        )
        self._thread.start()
        _log("memory compactor started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the thread and join it. Best-effort, never raises."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        _log("memory compactor stopped")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- the background loop ----------------------------------------------- #

    def _run(self) -> None:
        while True:
            stopped = self._wait(self.interval_s)
            if stopped:
                break
            self._safe_compact()
        # One more compaction at shutdown, so work done in the last partial
        # interval is not lost to a restart.
        self._safe_compact()

    def _safe_compact(self) -> None:
        try:
            self.compact()
        except Exception as err:  # noqa: BLE001 - never kill the thread
            logger.warning("memory_compactor: compact() raised: %s", err, exc_info=True)

    def _should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        return self._external_stop is not None and self._external_stop.is_set()

    def _wait(self, seconds: float) -> bool:
        """Sleep up to *seconds* of injected monotonic time. ``True`` if stopped."""
        deadline = self._monotonic() + seconds
        while not self._should_stop():
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                return False
            self._stop.wait(min(remaining, _WAIT_SLICE_S))
        return True

    # -- the Bedrock client -------------------------------------------------- #

    @property
    def client(self):
        """The bedrock-runtime client, built lazily and then cached."""
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _invoke_lite(self, transcript_text: str) -> dict[str, Any] | None:
        body = {
            "schemaVersion": "messages-v1",
            "system": [{"text": SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": [{"text": transcript_text}]}],
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "topP": 0.9,
                "temperature": 0.2,
            },
        }
        response = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        text = result["output"]["message"]["content"][0]["text"]
        if not isinstance(text, str):
            return None
        return _extract_json_object(text)

    # -- compaction ---------------------------------------------------------- #

    def compact(self) -> bool:
        """Distil the last ``max_age_s`` of the ledger into the memory file.

        Returns ``True`` when a new memory file was written, ``False`` when
        there was nothing to do (empty or unchanged ledger) or the Lite call
        failed/returned something unparseable — in the failure case the
        previous file is left exactly as it was. Never raises: every
        failure path is caught and turned into one named senselog line.
        """
        try:
            return self._compact()
        except Exception as err:  # noqa: BLE001 - h7: never raises out of the caller
            self.failures += 1
            _log(f"dropped reason={REASON_COMPACTION_FAILED}: unexpected error: {err}")
            return False

    def _compact(self) -> bool:
        now = self._clock()
        records = self.ledger.read(since_ts=now - self.max_age_s)
        if not records:
            return False

        newest_ledger_ts = max(_record_ts(r) for r in records)
        if self._last_ledger_ts is not None and newest_ledger_ts <= self._last_ledger_ts:
            return False  # nothing new since the last successful compaction

        transcript = self._render_transcript(records)
        if not transcript.strip():
            return False

        try:
            parsed = self._invoke_lite(transcript)
        except Exception as err:  # noqa: BLE001 - a Lite outage must not touch the file
            self.failures += 1
            _log(f"dropped reason={REASON_COMPACTION_FAILED}: lite call failed: {err}")
            return False

        if not isinstance(parsed, dict):
            self.failures += 1
            _log(f"dropped reason={REASON_COMPACTION_FAILED}: lite reply had no parsable JSON object")
            return False

        new_topics = _coerce_entries(parsed.get("topics"), now, with_kind=False)
        new_items = _coerce_entries(parsed.get("items"), now, with_kind=True)

        existing = self._read_memory_file()
        merged_topics = _merge_entries(existing["topics"], new_topics)
        merged_items = _merge_entries(existing["items"], new_items)

        newest_existing_ts = max(
            (_entry_ts(e) for e in existing["topics"] + existing["items"]), default=0.0
        )
        if newest_existing_ts and now < newest_existing_ts:
            # Stale boot clock (no RTC): expiring by age would wipe memory
            # written under a clock NTP has not yet corrected. Merge still
            # happens; only expiry is skipped, and only logged once.
            if not self._stale_clock_warned:
                _log(
                    "clock behind newest memory entry "
                    f"(now={now} < newest={newest_existing_ts}) — skipping expiry this run"
                )
                self._stale_clock_warned = True
        else:
            self._stale_clock_warned = False
            merged_topics, dropped_t = _drop_expired(merged_topics, now, self.max_age_s)
            merged_items, dropped_i = _drop_expired(merged_items, now, self.max_age_s)
            self.expired += dropped_t + dropped_i

        self._atomic_write({"topics": merged_topics, "items": merged_items})
        self.ledger.truncate(now=now, max_age_s=self.max_age_s)

        self.compactions += 1
        self._last_ledger_ts = newest_ledger_ts
        return True

    def _render_transcript(self, records: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for record in records:
            kind = str(record.get("kind", ""))
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            if kind == "USER":
                lines.append(f"User: {text}")
            elif kind == "ASSISTANT":
                lines.append(f"Nova: {text}")
            else:
                lines.append(f"[{kind or 'sense'}] {text}")
        return "\n".join(lines)

    # -- the memory file ------------------------------------------------------ #

    def _read_memory_file(self) -> dict[str, list[dict[str, Any]]]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError:
            return {"topics": [], "items": []}
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {"topics": [], "items": []}
        if not isinstance(data, dict):
            return {"topics": [], "items": []}
        topics = data.get("topics")
        items = data.get("items")
        return {
            "topics": topics if isinstance(topics, list) else [],
            "items": items if isinstance(items, list) else [],
        }

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f"{self._path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self._path)

    def memory(self) -> dict[str, list[dict[str, Any]]]:
        """The parsed memory file, or ``{"topics": [], "items": []}`` when absent."""
        return self._read_memory_file()

    # -- session-replay view (c12, c31) ---------------------------------------- #

    #: How many trailing USER/ASSISTANT exchanges ride along after the
    #: context block.
    _HISTORY_EXCHANGES = 3

    def history(self, max_chars: int = 2000) -> list[dict[str, str]]:
        """Conversation-history blocks for :mod:`nova_sonic`'s replay hook.

        ``[]`` when nothing is remembered and the ledger is empty. Otherwise
        one USER-role context block ("(earlier today we talked about: ...")
        summarising :meth:`memory`, followed by the last few USER/ASSISTANT
        ledger lines verbatim (sense lines are skipped — they are not
        conversation). The whole thing is trimmed to fit *max_chars*: older
        exchanges are dropped first, and if the context block alone still
        will not fit, IT is truncated rather than dropping it entirely —
        Sonic's replay hook (c12) sends these BEFORE the audio contentStart,
        and the context block is what keeps a rotation from reading to Nova
        like a blank slate (c31: it must not greet again).
        """
        memory = self.memory()
        topics = memory["topics"]
        items = memory["items"]
        exchanges = self._recent_exchanges()

        if not topics and not items and not exchanges:
            return []

        blocks: list[dict[str, str]] = []
        if topics or items:
            topics_text = ", ".join(e["text"] for e in topics) or "nothing in particular"
            items_text = "; ".join(e["text"] for e in items) or "nothing in particular"
            context = (
                f"(earlier today we talked about: {topics_text}; things to "
                f"remember: {items_text}. Do not greet or comment on this; "
                "just carry on when spoken to.)"
            )
            blocks.append({"role": "USER", "text": context})
        blocks.extend(exchanges)

        return _fit_to_cap(blocks, max_chars)

    def _recent_exchanges(self) -> list[dict[str, str]]:
        records = self.ledger.read()
        exchanges = [
            {"role": str(r.get("kind")), "text": str(r.get("text", ""))}
            for r in records
            if r.get("kind") in ("USER", "ASSISTANT") and str(r.get("text", "")).strip()
        ]
        return exchanges[-self._HISTORY_EXCHANGES :]


def _drop_expired(
    entries: list[dict[str, Any]], now: float, max_age_s: float
) -> tuple[list[dict[str, Any]], int]:
    survivors = [e for e in entries if now - _entry_ts(e) <= max_age_s]
    return survivors, len(entries) - len(survivors)


def _fit_to_cap(blocks: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    """Trim *blocks* to fit *max_chars* total text length.

    Drops the OLDEST exchange first (never the context block, always
    ``blocks[0]`` when present) so the most recent conversation survives
    longest; if a single block still exceeds the cap on its own, that
    block's text is truncated rather than the whole history disappearing.
    """

    def total_len(items: list[dict[str, str]]) -> int:
        return sum(len(b["text"]) for b in items)

    trimmed = list(blocks)
    while len(trimmed) > 1 and total_len(trimmed) > max_chars:
        del trimmed[1]  # the oldest exchange, right after the context block

    if trimmed and total_len(trimmed) > max_chars:
        remaining = max_chars
        capped: list[dict[str, str]] = []
        for block in trimmed:
            if remaining <= 0:
                break
            text = block["text"][:remaining]
            capped.append({"role": block["role"], "text": text})
            remaining -= len(text)
        trimmed = capped

    return trimmed
