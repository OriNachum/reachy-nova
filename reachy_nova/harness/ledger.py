"""Conversation ledger — locked NDJSON append, quiet-aware, 24 h truncation.

The raw half of c11's memory story (t10's ``memory_compactor.py`` is the
distilled half): every USER/ASSISTANT transcript ``on_transcript`` already
sees, and every sense the bus delivers, is appended as one compact JSON line
to ``<state>/nova-conversation.jsonl`` so a background job can later ask
Nova 2 Lite what the day was about. This module does none of that
summarising itself — it only keeps the raw record, bounded and safe to write
from two different threads at once (transcripts arrive on Sonic's thread,
senses on the MQTT thread).

Three properties are load-bearing (c30, c36, h22):

* **Locked, not lossy.** :meth:`append` serialises every writer through one
  lock, so two threads writing 500 lines each still produce well-formed
  NDJSON — never an interleaved, unparsable line.
* **Quiet-aware.** While a timed quiet is armed (``QuietState.active()``)
  nothing is written at all, transcripts or senses — a plaintext record of
  a conversation someone asked the robot to be quiet through is exactly the
  data that policy exists to withhold. The skip is counted, not logged: it
  is the ordinary, expected shape of a quiet window, not a fault.
* **Never on the voice path.** A write failure (a full disk, a read-only
  state dir) must never raise into Sonic's thread or the MQTT thread. It
  becomes ONE latched senselog drop line for the whole run of failures —
  never a flood — and the first success afterwards emits ONE recovery line,
  so the journal shows both edges of the outage. :meth:`truncate` rewrites
  the file atomically (temp file + ``os.replace``, the same pattern
  ``reachy_nova.harness.quiet`` uses for its persisted deadline) so a
  kill -9 mid-truncation leaves either the old or the new file, never a torn
  one.

stdlib only; never imports ``reachy_mini``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reachy_nova import sensory_log
from reachy_nova.harness import statedir
from reachy_nova.harness.quiet import QuietState

STAGE = "memory"
SOURCE = "nova"
EVENT = "ledger"

#: Latched drop reason for a failed append/truncate write.
REASON_WRITE_FAILED = "ledger-write-failed"

#: Default truncation window — see the "Raw ledger policy" decision.
DEFAULT_MAX_AGE_S = 86400.0


class Ledger:
    """Locked NDJSON append of transcripts and delivered senses.

    Args:
        path: file to append/read/truncate. Defaults to
            :func:`reachy_nova.harness.statedir.ledger_path`.
        quiet: when given and :meth:`~reachy_nova.harness.quiet.QuietState.active`
            is true, :meth:`append` becomes a no-op. ``None`` (the default)
            means every append lands — a caller with no quiet story yet
            (e.g. a standalone test) should not have to fabricate one.
        clock: wall-clock source (epoch seconds) used as the default ``ts``
            for :meth:`append` and ``now`` for :meth:`truncate`; injectable
            for tests.
    """

    def __init__(
        self,
        path: Path | None = None,
        quiet: QuietState | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = statedir.ledger_path() if path is None else Path(path)
        self._quiet = quiet
        self._clock = clock
        self._lock = threading.Lock()
        #: Successfully written lines.
        self.appended = 0
        #: Appends that were no-ops because quiet was armed.
        self.skipped_quiet = 0
        #: Failed writes (append or truncate), each latched to at most one
        #: senselog line until a write succeeds again.
        self.drops = 0
        #: Lines seen by :meth:`read`/:meth:`truncate` that did not parse.
        self.malformed = 0
        self._write_faulted = False

    # -- write ---------------------------------------------------------- #

    def append(self, kind: str, text: str, ts: float | None = None, **fields: Any) -> bool:
        """Append one ``{"ts", "kind", "text", ...fields}`` NDJSON line.

        Returns ``False`` without writing anything while quiet is armed, or
        when the write itself fails — either way the failure is counted,
        never raised.
        """
        record: dict[str, Any] = {
            "ts": self._clock() if ts is None else ts,
            "kind": kind,
            "text": text,
        }
        record.update(fields)
        line = json.dumps(record, separators=(",", ":")) + "\n"

        with self._lock:
            # Decided UNDER the write lock (PR #24 review): a quiet armed
            # between an earlier check and the write could otherwise leak
            # one record into a window that promised none.
            if self._quiet is not None and self._quiet.active():
                self.skipped_quiet += 1
                return False
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as err:
                self._note_write_failure(err)
                return False
            self.appended += 1
            self._note_write_success()
            return True

    def truncate(self, now: float | None = None, max_age_s: float = DEFAULT_MAX_AGE_S) -> int:
        """Rewrite the ledger keeping only lines with ``ts >= now - max_age_s``.

        Atomic (temp file + ``os.replace``): a crash mid-truncation leaves
        either the untouched original or the fully-written replacement,
        never a partial file. Returns the number of lines dropped (aged out
        or malformed); missing/unreadable input is treated as nothing to
        drop, never raised.
        """
        cutoff = (self._clock() if now is None else now) - max_age_s
        with self._lock:
            try:
                raw = self._path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return 0
            except OSError as err:
                self._note_write_failure(err)
                return 0

            survivors: list[str] = []
            dropped = 0
            for raw_line in raw.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    self.malformed += 1
                    dropped += 1
                    continue
                ts = record.get("ts") if isinstance(record, dict) else None
                if isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts >= cutoff:
                    survivors.append(line)
                else:
                    dropped += 1

            new_text = "".join(f"{line}\n" for line in survivors)
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_name(
                    f"{self._path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
                )
                tmp.write_text(new_text, encoding="utf-8")
                os.replace(tmp, self._path)
            except OSError as err:
                self._note_write_failure(err)
                return 0
            self._note_write_success()
            return dropped

    # -- read ------------------------------------------------------------- #

    def read(self, since_ts: float | None = None) -> list[dict[str, Any]]:
        """Parse the ledger in file order, skipping (and counting) bad lines."""
        with self._lock:
            try:
                raw = self._path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return []
            except OSError:
                return []

        out: list[dict[str, Any]] = []
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                self.malformed += 1
                continue
            if not isinstance(record, dict):
                self.malformed += 1
                continue
            if since_ts is not None:
                ts = record.get("ts")
                if not isinstance(ts, (int, float)) or isinstance(ts, bool) or ts < since_ts:
                    continue
            out.append(record)
        return out

    # -- fault latch (call while holding self._lock) --------------------- #

    def _note_write_failure(self, err: OSError) -> None:
        self.drops += 1
        if not self._write_faulted:
            self._write_faulted = True
            sensory_log.stage(
                STAGE, SOURCE, EVENT, f"dropped reason={REASON_WRITE_FAILED}: {err}"
            )

    def _note_write_success(self) -> None:
        if self._write_faulted:
            self._write_faulted = False
            sensory_log.stage(
                STAGE, SOURCE, EVENT, "ledger recovered (write succeeded after a prior failure)"
            )
