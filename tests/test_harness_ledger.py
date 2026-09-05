"""Conversation ledger (t4) — ``reachy_nova/harness/ledger.py``.

``Ledger`` is the raw half of c11's memory story (the compaction half is
t10): every USER/ASSISTANT transcript and every delivered sense is appended
to one NDJSON file so a later background job can distil it. Three properties
are load-bearing here and each has a test below:

1. **Locked, not lossy.** ``append()`` is called from Sonic's thread
   (transcripts) and the MQTT thread (senses) — concurrently, and it must
   never interleave two writers' bytes into one corrupt line (c30).
2. **Quiet-aware and bounded.** Nothing is written while a timed quiet is
   armed (c36), and ``truncate()`` keeps only the last 24 h, atomically
   (temp file + ``os.replace``, the same pattern as ``quiet.py``).
3. **A write failure never becomes a raised exception on the voice path**
   (h22): a read-only state dir latches ONE named senselog drop line for the
   whole run of failing appends, and a later successful append (after
   permissions are restored) emits ONE recovery line.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import pytest

from reachy_nova.harness import statedir
from reachy_nova.harness.ledger import Ledger
from reachy_nova.harness.quiet import QuietState


class FakeClock:
    """Injectable wall clock (epoch seconds)."""

    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    return tmp_path


def sense_lines(caplog) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if "[SENSE" in rec.getMessage()]


def ledger_lines(caplog) -> list[str]:
    return [line for line in sense_lines(caplog) if "event=ledger]" in line]


# --------------------------------------------------------------------------- #
# statedir helper                                                             #
# --------------------------------------------------------------------------- #


def test_ledger_path_lives_under_state_dir(state_dir):
    assert statedir.ledger_path() == state_dir / "nova-conversation.jsonl"


# --------------------------------------------------------------------------- #
# 1. concurrent append -> well-formed NDJSON, no interleaving                 #
# --------------------------------------------------------------------------- #


def test_concurrent_append_yields_well_formed_ndjson(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path=path)
    n_threads = 2
    n_per_thread = 500

    def writer(idx: int) -> None:
        for i in range(n_per_thread):
            ledger.append("USER", f"writer-{idx}-line-{i}")

    threads = [threading.Thread(target=writer, args=(idx,)) for idx in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert len(lines) == n_threads * n_per_thread
    parsed = [json.loads(line) for line in lines]  # raises on any interleaved/corrupt line
    assert all(record["kind"] == "USER" for record in parsed)
    assert ledger.appended == n_threads * n_per_thread
    assert ledger.drops == 0


def test_append_writes_compact_json_line_with_required_fields(tmp_path):
    path = tmp_path / "ledger.jsonl"
    clock = FakeClock()
    ledger = Ledger(path=path, clock=clock)

    assert ledger.append("USER", "hello there") is True

    (line,) = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert record["ts"] == clock.t
    assert record["kind"] == "USER"
    assert record["text"] == "hello there"


def test_append_accepts_explicit_ts_and_extra_fields(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path=path)

    ledger.append("sense", "a pat", ts=123.0, source="touch", rule="pat-acknowledge")

    (line,) = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert record["ts"] == 123.0
    assert record["kind"] == "sense"
    assert record["text"] == "a pat"
    assert record["source"] == "touch"
    assert record["rule"] == "pat-acknowledge"


# --------------------------------------------------------------------------- #
# 2. quiet-aware append + truncation                                         #
# --------------------------------------------------------------------------- #


def test_append_is_a_noop_while_quiet_is_armed(tmp_path, state_dir):
    path = tmp_path / "ledger.jsonl"
    clock = FakeClock()
    quiet = QuietState(clock=clock)
    ledger = Ledger(path=path, quiet=quiet, clock=clock)

    quiet.arm(10.0)
    assert quiet.active() is True

    result = ledger.append("USER", "should not land")

    assert result is False
    assert ledger.skipped_quiet == 1
    assert ledger.appended == 0
    assert not path.exists() or path.read_text(encoding="utf-8") == ""


def test_append_resumes_once_quiet_has_expired(tmp_path):
    path = tmp_path / "ledger.jsonl"
    clock = FakeClock()
    quiet = QuietState(clock=clock)
    ledger = Ledger(path=path, quiet=quiet, clock=clock)

    quiet.arm(10.0)
    ledger.append("USER", "dropped while quiet")
    clock.advance(10.0 * 60.0 + 1.0)
    assert quiet.active() is False

    assert ledger.append("USER", "lands now") is True
    assert ledger.appended == 1
    assert ledger.skipped_quiet == 1


def test_truncate_drops_lines_older_than_24h_atomically(tmp_path):
    path = tmp_path / "ledger.jsonl"
    clock = FakeClock(t=1_000_000.0)
    ledger = Ledger(path=path, clock=clock)

    ledger.append("USER", "old-1", ts=clock.t - 90000.0)  # 25h old
    ledger.append("ASSISTANT", "old-2", ts=clock.t - 86401.0)  # just over 24h
    ledger.append("USER", "recent-1", ts=clock.t - 3600.0)  # 1h old
    ledger.append("ASSISTANT", "recent-2", ts=clock.t)

    dropped = ledger.truncate(now=clock.t, max_age_s=86400.0)

    assert dropped == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    remaining_texts = [json.loads(line)["text"] for line in lines]
    assert remaining_texts == ["recent-1", "recent-2"]
    # no leftover temp file from the atomic rewrite
    tmp_leftovers = [p for p in tmp_path.iterdir() if p.name != path.name]
    assert tmp_leftovers == []


def test_truncate_survivors_all_parse(tmp_path):
    path = tmp_path / "ledger.jsonl"
    clock = FakeClock(t=2_000_000.0)
    ledger = Ledger(path=path, clock=clock)
    for i in range(10):
        ledger.append("USER", f"msg-{i}", ts=clock.t - i * 10000.0)

    ledger.truncate(now=clock.t, max_age_s=86400.0)

    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)  # must not raise


def test_truncate_is_a_noop_when_file_absent(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path=path)
    assert ledger.truncate(now=1_000.0) == 0
    assert not path.exists()


# --------------------------------------------------------------------------- #
# read()                                                                      #
# --------------------------------------------------------------------------- #


def test_read_round_trips_appended_records(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path=path)
    ledger.append("USER", "one", ts=1.0)
    ledger.append("ASSISTANT", "two", ts=2.0)

    records = ledger.read()

    assert [r["text"] for r in records] == ["one", "two"]


def test_read_filters_by_since_ts(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path=path)
    ledger.append("USER", "old", ts=1.0)
    ledger.append("ASSISTANT", "new", ts=100.0)

    records = ledger.read(since_ts=50.0)

    assert [r["text"] for r in records] == ["new"]


def test_read_skips_and_counts_malformed_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path=path)
    ledger.append("USER", "good", ts=1.0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write("\n")  # blank lines are ignored, not malformed
        fh.write("42\n")  # valid JSON, not an object

    records = ledger.read()

    assert [r["text"] for r in records] == ["good"]
    assert ledger.malformed == 2


def test_read_returns_empty_list_when_file_absent(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path=path)
    assert ledger.read() == []


# --------------------------------------------------------------------------- #
# 3. write failures: latched drop, then one recovery line                    #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permissions")
def test_readonly_state_dir_latches_one_drop_and_recovers(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="nova.sensory")
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    path = ro_dir / "ledger.jsonl"
    ledger = Ledger(path=path)

    ro_dir.chmod(0o500)
    try:
        for i in range(5):
            result = ledger.append("USER", f"line-{i}")
            assert result is False
    finally:
        ro_dir.chmod(0o700)

    drop_lines = [
        line for line in ledger_lines(caplog) if "dropped reason=ledger-write-failed" in line
    ]
    assert len(drop_lines) == 1
    assert ledger.drops == 5
    assert ledger.appended == 0

    caplog.clear()
    assert ledger.append("USER", "back online") is True

    recovery_lines = [line for line in ledger_lines(caplog) if "recovered" in line]
    assert len(recovery_lines) == 1
    assert ledger.appended == 1
