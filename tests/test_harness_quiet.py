"""Timed quiet (task t11) — ``reachy_nova/harness/quiet.py``.

``QuietState`` is a deadline, not a mode: ``arm(minutes)`` closes the mouth
until a wall-clock instant, ``release()`` opens it early, and the deadline is
persisted atomically to ``<state>/nova-quiet.json`` so a restart mid-quiet
comes back quiet instead of loudly re-introducing itself at 2am.

Later always wins: a SHORTER request while armed never shortens the quiet
(note ``"kept"``), a longer one extends it (note ``"extended"``).

All tests here drive an injected clock — no sleeping, no wall time.
"""

from __future__ import annotations

import json
import logging

import pytest

from reachy_nova.harness import statedir
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


@pytest.fixture
def clock():
    return FakeClock()


def sense_lines(caplog) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if "[SENSE" in rec.getMessage()]


def quiet_lines(caplog) -> list[str]:
    return [line for line in sense_lines(caplog) if "event=quiet]" in line]


# --------------------------------------------------------------------------- #
# 1. arm / active / remaining                                                 #
# --------------------------------------------------------------------------- #


def test_arm_returns_the_deadline_and_the_armed_note(state_dir, clock):
    quiet = QuietState(clock=clock)
    result = quiet.arm(10)
    assert result["note"] == "armed"
    assert result["until"] == pytest.approx(clock.t + 600.0)
    assert quiet.active() is True
    assert quiet.remaining_s() == pytest.approx(600.0)
    assert quiet.until == pytest.approx(clock.t + 600.0)


def test_a_fresh_state_is_not_armed(state_dir, clock):
    quiet = QuietState(clock=clock)
    assert quiet.active() is False
    assert quiet.remaining_s() == 0.0
    assert quiet.until is None


def test_a_shorter_request_while_armed_keeps_the_later_deadline(state_dir, clock):
    quiet = QuietState(clock=clock)
    first = quiet.arm(30)
    clock.advance(60.0)
    second = quiet.arm(5)
    assert second["note"] == "kept"
    assert second["until"] == pytest.approx(first["until"])
    assert quiet.remaining_s() == pytest.approx(1800.0 - 60.0)


def test_a_longer_request_while_armed_extends(state_dir, clock):
    quiet = QuietState(clock=clock)
    quiet.arm(5)
    clock.advance(60.0)
    second = quiet.arm(30)
    assert second["note"] == "extended"
    assert second["until"] == pytest.approx(clock.t + 1800.0)


def test_expiry_makes_the_state_inactive(state_dir, clock):
    quiet = QuietState(clock=clock)
    quiet.arm(1)
    clock.advance(61.0)
    assert quiet.active() is False
    assert quiet.remaining_s() == 0.0


# --------------------------------------------------------------------------- #
# 2. release                                                                  #
# --------------------------------------------------------------------------- #


def test_release_reports_whether_it_was_armed(state_dir, clock):
    quiet = QuietState(clock=clock)
    assert quiet.release() == {"was_armed": False}
    quiet.arm(10)
    assert quiet.release() == {"was_armed": True}
    assert quiet.active() is False
    assert quiet.release() == {"was_armed": False}


# --------------------------------------------------------------------------- #
# 3. Persistence — a restart mid-quiet stays quiet                            #
# --------------------------------------------------------------------------- #


def test_arm_persists_the_deadline_atomically(state_dir, clock):
    quiet = QuietState(clock=clock)
    quiet.arm(10)
    path = statedir.quiet_state_path()
    assert path.name == "nova-quiet.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["until"] == pytest.approx(clock.t + 600.0)
    # No temp file left behind by the tmp + os.replace write.
    assert [p.name for p in path.parent.glob("nova-quiet.json.tmp*")] == []


def test_a_restart_with_a_future_deadline_comes_up_armed(state_dir, clock):
    QuietState(clock=clock).arm(10)
    clock.advance(60.0)
    restarted = QuietState(clock=clock)
    assert restarted.active() is True
    assert restarted.remaining_s() == pytest.approx(540.0)


def test_a_restart_after_the_deadline_ignores_and_removes_the_file(state_dir, clock):
    QuietState(clock=clock).arm(1)
    clock.advance(120.0)
    restarted = QuietState(clock=clock)
    assert restarted.active() is False
    assert restarted.until is None
    assert not statedir.quiet_state_path().exists()


def test_release_removes_the_persisted_file(state_dir, clock):
    quiet = QuietState(clock=clock)
    quiet.arm(10)
    assert statedir.quiet_state_path().exists()
    quiet.release()
    assert not statedir.quiet_state_path().exists()


def test_a_corrupt_file_is_ignored_and_removed(state_dir, clock):
    path = statedir.quiet_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    quiet = QuietState(clock=clock)
    assert quiet.active() is False
    assert not path.exists()


# --------------------------------------------------------------------------- #
# 4. Named, latched logging                                                   #
# --------------------------------------------------------------------------- #


def test_arm_logs_one_supervise_line_with_the_iso_deadline(state_dir, clock, caplog):
    quiet = QuietState(clock=clock)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        quiet.arm(10)
    lines = quiet_lines(caplog)
    assert len(lines) == 1
    assert "[SENSE stage=supervise source=nova event=quiet]" in lines[0]
    assert "until=" in lines[0]
    assert "T" in lines[0].split("until=")[1]  # an ISO timestamp, not an epoch


def test_a_kept_request_logs_nothing_new_but_an_extension_does(state_dir, clock, caplog):
    quiet = QuietState(clock=clock)
    quiet.arm(30)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        quiet.arm(5)
        assert quiet_lines(caplog) == []
        quiet.arm(60)
    assert len(quiet_lines(caplog)) == 1


def test_release_logs_the_reason(state_dir, clock, caplog):
    quiet = QuietState(clock=clock)
    quiet.arm(10)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        quiet.release()
    lines = quiet_lines(caplog)
    assert len(lines) == 1
    assert "released reason=ended" in lines[0]


def test_expiry_logs_released_reason_expired_exactly_once(state_dir, clock, caplog):
    quiet = QuietState(clock=clock)
    quiet.arm(1)
    clock.advance(61.0)
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert quiet.active() is False
        assert quiet.active() is False
        assert quiet.remaining_s() == 0.0
    lines = quiet_lines(caplog)
    assert len(lines) == 1
    assert "released reason=expired" in lines[0]


# --------------------------------------------------------------------------- #
# 5. Post-acknowledgement arming                                              #
# --------------------------------------------------------------------------- #


def test_arm_leaves_the_first_utterance_pending(state_dir, clock):
    quiet = QuietState(clock=clock)
    quiet.arm(10)
    assert quiet.pending_first_utterance is True


def test_the_first_utterance_after_arm_is_allowed_then_the_mouth_closes(state_dir, clock):
    quiet = QuietState(clock=clock)
    quiet.arm(10)
    assert quiet.allow_utterance() is True  # "okay, quiet for ten minutes"
    assert quiet.pending_first_utterance is False
    assert quiet.allow_utterance() is False


def test_the_grace_closes_the_mouth_when_no_utterance_arrives(state_dir, clock):
    quiet = QuietState(clock=clock, grace_s=2.0)
    quiet.arm(10)
    clock.advance(2.5)
    assert quiet.allow_utterance() is False
    assert quiet.pending_first_utterance is False


def test_an_unarmed_state_allows_everything(state_dir, clock):
    quiet = QuietState(clock=clock)
    assert quiet.allow_utterance() is True
    quiet.arm(1)
    quiet.allow_utterance()  # consume the acknowledgement
    clock.advance(61.0)
    assert quiet.allow_utterance() is True


# --------------------------------------------------------------------------- #
# 6. body-ownership latches (finding F1)                                      #
#                                                                             #
# The deadline alone is not enough to survive a restart: whoever muted the    #
# runtime's own mouth has to be remembered too, or the restored quiet ends    #
# with nobody willing to undo the mute.                                       #
# --------------------------------------------------------------------------- #


def test_a_fresh_state_owns_no_body_latches(state_dir, clock):
    q = QuietState(clock=clock, path=statedir.quiet_state_path())
    assert q.body_latches() == (False, False)


def test_body_latches_round_trip_through_the_file(state_dir, clock):
    path = statedir.quiet_state_path()
    q = QuietState(clock=clock, path=path)
    q.arm(10)
    q.set_body_latches(True, False)
    assert json.loads(path.read_text(encoding="utf-8"))["body"] == {
        "added_speak": True,
        "muted_voice": False,
    }
    assert QuietState(clock=clock, path=path).body_latches() == (True, False)


def test_arm_keeps_already_owned_latches_in_the_file(state_dir, clock):
    path = statedir.quiet_state_path()
    q = QuietState(clock=clock, path=path)
    q.arm(10)
    q.set_body_latches(True, True)
    q.arm(20)
    assert json.loads(path.read_text(encoding="utf-8"))["body"] == {
        "added_speak": True,
        "muted_voice": True,
    }


def test_an_old_shape_file_without_latches_still_loads(state_dir, clock):
    path = statedir.quiet_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"until": clock.t + 600}), encoding="utf-8")
    q = QuietState(clock=clock, path=path)
    assert q.active() is True
    assert q.body_latches() == (False, False)


def test_a_malformed_body_field_reads_as_unowned(state_dir, clock):
    path = statedir.quiet_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"until": clock.t + 600, "body": "yes"}), encoding="utf-8"
    )
    q = QuietState(clock=clock, path=path)
    assert q.active() is True
    assert q.body_latches() == (False, False)


def test_release_clears_the_body_latches(state_dir, clock):
    q = QuietState(clock=clock, path=statedir.quiet_state_path())
    q.arm(10)
    q.set_body_latches(True, True)
    q.release()
    assert q.body_latches() == (False, False)


def test_latches_set_while_unarmed_are_not_persisted(state_dir, clock):
    path = statedir.quiet_state_path()
    q = QuietState(clock=clock, path=path)
    q.set_body_latches(True, True)
    assert q.body_latches() == (True, True)
    assert not path.exists()
