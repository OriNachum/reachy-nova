"""Tests for reachy_nova.sensory_log and the loud inject_text drop paths.

sensory_log is the stdlib-only per-stage sensory logging helper (t3). It must
emit one parseable INFO line per stage under the "nova.sensory" logger. The
two silent DEBUG paths in NovaSonic.inject_text (the 3s throttle and the
skipped-while-speaking guard) must go through this helper at INFO, carrying
the fate and the first 60 chars of the text.

Since t9 those two fates differ: the throttle still *drops*
("dropped reason=throttled"), while the speaking guard *defers* — the cue is
parked in DeferredCues and logged "deferred class=<sense class>", then
delivered when the utterance ends (tests/test_harness_deferred_cues.py owns
the deferral behaviour itself; this module owns the log line's shape).
"""

from __future__ import annotations

import logging
import re
import time

from reachy_nova import sensory_log
from reachy_nova.nova_sonic import NovaSonic

_SENSE_LINE_RE = re.compile(
    r"^\[SENSE stage=(?P<stage>\S+) source=(?P<source>\S+) event=(?P<event>\S+)\] (?P<detail>.*)$"
)


def _sensory_records(caplog):
    return [r for r in caplog.records if r.name == "nova.sensory"]


class TestSensoryLogStage:
    """sensory_log.stage() format contract (acceptance criterion 1)."""

    def test_emits_one_info_line_under_nova_sensory_logger(self, caplog):
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sensory_log.stage("vad", "speech", "evt-123", "utterance detected")

        records = _sensory_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.INFO

    def test_line_is_parseable_and_carries_all_fields(self, caplog):
        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sensory_log.stage("vad", "speech", "evt-123", "utterance detected")

        message = _sensory_records(caplog)[0].getMessage()
        match = _SENSE_LINE_RE.match(message)
        assert match is not None, f"unparseable sensory log line: {message!r}"
        assert match.group("stage") == "vad"
        assert match.group("source") == "speech"
        assert match.group("event") == "evt-123"
        assert match.group("detail") == "utterance detected"

    def test_uses_nova_sensory_logger_name(self):
        assert sensory_log.logger.name == "nova.sensory"


class TestInjectTextDropPaths:
    """Both silent inject_text skip paths now log loudly (acceptance criterion 2)."""

    def _make_ready_sonic(self) -> NovaSonic:
        """A cheaply-constructed NovaSonic, marked active with a stub loop.

        inject_text's two skip checks (speaking guard, throttle) both return
        before the code ever touches ``self._loop`` for real scheduling, so a
        plain truthy sentinel is enough to get past the leading
        ``not self._active or not self._loop`` guard.
        """
        sonic = NovaSonic()
        sonic._active = True
        sonic._loop = object()
        return sonic

    def test_skipped_while_speaking_logs_info_with_the_fate_and_text_snippet(self, caplog):
        """Since t9 the fate is 'deferred', not 'dropped' — the cue is parked."""
        sonic = self._make_ready_sonic()
        sonic._speaking = True
        text = "this inject arrives mid-utterance and should be parked loudly " * 2

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sonic.inject_text(text, force=False, sense_class="pat")

        records = _sensory_records(caplog)
        assert len(records) == 1
        message = records[0].getMessage()
        match = _SENSE_LINE_RE.match(message)
        assert match is not None, f"unparseable sensory log line: {message!r}"
        assert match.group("stage") == "inject"
        assert match.group("source") == "speech"
        assert match.group("event"), "expected a non-empty event id"
        detail = match.group("detail")
        assert detail.startswith("deferred ")  # the fate
        assert "class=pat" in detail
        assert "dropped" not in detail
        assert text[:60] in detail

    def test_force_bypasses_the_speaking_guard_and_defers_nothing(self, caplog):
        sonic = self._make_ready_sonic()
        sonic._speaking = True

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            # force=True skips the speaking guard; it will proceed to the
            # throttle check and then attempt real scheduling against the
            # stub loop, which raises — that's fine, we only care that no
            # "skipped-while-speaking" line was logged.
            try:
                sonic.inject_text("hello", force=True)
            except Exception:
                pass

        records = _sensory_records(caplog)
        assert not any("deferred" in r.getMessage() for r in records)
        assert sonic._deferred.pending() == 0

    def test_throttled_logs_info_with_reason_and_text_snippet(self, caplog):
        sonic = self._make_ready_sonic()
        sonic._speaking = False
        sonic._last_inject_time = time.time()  # just injected -> next is throttled
        text = "vision says the room has a red chair near the window right now " * 2

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sonic.inject_text(text, force=False)

        records = _sensory_records(caplog)
        assert len(records) == 1
        message = records[0].getMessage()
        match = _SENSE_LINE_RE.match(message)
        assert match is not None, f"unparseable sensory log line: {message!r}"
        assert match.group("stage") == "inject"
        assert match.group("source") == "speech"
        assert match.group("event"), "expected a non-empty event id"
        detail = match.group("detail")
        assert "throttled" in detail  # the reason
        assert text[:60] in detail

    def test_throttle_and_speaking_lines_use_distinct_event_ids(self, caplog):
        """Each line gets its own event id — never a shared/static placeholder."""
        sonic = self._make_ready_sonic()
        sonic._speaking = True

        with caplog.at_level(logging.INFO, logger="nova.sensory"):
            sonic.inject_text("first cue, deferred while speaking", force=False)
            sonic._speaking = False
            sonic._last_inject_time = time.time()
            sonic.inject_text("second cue, dropped via throttle", force=False)

        records = _sensory_records(caplog)
        assert len(records) == 2
        events = [_SENSE_LINE_RE.match(r.getMessage()).group("event") for r in records]
        assert events[0] != events[1]
