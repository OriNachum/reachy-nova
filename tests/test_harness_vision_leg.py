"""Vision leg — the runtime's rolling camera clip becomes one Sonic inject (t9).

The reachy-mini-cli runtime owns the camera. Its "clip rider" continuously
re-encodes a short rolling clip to ONE overwrite-in-place file and announces it
on the RETAINED MQTT topic ``reachy/state/clip``::

    {"available": true, "reason": null, "path": "/…/clip.mp4",
     "ts": 1.79e9, "duration_s": 4.0, "frame_count": 60}

    {"available": false, "reason": "vision-extra-absent", "path": null, …}
    (observed live on the wireless, 2026-08-11)

:class:`~reachy_nova.harness.vision_leg.VisionLeg` is the consumer of that
state. It never opens a camera, never decodes a frame and never subscribes to
anything itself: the app hands it ``get_clip_state`` (the retained payload),
``understand`` (``NovaOmni``) and ``on_answer`` (``sonic.inject_text``), and
this suite drives all three with plain fakes — no MQTT, no Bedrock, no
``cv2``, no ``reachy_mini``.

Three properties are load-bearing and each has tests below:

1. **An absent clip is the ordinary resting state.** ``available: false`` is
   what the device actually reports today, so it must cost exactly ONE named,
   latched senselog line and a retry next cycle — never a crash, never a
   per-cycle log flood, and never an ``understand`` call against a null path.
2. **The payload's own ``reason`` is the drop's name.** The runtime already
   said *why* (``vision-extra-absent``); inventing a second vocabulary for it
   would make the log unsearchable against the runtime's.
3. **Every failure downstream is named too** — a raising ``understand``, an
   empty answer, a raising inject: all latched named drops, and the reader
   thread survives all of them.
"""

from __future__ import annotations

import ast
import logging
import threading
import time
from pathlib import Path

import pytest

from reachy_nova.harness import vision_leg
from reachy_nova.harness.vision_leg import VisionLeg

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "reachy_nova" / "harness" / "vision_leg.py"


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class Recorder:
    """Records exactly how a callback was called — args AND kwargs."""

    def __init__(self, answer: str = "", raises: Exception | None = None) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self.answer = answer
        self.raises = raises
        self.called = threading.Event()

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.called.set()
        if self.raises is not None:
            raise self.raises
        return self.answer

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def texts(self) -> list[str]:
        return [args[0] for args, _ in self.calls if args]


class FakeOmni:
    """A ``NovaOmni``-shaped object: the leg may be handed the INSTANCE."""

    def __init__(self, answer: str = "a hand waves at the camera") -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def understand(self, clip_path=None, image_bytes=None, context="") -> str:
        self.calls.append(
            {"clip_path": clip_path, "image_bytes": image_bytes, "context": context}
        )
        return self.answer


class FakeOmniWithEvent(FakeOmni):
    """A future ``NovaOmni`` whose ``understand`` also takes an ``event`` id."""

    def understand(self, clip_path=None, image_bytes=None, context="", event="") -> str:
        self.calls.append(
            {
                "clip_path": clip_path,
                "image_bytes": image_bytes,
                "context": context,
                "event": event,
            }
        )
        return self.answer


def sense_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "nova.sensory"]


def drops(caplog: pytest.LogCaptureFixture, reason: str) -> list[str]:
    return [line for line in sense_lines(caplog) if f"reason={reason}" in line]


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A real (tiny) file standing in for the rider's overwrite-in-place clip."""
    path = tmp_path / "rolling.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return path


def available(path: Path, ts: float = 1000.0) -> dict:
    return {
        "available": True,
        "reason": None,
        "path": str(path),
        "ts": ts,
        "duration_s": 4.0,
        "frame_count": 60,
    }


UNAVAILABLE = {
    "available": False,
    "reason": "vision-extra-absent",
    "path": None,
    "ts": 1000.0,
    "duration_s": None,
    "frame_count": 0,
}


def leg_for(state, understand, on_answer, **kwargs) -> VisionLeg:
    """A leg over a fixed (or callable) clip state, with dedupe off by default."""
    getter = state if callable(state) else (lambda: state)
    kwargs.setdefault("skip_unchanged", False)
    kwargs.setdefault("interval_s", 0.01)
    return VisionLeg(getter, understand, on_answer, **kwargs)


# --------------------------------------------------------------------------- #
# 1. The happy path: retained clip state -> understand -> inject               #
# --------------------------------------------------------------------------- #


def test_available_clip_is_understood_and_the_answer_is_injected(clip: Path) -> None:
    omni = FakeOmni("two people are talking")
    inject = Recorder()
    leg = leg_for(available(clip), omni, inject)

    assert leg.cycle() is True

    assert len(omni.calls) == 1
    assert omni.calls[0]["clip_path"] == str(clip)
    assert inject.texts == ["two people are talking"]


def test_the_clip_path_is_passed_by_keyword_with_the_context(clip: Path) -> None:
    understand = Recorder(answer="ok")
    leg = leg_for(available(clip), understand, Recorder(), context="what changed?")

    leg.cycle()

    args, kwargs = understand.calls[0]
    assert args == ()
    assert kwargs["clip_path"] == str(clip)
    assert kwargs["context"] == "what changed?"


def test_a_bare_callable_is_accepted_as_well_as_a_novaomni_instance(clip: Path) -> None:
    omni = FakeOmni("instance answer")
    inject = Recorder()
    leg_for(available(clip), omni.understand, inject).cycle()

    assert inject.texts == ["instance answer"]
    assert len(omni.calls) == 1


def test_the_inject_is_called_with_exactly_one_positional_argument(clip: Path) -> None:
    """``sonic.inject_text``'s speaking guard and throttle stay in the path."""
    inject = Recorder()
    leg_for(available(clip), FakeOmni("hello"), inject).cycle()

    args, kwargs = inject.calls[0]
    assert len(args) == 1 and kwargs == {}


def test_a_context_provider_is_called_fresh_every_cycle(clip: Path) -> None:
    contexts = iter(["first", "second"])
    understand = Recorder(answer="ok")
    leg = leg_for(available(clip), understand, Recorder(), context=lambda: next(contexts))

    leg.cycle()
    leg.cycle()

    assert [kwargs["context"] for _, kwargs in understand.calls] == ["first", "second"]


def test_a_raising_context_provider_falls_back_to_the_default_context(
    clip: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def boom() -> str:
        raise RuntimeError("no state yet")

    understand = Recorder(answer="ok")
    inject = Recorder()
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for(available(clip), understand, inject, context=boom).cycle() is True

    assert understand.calls[0][1]["context"] == vision_leg.DEFAULT_CONTEXT
    assert inject.count == 1
    assert drops(caplog, vision_leg.REASON_CONTEXT_FAILED)


def test_the_default_context_is_used_when_none_is_configured(clip: Path) -> None:
    understand = Recorder(answer="ok")
    leg_for(available(clip), understand, Recorder()).cycle()

    assert understand.calls[0][1]["context"] == vision_leg.DEFAULT_CONTEXT


def test_an_event_id_is_passed_only_to_an_understand_that_accepts_one(clip: Path) -> None:
    with_event = FakeOmniWithEvent()
    leg_for(available(clip), with_event, Recorder()).cycle()
    assert with_event.calls[0]["event"] == vision_leg.EVENT

    without_event = FakeOmni()
    leg_for(available(clip), without_event, Recorder()).cycle()
    assert "event" not in without_event.calls[0]


def test_counters_track_cycles_and_answers(clip: Path) -> None:
    leg = leg_for(available(clip), FakeOmni("ok"), Recorder())
    leg.cycle()
    leg.cycle()

    assert leg.cycles == 2
    assert leg.answers == 2
    assert leg.drops == 0


# --------------------------------------------------------------------------- #
# 2. The ordinary resting state: no clip. One named, latched drop.             #
# --------------------------------------------------------------------------- #


def test_an_unavailable_clip_is_a_named_drop_carrying_the_payloads_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    understand = Recorder()
    inject = Recorder()
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for(UNAVAILABLE, understand, inject).cycle() is False

    assert understand.count == 0 and inject.count == 0
    assert drops(caplog, "vision-extra-absent"), sense_lines(caplog)


def test_an_unavailable_clip_without_a_reason_still_gets_a_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = {"available": False, "reason": None, "path": None}
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for(state, Recorder(), Recorder()).cycle() is False

    assert drops(caplog, vision_leg.REASON_CLIP_UNAVAILABLE), sense_lines(caplog)


def test_the_unavailable_drop_is_latched_to_one_line_across_many_cycles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leg = leg_for(UNAVAILABLE, Recorder(), Recorder())
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        for _ in range(5):
            leg.cycle()

    assert len(drops(caplog, "vision-extra-absent")) == 1, sense_lines(caplog)
    assert leg.drops == 5


def test_the_latch_reopens_after_the_clip_comes_back_and_goes_again(
    clip: Path, caplog: pytest.LogCaptureFixture
) -> None:
    states = [UNAVAILABLE, UNAVAILABLE, available(clip), UNAVAILABLE, UNAVAILABLE]
    pending = iter(states)
    leg = leg_for(lambda: next(pending), FakeOmni("back again"), Recorder())

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        for _ in states:
            leg.cycle()

    assert len(drops(caplog, "vision-extra-absent")) == 2, sense_lines(caplog)


def test_a_changed_reason_is_logged_even_while_the_clip_stays_absent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    states = iter(
        [
            {"available": False, "reason": "vision-extra-absent", "path": None},
            {"available": False, "reason": "camera-busy", "path": None},
        ]
    )
    leg = leg_for(lambda: next(states), Recorder(), Recorder())
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        leg.cycle()
        leg.cycle()

    assert drops(caplog, "vision-extra-absent")
    assert drops(caplog, "camera-busy")


def test_no_retained_state_at_all_is_a_named_drop(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for(None, Recorder(), Recorder()).cycle() is False

    assert drops(caplog, vision_leg.REASON_NO_CLIP_STATE), sense_lines(caplog)


def test_a_non_mapping_state_is_a_named_drop(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for("not-a-dict", Recorder(), Recorder()).cycle() is False

    assert drops(caplog, vision_leg.REASON_BAD_CLIP_STATE), sense_lines(caplog)


def test_a_raising_state_getter_is_a_named_drop_not_a_crash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom():
        raise RuntimeError("broker down")

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for(boom, Recorder(), Recorder()).cycle() is False

    assert drops(caplog, vision_leg.REASON_CLIP_STATE_FAILED), sense_lines(caplog)


def test_available_but_pathless_state_is_a_named_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    understand = Recorder()
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        state = {"available": True, "reason": None, "path": None}
        assert leg_for(state, understand, Recorder()).cycle() is False

    assert understand.count == 0
    assert drops(caplog, vision_leg.REASON_NO_CLIP_PATH), sense_lines(caplog)


def test_a_path_that_is_not_on_disk_is_a_named_drop_not_a_blind_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A retained payload outlives the file it names — never hand that to Omni."""
    understand = Recorder()
    state = available(tmp_path / "vanished.mp4")
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for(state, understand, Recorder()).cycle() is False

    assert understand.count == 0
    assert drops(caplog, vision_leg.REASON_CLIP_FILE_MISSING), sense_lines(caplog)


def test_an_empty_clip_file_is_a_named_drop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    understand = Recorder()
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for(available(empty), understand, Recorder()).cycle() is False

    assert understand.count == 0
    assert drops(caplog, vision_leg.REASON_CLIP_FILE_EMPTY), sense_lines(caplog)


# --------------------------------------------------------------------------- #
# 3. Downstream failures are named too, and never kill the leg                 #
# --------------------------------------------------------------------------- #


def test_a_raising_understand_is_a_named_latched_drop(
    clip: Path, caplog: pytest.LogCaptureFixture
) -> None:
    understand = Recorder(raises=RuntimeError("bedrock said no"))
    inject = Recorder()
    leg = leg_for(available(clip), understand, inject)

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg.cycle() is False
        assert leg.cycle() is False

    assert inject.count == 0
    assert understand.count == 2
    assert len(drops(caplog, vision_leg.REASON_UNDERSTAND_FAILED)) == 1, sense_lines(caplog)


def test_an_empty_answer_is_a_named_drop(clip: Path, caplog: pytest.LogCaptureFixture) -> None:
    inject = Recorder()
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for(available(clip), Recorder(answer="   "), inject).cycle() is False

    assert inject.count == 0
    assert drops(caplog, vision_leg.REASON_EMPTY_ANSWER), sense_lines(caplog)


def test_a_raising_inject_is_a_named_drop_not_a_crash(
    clip: Path, caplog: pytest.LogCaptureFixture
) -> None:
    inject = Recorder(raises=RuntimeError("sonic stream is down"))
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg_for(available(clip), FakeOmni("seen"), inject).cycle() is False

    assert drops(caplog, vision_leg.REASON_INJECT_FAILED), sense_lines(caplog)


def test_a_recovered_understand_clears_its_latch(
    clip: Path, caplog: pytest.LogCaptureFixture
) -> None:
    outcomes = iter([RuntimeError("x"), None, RuntimeError("x")])

    def understand(**kwargs) -> str:
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome
        return "recovered"

    leg = leg_for(available(clip), understand, Recorder())
    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        for _ in range(3):
            leg.cycle()

    assert len(drops(caplog, vision_leg.REASON_UNDERSTAND_FAILED)) == 2


# --------------------------------------------------------------------------- #
# 4. Not re-describing the same clip                                           #
# --------------------------------------------------------------------------- #


def test_an_unchanged_clip_timestamp_is_a_named_drop_when_dedupe_is_on(
    clip: Path, caplog: pytest.LogCaptureFixture
) -> None:
    understand = Recorder(answer="ok")
    leg = leg_for(available(clip, ts=42.0), understand, Recorder(), skip_unchanged=True)

    with caplog.at_level(logging.INFO, logger="nova.sensory"):
        assert leg.cycle() is True
        assert leg.cycle() is False

    assert understand.count == 1
    assert drops(caplog, vision_leg.REASON_CLIP_UNCHANGED), sense_lines(caplog)


def test_a_new_clip_timestamp_is_understood_again(clip: Path) -> None:
    stamps = iter([available(clip, ts=1.0), available(clip, ts=2.0)])
    understand = Recorder(answer="ok")
    leg = leg_for(lambda: next(stamps), understand, Recorder(), skip_unchanged=True)

    leg.cycle()
    leg.cycle()

    assert understand.count == 2


def test_dedupe_does_nothing_when_the_payload_carries_no_timestamp(clip: Path) -> None:
    state = {"available": True, "reason": None, "path": str(clip)}
    understand = Recorder(answer="ok")
    leg = leg_for(state, understand, Recorder(), skip_unchanged=True)

    leg.cycle()
    leg.cycle()

    assert understand.count == 2


# --------------------------------------------------------------------------- #
# 5. Interval resolution — env-overridable, 0/absent-safe                      #
# --------------------------------------------------------------------------- #


def test_the_interval_defaults_when_the_env_var_is_absent(monkeypatch) -> None:
    monkeypatch.delenv(vision_leg.INTERVAL_ENV, raising=False)
    assert vision_leg.resolve_interval() == vision_leg.DEFAULT_INTERVAL_S


@pytest.mark.parametrize("raw", ["", "   ", "0", "-5", "not-a-number", "None"])
def test_an_unusable_interval_env_value_falls_back_to_the_default(monkeypatch, raw: str) -> None:
    monkeypatch.setenv(vision_leg.INTERVAL_ENV, raw)
    assert vision_leg.resolve_interval() == vision_leg.DEFAULT_INTERVAL_S


def test_the_env_var_overrides_the_interval(monkeypatch) -> None:
    monkeypatch.setenv(vision_leg.INTERVAL_ENV, "12.5")
    assert vision_leg.resolve_interval() == 12.5
    leg = VisionLeg(lambda: None, Recorder(), Recorder())
    assert leg.interval_s == 12.5


def test_an_explicit_interval_wins_over_the_env(monkeypatch) -> None:
    monkeypatch.setenv(vision_leg.INTERVAL_ENV, "12.5")
    leg = VisionLeg(lambda: None, Recorder(), Recorder(), interval_s=3.0)
    assert leg.interval_s == 3.0


# --------------------------------------------------------------------------- #
# 6. Thread lifecycle                                                          #
# --------------------------------------------------------------------------- #


def test_start_runs_cycles_on_a_daemon_thread_and_stop_joins(clip: Path) -> None:
    inject = Recorder()
    leg = leg_for(available(clip), FakeOmni("moving"), inject, interval_s=0.01)
    stop = threading.Event()

    leg.start(stop)
    try:
        assert inject.called.wait(5.0), "the leg never injected an answer"
        assert leg.is_alive()
        assert leg._thread.daemon is True
    finally:
        leg.stop()

    assert not leg.is_alive()


def test_the_external_stop_event_ends_the_thread(clip: Path) -> None:
    leg = leg_for(available(clip), FakeOmni("x"), Recorder(), interval_s=0.01)
    stop = threading.Event()
    leg.start(stop)

    stop.set()
    deadline = time.monotonic() + 5.0
    while leg.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not leg.is_alive()


def test_start_is_idempotent_while_the_thread_lives(clip: Path) -> None:
    leg = leg_for(available(clip), FakeOmni("x"), Recorder(), interval_s=0.05)
    stop = threading.Event()
    leg.start(stop)
    try:
        first = leg._thread
        leg.start(stop)
        assert leg._thread is first
    finally:
        leg.stop()


def test_a_permanently_failing_cycle_never_kills_the_thread() -> None:
    def boom():
        raise RuntimeError("nothing works")

    leg = leg_for(boom, Recorder(), Recorder(), interval_s=0.01)
    stop = threading.Event()
    leg.start(stop)
    try:
        deadline = time.monotonic() + 3.0
        while leg.cycles < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert leg.cycles >= 3
        assert leg.is_alive()
    finally:
        leg.stop()


def test_stop_is_safe_before_start() -> None:
    VisionLeg(lambda: None, Recorder(), Recorder()).stop()


# --------------------------------------------------------------------------- #
# 7. The boundary: no robot SDK, no OpenCV                                     #
# --------------------------------------------------------------------------- #

_FORBIDDEN_IMPORT_ROOTS = ("reachy_mini", "cv2")


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("forbidden", _FORBIDDEN_IMPORT_ROOTS)
def test_the_vision_leg_imports_neither_the_robot_sdk_nor_opencv(forbidden: str) -> None:
    assert forbidden not in _imported_roots(MODULE_PATH), (
        f"reachy_nova/harness/vision_leg.py must never import '{forbidden}' — the "
        "runtime owns the camera; the leg only reads the clip state it publishes."
    )


def test_the_ast_boundary_check_would_actually_catch_a_violation(tmp_path: Path) -> None:
    offender = tmp_path / "offender.py"
    offender.write_text("import cv2\nfrom reachy_mini import Robot\n", encoding="utf-8")
    assert _imported_roots(offender) >= {"cv2", "reachy_mini"}
