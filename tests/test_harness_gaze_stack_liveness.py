"""The base-layer revive: ``feel-alive`` re-issued whenever the runtime lost it.

Live finding, 2026-09-06 11:51 BST ("I don't see it moving"): the runtime's
face lock inhibits ``feel-alive``, its intents driver EVICTS an inhibited
behaviour every tick, and nothing re-seeds the base layer once the inhibition
clears (reachy-mini-cli #183). After the first lock of a session the robot is
still for good. Until the runtime re-seeds it, this layer re-issues a bounded
passive ``run_behavior feel-alive`` whenever the runtime's active list lacks it
and nothing inhibits it — in every layer, never under a held lock (the sway
covers the hold), rate-limited so a refusal or an unreadable state file costs
one quiet retry, not one op per tick.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from reachy_nova.harness.gaze_stack import (
    BASE_LAYER_BEHAVIOR,
    BASE_REVIVE_DURATION_S,
    BASE_REVIVE_MARGIN_S,
    BASE_REVIVE_RETRY_S,
    GazeStack,
)
from reachy_nova.harness.lock_state import LockState

TICK = 0.02
DEADLINE = 5.0


class FakeIntents:
    """Records ops in order; answers per tool name from ``results``."""

    def __init__(self, results: dict[str, dict] | None = None) -> None:
        self._lock = threading.Lock()
        self.ops: list[tuple[int, str, dict]] = []
        self.results = dict(results or {})
        self._seq = 0

    def execute(self, tool_name: str, params: dict) -> str:
        with self._lock:
            self._seq += 1
            self.ops.append((self._seq, tool_name, dict(params)))
        return json.dumps(self.results.get(tool_name, {"ok": True}))

    def snapshot(self) -> list[tuple[int, str, dict]]:
        with self._lock:
            return list(self.ops)

    def names(self) -> list[str]:
        return [name for _seq, name, _params in self.snapshot()]

    def forget(self) -> None:
        with self._lock:
            self.ops.clear()


class FakeAttention:
    def __init__(self, live: bool = False) -> None:
        self.conversation_live = live


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self._lock = threading.Lock()
        self.now = now

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class ActiveNames:
    """The runtime's ``state.json`` active list, as the test wants it read."""

    def __init__(self, names: list[str] | None = None, raises: bool = False) -> None:
        self._lock = threading.Lock()
        self.names = list(names or [])
        self.raises = raises
        self.reads = 0

    def __call__(self) -> list[str]:
        with self._lock:
            self.reads += 1
            if self.raises:
                raise OSError("state.json unreadable")
            return list(self.names)

    def set(self, names: list[str]) -> None:
        with self._lock:
            self.names = list(names)


def wait_for(predicate, deadline: float = DEADLINE, message: str = "condition") -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {message}")


def revive_ops(intents):
    return [
        op
        for op in intents.snapshot()
        if op[1] == "run_behavior" and op[2].get("name") == BASE_LAYER_BEHAVIOR
    ]


def started(intents, active, clock=None, attention=None, lock_state=None, **kwargs):
    stack = GazeStack(
        intents,
        attention=attention,
        lock_state=lock_state,
        clock=clock or time.monotonic,
        tick_s=TICK,
        active_names=active,
        **kwargs,
    )
    stack.start(threading.Event())
    wait_for(lambda: intents.names()[:2] == ["release_face", "declare_goal"], message="hygiene")
    return stack


def test_constants_are_the_documented_ones():
    assert BASE_LAYER_BEHAVIOR == "feel-alive"
    assert BASE_REVIVE_DURATION_S == 300.0
    assert BASE_REVIVE_MARGIN_S == 15.0
    assert BASE_REVIVE_RETRY_S == 30.0


def test_a_live_base_layer_is_left_alone():
    intents = FakeIntents()
    active = ActiveNames([BASE_LAYER_BEHAVIOR])
    stack = started(intents, active)
    try:
        wait_for(lambda: active.reads >= 5, message="a few ticks")
        assert revive_ops(intents) == []
        assert stack.status()["base_revive_until"] is None
    finally:
        stack.stop()


def test_a_missing_base_layer_is_revived_once_bounded_and_passive_by_name():
    clock = FakeClock()
    intents = FakeIntents()
    active = ActiveNames([])
    stack = started(intents, active, clock=clock)
    try:
        wait_for(lambda: len(revive_ops(intents)) == 1, message="the revive")
        _seq, _name, params = revive_ops(intents)[0]
        assert params == {"name": BASE_LAYER_BEHAVIOR, "duration": BASE_REVIVE_DURATION_S}
        # The runtime now lists it; nothing more is issued.
        active.set([BASE_LAYER_BEHAVIOR])
        time.sleep(TICK * 8)
        assert len(revive_ops(intents)) == 1
        assert stack.status()["base_revive_until"] == pytest.approx(
            clock.now + BASE_REVIVE_DURATION_S
        )
    finally:
        stack.stop()


def test_a_stale_active_list_does_not_re_issue_before_the_margin():
    clock = FakeClock()
    intents = FakeIntents()
    active = ActiveNames([])  # never updated: the runtime "never lists" it
    stack = started(intents, active, clock=clock)
    try:
        wait_for(lambda: len(revive_ops(intents)) == 1, message="the revive")
        time.sleep(TICK * 8)
        assert len(revive_ops(intents)) == 1
        clock.advance(BASE_REVIVE_DURATION_S - BASE_REVIVE_MARGIN_S + 1.0)
        wait_for(lambda: len(revive_ops(intents)) == 2, message="the re-issue")
        time.sleep(TICK * 8)
        assert len(revive_ops(intents)) == 2
    finally:
        stack.stop()


def test_a_refused_revive_is_retried_after_the_backoff_not_every_tick():
    clock = FakeClock()
    intents = FakeIntents(
        results={"run_behavior": {"ok": False, "error": "'feel-alive' is inhibited"}}
    )
    active = ActiveNames([])
    stack = started(intents, active, clock=clock)
    try:
        wait_for(lambda: len(revive_ops(intents)) == 1, message="the first attempt")
        time.sleep(TICK * 8)
        assert len(revive_ops(intents)) == 1
        assert stack.status()["base_revive_until"] is None
        clock.advance(BASE_REVIVE_RETRY_S + 1.0)
        wait_for(lambda: len(revive_ops(intents)) == 2, message="the retry")
    finally:
        stack.stop()


def test_an_unreadable_active_list_counts_as_missing_but_is_rate_limited():
    clock = FakeClock()
    intents = FakeIntents()
    active = ActiveNames(raises=True)
    stack = started(intents, active, clock=clock)
    try:
        wait_for(lambda: len(revive_ops(intents)) == 1, message="the revive")
        time.sleep(TICK * 8)
        assert len(revive_ops(intents)) == 1
        clock.advance(BASE_REVIVE_DURATION_S - BASE_REVIVE_MARGIN_S + 1.0)
        wait_for(lambda: len(revive_ops(intents)) == 2, message="the re-issue")
    finally:
        stack.stop()


def test_an_inhibited_base_layer_is_never_revived():
    intents = FakeIntents()
    active = ActiveNames([])
    stack = started(intents, active, current_inhibitions=lambda: [BASE_LAYER_BEHAVIOR])
    try:
        wait_for(lambda: active.reads >= 1 or True, message="ticks")
        time.sleep(TICK * 8)
        assert revive_ops(intents) == []
    finally:
        stack.stop()


def test_lock_liveness_off_never_revives():
    intents = FakeIntents()
    active = ActiveNames([])
    stack = started(intents, active, lock_liveness=False)
    try:
        time.sleep(TICK * 8)
        assert revive_ops(intents) == []
        assert active.reads == 0
    finally:
        stack.stop()


def test_not_under_the_held_lock_but_right_after_the_fade():
    clock = FakeClock()
    intents = FakeIntents()
    attention = FakeAttention(live=True)
    active = ActiveNames([BASE_LAYER_BEHAVIOR])
    lock_state = LockState(clock=clock)
    stack = started(intents, active, clock=clock, attention=attention, lock_state=lock_state)
    try:
        wait_for(lambda: stack.lock_held is True, message="the lock")
        active.set([])  # the runtime evicted the base layer under the lock
        time.sleep(TICK * 8)
        assert revive_ops(intents) == []  # the sway covers the hold; feel-alive is inhibited
        attention.conversation_live = False
        wait_for(lambda: "release_face" in intents.names()[2:], message="the release")
        wait_for(lambda: len(revive_ops(intents)) == 1, message="the revive after the fade")
        release_seq = [s for s, n, _p in intents.snapshot() if n == "release_face"][-1]
        assert revive_ops(intents)[0][0] > release_seq
    finally:
        stack.stop()


def test_the_runtime_dropping_the_lock_revives_at_once_even_mid_deadline():
    clock = FakeClock()
    intents = FakeIntents()
    active = ActiveNames([])
    stack = started(intents, active, clock=clock)
    try:
        wait_for(lambda: len(revive_ops(intents)) == 1, message="the first revive")
        clock.advance(10.0)  # well inside the 300 s deadline
        stack.on_lock_released("max-hold")  # the runtime evicted everything it inhibited
        wait_for(lambda: len(revive_ops(intents)) == 2, message="the immediate revive")
    finally:
        stack.stop()


def test_browsing_posture_keeps_the_base_layer_alive_too():
    intents = FakeIntents()
    active = ActiveNames([])
    stack = started(intents, active)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: stack.layer == "browsing", message="the browsing layer")
        wait_for(lambda: len(revive_ops(intents)) == 1, message="the revive while browsing")
        assert "declare_goal" in intents.names()
    finally:
        stack.stop()


def test_active_names_is_injectable_and_defaults_to_the_runtime_reader():
    from reachy_nova.harness import tools

    stack = GazeStack(FakeIntents())
    assert stack._active_names is tools.current_active_names


# --------------------------------------------------------------------------- #
# The runtime reader behind the default                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    return tmp_path


def _write_state(payload) -> None:
    from reachy_nova.harness import statedir

    path = statedir.state_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_current_active_names_reads_the_engines_active_list(state_dir):
    from reachy_nova.harness.tools import current_active_names

    _write_state(
        {
            "active": [
                {"id": "feel-alive-1", "name": "feel-alive", "base": True},
                {"id": "rule:look-toward-sound:8", "name": "orient-to-sound"},
                {"name": 7},
                "not a dict",
            ],
            "intents": {"goal": None, "inhibitions": [], "mode": None},
        }
    )
    assert current_active_names() == ["feel-alive", "orient-to-sound"]


@pytest.mark.parametrize(
    "payload",
    [None, "not = json", {"active": "nope"}, {"intents": {}}, {"active": [1, 2]}],
)
def test_current_active_names_degrades_to_empty(state_dir, payload):
    from reachy_nova.harness import statedir
    from reachy_nova.harness.tools import current_active_names

    if payload is None:
        assert not statedir.state_json_path().exists()
    elif isinstance(payload, str):
        path = statedir.state_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    else:
        _write_state(payload)
    assert current_active_names() == []
