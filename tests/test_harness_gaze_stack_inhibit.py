"""Tests for the BROWSING layer's runtime head-reflex inhibits (task t10).

A standing ``gaze-hold`` goal (t8) is not enough to keep a browse's thinking
pose stable: the runtime's own ``orient-to-sound`` and ``nod`` reflexes still
compete for the head by recency. So entering BROWSING also merges
:data:`~reachy_nova.harness.gaze_stack.BROWSING_INHIBITS` into the runtime's
CURRENT inhibited set (via ``set_inhibition``, which REPLACES the whole set),
remembering exactly which names it added; leaving BROWSING for WANDER gives
back only those names, re-reading the live set first so a later operator
change survives; a conversation in between leaves them standing (the face
lock re-asserts its own inhibitions anyway); and returning from conversation
to browsing re-adds only whatever is missing.

Everything here drives :class:`~reachy_nova.harness.gaze_stack.GazeStack`
against the SAME recording fake ``IntentTools`` used in
``test_harness_gaze_stack.py`` (reproduced here so this file stands alone),
plus a small injectable stand-in for
:func:`reachy_nova.harness.tools.current_inhibitions`.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from reachy_nova.harness.gaze_stack import (
    BROWSING_INHIBITS,
    LAYER_BROWSING,
    LAYER_CONVERSATION,
    LAYER_WANDER,
    GazeStack,
)

TICK = 0.02
DEADLINE = 5.0


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class FakeIntents:
    """Records ops in order, with a sequence number. Same shape as t8's fake."""

    def __init__(self, call_delay: float = 0.0, result: dict | None = None) -> None:
        self._lock = threading.Lock()
        self.ops: list[tuple[int, str, dict]] = []
        self.call_delay = call_delay
        self.result = result if result is not None else {"ok": True}
        self._seq = 0

    def execute(self, tool_name: str, params: dict) -> str:
        with self._lock:
            self._seq += 1
            seq = self._seq
            self.ops.append((seq, tool_name, dict(params)))
        if self.call_delay:
            time.sleep(self.call_delay)
        return json.dumps(self.result)

    def forget(self) -> None:
        with self._lock:
            self.ops.clear()

    def snapshot(self) -> list[tuple[int, str, dict]]:
        with self._lock:
            return list(self.ops)

    def names(self) -> list[str]:
        return [name for _seq, name, _params in self.snapshot()]

    def inhibit_ops(self) -> list[tuple[int, str, dict]]:
        return [op for op in self.snapshot() if op[1] == "set_inhibition"]


class FakeAttention:
    def __init__(self, live: bool = False) -> None:
        self.conversation_live = live


class FakeLiveSet:
    """A mutable stand-in for :func:`tools.current_inhibitions`.

    Lets a test simulate the runtime's own state changing underneath the
    stack — an operator adding a name, or a face lock's replacement dropping
    one — between reads.
    """

    def __init__(self, names: list[str] | None = None) -> None:
        self._names = list(names or [])

    def __call__(self) -> list[str]:
        return list(self._names)

    def set(self, names: list[str]) -> None:
        self._names = list(names)


def wait_for(predicate, deadline: float = DEADLINE, message: str = "condition") -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {message}")


def running_stack(intents, **kwargs):
    stack = GazeStack(intents, tick_s=TICK, **kwargs)
    stop = threading.Event()
    stack.start(stop)
    # The hygiene runs on the WORKER now (PR #26 review), so wait for it
    # before forgetting it, or it lands in the middle of the ops a test counts.
    wait_for(lambda: len(intents.snapshot()) >= 2, message="the start hygiene")
    intents.forget()
    return stack, stop


# --------------------------------------------------------------------------- #
# Entering browsing merges the live set with BROWSING_INHIBITS                #
# --------------------------------------------------------------------------- #


def test_entering_browsing_merges_live_set_with_browsing_inhibits():
    intents = FakeIntents()
    live = FakeLiveSet(["speak"])
    stack, _stop = running_stack(intents, current_inhibitions=live)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing layer")
        wait_for(lambda: len(intents.inhibit_ops()) == 1, message="the set_inhibition op")

        _seq, name, params = intents.inhibit_ops()[0]
        assert name == "set_inhibition"
        assert set(params["behaviors"]) == {"speak", *BROWSING_INHIBITS}
        assert set(stack.status()["browsing_inhibits"]) == set(BROWSING_INHIBITS)
    finally:
        stack.stop()


def test_added_names_are_only_the_ones_not_already_live():
    """A name already inhibited for another reason is never counted as ours."""
    intents = FakeIntents()
    live = FakeLiveSet(["speak", "nod"])  # 'nod' already inhibited by someone else
    stack, _stop = running_stack(intents, current_inhibitions=live)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: len(intents.inhibit_ops()) == 1, message="the set_inhibition op")
        assert stack.status()["browsing_inhibits"] == ["orient-to-sound"]
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# Leaving browsing for wander restores only the added names                   #
# --------------------------------------------------------------------------- #


def test_leaving_browsing_for_wander_restores_only_the_added_names():
    intents = FakeIntents()
    live = FakeLiveSet(["speak"])
    stack, _stop = running_stack(intents, current_inhibitions=live)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: len(intents.inhibit_ops()) == 1, message="the enter op")

        # The runtime now reflects our add, AND the operator added something
        # of their own in the meantime.
        live.set(["speak", *BROWSING_INHIBITS, "antenna-sway"])

        stack.on_browser_state("idle")
        wait_for(lambda: stack.layer == LAYER_WANDER, message="wander layer")
        wait_for(lambda: len(intents.inhibit_ops()) == 2, message="the leave op")

        _seq, _name, params = intents.inhibit_ops()[1]
        assert set(params["behaviors"]) == {"speak", "antenna-sway"}
        assert stack.status()["browsing_inhibits"] == []
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# A conversation in between does not touch the inhibits                       #
# --------------------------------------------------------------------------- #


def test_conversation_does_not_remove_browsing_inhibits():
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    live = FakeLiveSet(["speak"])
    stack, _stop = running_stack(intents, attention=attention, current_inhibitions=live)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: len(intents.inhibit_ops()) == 1, message="the enter op")

        attention.conversation_live = True
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")
        time.sleep(TICK * 5)

        # No extra set_inhibition submitted just for entering conversation.
        assert len(intents.inhibit_ops()) == 1
        assert set(stack.status()["browsing_inhibits"]) == set(BROWSING_INHIBITS)
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# Conversation -> browsing re-adds only what is missing                       #
# --------------------------------------------------------------------------- #


def test_returning_to_browsing_from_conversation_readds_only_missing_names():
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    live = FakeLiveSet(["speak"])
    stack, _stop = running_stack(intents, attention=attention, current_inhibitions=live)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: len(intents.inhibit_ops()) == 1, message="the enter op")

        attention.conversation_live = True
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")

        # A face-lock replacement re-asserted 'orient-to-sound' (the runtime's
        # own) but dropped 'nod' (the harness's addition) along the way.
        live.set(["speak", "orient-to-sound"])

        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing again")
        wait_for(lambda: len(intents.inhibit_ops()) == 2, message="the re-add op")

        _seq, _name, params = intents.inhibit_ops()[1]
        assert set(params["behaviors"]) == {"speak", "orient-to-sound", "nod"}
        assert set(stack.status()["browsing_inhibits"]) == set(BROWSING_INHIBITS)
    finally:
        stack.stop()


def test_returning_to_browsing_with_nothing_missing_submits_no_extra_op():
    intents = FakeIntents()
    attention = FakeAttention(live=False)
    live = FakeLiveSet(["speak"])
    stack, _stop = running_stack(intents, attention=attention, current_inhibitions=live)
    try:
        stack.on_browser_state("busy")
        wait_for(lambda: len(intents.inhibit_ops()) == 1, message="the enter op")

        attention.conversation_live = True
        wait_for(lambda: stack.layer == LAYER_CONVERSATION, message="conversation layer")

        # Nothing dropped: both names are still live.
        live.set(["speak", *BROWSING_INHIBITS])

        attention.conversation_live = False
        wait_for(lambda: stack.layer == LAYER_BROWSING, message="browsing again")
        time.sleep(TICK * 5)

        assert len(intents.inhibit_ops()) == 1
        assert set(stack.status()["browsing_inhibits"]) == set(BROWSING_INHIBITS)
    finally:
        stack.stop()


# --------------------------------------------------------------------------- #
# status() and the default injectable                                         #
# --------------------------------------------------------------------------- #


def test_status_reports_browsing_inhibits_empty_by_default():
    intents = FakeIntents()
    stack = GazeStack(intents, tick_s=TICK)
    assert stack.status()["browsing_inhibits"] == []


def test_current_inhibitions_is_injectable_and_defaults_to_the_runtime_reader():
    intents = FakeIntents()
    # No current_inhibitions kwarg: must not raise at construction time, and
    # must fall back to reachy_nova.harness.tools.current_inhibitions (which
    # itself degrades to [] with no state dir wired up in this test).
    stack = GazeStack(intents, tick_s=TICK)
    assert stack._current_inhibitions() == []


# --------------------------------------------------------------------------- #
# Stopping while browsing gives the inhibits back (PR #26, comment 3943444434) #
# --------------------------------------------------------------------------- #


def test_stopping_while_browsing_restores_the_added_inhibits():
    """Stopping mid-browse must not leave orient-to-sound and nod disabled in
    the runtime after the harness that disabled them is gone."""
    intents = FakeIntents()
    live = FakeLiveSet(["speak"])
    stack, _stop = running_stack(intents, current_inhibitions=live)
    stack.on_browser_state("busy")
    wait_for(lambda: len(intents.inhibit_ops()) == 1, message="the enter op")
    assert set(stack.status()["browsing_inhibits"]) == set(BROWSING_INHIBITS)
    # The runtime now reflects our add, plus an operator's own later addition.
    live.set(["speak", *BROWSING_INHIBITS, "antenna-sway"])

    stack.stop()

    ops = intents.inhibit_ops()
    assert len(ops) == 2, intents.names()
    assert set(ops[1][2]["behaviors"]) == {"speak", "antenna-sway"}
    assert stack.status()["browsing_inhibits"] == []
    # ...alongside the usual stop hygiene: the standing goal is cleared too.
    assert intents.names().count("declare_goal") == 2  # the browse declare + the clear
    assert intents.names()[-2:] == ["declare_goal", "set_inhibition"] or intents.names()[
        -2:
    ] == ["set_inhibition", "declare_goal"]


def test_stopping_in_wander_submits_no_inhibition_op():
    intents = FakeIntents()
    live = FakeLiveSet(["speak"])
    stack, _stop = running_stack(intents, current_inhibitions=live)
    stack.stop()
    assert intents.inhibit_ops() == []
