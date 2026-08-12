"""Memory leg v1 — qq knowledge context routed through the nervous system (t14).

Everything here runs with **no network, no MongoDB, no Neo4j and no Bedrock**:
``NovaMemory`` is replaced by a fake object carrying the same callback surface
(``on_context`` / ``on_result`` attributes plus a ``query`` method), and the
rules consulted are the repository's REAL
``config/nervous-system/rules.yaml`` — the point of the task is that the leg
goes through that file, not around it.

The load-bearing property under test is the *absence* of a bypass:

- a memory context/result only ever reaches ``on_inject`` when
  :func:`reachy_nova.harness.bus.route_event` returned a rendered string for
  its ``"memory/<type>"`` key;
- an entry without an ``inject_template`` produces no inject and exactly one
  named ``[SENSE stage=memory ... reason=no-template]`` line;
- the wired callable is called with exactly ONE positional argument and no
  keywords, so ``NovaSonic.inject_text``'s own 3s throttle and speaking guard
  stay in the path (``force`` is never passed, and the module never so much as
  names it).
"""

from __future__ import annotations

import ast
import logging
import threading
from pathlib import Path

import pytest
import yaml

from reachy_nova.harness import bus
from reachy_nova.harness.memory_leg import MemoryLeg

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "config" / "nervous-system" / "rules.yaml"
MODULE_PATH = REPO_ROOT / "reachy_nova" / "harness" / "memory_leg.py"


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class FakeMemory:
    """Stands in for ``NovaMemory``: same callback surface, zero backends."""

    def __init__(self, answer: str = "", raises: Exception | None = None) -> None:
        self.on_context = None
        self.on_result = None
        self.on_progress = None
        self.on_state_change = None
        self.answer = answer
        self.raises = raises
        self.queries: list[str] = []
        self.gate = threading.Event()
        self.gate.set()
        self.called = threading.Event()

    def query(self, text: str) -> str:
        self.queries.append(text)
        self.called.set()
        self.gate.wait(5)
        if self.raises is not None:
            raise self.raises
        # Real NovaMemory fires on_result from inside query().
        if self.on_result is not None:
            self.on_result(self.answer)
        return self.answer


class Recorder:
    """Records exactly how ``on_inject`` was called — args AND kwargs."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))

    @property
    def texts(self) -> list[str]:
        return [args[0] for args, _ in self.calls if args]


@pytest.fixture
def real_rules() -> dict:
    """The repository's real nervous-system rules, loaded from disk."""
    return bus.load_rules(RULES_PATH)


def sense_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "nova.sensory"]


# --------------------------------------------------------------------------- #
# 1. rules.yaml carries the memory leg's entries                              #
# --------------------------------------------------------------------------- #


def test_rules_yaml_covers_the_memory_leg_event_types() -> None:
    rules = yaml.safe_load(RULES_PATH.read_text())["rules"]
    for key in ("memory/context", "memory/result"):
        assert key in rules, f"rules.yaml is missing {key}"
        entry = rules[key]
        assert "priority" in entry and "urgency" in entry and "llm_evaluate" in entry
        assert "{text}" in entry["inject_template"], (
            f"{key}'s inject_template must carry the text field"
        )
    # The untemplated memory entry the drop test relies on is still untemplated.
    assert not rules["memory/store_result"].get("inject_template")


# --------------------------------------------------------------------------- #
# 2. The happy path: context -> real rules -> on_inject                       #
# --------------------------------------------------------------------------- #


def test_context_event_reaches_on_inject_rendered_by_the_real_rules(real_rules) -> None:
    inject = Recorder()
    leg = MemoryLeg(FakeMemory(), inject, rules_cfg=real_rules)

    leg.on_context("Ori built you last spring")

    template = real_rules["rules"]["memory/context"]["inject_template"]
    expected = template.format(text="Ori built you last spring", context="Ori built you last spring")
    assert inject.texts == [expected]
    assert "Ori built you last spring" in inject.texts[0]


def test_result_event_reaches_on_inject_rendered_by_the_real_rules(real_rules) -> None:
    inject = Recorder()
    leg = MemoryLeg(FakeMemory(), inject, rules_cfg=real_rules)

    leg.on_result("the spare antennas are in the blue drawer")

    assert len(inject.texts) == 1
    assert "blue drawer" in inject.texts[0]


def test_inject_emits_a_named_sense_line(real_rules, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    leg = MemoryLeg(FakeMemory(), Recorder(), rules_cfg=real_rules)

    leg.on_context("something worth saying")

    lines = sense_lines(caplog)
    assert any("stage=memory" in line and "event=memory/context" in line for line in lines)
    assert any("injecting" in line and "priority=" in line for line in lines)


# --------------------------------------------------------------------------- #
# 3. No template -> no inject, and never silent                               #
# --------------------------------------------------------------------------- #


def test_untemplated_rule_does_not_reach_on_inject_and_is_named(real_rules, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    inject = Recorder()
    leg = MemoryLeg(FakeMemory(), inject, rules_cfg=real_rules)

    injected = leg.emit("store_result", {"text": "stored that for you"})

    assert injected is False
    assert inject.calls == []
    lines = sense_lines(caplog)
    assert any(
        "event=memory/store_result" in line and f"reason={bus.REASON_NO_TEMPLATE}" in line
        for line in lines
    ), lines


def test_unknown_event_type_falls_to_the_default_rule_and_drops(real_rules, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    inject = Recorder()
    leg = MemoryLeg(FakeMemory(), inject, rules_cfg=real_rules)

    assert leg.emit("no_such_memory_event", {"text": "hello"}) is False
    assert inject.calls == []
    assert any(f"reason={bus.REASON_NO_TEMPLATE}" in line for line in sense_lines(caplog))


def test_empty_text_never_reaches_the_router_or_on_inject(real_rules, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    inject = Recorder()
    routed: list[tuple] = []

    def spy_route(rules_cfg, source, event_type, payload):
        routed.append((source, event_type, payload))
        return "should never be injected", bus.REASON_INJECT

    leg = MemoryLeg(FakeMemory(), inject, route=spy_route, rules_cfg=real_rules)
    assert leg.on_context("   ") is False
    assert leg.on_result("") is False

    assert routed == []
    assert inject.calls == []
    assert any("reason=empty-text" in line for line in sense_lines(caplog))


# --------------------------------------------------------------------------- #
# 4. The gate itself: only a routed, non-None string may pass                 #
# --------------------------------------------------------------------------- #


def test_router_veto_blocks_the_inject_even_for_a_templated_type(real_rules) -> None:
    inject = Recorder()
    leg = MemoryLeg(
        FakeMemory(),
        inject,
        route=lambda *_args: (None, "vetoed-by-test"),
        rules_cfg=real_rules,
    )

    assert leg.on_context("this would otherwise be injected") is False
    assert inject.calls == []


def test_every_inject_is_preceded_by_a_route_call(real_rules) -> None:
    """No path may call ``on_inject`` without ``route_event`` deciding first."""
    order: list[str] = []
    inject = Recorder()

    def spy_route(rules_cfg, source, event_type, payload):
        order.append(f"route:{source}/{event_type}")
        return bus.route_event(rules_cfg, source, event_type, payload)

    def recording_inject(text):
        order.append("inject")
        inject(text)

    leg = MemoryLeg(FakeMemory(), recording_inject, route=spy_route, rules_cfg=real_rules)
    leg.on_context("a")
    leg.on_result("b")
    leg.emit("store_result", {"text": "c"})

    assert order == [
        "route:memory/context",
        "inject",
        "route:memory/result",
        "inject",
        "route:memory/store_result",
    ]


def test_the_leg_uses_bus_route_event_by_default(real_rules, monkeypatch) -> None:
    seen: list[tuple] = []
    real = bus.route_event

    def wrapper(rules_cfg, source, event_type, payload):
        seen.append((source, event_type))
        return real(rules_cfg, source, event_type, payload)

    monkeypatch.setattr(bus, "route_event", wrapper)
    leg = MemoryLeg(FakeMemory(), Recorder(), rules_cfg=real_rules)
    leg.on_context("routed through the module-level default")

    assert seen == [("memory", "context")]


# --------------------------------------------------------------------------- #
# 5. The throttle is preserved: bare, single-positional-arg inject calls      #
# --------------------------------------------------------------------------- #


def test_on_inject_is_called_with_exactly_one_positional_text_argument(real_rules) -> None:
    inject = Recorder()
    leg = MemoryLeg(FakeMemory(), inject, rules_cfg=real_rules)

    leg.on_context("hello")

    (args, kwargs), = inject.calls
    assert len(args) == 1
    assert isinstance(args[0], str)
    assert kwargs == {}, "a keyword (e.g. force=True) would bypass the Sonic guard"


def test_module_never_names_the_force_parameter() -> None:
    """``inject_text(text, force=False)``: the leg must never touch ``force``."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    offences = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "force":
            offences.append(f"line {node.value.lineno}: force= keyword")
        elif isinstance(node, ast.Name) and node.id == "force":
            offences.append(f"line {node.lineno}: name 'force'")
        elif isinstance(node, ast.Attribute) and node.attr == "force":
            offences.append(f"line {node.lineno}: attribute '.force'")
        elif isinstance(node, ast.arg) and node.arg == "force":
            offences.append(f"line {node.lineno}: parameter 'force'")
    assert not offences, f"memory_leg.py references inject_text's force flag: {offences}"


def test_every_on_inject_call_site_passes_one_bare_positional_arg() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    call_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_on_inject"
    ]
    assert call_sites, "expected at least one self._on_inject(...) call site"
    for call in call_sites:
        assert len(call.args) == 1, f"line {call.lineno}: on_inject must take one positional arg"
        assert not call.keywords, f"line {call.lineno}: on_inject must take no keywords"


# --------------------------------------------------------------------------- #
# 6. query(): background dispatch, never blocking the caller                  #
# --------------------------------------------------------------------------- #


def test_query_dispatches_in_the_background_without_blocking(real_rules) -> None:
    memory = FakeMemory(answer="the blue drawer")
    memory.gate.clear()  # hold query() open inside its thread
    inject = Recorder()
    leg = MemoryLeg(memory, inject, rules_cfg=real_rules)

    thread = leg.query("where are the antennas")

    assert memory.called.wait(5), "query was never dispatched"
    assert thread is not None and thread.is_alive()
    memory.gate.set()
    thread.join(5)
    assert memory.queries == ["where are the antennas"]
    assert leg.pending == 0


def test_query_result_returned_by_a_detached_memory_is_still_routed(real_rules) -> None:
    """A memory that returns instead of calling back still goes through the rules."""
    memory = FakeMemory(answer="a returned answer")
    memory.on_result = None  # never fires the callback
    inject = Recorder()
    leg = MemoryLeg(memory, inject, rules_cfg=real_rules)

    leg.query("anything").join(5)

    assert len(inject.texts) == 1
    assert "a returned answer" in inject.texts[0]


def test_query_does_not_double_inject_when_attached(real_rules) -> None:
    memory = FakeMemory(answer="one answer only")
    inject = Recorder()
    leg = MemoryLeg(memory, inject, rules_cfg=real_rules)
    leg.attach()

    leg.query("anything").join(5)

    assert len(inject.texts) == 1, inject.texts


def test_empty_query_is_dropped_by_name(real_rules, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    memory = FakeMemory()
    leg = MemoryLeg(memory, Recorder(), rules_cfg=real_rules)

    assert leg.query("   ") is None
    assert memory.queries == []
    assert any("reason=empty-query" in line for line in sense_lines(caplog))


# --------------------------------------------------------------------------- #
# 7. Graceful degradation                                                     #
# --------------------------------------------------------------------------- #


def test_backend_failure_inside_query_never_escapes_the_thread(real_rules, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    memory = FakeMemory(raises=RuntimeError("mongo unavailable"))
    inject = Recorder()
    leg = MemoryLeg(memory, inject, rules_cfg=real_rules)

    thread = leg.query("what do you know")
    thread.join(5)

    assert not thread.is_alive()
    assert inject.calls == []
    assert any("reason=query-failed" in line for line in sense_lines(caplog))
    assert leg.pending == 0


def test_a_raising_inject_callback_is_contained(real_rules, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")

    def boom(_text):
        raise RuntimeError("sonic stream is down")

    leg = MemoryLeg(FakeMemory(), boom, rules_cfg=real_rules)

    assert leg.on_context("still fine") is False
    assert any("reason=inject-failed" in line for line in sense_lines(caplog))


def test_missing_rules_file_degrades_to_the_default_rule(tmp_path) -> None:
    inject = Recorder()
    leg = MemoryLeg(FakeMemory(), inject, rules_path=tmp_path / "nope.yaml")

    assert leg.on_context("nothing to render this with") is False
    assert inject.calls == []


def test_on_inject_is_required() -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryLeg(FakeMemory(), None)


# --------------------------------------------------------------------------- #
# 8. attach(): wiring NovaMemory's callbacks without stealing them            #
# --------------------------------------------------------------------------- #


def test_attach_wires_memory_callbacks_to_the_routed_path(real_rules) -> None:
    memory = FakeMemory()
    inject = Recorder()
    leg = MemoryLeg(memory, inject, rules_cfg=real_rules)

    leg.attach()
    memory.on_context("wired context")
    memory.on_result("wired result")

    assert len(inject.texts) == 2
    assert "wired context" in inject.texts[0]
    assert "wired result" in inject.texts[1]


def test_attach_preserves_callbacks_the_app_already_wired(real_rules) -> None:
    seen: list[str] = []
    memory = FakeMemory()
    memory.on_result = seen.append  # e.g. main.py's state.update(...)
    leg = MemoryLeg(memory, Recorder(), rules_cfg=real_rules)

    leg.attach()
    memory.on_result("both must run")

    assert seen == ["both must run"]


def test_attach_survives_a_raising_pre_existing_callback(real_rules) -> None:
    memory = FakeMemory()

    def boom(_text):
        raise RuntimeError("dashboard update failed")

    memory.on_context = boom
    inject = Recorder()
    leg = MemoryLeg(memory, inject, rules_cfg=real_rules)

    leg.attach()
    memory.on_context("still reaches the rules path")

    assert len(inject.texts) == 1
