"""Memory leg — qq knowledge context, prioritized before it is ever spoken.

``reachy_nova/nova_memory.py`` already knows how to *retrieve*: local qq
markdown, MongoDB notes, the Neo4j graph, Bedrock synthesis, all of it
configured from the environment and all of it degrading quietly when a backend
is missing. This module does not re-implement any of that. It answers the one
question retrieval does not: **may this memory be said out loud right now, and
in what words?**

The rule the harness lives by
-----------------------------

Every sense Nova has reaches the conversation through exactly one gate —
``config/nervous-system/rules.yaml``, keyed ``"<source>/<type>"`` and rendered
by :func:`reachy_nova.harness.bus.route_event`. Memory is not special. So:

- a context/result arrives (from ``NovaMemory``'s ``on_context`` /
  ``on_result`` callbacks, or from a query this leg dispatched);
- it is keyed ``memory/context`` or ``memory/result`` and routed;
- **only** a non-``None`` routed string is handed to ``on_inject``.

There is no other path to ``on_inject`` in this module, and
``tests/test_harness_memory_leg.py`` asserts that statically as well as
behaviourally. A rule without an ``inject_template`` (``memory/store_result``,
and anything falling to the permissive ``default``) therefore produces no
inject at all — and exactly one named ``[SENSE stage=memory ...
reason=no-template]`` line, so a swallowed memory is never a silent one.

Why the inject callable is passed in bare
-----------------------------------------

The integrator wires ``on_inject=sonic.inject_text`` — the bound method, no
partial, no flags. ``NovaSonic.inject_text(text, force=False)`` carries its own
speaking guard and 3s throttle, and those are exactly the protections that keep
a burst of recalled notes from destabilizing the Bedrock bidirectional stream.
Passing ``force=True`` (or wrapping the callable to do so) would defeat them,
so this module never names that parameter and always calls the callback with
one positional argument.

Threading
---------

:meth:`MemoryLeg.query` dispatches ``memory.query`` on a daemon thread and
returns immediately — the 50Hz main loop must never wait on Mongo/Neo4j/
Bedrock. Nothing raised inside that thread escapes it: a dead backend resolves
to a named ``reason=query-failed`` line and an unchanged conversation.

stdlib + pyyaml only (via :mod:`reachy_nova.harness.bus`); ``reachy_mini`` is
never imported.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reachy_nova import sensory_log
from reachy_nova.harness import bus

# --------------------------------------------------------------------------- #
# Senselog identity                                                           #
# --------------------------------------------------------------------------- #

#: ``[SENSE stage=memory source=nova event=...]`` — every line this leg emits.
STAGE = "memory"
SOURCE = "nova"

# --------------------------------------------------------------------------- #
# Rules keys                                                                  #
# --------------------------------------------------------------------------- #

#: The rules.yaml source half of this leg's keys.
EVENT_SOURCE = "memory"
#: Knowledge Nova *has* (startup identity, recalled notes) — ``memory/context``.
EVENT_CONTEXT = "context"
#: The answer to a query this leg dispatched — ``memory/result``.
EVENT_RESULT = "result"

# --------------------------------------------------------------------------- #
# Named drop reasons (bus supplies the routing verdicts; these are ours)      #
# --------------------------------------------------------------------------- #

REASON_EMPTY_TEXT = "empty-text"
REASON_EMPTY_QUERY = "empty-query"
REASON_QUERY_FAILED = "query-failed"
REASON_INJECT_FAILED = "inject-failed"

#: ``(rules_cfg, source, event_type, payload) -> (text | None, reason)``.
Router = Callable[[dict[str, Any], str, str, dict[str, Any]], "tuple[str | None, str]"]


class MemoryLeg:
    """Routes qq memory context into the conversation through the rules gate.

    Args:
        memory: a ``NovaMemory``-shaped object — ``query(text)`` plus the
            assignable ``on_context`` / ``on_result`` callback attributes.
            Never constructed here: the app owns its env configuration.
        on_inject: **required.** Called with one positional string whenever a
            memory event routes to a rendered inject. Wired to
            ``NovaSonic.inject_text`` in the app, bare, so that method's own
            speaking guard and throttle stay in the path.
        route: the routing function, injectable for tests. ``None`` uses
            :func:`reachy_nova.harness.bus.route_event`, resolved at call time.
        rules_cfg: an already-loaded rules mapping (``{"rules": ..., "default":
            ...}``). ``None`` loads *rules_path*.
        rules_path: nervous-system rules file; ``None`` uses the repo's (or
            ``NOVA_RULES_PATH``). A missing file degrades to the permissive
            default rule, which carries no template — so it injects nothing.
    """

    def __init__(
        self,
        memory: Any,
        on_inject: Callable[[str], None],
        route: Router | None = None,
        rules_cfg: dict[str, Any] | None = None,
        rules_path: str | Path | None = None,
    ) -> None:
        if on_inject is None or not callable(on_inject):
            raise ValueError("MemoryLeg requires an on_inject callable (e.g. sonic.inject_text)")
        self._memory = memory
        self._on_inject = on_inject
        self._route = route
        self.rules = rules_cfg if rules_cfg is not None else bus.load_rules(rules_path)
        self._attached = False
        self._lock = threading.Lock()
        self._pending = 0

    # -- read-only status ---------------------------------------------------

    @property
    def attached(self) -> bool:
        """Have ``NovaMemory``'s callbacks been wired to this leg?"""
        return self._attached

    @property
    def pending(self) -> int:
        """Number of dispatched queries still running (tests/dashboards)."""
        with self._lock:
            return self._pending

    # -- wiring -------------------------------------------------------------

    def attach(self) -> None:
        """Wire the memory object's ``on_context``/``on_result`` to this leg.

        Any callback the app already installed (``state.update(...)`` and
        friends) is preserved and called first — attaching the nervous system
        must not steal the dashboard's updates. Idempotent.
        """
        if self._attached:
            return
        self._memory.on_context = _chain(getattr(self._memory, "on_context", None), self.on_context)
        self._memory.on_result = _chain(getattr(self._memory, "on_result", None), self.on_result)
        self._attached = True
        sensory_log.stage(STAGE, SOURCE, "attach", "wired on_context/on_result to the rules path")

    # -- inbound events -----------------------------------------------------

    def on_context(self, text: str) -> bool:
        """``NovaMemory.on_context`` sink — knowledge Nova already holds."""
        return self.emit(EVENT_CONTEXT, {"text": text, "context": text})

    def on_result(self, text: str) -> bool:
        """``NovaMemory.on_result`` sink — the answer to a memory query."""
        return self.emit(EVENT_RESULT, {"text": text, "result": text})

    def emit(self, event_type: str, payload: dict[str, Any] | None) -> bool:
        """Route one memory event; return whether it produced an inject.

        This is the ONLY place ``on_inject`` is called, and it is reached only
        after the router returned a rendered, non-empty string for
        ``"memory/<event_type>"``.
        """
        key = f"{EVENT_SOURCE}/{event_type}"
        fields: dict[str, Any] = dict(payload or {})
        text = str(fields.get("text") or "").strip()
        if not text:
            sensory_log.stage(STAGE, SOURCE, key, f"dropped reason={REASON_EMPTY_TEXT}")
            return False
        fields["text"] = text

        router = self._route or bus.route_event
        rendered, reason = router(self.rules, EVENT_SOURCE, event_type, fields)
        if rendered is None:
            sensory_log.stage(
                STAGE, SOURCE, key, f"dropped reason={reason} chars={len(text)}"
            )
            return False

        rule = bus.rule_for(self.rules, EVENT_SOURCE, event_type)
        sensory_log.stage(
            STAGE,
            SOURCE,
            key,
            f"injecting priority={rule.get('priority')} urgency={rule.get('urgency')} "
            f"chars={len(rendered)}",
        )
        try:
            # One positional argument, always: inject_text's speaking guard and
            # 3s throttle are the harness's flood protection and stay in path.
            self._on_inject(rendered)
        except Exception as err:
            sensory_log.stage(
                STAGE, SOURCE, key, f"dropped reason={REASON_INJECT_FAILED}: {err}"
            )
            return False
        return True

    # -- outbound queries ---------------------------------------------------

    def query(self, text: str) -> threading.Thread | None:
        """Ask the knowledge system, off the caller's thread.

        Returns the dispatch thread (``None`` if the query was empty). The
        caller never blocks and never sees an exception: a backend that is
        down resolves to a named senselog drop.

        When :meth:`attach` has been called the answer arrives through
        ``NovaMemory.on_result``; when it has not, the value ``query()``
        returned is routed here instead — either way through the rules gate,
        and never twice.
        """
        query_text = str(text or "").strip()
        if not query_text:
            sensory_log.stage(STAGE, SOURCE, "query", f"dropped reason={REASON_EMPTY_QUERY}")
            return None
        sensory_log.stage(
            STAGE, SOURCE, "query", f"dispatching chars={len(query_text)} attached={self._attached}"
        )
        with self._lock:
            self._pending += 1
        thread = threading.Thread(
            target=self._run_query,
            args=(query_text,),
            name="memory-leg-query",
            daemon=True,
        )
        thread.start()
        return thread

    def _run_query(self, query_text: str) -> None:
        """Body of the dispatch thread. Nothing escapes it."""
        try:
            answer = self._memory.query(query_text)
            if not self._attached and answer:
                # A detached memory object returns instead of calling back;
                # the answer still has to pass the same gate.
                self.on_result(str(answer))
            else:
                sensory_log.stage(
                    STAGE, SOURCE, "query", f"complete chars={len(str(answer or ''))}"
                )
        except Exception as err:
            # NovaMemory already degrades per-backend; this catches whatever it
            # cannot (no Bedrock client, an import error, a hung driver).
            sensory_log.stage(
                STAGE, SOURCE, "query", f"dropped reason={REASON_QUERY_FAILED}: {err}"
            )
        finally:
            with self._lock:
                self._pending -= 1


def _chain(
    previous: Callable[[str], Any] | None, ours: Callable[[str], Any]
) -> Callable[[str], Any]:
    """Call *previous* (guarded), then *ours*. Used to attach without stealing."""
    if previous is None:
        return ours

    def chained(text: str) -> Any:
        try:
            previous(text)
        except Exception as err:
            sensory_log.stage(
                STAGE, SOURCE, "attach", f"pre-existing callback raised, continuing: {err}"
            )
        return ours(text)

    return chained
