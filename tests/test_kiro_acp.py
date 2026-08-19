"""Tests for reachy_nova.kiro_acp — the stdlib ACP client driving kiro-cli.

All tests drive a *fake* subprocess (never a real kiro-cli): a small stand-in
object exposing the same surface KiroAcpSession relies on
(``stdin``/``stdout``/``poll()``/``terminate()``/``wait()``/``kill()``).
``stdin`` is a recording buffer so tests can assert exact JSON-RPC framing;
``stdout`` is a blocking queue-backed line iterator so tests can script
responses/notifications (or withhold them, to exercise timeouts) independent
of what was written to stdin.
"""

from __future__ import annotations

import ast
import json
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

from reachy_nova import kiro_acp
from reachy_nova.kiro_acp import KiroAcpError, KiroAcpSession

MODULE_PATH = Path(__file__).resolve().parents[1] / "reachy_nova" / "kiro_acp.py"


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class FakeStdin:
    """Records every write; each ``write()`` call is one JSON-RPC line.

    ``on_write`` (if given) fires synchronously after recording, so a fake
    process can react to a request the moment it lands — this is what lets
    :class:`FakeProcess` answer each request with its scripted response
    instead of racing a reader thread against pre-queued stdout lines.
    """

    def __init__(self, on_write=None) -> None:
        self.lines: list[str] = []
        self.closed = False
        self._on_write = on_write

    def write(self, data: str) -> int:
        self.lines.append(data)
        if self._on_write is not None:
            self._on_write(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeStdout:
    """A blocking, queue-backed iterator standing in for a subprocess pipe.

    Lines are pushed with :meth:`push`; :meth:`end` pushes the sentinel that
    makes iteration stop (simulating EOF, i.e. the child process exiting). A
    test that never calls :meth:`end` and never pushes the awaited response
    exercises the timeout path — the reader thread just blocks on the queue
    forever, harmless because it is a daemon thread.
    """

    _SENTINEL = object()

    def __init__(self) -> None:
        self._q: "queue.Queue" = queue.Queue()

    def push(self, obj: dict) -> None:
        self._q.put(json.dumps(obj) + "\n")

    def end(self) -> None:
        self._q.put(self._SENTINEL)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is self._SENTINEL:
            raise StopIteration
        return item


class FakeProcess:
    """Stands in for ``subprocess.Popen`` in tests.

    Responses are *scripted* per JSON-RPC method (:meth:`script`) rather than
    pre-loaded onto stdout: the moment a matching request is written to
    stdin, this fake pushes the scripted notifications and then the response
    onto stdout. That keeps the fake honest to the real protocol (a response
    can never precede its request) and avoids racing the reader thread
    against a queue that already held the answer before the question was
    asked.
    """

    def __init__(self, argv: list) -> None:
        self.argv = argv
        self.stdout = FakeStdout()
        self.stdin = FakeStdin(on_write=self._on_write)
        self._returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._script: dict[str, list[dict]] = {}

    def script(
        self,
        method: str,
        *,
        notifications: list[dict] | None = None,
        result: dict | None = None,
        error: dict | None = None,
    ) -> None:
        """Queue one scripted answer for the next request calling *method*."""
        self._script.setdefault(method, []).append(
            {"notifications": notifications or [], "result": result, "error": error}
        )

    def _on_write(self, line: str) -> None:
        msg = json.loads(line)
        method = msg.get("method")
        req_id = msg.get("id")
        queued = self._script.get(method)
        if not queued:
            return
        spec = queued.pop(0)
        for note in spec["notifications"]:
            self.stdout.push(note)
        if spec.get("error") is not None:
            self.stdout.push({"jsonrpc": "2.0", "id": req_id, "error": spec["error"]})
        else:
            self.stdout.push({"jsonrpc": "2.0", "id": req_id, "result": spec.get("result") or {}})

    def poll(self):
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0
        self.stdout.end()

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9
        self.stdout.end()

    def wait(self, timeout: float | None = None) -> int:
        return self._returncode if self._returncode is not None else 0

    def die(self, code: int = 1) -> None:
        """Test helper: simulate the child exiting on its own."""
        self._returncode = code
        self.stdout.end()


def _written_messages(fake_stdin: FakeStdin) -> list[dict]:
    return [json.loads(line) for line in fake_stdin.lines]


def _make_session(**kwargs) -> tuple[KiroAcpSession, FakeProcess]:
    holder: dict = {}

    def factory(argv: list) -> FakeProcess:
        proc = FakeProcess(argv)
        holder["process"] = proc
        return proc

    session = KiroAcpSession(process_factory=factory, env={}, **kwargs)
    return session, holder


# --------------------------------------------------------------------------- #
# argv construction (engine / model / binary)                                #
# --------------------------------------------------------------------------- #


def test_argv_defaults_with_no_env():
    session = KiroAcpSession(env={})
    assert session.argv == [
        "kiro-cli",
        "acp",
        "--trust-all-tools",
        "--model",
        "minimax-m2.5",
        "--agent-engine",
        "v2",
    ]


def test_argv_reads_from_env():
    env = {
        "KIRO_CLI_BIN": "/opt/kiro/kiro-cli",
        "KIRO_MODEL": "some-other-model",
        "KIRO_AGENT_ENGINE": "v3",
    }
    session = KiroAcpSession(env=env)
    assert session.argv == [
        "/opt/kiro/kiro-cli",
        "acp",
        "--trust-all-tools",
        "--model",
        "some-other-model",
        "--agent-engine",
        "v3",
    ]


def test_argv_kwargs_override_env():
    env = {"KIRO_CLI_BIN": "env-bin", "KIRO_MODEL": "env-model", "KIRO_AGENT_ENGINE": "v3"}
    session = KiroAcpSession(binary="kw-bin", model="kw-model", agent_engine="v1", env=env)
    assert session.argv == [
        "kw-bin",
        "acp",
        "--trust-all-tools",
        "--model",
        "kw-model",
        "--agent-engine",
        "v1",
    ]


@pytest.mark.parametrize("engine", ["v1", "v2", "v3"])
def test_valid_agent_engines_accepted(engine):
    session = KiroAcpSession(agent_engine=engine, env={})
    assert f"--agent-engine" in session.argv
    assert session.argv[session.argv.index("--agent-engine") + 1] == engine


def test_invalid_agent_engine_kwarg_rejected():
    with pytest.raises(ValueError, match="v1.*v2.*v3|agent engine"):
        KiroAcpSession(agent_engine="v99", env={})


def test_invalid_agent_engine_env_rejected():
    with pytest.raises(ValueError):
        KiroAcpSession(env={"KIRO_AGENT_ENGINE": "nightly"})


def test_engine_default_is_v2_not_v3():
    # v3 is pass-through / opt-in only — never the silent default.
    session = KiroAcpSession(env={})
    assert session._agent_engine == "v2"


# --------------------------------------------------------------------------- #
# Full round trip: initialize -> session/new -> session/prompt               #
# --------------------------------------------------------------------------- #


def test_full_round_trip_accumulates_text_and_frames_requests():
    session, holder = _make_session()
    session.start()
    process = holder["process"]

    # Script the child's replies: each fires the moment the matching request
    # is written to stdin, so a response can never precede its request.
    process.script("initialize", result={"protocolVersion": 1})
    process.script("session/new", result={"sessionId": "sess-123"})
    process.script(
        "session/prompt",
        notifications=[
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "sess-123",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "Hello, "},
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "sess-123",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "world!"},
                    },
                },
            },
        ],
        result={"stopReason": "end_turn"},
    )

    init_result = session.initialize(timeout=5)
    assert init_result == {"protocolVersion": 1}

    session_id = session.new_session(cwd="/tmp/somewhere", timeout=5)
    assert session_id == "sess-123"
    assert session.session_id == "sess-123"

    reply = session.prompt("say hi", timeout=5)
    assert reply == "Hello, world!"

    # -- exact JSON-RPC request framing --------------------------------------
    written = _written_messages(process.stdin)
    assert len(written) == 3

    for raw_line in process.stdin.lines:
        assert raw_line.endswith("\n")
        assert raw_line.count("\n") == 1  # newline-delimited: exactly one line
        json.loads(raw_line)  # each write is independently valid JSON

    assert [m["jsonrpc"] for m in written] == ["2.0", "2.0", "2.0"]
    assert [m["id"] for m in written] == [1, 2, 3]
    assert [m["method"] for m in written] == ["initialize", "session/new", "session/prompt"]

    assert written[0]["params"]["protocolVersion"] == kiro_acp.PROTOCOL_VERSION
    assert written[1]["params"] == {"cwd": "/tmp/somewhere", "mcpServers": []}
    assert written[2]["params"] == {
        "sessionId": "sess-123",
        "prompt": [{"type": "text", "text": "say hi"}],
    }

    process.stdout.end()
    session.close(grace=1)


def test_prompt_buffer_resets_between_turns():
    session, holder = _make_session()
    session.start()
    process = holder["process"]

    process.script("initialize", result={})
    process.script("session/new", result={"sessionId": "s1"})
    process.script(
        "session/prompt",
        notifications=[
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "first"},
                    }
                },
            }
        ],
        result={"stopReason": "end_turn"},
    )
    process.script(
        "session/prompt",
        notifications=[
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "second"},
                    }
                },
            }
        ],
        result={"stopReason": "end_turn"},
    )

    session.initialize(timeout=5)
    session.new_session(cwd="/tmp", timeout=5)

    first = session.prompt("one", timeout=5)
    second = session.prompt("two", timeout=5)

    assert first == "first"
    assert second == "second"  # not "firstsecond" — the buffer must reset

    process.stdout.end()
    session.close(grace=1)


# --------------------------------------------------------------------------- #
# Failure modes                                                              #
# --------------------------------------------------------------------------- #


def test_prompt_before_session_raises():
    session, holder = _make_session()
    session.start()
    process = holder["process"]
    process.script("initialize", result={})
    session.initialize(timeout=5)

    with pytest.raises(KiroAcpError, match="new_session"):
        session.prompt("too early")

    process.stdout.end()
    session.close(grace=1)


def test_prompt_on_dead_process_raises_promptly():
    session, holder = _make_session()
    session.start()
    process = holder["process"]

    process.script("initialize", result={})
    process.script("session/new", result={"sessionId": "s1"})
    session.initialize(timeout=5)
    session.new_session(cwd="/tmp", timeout=5)

    # Simulate the child dying on its own.
    process.die(code=1)

    start = time.monotonic()
    with pytest.raises(KiroAcpError, match="not running"):
        session.prompt("hello", timeout=30)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # must fail fast, not wait on the (huge) timeout

    session.close(grace=1)


def test_prompt_timeout_raises_and_does_not_hang():
    session, holder = _make_session()
    session.start()
    process = holder["process"]

    process.script("initialize", result={})
    process.script("session/new", result={"sessionId": "s1"})
    session.initialize(timeout=5)
    session.new_session(cwd="/tmp", timeout=5)

    # session/prompt is never scripted — it must time out rather
    # than hang. The reader thread stays blocked reading from FakeStdout's
    # queue, but it is a daemon thread so the process/test can still exit.
    start = time.monotonic()
    with pytest.raises(KiroAcpError, match="timed out"):
        session.prompt("hello?", timeout=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0

    process.stdout.end()
    session.close(grace=1)


def test_initialize_raises_on_jsonrpc_error():
    session, holder = _make_session()
    session.start()
    process = holder["process"]
    process.script("initialize", error={"code": -32000, "message": "boom"})

    with pytest.raises(KiroAcpError, match="boom"):
        session.initialize(timeout=5)

    process.stdout.end()
    session.close(grace=1)


def test_send_before_start_raises():
    session, _holder = _make_session()
    with pytest.raises(KiroAcpError, match="not running"):
        session.initialize(timeout=5)


def test_double_start_raises():
    session, _holder = _make_session()
    session.start()
    with pytest.raises(KiroAcpError):
        session.start()
    session.close(grace=1)


# --------------------------------------------------------------------------- #
# alive / is_alive / close                                                    #
# --------------------------------------------------------------------------- #


def test_alive_reflects_process_state():
    session, holder = _make_session()
    assert session.alive is False
    assert session.is_alive() is False

    session.start()
    process = holder["process"]
    assert session.alive is True
    assert session.is_alive() is True

    process.die(code=0)
    assert session.alive is False
    session.close(grace=1)


def test_close_terminates_and_is_idempotent():
    session, holder = _make_session()
    session.start()
    process = holder["process"]
    process.script("initialize", result={})
    session.initialize(timeout=5)

    session.close(grace=1)
    assert process.terminated is True
    assert process.stdin.closed is True

    # Idempotent: closing an already-closed session must not raise.
    session.close(grace=1)


def test_close_on_never_started_session_is_a_noop():
    session, _holder = _make_session()
    session.close(grace=1)  # must not raise


# --------------------------------------------------------------------------- #
# Stdlib-only import guarantee                                                #
# --------------------------------------------------------------------------- #


def test_module_imports_stdlib_only():
    source = MODULE_PATH.read_text()
    tree = ast.parse(source, filename=str(MODULE_PATH))

    stdlib_names = set(sys.stdlib_module_names)
    # `__future__` is a real stdlib pseudo-module used for annotations import;
    # it's already covered by stdlib_module_names but keep this explicit for
    # readability/robustness across Python versions.
    stdlib_names.add("__future__")

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # A relative import (`from . import x`) — not a third-party
                # or even absolute stdlib import; skip it (kiro_acp.py has
                # none, but this keeps the check honest if that ever changes).
                continue
            if node.module:
                roots.add(node.module.split(".")[0])

    assert roots, "expected at least one import in kiro_acp.py"

    non_stdlib = roots - stdlib_names
    assert not non_stdlib, f"kiro_acp.py imports non-stdlib modules: {sorted(non_stdlib)}"

    # And, doubly explicit: reachy_nova itself must not appear (no importing
    # sibling application modules — this file is meant to be self-contained).
    assert "reachy_nova" not in roots
    assert "cultureagent" not in roots


# --------------------------------------------------------------------------- #
# Reader thread robustness                                                    #
# --------------------------------------------------------------------------- #


def test_unparseable_line_is_skipped_not_fatal():
    session, holder = _make_session()
    session.start()
    process = holder["process"]

    process.stdout._q.put("not json at all\n")
    process.script("initialize", result={"ok": True})

    result = session.initialize(timeout=5)
    assert result == {"ok": True}

    process.stdout.end()
    session.close(grace=1)


def test_reader_thread_is_daemon():
    session, holder = _make_session()
    session.start()
    assert session._reader_thread is not None
    assert session._reader_thread.daemon is True
    process = holder["process"]
    process.stdout.end()
    session.close(grace=1)


# --------------------------------------------------------------------------- #
# --agent selection (KIRO_AGENT / agent kwarg) — the nova-writer seam         #
# --------------------------------------------------------------------------- #


def test_agent_flag_absent_by_default():
    session = KiroAcpSession(env={})
    assert "--agent" not in session.argv


def test_agent_flag_from_env_and_kwarg():
    from_env = KiroAcpSession(env={"KIRO_AGENT": "nova-writer"})
    assert from_env.argv[-2:] == ["--agent", "nova-writer"]
    kwarg_wins = KiroAcpSession(agent="other", env={"KIRO_AGENT": "nova-writer"})
    assert kwarg_wins.argv[-2:] == ["--agent", "other"]
    empty_env_means_none = KiroAcpSession(env={"KIRO_AGENT": ""})
    assert "--agent" not in empty_env_means_none.argv
