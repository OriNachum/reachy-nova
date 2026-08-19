"""Stdlib-only Agent Client Protocol (ACP) client for kiro-cli.

Reachy Nova's skill-forge and future self-extension work want a way to drive
a real coding agent (Kiro CLI 2.18.1, via ``kiro-cli acp``) as a subprocess
and hold a conversation with it. ACP — Zed's Agent Client Protocol — is
newline-delimited JSON-RPC 2.0 over the child process's stdin/stdout, so no
network client or SDK is required: a subprocess plus a line reader is the
whole transport. This module is deliberately stdlib-only (``subprocess``,
``threading``, ``json``, ``concurrent.futures``, ``logging``) so it carries no
new dependency for a self-extension feature that is still exploratory.

Protocol flow
--------------
1. Spawn ``kiro-cli acp --trust-all-tools --model <model> --agent-engine
   <engine>``. ``--trust-all-tools`` means kiro auto-approves its own tool
   calls, so ``session/request_permission`` is not expected on the happy
   path (a stray one is logged, not handled, per the task brief).
2. ``initialize`` — negotiate protocol version and client capabilities.
3. ``session/new`` — given a ``cwd``, returns a ``sessionId`` that every
   subsequent ``session/prompt`` call rides on.
4. ``session/prompt`` — one blocking round trip per conversational turn.
   While the turn is in flight the agent streams ``session/update``
   notifications (assistant message chunks, thoughts, tool calls); this
   client accumulates the text of every ``agent_message_chunk`` update and
   hands it back as the return value of :meth:`KiroAcpSession.prompt` once
   the matching response (carrying ``stopReason``) arrives.

Threading model
----------------
A single background reader thread (started by :meth:`start`) owns
``process.stdout``: it decodes each newline-delimited JSON object and either
resolves a pending request's :class:`concurrent.futures.Future` (a message
carrying ``id`` and ``result``/``error``) or dispatches a notification
(``session/update``). Callers (``initialize``/``new_session``/``prompt``) are
synchronous from their own thread's point of view — they write a request line
and block on that request's ``Future.result(timeout=...)``. Because
``Future`` is itself thread-safe, the only additional lock is a short one
guarding the request-id counter and the pending-requests dict.

If the child process exits (stdout hits EOF) the reader thread fails every
still-pending future with :class:`KiroAcpError` rather than leaving a caller
blocked forever; :meth:`prompt` also rejects immediately, before writing
anything, if the process has already exited.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Env-configured defaults                                                     #
# --------------------------------------------------------------------------- #

#: Env var naming the kiro-cli executable; a bare name is resolved via PATH.
BINARY_ENV = "KIRO_CLI_BIN"
DEFAULT_BINARY = "kiro-cli"

#: Env var naming the model kiro-cli should drive.
MODEL_ENV = "KIRO_MODEL"
DEFAULT_MODEL = "minimax-m2.5"

#: Env var selecting kiro's agent engine. v3 is pass-through (opt-in only) —
#: it is never the default.
AGENT_ENGINE_ENV = "KIRO_AGENT_ENGINE"
DEFAULT_AGENT_ENGINE = "v2"
VALID_AGENT_ENGINES = frozenset({"v1", "v2", "v3"})

# ACP protocol constants for the `initialize` handshake.
PROTOCOL_VERSION = 1
CLIENT_NAME = "reachy-nova"
CLIENT_VERSION = "0.1.0"

DEFAULT_INITIALIZE_TIMEOUT = 30.0
DEFAULT_PROMPT_TIMEOUT = 300.0
DEFAULT_CLOSE_GRACE = 5.0

#: Factory building the child process: takes the argv list, returns an object
#: with the ``subprocess.Popen`` surface this module relies on (``stdin``,
#: ``stdout``, ``poll()``, ``terminate()``, ``wait(timeout=...)``, ``kill()``).
#: Injectable so tests never spawn a real kiro-cli.
ProcessFactory = Callable[[list], Any]


class KiroAcpError(RuntimeError):
    """Raised for ACP protocol errors, a dead child process, or a timeout."""


def _default_process_factory(argv: list) -> subprocess.Popen:
    """Spawn *argv* as a real subprocess, text-mode, line-buffered pipes."""
    return subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


class KiroAcpSession:
    """Drives one ``kiro-cli acp`` subprocess through the ACP handshake and turns.

    Args:
        binary: kiro-cli executable; defaults to ``KIRO_CLI_BIN`` env or
            :data:`DEFAULT_BINARY`.
        model: model name passed as ``--model``; defaults to ``KIRO_MODEL``
            env or :data:`DEFAULT_MODEL`.
        agent_engine: one of ``"v1"``/``"v2"``/``"v3"``, passed as
            ``--agent-engine``; defaults to ``KIRO_AGENT_ENGINE`` env or
            :data:`DEFAULT_AGENT_ENGINE`. Anything else raises ``ValueError``
            at construction time.
        process_factory: zero-dependency injection point for tests — a
            callable ``argv -> process-like object``. Defaults to spawning a
            real ``subprocess.Popen``.
        env: mapping consulted instead of ``os.environ`` for the three env
            vars above (test convenience; production code omits this).
    """

    def __init__(
        self,
        *,
        binary: str | None = None,
        model: str | None = None,
        agent_engine: str | None = None,
        process_factory: ProcessFactory | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        source: Mapping[str, str] = os.environ if env is None else env

        self._binary = binary if binary is not None else source.get(BINARY_ENV, DEFAULT_BINARY)
        self._model = model if model is not None else source.get(MODEL_ENV, DEFAULT_MODEL)

        resolved_engine = (
            agent_engine if agent_engine is not None else source.get(AGENT_ENGINE_ENV, DEFAULT_AGENT_ENGINE)
        )
        if resolved_engine not in VALID_AGENT_ENGINES:
            raise ValueError(
                f"invalid kiro agent engine {resolved_engine!r}; expected one of "
                f"{sorted(VALID_AGENT_ENGINES)}"
            )
        self._agent_engine = resolved_engine

        self._process_factory = process_factory or _default_process_factory

        self._process: Any | None = None
        self._reader_thread: threading.Thread | None = None

        self._id_lock = threading.Lock()
        self._request_id = 0

        self._pending_lock = threading.Lock()
        self._pending: dict[int, Future] = {}

        self._session_id: str | None = None

        # Guards the accumulated-text buffer, written by the reader thread
        # (session/update notifications) and read/reset by prompt() on the
        # caller's thread.
        self._text_lock = threading.Lock()
        self._accumulated_text: list[str] = []

    # -- construction / argv -------------------------------------------------

    @property
    def argv(self) -> list[str]:
        """The exact argv :meth:`start` spawns."""
        return [
            self._binary,
            "acp",
            "--trust-all-tools",
            "--model",
            self._model,
            "--agent-engine",
            self._agent_engine,
        ]

    # -- status ---------------------------------------------------------------

    @property
    def alive(self) -> bool:
        """Is the child process running right now?"""
        return self._process is not None and self._process.poll() is None

    def is_alive(self) -> bool:
        """Method form of :attr:`alive`, for callers that prefer a verb."""
        return self.alive

    @property
    def session_id(self) -> str | None:
        """The ACP ``sessionId`` returned by :meth:`new_session`, if any."""
        return self._session_id

    # -- lifecycle --------------------------------------------------------------

    def start(self) -> None:
        """Spawn the kiro-cli subprocess and start the background reader thread."""
        if self._process is not None:
            raise KiroAcpError("KiroAcpSession.start() called twice")
        argv = self.argv
        logger.info("kiro_acp: spawning %s", " ".join(argv))
        self._process = self._process_factory(argv)
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="kiro-acp-reader", daemon=True
        )
        self._reader_thread.start()

    def initialize(self, timeout: float = DEFAULT_INITIALIZE_TIMEOUT) -> dict:
        """Run the ACP ``initialize`` handshake; returns its ``result`` dict."""
        return self._send_request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {
                    "fs": {"readTextFile": True, "writeTextFile": True},
                    "terminal": True,
                },
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
            timeout=timeout,
        )

    def new_session(self, cwd: str, timeout: float = DEFAULT_INITIALIZE_TIMEOUT) -> str:
        """Open a new ACP session rooted at *cwd*; returns its ``sessionId``."""
        result = self._send_request(
            "session/new",
            {"cwd": cwd, "mcpServers": []},
            timeout=timeout,
        )
        session_id = result.get("sessionId")
        if not session_id:
            raise KiroAcpError(f"session/new returned no sessionId: {result!r}")
        self._session_id = session_id
        return session_id

    def prompt(self, text: str, timeout: float = DEFAULT_PROMPT_TIMEOUT) -> str:
        """Send one conversational turn; returns the accumulated assistant text.

        Raises :class:`KiroAcpError` promptly (no waiting) if the child
        process has already exited, if the request times out, or if the
        response carries a JSON-RPC error.
        """
        if self._session_id is None:
            raise KiroAcpError("prompt() called before a session exists — call new_session() first")

        with self._text_lock:
            self._accumulated_text = []

        result = self._send_request(
            "session/prompt",
            {"sessionId": self._session_id, "prompt": [{"type": "text", "text": text}]},
            timeout=timeout,
        )
        stop_reason = result.get("stopReason")
        if stop_reason is not None and stop_reason != "end_turn":
            logger.warning("kiro_acp: prompt turn ended with stopReason=%r", stop_reason)

        with self._text_lock:
            return "".join(self._accumulated_text)

    def close(self, grace: float = DEFAULT_CLOSE_GRACE) -> None:
        """Terminate the child gracefully: terminate, wait, kill as last resort.

        Idempotent — safe to call on a never-started or already-closed session.
        """
        process = self._process
        if process is None:
            return

        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception as err:  # noqa: BLE001 - best-effort during shutdown
            logger.debug("kiro_acp: closing stdin raised: %s", err)

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                logger.warning("kiro_acp: process did not exit within %ss of terminate(); killing", grace)
                process.kill()
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    logger.warning("kiro_acp: process still alive after kill()")
            except Exception as err:  # noqa: BLE001 - best-effort during shutdown
                logger.debug("kiro_acp: terminate/wait raised: %s", err)

        self._fail_pending("session closed")

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=grace)

    # -- request/response plumbing --------------------------------------------

    def _next_id(self) -> int:
        with self._id_lock:
            self._request_id += 1
            return self._request_id

    def _send_request(self, method: str, params: dict, timeout: float) -> dict:
        process = self._process
        if process is None or process.poll() is not None:
            raise KiroAcpError(f"cannot send {method!r}: kiro-cli process is not running")

        req_id = self._next_id()
        future: Future = Future()
        with self._pending_lock:
            self._pending[req_id] = future

        message = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        line = json.dumps(message) + "\n"
        try:
            process.stdin.write(line)
            process.stdin.flush()
        except Exception as err:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise KiroAcpError(f"failed writing {method!r} to kiro-cli stdin: {err}") from err

        try:
            response = future.result(timeout=timeout)
        except FutureTimeoutError:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise KiroAcpError(f"{method!r} timed out after {timeout}s waiting on kiro-cli") from None

        if "error" in response:
            raise KiroAcpError(f"{method!r} returned a JSON-RPC error: {response['error']!r}")
        return response.get("result") or {}

    def _read_loop(self) -> None:
        """Background reader: decode each stdout line and dispatch it."""
        process = self._process
        assert process is not None
        try:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("kiro_acp: unparseable line from kiro-cli: %s", line[:200])
                    continue
                self._dispatch(msg)
        except Exception as err:  # noqa: BLE001 - reader thread must never crash silently
            logger.warning("kiro_acp: reader loop error: %s", err)
        finally:
            self._fail_pending("kiro-cli process exited")

    def _dispatch(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):
            req_id = msg["id"]
            with self._pending_lock:
                future = self._pending.pop(req_id, None)
            if future is not None and not future.done():
                future.set_result(msg)
            return

        method = msg.get("method")
        if method == "session/update":
            self._handle_update(msg.get("params") or {})
        elif method == "session/request_permission":
            # --trust-all-tools means we do not expect these; log rather than
            # silently drop, since a stray one signals a config mismatch.
            logger.warning("kiro_acp: unexpected session/request_permission (--trust-all-tools set): %s", msg)
        else:
            logger.debug("kiro_acp: ignoring unhandled notification: %s", msg)

    def _handle_update(self, params: dict) -> None:
        update = params.get("update", params)
        kind = update.get("sessionUpdate", "")
        if kind != "agent_message_chunk":
            return
        content = update.get("content") or {}
        if content.get("type") != "text":
            return
        text = content.get("text", "")
        if not text:
            return
        with self._text_lock:
            self._accumulated_text.append(text)

    def _fail_pending(self, reason: str) -> None:
        with self._pending_lock:
            pending = list(self._pending.items())
            self._pending.clear()
        for _req_id, future in pending:
            if not future.done():
                future.set_exception(KiroAcpError(reason))
