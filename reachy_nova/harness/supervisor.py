"""Process supervisor for the harness: exclusivity, lifecycle, observability.

This is the harness's own control plane — what ``python -m reachy_nova.harness``
runs, and what ``reachy-nova-harness.service`` (:mod:`reachy_nova.harness.unit`)
starts on the robot. It mirrors ``reachy-mini-cli``'s PID-file + argv-identity
idiom (``reachy/procsup.py``, ``reachy/embody/supervisor.py``) rather than
inventing a second one, but it is deliberately the FOREGROUND half only: systemd
owns backgrounding, restarts and logging, so nothing here spawns detached
processes or escalates signals.

Three jobs, in the order they matter:

**1. Exclusivity.** The ``agent embody`` layer and this harness are two AI
attachments to the same robot, and they would fight over the same three scarce
resources — the audio-tee socket (one reader per frame budget), the intent spool
(two writers interleaving contradictory intents) and the speaker. So a live
embody layer REFUSES this start outright (exit :data:`EXIT_EMBODY_LIVE`) with a
named ``[SENSE …]`` line. A warning would not do: the failure mode of running
both is a robot that stutters and contradicts itself, which reads as a hardware
fault rather than a configuration one.

**2. Identity.** Our own PID file (``<state>/nova-harness.pid``) is claimed on
start. A *stale* file (dead PID) is reclaimed silently-but-named; a *live* file
whose process really is another harness refuses (exit
:data:`EXIT_ALREADY_RUNNING`). Liveness alone is never enough — PIDs are reused
— so identity is confirmed with EXACT argv tokens (see :func:`_is_our_harness`),
never a substring of the joined command line, which any process launched from a
checkout named after this package would satisfy.

**3. Observability.** The engine heartbeat is polled and every TRANSITION is a
named line (``engine live`` / ``dropped reason=engine-heartbeat-lost``). The
harness keeps running when the engine drops — it is a peripheral and re-attaches
— so without this line "the runtime died and nothing reacted" and "everything is
fine and nobody spoke" look identical in the log.

Standard library only (plus an optional, lazily imported ``python-dotenv``).
This package must never import ``reachy_mini`` or ``reachy``, and contains no
motion code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import signal
import sys
import subprocess  # nosec B404 - passed through to unit.install_unit's fixed argv
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from .. import sensory_log
from . import quiet as quiet_mod
from . import statedir, unit

# Exit codes (operator-facing contract; 1 stays "unexpected error").
EXIT_OK = 0
#: Another harness already holds the PID file.
EXIT_ALREADY_RUNNING = 2
#: The ``agent embody`` layer is live — we would fight it over audio + spool.
EXIT_EMBODY_LIVE = 3
#: Composition AND the degraded fallback both produced zero components — a
#: harness with nothing to run must fail (systemd restarts it), not sit inert.
EXIT_NO_COMPONENTS = 4

#: How often the watch loop re-reads the engine heartbeat.
DEFAULT_POLL_INTERVAL = 2.0

#: The exact argv token our own spawn line carries (``-m reachy_nova.harness``).
IDENTITY_TOKEN = "reachy_nova.harness"

#: Optional sibling modules the harness composes itself from. Each may be
#: absent (a partial install, an in-flight branch); an absent one is NAMED and
#: skipped, never fatal. A present one must expose ``build_component()``
#: returning an object with ``start(stop_event)`` / ``stop()``.
OPTIONAL_COMPONENTS: tuple[str, ...] = ("bus", "hearing", "speaking", "tools")

_STAGE = "supervise"
_SOURCE = "nova"


def _log(event: str, detail: str) -> None:
    sensory_log.stage(_STAGE, _SOURCE, event, detail)


# --------------------------------------------------------------------------- #
# Process identity.
# --------------------------------------------------------------------------- #


def _is_alive(pid: int) -> bool:
    """Does *pid* exist? Signal 0 asks without delivering anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process — it exists, we just may not signal it.
        return True
    return True


def _pid_argv(pid: int) -> list[str]:
    """*pid*'s real argv, split on NUL as the kernel stores it.

    Deliberately a local copy of :func:`reachy_nova.harness.statedir._pid_argv`
    rather than a call into that module's private name: this is the seam tests
    monkeypatch to fake another process's command line, and it must be the
    function THIS module resolves at call time.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _is_our_harness(pid: int) -> bool:
    """Is *pid* really another harness — or a stranger that reused the PID?

    Matched as an EXACT argv token. A substring test against the joined command
    line would also match ``/home/pi/git/reachy_nova.harness/bin/python`` and
    any other path that merely CONTAINS the module name, which would make a
    reclaimable stale file look like a live sibling forever. Where ``/proc`` is
    unavailable we cannot verify, so we trust the PID file (refuse) — the safe
    direction, since the cost is a refused start an operator can see, not a
    second harness fighting the first one in silence.
    """
    if not Path("/proc").is_dir():
        return True
    return IDENTITY_TOKEN in _pid_argv(pid)


# --------------------------------------------------------------------------- #
# The PID file.
# --------------------------------------------------------------------------- #


def read_pid() -> int | None:
    """The recorded harness PID, or ``None`` if absent/unparseable."""
    try:
        return int(statedir.harness_pid_path().read_text().strip())
    except (OSError, ValueError):
        return None


def acquire_pid_file() -> bool:
    """Claim the harness PID file. ``False`` means a live sibling holds it.

    The claim is ATOMIC (``O_CREAT | O_EXCL``): two concurrent starts race the
    kernel, not a read-then-write window, so exactly one wins. A stale file
    (dead PID, or a live PID that is NOT a harness — PID reuse) is reclaimed
    and named in the log; the reclaim unlinks and retries the atomic create
    once, so losing that second race also refuses cleanly. Only a genuine live
    sibling refuses with its file left untouched.
    """
    path = statedir.harness_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = read_pid()
            if (
                existing is not None
                and existing != os.getpid()
                and _is_alive(existing)
                and _is_our_harness(existing)
            ):
                _log("start", f"refused reason=already-running pid={existing}")
                return False
            _log("start", f"reclaimed stale pid={existing}")
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        return True
    _log("start", "refused reason=already-running (lost the reclaim race)")
    return False


def release_pid_file() -> None:
    """Remove the PID file — but only while it still names US.

    A file that names another PID belongs to whoever wrote it after us; removing
    it would hand a third harness a free claim over a live sibling.
    """
    if read_pid() != os.getpid():
        return
    try:
        statedir.harness_pid_path().unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------- #
# Components.
# --------------------------------------------------------------------------- #


def _component_name(component: object) -> str:
    return str(getattr(component, "name", type(component).__name__))


def build_components(names: Iterable[str] = OPTIONAL_COMPONENTS) -> list[object]:
    """Import each optional sibling module and collect its component.

    Every import is LAZY and individually guarded: the harness's subsystems land
    over several changes, and a supervisor that cannot start until all of them
    exist is a supervisor nobody can run in between. An absent module — or a
    present one with no ``build_component()`` factory — is named in the log and
    skipped, so "we started without hearing" is visible rather than inferred.
    """
    components: list[object] = []
    for name in names:
        try:
            module = importlib.import_module(f"{__package__}.{name}")
        except ImportError:
            _log("component", f"component absent name={name} reason=import-error")
            continue
        factory = getattr(module, "build_component", None)
        if factory is None:
            _log("component", f"component absent name={name} reason=no-factory")
            continue
        try:
            components.append(factory())
        except Exception as err:  # noqa: BLE001 - one component must not sink the rest
            _log("component", f"component absent name={name} reason=build-failed detail={err}")
    return components


def _composed_components() -> list[object]:
    """The real wired graph (app.build_app), degrading to bare components.

    ``build_components`` starts whatever exists un-wired — enough to observe a
    box, never enough to talk. The composition root is preferred; if it cannot
    build (a missing heavy dependency, a bad env), the failure is named and the
    harness still comes up in the degraded shape.
    """
    try:
        from . import app

        return app.build_app()
    except Exception as err:  # noqa: BLE001 - degraded beats dead
        _log("component", f"composition failed reason={err}; running degraded components")
        return build_components()


def _start_components(components: Sequence[object], stop_event: threading.Event) -> list[object]:
    started: list[object] = []
    for component in components:
        name = _component_name(component)
        try:
            component.start(stop_event)  # type: ignore[attr-defined]
        except Exception as err:  # noqa: BLE001 - a dead mic must not cost us the voice
            _log("component", f"start failed name={name} detail={err}")
            continue
        started.append(component)
        _log("component", f"started name={name}")
    return started


def _stop_components(started: Sequence[object]) -> None:
    for component in reversed(list(started)):
        name = _component_name(component)
        try:
            component.stop()  # type: ignore[attr-defined]
        except Exception as err:  # noqa: BLE001 - shutdown reports, never raises
            _log("component", f"stop failed name={name} detail={err}")


# --------------------------------------------------------------------------- #
# The watch loop.
# --------------------------------------------------------------------------- #


def _find_lock_state(components: Sequence[object]) -> object | None:
    """The first component exposing a non-``None`` ``.lock_state`` attribute.

    ``IntentTools`` (``tools.py``) carries the harness's one
    :class:`~reachy_nova.harness.lock_state.LockState` belief as a public
    attribute rather than a second constructor argument threaded through
    every layer between ``app.build_app`` and this module — this is the seam
    that finds it again on the component list :func:`run` already receives.
    """
    for component in components:
        candidate = getattr(component, "lock_state", None)
        if candidate is not None:
            return candidate
    return None


def run(
    components: Sequence[object],
    stop_event: threading.Event,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    tick_hook: Callable[[int], None] | None = None,
    lock_state: object | None = None,
) -> None:
    """Start every component, then watch the engine heartbeat until stopped.

    The loop itself does almost nothing on purpose: each component owns its own
    thread, so this is only the place that (a) keeps the process alive and (b)
    turns the engine's heartbeat into named transitions. *tick_hook* is the test
    seam — it is called with the tick count after each observation, which is how
    a heartbeat transition is exercised inside a single :func:`run` call.

    *lock_state* (duck-typed: anything with an ``on_engine_dropped()`` method,
    e.g. :class:`~reachy_nova.harness.lock_state.LockState`) is notified on
    the live -> dropped transition, so a locally-believed gaze lock does not
    outlive the engine process that actually held it — see that method's
    docstring for why this is a supervisor concern rather than a bus one.
    ``None`` (the default, and every non-t13 caller) skips this entirely.
    """
    started = _start_components(components, stop_event)
    _log("start", f"harness up pid={os.getpid()} components={len(started)}")
    previous: bool | None = None
    ticks = 0
    try:
        while not stop_event.is_set():
            live = statedir.engine_is_live()
            if live != previous:
                _log_engine_transition(live, previous, lock_state)
                previous = live
            ticks += 1
            if tick_hook is not None:
                tick_hook(ticks)
            if stop_event.wait(poll_interval):
                break
    finally:
        _stop_components(started)
        _log("stop", f"harness down pid={os.getpid()} ticks={ticks}")


def _log_engine_transition(live: bool, previous: bool | None, lock_state: object | None) -> None:
    """Log the engine heartbeat TRANSITION and, on a live -> dropped edge,
    notify *lock_state* (see :func:`run`'s docstring for why).
    """
    if live:
        _log("engine", "engine live")
        return
    if previous is None:
        _log("engine", "engine absent")
        return
    _log("engine", "dropped reason=engine-heartbeat-lost")
    if lock_state is None:
        return
    try:
        lock_state.on_engine_dropped()  # type: ignore[attr-defined]
    except Exception as err:  # noqa: BLE001 - a belief update must not kill the loop
        _log("engine", f"lock-state update failed detail={err}")


def install_signal_handlers(stop_event: threading.Event) -> Callable[[], None]:
    """Make SIGTERM/SIGINT set *stop_event*; return a restore callable.

    systemd stops us with SIGTERM, and an operator with Ctrl-C: both must reach
    the same orderly shutdown (components stopped, PID file released) rather
    than killing the process mid-write. Installing only works on the main
    thread, so a non-main caller degrades to a no-op restore.
    """
    previous: dict[int, object] = {}

    def _handler(signum, _frame):  # pragma: no cover - exercised by the OS
        _log("stop", f"signal received signal={signal.Signals(signum).name}")
        stop_event.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, _handler)
        except (ValueError, OSError):
            return lambda: None

    def restore() -> None:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)  # type: ignore[arg-type]
            except (ValueError, OSError, TypeError):
                pass

    return restore


# --------------------------------------------------------------------------- #
# Commands.
# --------------------------------------------------------------------------- #


def _load_env_file(path: str) -> None:
    """Load *path* into the environment before anything reads a credential."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415 - optional, resolved at call time
    except ImportError:
        _log("start", f"env file skipped reason=no-dotenv path={path}")
        return
    load_dotenv(path, override=False)


def _configure_logging() -> None:
    """INFO-level stderr logging so every [SENSE] line reaches journald.

    Without a configured handler Python drops INFO records, which makes the
    harness a silent cognition death — exactly what the observability
    requirement forbids. Idempotent: an already-configured root is respected.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s",
        stream=sys.stderr,
    )


def cmd_run(env_file: str | None = None) -> int:
    """The ``run`` verb: refuse if we would fight a peer, else watch until stopped."""
    _configure_logging()
    if env_file:
        _load_env_file(env_file)
    if statedir.embody_is_live():
        # Named, not merely refused: two AI attachments on one robot show up as
        # stuttering audio, which reads like a hardware fault.
        _log("start", "refused reason=embody-live")
        return EXIT_EMBODY_LIVE
    if not acquire_pid_file():
        return EXIT_ALREADY_RUNNING
    components = _composed_components()
    if not components:
        # An inert harness looks alive to systemd while doing nothing at all —
        # the one shape worse than dead. Fail loudly and let Restart= retry.
        _log("start", "refused reason=no-components (composition and fallback both empty)")
        release_pid_file()
        return EXIT_NO_COMPONENTS
    stop_event = threading.Event()
    restore = install_signal_handlers(stop_event)
    try:
        run(components, stop_event, lock_state=_find_lock_state(components))
    finally:
        restore()
        release_pid_file()
    return EXIT_OK


def status(
    quiet: "quiet_mod.QuietState | None" = None,
    lock_state: object | None = None,
) -> dict[str, object]:
    """Engine / embody / own-PID liveness — the whole attachment picture.

    *quiet* is the running harness's own :class:`~reachy_nova.harness.quiet.QuietState`
    when there is one in this process. The ``status`` CLI runs in a DIFFERENT
    process from the harness, so with no instance to hand the persisted
    deadline is read side-effect-free instead — "why is it not talking?" must
    be answerable from outside the harness, which is where the operator is.

    *lock_state* is the running harness's own (in-process)
    :class:`~reachy_nova.harness.lock_state.LockState`, same seam as *quiet*
    (task t11). Unlike the quiet deadline, the lock belief has no on-disk
    mirror — the runtime does not (yet) publish lock state into state.json —
    so an out-of-process ``status`` CLI call always reports ``locked: None``
    (unknown), which is the honest answer for a belief that lives only inside
    the running harness's own memory.
    """
    pid = read_pid()
    running = pid is not None and _is_alive(pid) and _is_our_harness(pid)
    quiet_until = quiet.until_iso() if quiet is not None else quiet_mod.peek_until_iso()
    locked = lock_state.locked if lock_state is not None else None  # type: ignore[attr-defined]
    return {
        "state_dir": str(statedir.state_dir()),
        "engine_live": statedir.engine_is_live(),
        "embody_live": statedir.embody_is_live(),
        "harness_pid": pid,
        "harness_running": running,
        "quiet_until": quiet_until,
        "locked": locked,
        "unit": unit.HARNESS_UNIT,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reachy_nova.harness",
        description="Reachy Nova harness — an AI peripheral over the symbolic runtime.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run the harness in the foreground")
    run_parser.add_argument("--env-file", default=None, help="dotenv file to load before starting")

    install_parser = sub.add_parser("install-unit", help="write the systemd --user unit file")
    install_parser.add_argument("--env-file", default=None, help="env file to bake into ExecStart")

    sub.add_parser("status", help="print engine/embody/harness liveness as JSON")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> int:
    """CLI entry: ``run`` / ``install-unit`` / ``status``.

    *runner* is threaded through to :func:`reachy_nova.harness.unit.install_unit`
    so the one external command this package runs stays injectable end-to-end.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args.env_file)
    if args.command == "install-unit":
        path = unit.install_unit(env_file=args.env_file, runner=runner)
        print(json.dumps({"installed": str(path), "unit": unit.HARNESS_UNIT}))
        return EXIT_OK
    print(json.dumps(status(), indent=2))
    return EXIT_OK
