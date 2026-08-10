"""The harness's systemd ``--user`` unit — pure text, plus a thin installer.

Mirrors ``reachy-mini-cli``'s ``reachy/service/units.py`` grammar
(``Type=simple`` / ``Restart=on-failure`` / ``RestartSec=5`` /
``WantedBy=default.target`` / an ``ExecStart`` that re-invokes a named
interpreter against a ``-m`` module entry, so the unit is PATH-independent),
with one deliberate difference that is the whole point of this module:

**The harness is a PERIPHERAL, not a presence unit.** The presence units over
there (``reachy-demo-mode.service``, ``reachy-runtime.service``) ``Requires=``
the daemon, because a presence loop with no daemon has nothing to drive. This
unit orders ``After=reachy-runtime.service network-online.target`` and carries
**no ``Requires=`` at all**. The ordering says "if the runtime is coming up
this boot, come up after it"; a ``Requires=`` would additionally say "stop me
whenever it stops", and that is exactly the behaviour we must not have: the
runtime restarting (an upgrade, a rules reload, a crash-and-restart) must leave
the harness running so it can re-attach through the filesystem seams
(:mod:`reachy_nova.harness.statedir`) on its own. A harness that dies with the
runtime turns every runtime blip into a silent loss of hearing and voice.

``harness_unit_text`` is **pure**: it returns a ``str`` and touches nothing.
The one function with side effects (:func:`install_unit`) writes the file and
reloads the user manager — and stops there. It never ``enable``s and never
``start``s: turning the harness on is an explicit operator act, so that
installing the unit during a package upgrade can never quietly start a second
audio consumer on a robot that is already talking through something else.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - only ever runs the fixed systemctl argv below
import sys
from collections.abc import Callable
from pathlib import Path

#: Canonical unit name (CROSS-TASK CONTRACT — import it, never re-spell it).
HARNESS_UNIT = "reachy-nova-harness.service"

#: The presence unit we order after (owned by reachy-mini-cli, named by value:
#: this package must never import ``reachy`` or ``reachy_mini``).
RUNTIME_UNIT = "reachy-runtime.service"

DESCRIPTION = "Reachy Nova harness (on-device AI peripheral over the symbolic runtime)"


def _unit_arg(value: str) -> str:
    """Quote/escape one ``ExecStart`` argument for the systemd unit grammar.

    systemd splits ``ExecStart`` on whitespace and treats ``%`` as a specifier,
    so a path with spaces or ``%`` would corrupt the command line. Double quotes
    preserve spaces; ``%`` doubles, and ``"`` / ``\\`` are backslash-escaped.
    Matches ``reachy/service/units.py::_unit_arg`` exactly.
    """
    escaped = value.replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _default_python() -> str:
    """The interpreter to launch the module entry with — the running one."""
    return sys.executable


def harness_exec_start(python: str | None = None, env_file: str | None = None) -> str:
    """``ExecStart`` for the harness: ``<python> -m reachy_nova.harness run``.

    ``--env-file`` is forwarded only when given: an absent flag means the
    harness resolves credentials from the process environment it inherits
    (``environment.d``, a drop-in) rather than from a file that may not exist.
    """
    py = python or _default_python()
    line = f"{_unit_arg(py)} -m reachy_nova.harness run"
    if env_file:
        line += f" --env-file {_unit_arg(str(env_file))}"
    return line


def harness_unit_text(
    python: str | None = None,
    env_file: str | None = None,
    workdir: str | None = None,
) -> str:
    """Render ``reachy-nova-harness.service``. Pure — no writes, no systemctl.

    See the module docstring for why there is no ``Requires=`` line: a
    peripheral is ordered after the runtime, never bound to its lifetime.
    """
    workdir_line = f"WorkingDirectory={_unit_arg(str(workdir))}\n" if workdir else ""
    return (
        "[Unit]\n"
        f"Description={DESCRIPTION}\n"
        f"After={RUNTIME_UNIT} network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"{workdir_line}"
        f"ExecStart={harness_exec_start(python, env_file)}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def unit_dir() -> Path:
    """The per-user unit directory: ``$XDG_CONFIG_HOME/systemd/user``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "systemd" / "user"


def unit_path() -> Path:
    """Where :func:`install_unit` writes the unit file."""
    return unit_dir() / HARNESS_UNIT


def install_unit(
    *,
    python: str | None = None,
    env_file: str | None = None,
    workdir: str | None = None,
    runner: Callable[..., object] = subprocess.run,
) -> Path:
    """Write the unit file and ``daemon-reload``; never enable, never start.

    *runner* is injected (defaulting to :func:`subprocess.run`) so a test can
    observe the exact argv without a real user manager — and so the ONE
    external command this module runs stays visible in one place.
    """
    path = unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(harness_unit_text(python, env_file, workdir), encoding="utf-8")
    runner(["systemctl", "--user", "daemon-reload"], check=False)  # nosec B603 - fixed argv
    return path
