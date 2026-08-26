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

This module also carries **repo-reproducible templates for the two presence
units** it does not own (``reachy-runtime.service`` / ``reachy-demo-mode.service``
— upstream in reachy-mini-cli): :func:`runtime_unit_text` and
:func:`demo_mode_unit_text`. They exist here, alongside the harness's own
template, because the mutual-exclusion hardening between them (each
``Conflicts=`` the other, so systemd itself refuses to run both presence
loops against the same body at once) was hand-applied live on the device and
would otherwise silently regress on the next reinstall. ``reachy-demo-mode.service``
carries **no ``[Install]`` section at all** — it is reachable only via an
explicit ``systemctl --user start``, never via ``enable``, because a demo loop
that could come up unattended at boot is exactly the second presence this
hardening exists to rule out. ``scripts/install-device-units.sh`` writes all
three, enables the runtime and the harness (never the demo unit), masks the
legacy system-level ``reachy-nova-autostart.service``, and applies a bounded
journald persistence drop-in — see that script's header for the full list.
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

#: The OTHER presence unit (owned by reachy-mini-cli): a manual-only demo loop
#: that drives the same body as RUNTIME_UNIT. The two Conflicts= each other
#: (see :func:`runtime_unit_text` / :func:`demo_mode_unit_text`) so systemd
#: itself refuses to run both at once, rather than relying on an operator to
#: remember not to.
DEMO_UNIT = "reachy-demo-mode.service"

#: The legacy system-level unit (the old ReachyMiniApp autostart). Superseded
#: by RUNTIME_UNIT + HARNESS_UNIT; install masks it so a stale preset-enabled
#: unit from a prior install can never wake up at boot and fight the new
#: pair for the same audio/motor resources.
AUTOSTART_UNIT = "reachy-nova-autostart.service"

DESCRIPTION = "Reachy Nova harness (on-device AI peripheral over the symbolic runtime)"

#: Rendered into the harness unit's ``[Unit]`` section, immediately above the
#: ``After=`` line. LIVE FINDING (2026-08-26): ``network-online.target`` was
#: reached on this device BEFORE wlan0 actually associated, so the harness
#: started with no route, kiro-cli exited on spawn, and the writer stayed
#: absent until a manual restart. The ordering is kept (it is free, and on a
#: box where the target IS meaningful it saves a retry cycle) but the harness
#: must never DEPEND on it: there is no ``Wants=``/``Requires=`` on it, and the
#: network-less start is handled in code — ``KiroSessionUnit`` starts degraded
#: and retries under its watchdog, Sonic's stream loop reconnects with backoff,
#: and ``harness/network.py`` turns a real Wi-Fi join into an immediate restart
#: of both. The comment exists so the next reader does not "fix" the ordering
#: into a dependency and re-create the 2026-08-26 failure. It deliberately
#: spells no systemd directive verbatim, so the unit-text tests can assert
#: "no Wants=/Requires= anywhere in this file" as a flat string search.
NETWORK_ORDERING_COMMENT = (
    "# Ordering only - the harness does NOT depend on the network being up:\n"
    "# network-online.target is reached before wlan0 associates on this device\n"
    "# (2026-08-26), so a network-less start is handled in code (degraded Kiro\n"
    "# session + Sonic stream retry + harness/network.py restart on join).\n"
    "# Never turn this ordering into a Wants or Requires dependency.\n"
)

RUNTIME_DESCRIPTION = "Reachy Mini CLI symbolic runtime (presence loop)"

DEMO_DESCRIPTION = "Reachy Mini CLI demo mode (manual-only presence loop)"


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
        f"{NETWORK_ORDERING_COMMENT}"
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


def runtime_exec_start(python: str | None = None) -> str:
    """``ExecStart`` for the runtime: ``<cli-venv python> -m reachy behavior engine run``.

    *python* is the reachy-mini-cli venv's interpreter, not this harness's
    own — it is a DIFFERENT venv, so unlike :func:`harness_exec_start` there
    is no sensible same-process default; callers (the install script) must
    supply the real path. The ``_default_python`` fallback exists only so
    the renderer stays callable/testable with no arguments, mirroring the
    parameterization style of the harness renderer above.
    """
    py = python or _default_python()
    return f"{_unit_arg(py)} -m reachy behavior engine run"


def runtime_unit_text(python: str | None = None) -> str:
    """Render ``reachy-runtime.service`` — the symbolic runtime presence unit.

    ``Conflicts=reachy-demo-mode.service`` is the hand-applied hardening this
    renders reproducibly: the runtime and the demo loop both drive the body,
    and systemd Conflicts= is a semver-of-behaviour guarantee (starting one
    stops the other) that no amount of operator discipline gives you for
    free. Ordered ``After=network-online.target`` and enabled (``[Install]``)
    since the runtime is the presence a fresh boot should come up into.
    """
    return (
        "[Unit]\n"
        f"Description={RUNTIME_DESCRIPTION}\n"
        "After=network-online.target\n"
        f"Conflicts={DEMO_UNIT}\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={runtime_exec_start(python)}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def demo_mode_exec_start(python: str | None = None) -> str:
    """``ExecStart`` for demo mode: ``<cli-venv python> -m reachy demo-mode run --config %h/...``.

    ``%h`` is a systemd specifier (the invoking user's home directory), not a
    path this renderer resolves — it must reach the unit file literally so
    systemd expands it per-user at activation time, so it is written outside
    :func:`_unit_arg`'s ``%``-doubling (which is for literal ``%`` characters
    inside a path, not specifiers we want systemd itself to interpret).
    """
    py = python or _default_python()
    return f"{_unit_arg(py)} -m reachy demo-mode run --config %h/.config/reachy/demo-mode.json"


def demo_mode_unit_text(python: str | None = None) -> str:
    """Render ``reachy-demo-mode.service`` — manual-only, never at boot.

    Two deliberate omissions encode the hardening:

    * **no ``[Install]`` section at all** — with nothing ``WantedBy=``
      anything, ``systemctl --user enable`` has nothing to link and the unit
      can only ever be started explicitly (``systemctl --user start``). A
      demo loop that could come up unattended at boot is a second presence
      fighting the runtime for the same body, which is exactly what this
      hardening exists to prevent.
    * ``Conflicts=reachy-runtime.service`` — the other half of the mutual
      exclusion in :func:`runtime_unit_text`: whichever of the two starts,
      systemd stops the other first.
    """
    return (
        "[Unit]\n"
        f"Description={DEMO_DESCRIPTION}\n"
        f"Conflicts={RUNTIME_UNIT}\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={demo_mode_exec_start(python)}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
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
