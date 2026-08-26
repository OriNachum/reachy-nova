"""Network-awareness unit — the harness's own eyes on Wi-Fi (task t4).

The 2026-08-26 dual-network drill exposed a harness with **zero** network
awareness: only unit ordering (``After=network-online.target``) and Sonic's
180 s liveness/clock-step watchdogs reacted to anything at all, which cannot
meet a 60 s "the mind is back" bound (see
``docs/specs/2026-08-26-dual-network-never-downtime.md`` c17, h7). This module
is the harness's own poll of the machine's network state — never anything
that touches NetworkManager or the robot SDK, both out of scope for this
package (:mod:`reachy_nova.harness` boundary, ``tests/test_harness_boundary.py``).

What it observes, every ``NOVA_NET_POLL_S`` seconds (default 2 s):

* **a default route** — ``/proc/net/route`` has a ``00000000`` destination on
  ANY interface. Injectable as ``route_reader`` for tests; the default
  implementation is :func:`_default_route_present`.
* **a wlan address** — the machine's own IPv4 address. Injectable as
  ``addr_reader``; the default implementation (:func:`_default_wlan_address`)
  shells out to ``ip -4 -o addr show`` and is the ONE place this module runs a
  subprocess.
* **``<statedir>/network-change``** — a JSON ``{ssid, ip, ts}`` file, written
  atomically by the (future, t5+) NetworkManager dispatcher hook. Missing is
  fine (no SSID yet known); it is the only source of the SSID, since neither
  ``/proc/net/route`` nor ``ip addr`` carries one.

State machine
-------------
``online(ip, ssid)`` vs ``offline``. Every TRANSITION emits exactly one
``[SENSE stage=supervise source=nova event=network]`` line and fires every
registered :meth:`NetworkUnit.on_change` callback once, on the unit's own
thread — latched, so a stable state costs nothing per poll. Three shapes:

* ``joined`` — offline (or startup) -> online: ``joined=<ssid-or-unknown>
  ip=<addr>``.
* ``dropped`` — online (or startup) -> offline: ``dropped reason=no-route``.
* ``moved`` — online -> online with a DIFFERENT ip and/or ssid (a same-radio
  roam, not a drop): ``moved joined=<ssid-or-unknown> ip=<addr>``, ONE line —
  not a dropped+joined pair. This is a deliberate choice (see
  ``tests/test_harness_network.py::test_ip_change_while_online_is_a_single_transition_line``):
  a roam never actually lost the route, so counting it as a drop would make
  the "one joined and one dropped line per transition" honesty condition
  (h-line in the spec) overcount on every roam.

Startup always logs one line (joined or dropped) so the journal shows the
harness's initial network state, never a silent gap before the first
transition. That FIRST observation is a baseline, not a change: it carries
``info["initial"] = True`` so a consumer can tell "this is what the network
already was when I started" from "the network just moved under me". Every
later transition carries ``info["initial"] = False``.

Consumers (wired in t5, ``harness/app.py``'s ``NetworkReactor``): a
``joined``/``moved`` restarts the Sonic stream and the Kiro session; a
``dropped`` is logged only; and an ``initial`` observation restarts nothing —
the legs were constructed seconds earlier against exactly that network, so
restarting them at boot would cost a reconnect cycle and buy nothing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess  # nosec B404 - one fixed, injectable argv, see _default_wlan_address
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .. import sensory_log
from . import statedir

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Senselog identity                                                           #
# --------------------------------------------------------------------------- #

#: ``[SENSE stage=supervise source=nova event=network]`` — every line here.
STAGE = "supervise"
SOURCE = "nova"
EVENT = "network"

# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #

#: Env var overriding the poll interval.
POLL_ENV = "NOVA_NET_POLL_S"
DEFAULT_POLL_S = 2.0

PROC_NET_ROUTE = Path("/proc/net/route")
#: ``/proc/net/route``'s hex little-endian "any destination" value.
DEFAULT_ROUTE_DEST = "00000000"

#: A subprocess timeout short enough it never stalls the poll loop.
ADDR_READER_TIMEOUT_S = 2.0
_INET_ADDR_RE = re.compile(r"inet\s+(\d+\.\d+\.\d+\.\d+)/")

REASON_NO_ROUTE = "no-route"

EVENT_JOINED = "joined"
EVENT_DROPPED = "dropped"
EVENT_MOVED = "moved"

_WAIT_SLICE_S = 0.05


# --------------------------------------------------------------------------- #
# Pure helpers — no threads, trivially testable                               #
# --------------------------------------------------------------------------- #


def _parse_proc_net_route(text: str) -> bool:
    """Is there a ``00000000``-destination row in ``/proc/net/route`` text?"""
    lines = text.splitlines()
    for line in lines[1:]:  # first line is the column header
        fields = line.split()
        if len(fields) >= 2 and fields[1].upper() == DEFAULT_ROUTE_DEST:
            return True
    return False


def _read_default_route(path: Path = PROC_NET_ROUTE) -> bool:
    """Default ``route_reader``: read+parse ``/proc/net/route``. Never raises."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return _parse_proc_net_route(text)


def _default_route_present() -> bool:
    return _read_default_route(PROC_NET_ROUTE)


def _default_wlan_address() -> str | None:
    """Default ``addr_reader``: the machine's wlan IPv4 address, or ``None``.

    The ONE place this module runs a subprocess — kept in a single injectable
    function so tests never actually shell out. ``ip -4 -o addr show`` lists
    every interface's IPv4 addresses, one per line; the first ``wlan*`` line
    with an ``inet`` field wins.
    """
    try:
        result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell, no user input
            ["ip", "-4", "-o", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=ADDR_READER_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) < 2 or "wlan" not in fields[1]:
            continue
        match = _INET_ADDR_RE.search(line)
        if match:
            return match.group(1)
    return None


def _read_change_file(path: Path) -> dict | None:
    """The dispatcher hook's ``{ssid, ip, ts}`` drop file — missing is fine."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _resolve_poll_interval(env: dict[str, str] | None = None) -> float:
    """``$NOVA_NET_POLL_S``, or :data:`DEFAULT_POLL_S` when absent/bad."""
    source = os.environ if env is None else env
    raw = source.get(POLL_ENV)
    if not raw:
        return DEFAULT_POLL_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_POLL_S
    return value if value > 0.0 else DEFAULT_POLL_S


# --------------------------------------------------------------------------- #
# The unit                                                                     #
# --------------------------------------------------------------------------- #


class NetworkUnit:
    """Polls default-route + wlan address + the network-change file.

    Follows the harness unit contract (``start(stop_event)`` / ``stop()`` /
    ``is_alive()`` / ``status()``), owns one daemon thread, and emits a
    latched ``[SENSE stage=supervise source=nova event=network]`` line on
    every state transition (see the module docstring for the joined/dropped/
    moved shapes).

    Args:
        route_reader: zero-arg callable returning whether a default route is
            present. Defaults to :func:`_default_route_present`
            (``/proc/net/route``). Injected for tests.
        addr_reader: zero-arg callable returning the wlan IPv4 address, or
            ``None``. Defaults to :func:`_default_wlan_address` (``ip -4 -o
            addr show``). Injected for tests.
        change_file: path to the network-change drop file. Defaults to
            :func:`reachy_nova.harness.statedir.network_change_path`.
        poll_interval: seconds between observations. Defaults to
            :func:`_resolve_poll_interval` (``$NOVA_NET_POLL_S`` or 2.0).
    """

    def __init__(
        self,
        *,
        route_reader: Callable[[], bool] | None = None,
        addr_reader: Callable[[], str | None] | None = None,
        change_file: Path | str | None = None,
        poll_interval: float | None = None,
        name: str = "network",
    ) -> None:
        self.name = name
        self._route_reader = route_reader or _default_route_present
        self._addr_reader = addr_reader or _default_wlan_address
        self._change_file = (
            Path(change_file) if change_file is not None else statedir.network_change_path()
        )
        self._poll_interval = (
            float(poll_interval) if poll_interval is not None else _resolve_poll_interval()
        )

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._external_stop: threading.Event | None = None

        self._lock = threading.Lock()
        self._callbacks: list[Callable[[str, dict], None]] = []

        # Latched observed state. `None` online means "not yet observed".
        self._online: bool | None = None
        self._ip: str | None = None
        self._ssid: str | None = None

    # -- registration --------------------------------------------------------

    def on_change(self, callback: Callable[[str, dict], None]) -> None:
        """Register a transition callback: ``callback(event, info)``.

        *event* is one of ``"joined"``/``"dropped"``/``"moved"``; *info* is a
        fresh dict per call. Fired on the unit's own thread; an exception is
        caught and logged, never allowed to kill the poll loop.
        """
        with self._lock:
            self._callbacks.append(callback)

    # -- lifecycle ------------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        """Start the poll daemon thread, shutting down on *stop_event*."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._external_stop = stop_event
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="nova-network", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the poll loop to finish and join it (best effort, never raises)."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, object]:
        """The current latched observation — the ``/api`` + test surface."""
        with self._lock:
            return {
                "online": bool(self._online),
                "ip": self._ip,
                "ssid": self._ssid,
            }

    # -- the poll loop ----------------------------------------------------------

    def _run(self) -> None:
        while not self._should_stop():
            self._observe()
            self._wait(self._poll_interval)

    def _observe(self) -> None:
        has_route = self._safe_call(self._route_reader, False)
        ip = self._safe_call(self._addr_reader, None)
        change = _read_change_file(self._change_file)
        ssid = change.get("ssid") if isinstance(change, dict) else None
        online = bool(has_route and ip)

        with self._lock:
            prev_online = self._online
            prev_ip = self._ip
            prev_ssid = self._ssid
            first = prev_online is None

            if not first and online and prev_online and ip == prev_ip and ssid == prev_ssid:
                return  # stable online — latched, no line, no callback
            if not first and not online and not prev_online:
                return  # stable offline — latched, no line, no callback

            self._online = online
            self._ip = ip if online else None
            self._ssid = ssid if online else None

        if online:
            event = EVENT_MOVED if (not first and prev_online) else EVENT_JOINED
            self._fire(event, ip, ssid, initial=first)
        else:
            self._fire(EVENT_DROPPED, None, None, initial=first)

    @staticmethod
    def _safe_call(fn: Callable[[], object], default: object) -> object:
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 - a bad reader must not kill the poll loop
            logger.warning("network: reader failed: %s", err, exc_info=True)
            return default

    def _fire(self, event: str, ip: str | None, ssid: str | None, *, initial: bool) -> None:
        """Log the one transition line and hand it to every callback.

        *initial* marks the startup BASELINE observation — the state the
        network was already in when this unit started, as opposed to a change
        that happened while it was watching. It reaches consumers as
        ``info["initial"]``; the log line is emitted either way (the journal
        must show the initial state, never a silent gap).
        """
        if event == EVENT_DROPPED:
            detail = f"dropped reason={REASON_NO_ROUTE}"
            info: dict[str, object] = {"reason": REASON_NO_ROUTE}
        elif event == EVENT_MOVED:
            detail = f"moved joined={ssid or 'unknown'} ip={ip}"
            info = {"ip": ip, "ssid": ssid}
        else:
            detail = f"joined={ssid or 'unknown'} ip={ip}"
            info = {"ip": ip, "ssid": ssid}

        info["initial"] = initial
        if initial:
            detail = f"{detail} initial=true"
        sensory_log.stage(STAGE, SOURCE, EVENT, detail)

        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(event, dict(info))
            except Exception as err:  # noqa: BLE001 - a bad consumer must not kill the loop
                logger.warning("network: on_change callback failed: %s", err, exc_info=True)

    # -- plumbing ---------------------------------------------------------------

    def _should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        return self._external_stop is not None and self._external_stop.is_set()

    def _wait(self, seconds: float) -> None:
        """Sleep, interruptibly: either stop signal cuts it short."""
        deadline = time.monotonic() + seconds
        while not self._should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            self._stop.wait(min(remaining, _WAIT_SLICE_S))
