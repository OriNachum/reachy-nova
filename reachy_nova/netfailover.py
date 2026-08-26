"""The thin I/O driver around :mod:`reachy_nova.netpolicy` (task t3).

:mod:`reachy_nova.netpolicy` is the pure brain — no clock, no I/O, no
``subprocess``. **This** module is the only place in the repo that runs
``nmcli``, reads ``/proc/net/route`` or writes the failover state files. Keep
it that way: everything here is injectable (``nmcli=``, ``route_path=``,
``clock=``, ``sleep=``) so the whole driver is testable without a real
NetworkManager, and the policy stays provable on its own.

Why it exists (see ``docs/specs/2026-08-26-dual-network-never-downtime.md``
c22/h16/c26/h22 and ``docs/architecture.md`` §8) — three observed facts about
NetworkManager on this robot that no amount of profile priority fixes:

1. NM only auto-activates on *disconnect*, and it retries the **current**
   profile with growing wpa_supplicant temp-disable backoffs (10 s, 20 s, …)
   before it falls through to another profile at all (probe s12: 48 s with no
   attempt on the other profile).
2. NM never roams *up* to a higher-priority profile while it is connected. A
   robot that has fallen back to home Wi-Fi stays there forever unless
   something re-evaluates (c26).
3. One failed fallback attempt (``ssid-not-found`` — the iPhone stops
   beaconing when the Hotspot screen closes) stops further attempts for
   minutes (probe s15).

So: a NetworkManager dispatcher hook (``config/network/90-reachy-failover``)
runs ``python -m reachy_nova.netfailover --once`` as root on wlan0 events, and
starts a transient ``--loop`` unit that keeps re-evaluating every
``Decision.next_check_s`` seconds while the situation is unsettled and exits
by itself once the robot has been on the preferred network for
``PREFERRED_SETTLE_S``.

State files, both under the state dir
(:func:`reachy_nova.harness.statedir.state_dir` semantics —
``REACHY_STATE_DIR`` -> ``$XDG_STATE_HOME/reachy`` -> ``~/.local/state/reachy``):

- ``netfailover.json`` — the attempts/last-change record. This is what makes
  storm control (``Policy.min_attempt_gap_s``) survive the hook being a fresh
  short-lived process on every single NM event, and what latches the log line
  so a tight loop costs at most one line per minute (h22's "≤1 network line
  per minute").
- ``network-change`` — ``{"ssid", "ip", "ts"}``, written atomically whenever
  the OBSERVED network differs from what the file already holds — not only
  after one of *our* activations. NetworkManager (or a human with ``nmcli``)
  can join a network without us, and the harness's ``NetworkUnit`` (task t4,
  running as ``pollen``) treats this file as its *only* SSID source, so a
  stale file means a wrong ssid in every ``joined``/``moved`` line. Never
  written while disconnected: ``NetworkUnit`` derives ``dropped`` from the
  route itself. The dispatcher hook chowns it to pollen after each root-side
  run.
- ``netfailover.lock`` — the ``flock`` that serialises the whole
  read->decide->activate->write round. The dispatcher starts a *distinct*
  transient ``--once`` unit per NM event while the ``--loop`` unit is also
  running, so overlapping rounds are the normal case, not the exotic one:
  unserialised they read the same stale ``last_attempt_at``, both activate the
  same SSID, and each overwrites the other's record — exactly the attempt
  storm ``Policy.min_attempt_gap_s`` exists to prevent.

Environment:

- ``REACHY_NET_PREFERRED`` (default ``"iPhone (5)"``)
- ``REACHY_NET_FALLBACK`` (default ``"bar-nachum"``)
- ``REACHY_NMCLI`` — nmcli binary override (default ``nmcli`` from PATH)
- ``REACHY_NET_RESCAN=1`` — force ``nmcli device wifi rescan`` before scanning
  (needs root; failure is ignored and the cached scan is used)
- ``REACHY_STATE_DIR`` — state dir override

CLI::

    python -m reachy_nova.netfailover [--once|--loop] [--dry-run] [--self-test]
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import signal
import subprocess  # nosec B404 - fixed nmcli argv only, never a shell
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import netpolicy

logger = logging.getLogger("nova.netfailover")

#: Defaults for this robot's two profiles (overridable via the env, above).
DEFAULT_PREFERRED = "iPhone (5)"
DEFAULT_FALLBACK = "bar-nachum"

#: The single radio on the CM4.
WIFI_IFACE = "wlan0"

RECORD_FILENAME = "netfailover.json"
NETWORK_CHANGE_FILENAME = "network-change"
LOCK_FILENAME = "netfailover.lock"

ROUTE_PATH = Path("/proc/net/route")

#: nmcli call timeouts. ``connection up`` genuinely takes tens of seconds on a
#: hotspot join, so it gets its own, much longer budget than a read command.
NMCLI_TIMEOUT_S = 15.0
NMCLI_ACTIVATE_TIMEOUT_S = 45.0

#: How long a round waits for the inter-process round lock before giving up.
#: Bounded, and deliberately a little longer than the slowest thing the lock
#: is ever held across (one ``nmcli connection up``): a waiter that timed out
#: earlier than that would routinely skip rounds that were about to be free,
#: and one that waited forever would pile transient ``--once`` units up behind
#: a wedged round.
LOCK_TIMEOUT_S = NMCLI_ACTIVATE_TIMEOUT_S + 5.0

#: Poll cadence while waiting for the lock (flock has no timed variant).
LOCK_POLL_S = 0.05

#: ``--loop`` clamps whatever the policy asks for into this band, so a bad
#: policy value can never busy-spin nor effectively hang the transient unit.
MIN_INTERVAL_S = 5.0
MAX_INTERVAL_S = 300.0

#: ``--loop`` is self-terminating: once the robot has been continuously on the
#: preferred network this long, there is nothing left to supervise and the
#: transient systemd unit exits (the dispatcher restarts it on the next event).
PREFERRED_SETTLE_S = 300.0

NmcliFn = Callable[..., "tuple[int, str]"]


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def nmcli_binary() -> str:
    """The nmcli binary to run — ``REACHY_NMCLI`` or plain ``nmcli``."""
    return os.environ.get("REACHY_NMCLI") or "nmcli"


def rescan_from_env() -> bool:
    """Whether to force a fresh ``nmcli device wifi rescan`` (needs root)."""
    return os.environ.get("REACHY_NET_RESCAN", "").strip() in ("1", "true", "yes", "on")


def policy_from_env() -> netpolicy.Policy:
    """Build the pure policy from ``REACHY_NET_PREFERRED`` / ``REACHY_NET_FALLBACK``."""
    return netpolicy.Policy(
        preferred=os.environ.get("REACHY_NET_PREFERRED") or DEFAULT_PREFERRED,
        fallback=os.environ.get("REACHY_NET_FALLBACK") or DEFAULT_FALLBACK,
    )


def default_statedir() -> Path:
    """State dir, resolved exactly like :mod:`reachy_nova.harness.statedir`.

    Restated rather than imported: this module runs as **root** from the NM
    dispatcher with ``REACHY_STATE_DIR`` set explicitly, and must not drag the
    harness package (and its optional dependencies) into that process.
    """
    explicit = os.environ.get("REACHY_STATE_DIR")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "reachy"
    return Path.home() / ".local" / "state" / "reachy"


# --------------------------------------------------------------------------
# nmcli plumbing
# --------------------------------------------------------------------------


def run_nmcli(args, timeout: float | None = None) -> tuple[int, str]:
    """Run ``nmcli <args>`` and return ``(returncode, stdout)``.

    Never raises for a non-zero exit or a timeout — those are ordinary
    outcomes here (nmcli exits 4 on an activation timeout, 10 on an unknown
    connection). Only a genuinely missing binary raises ``OSError``, which the
    CLI turns into a warning + rc 1 rather than a traceback into journald.
    """
    argv = [nmcli_binary(), *[str(a) for a in args]]
    try:
        proc = subprocess.run(  # nosec B603 - fixed argv, shell=False
            argv,
            capture_output=True,
            text=True,
            timeout=NMCLI_TIMEOUT_S if timeout is None else timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (124, "")
    return (proc.returncode, proc.stdout or "")


def _split_terse(line: str) -> list[str]:
    """Split one ``nmcli -t`` row on unescaped ``:`` and unescape the fields.

    nmcli's terse output escapes ``:`` as ``\\:`` and ``\\`` as ``\\\\`` inside
    values — an SSID containing a colon (or a backslash) otherwise silently
    shifts every field to the right.
    """
    fields: list[str] = []
    current: list[str] = []
    escaped = False
    for ch in line:
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
    fields.append("".join(current))
    return fields


def parse_scan(text: str) -> dict[str, int]:
    """Parse ``nmcli -t -f SSID,SIGNAL,ACTIVE device wifi list`` into ``{ssid: signal}``.

    Duplicates (the same SSID seen on several APs/bands) collapse to the
    **strongest** signal. Hidden networks (empty SSID or nmcli's ``--``
    placeholder) and unparseable rows are dropped rather than guessed at.
    """
    out: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        fields = _split_terse(line)
        if len(fields) < 2:
            continue
        ssid = fields[0].strip()
        if not ssid or ssid == "--":
            continue
        try:
            signal_pct = int(fields[1].strip())
        except (TypeError, ValueError):
            continue
        if signal_pct > out.get(ssid, -1):
            out[ssid] = signal_pct
    return out


def scan(nmcli: NmcliFn | None = None, rescan: bool = False) -> dict[str, int]:
    """Return ``{ssid: signal}`` currently visible on the radio.

    With *rescan* true, ask NM for a fresh scan first (root only) — a stale
    scan cache is exactly how probe s15 ended up attempting an SSID that had
    stopped beaconing. A failed rescan is ignored: the cached list is still
    better than no decision at all.
    """
    call = nmcli or run_nmcli
    if rescan:
        try:
            call(["device", "wifi", "rescan"], timeout=NMCLI_TIMEOUT_S)
        except OSError:
            raise
        except Exception:  # nosec B110 - a failed rescan is explicitly tolerated
            logger.debug("wifi rescan failed; using the cached scan", exc_info=True)
    rc, out = call(["-t", "-f", "SSID,SIGNAL,ACTIVE", "device", "wifi", "list"])
    if rc != 0 and not out:
        return {}
    return parse_scan(out)


def active_ssid(nmcli: NmcliFn | None = None) -> str | None:
    """The profile name active on wlan0, from ``connection show --active``.

    Profile names and SSIDs are one and the same on this robot (both profiles
    were created by name), which is what lets the policy compare them
    directly.
    """
    call = nmcli or run_nmcli
    rc, out = call(["-t", "-f", "NAME,DEVICE", "connection", "show", "--active"])
    if rc != 0 and not out:
        return None
    fallback_name: str | None = None
    for raw in out.splitlines():
        if not raw.strip():
            continue
        fields = _split_terse(raw.rstrip("\r"))
        if len(fields) < 2:
            continue
        name, device = fields[0].strip(), fields[1].strip()
        if not name:
            continue
        if device == WIFI_IFACE:
            return name
        if device and device != "lo" and fallback_name is None:
            fallback_name = name
    return fallback_name


def has_route(route_path: Path | None = None) -> bool:
    """Is there a default route? (``/proc/net/route`` destination ``00000000``.)

    This — not "is a connection active" — is what "link lost" means here: a
    connection stays nominally active through a whole supplicant retry storm
    while having no usable route.
    """
    path = ROUTE_PATH if route_path is None else route_path
    try:
        text = path.read_text()
    except OSError:
        return False
    for raw in text.splitlines()[1:]:
        fields = raw.split()
        if len(fields) >= 2 and fields[1] == "00000000":
            return True
    return False


def current_ip(nmcli: NmcliFn | None = None, iface: str = WIFI_IFACE) -> str | None:
    """The IPv4 address on *iface*, or ``None`` — recorded in ``network-change``."""
    call = nmcli or run_nmcli
    try:
        rc, out = call(["-t", "-f", "IP4.ADDRESS", "device", "show", iface])
    except Exception:
        return None
    if rc != 0 and not out:
        return None
    for raw in out.splitlines():
        fields = _split_terse(raw.rstrip("\r"))
        if len(fields) < 2 or not fields[1].strip():
            continue
        return fields[1].strip().split("/")[0] or None
    return None


def activate(profile: str, nmcli: NmcliFn | None = None) -> bool:
    """``nmcli connection up "<profile>"`` — True on success.

    Bounded by ``NMCLI_ACTIVATE_TIMEOUT_S``: a hotspot join that never
    completes must not wedge the dispatcher hook (which NM runs serially).
    """
    call = nmcli or run_nmcli
    rc, _ = call(["connection", "up", profile], timeout=NMCLI_ACTIVATE_TIMEOUT_S)
    return rc == 0


# --------------------------------------------------------------------------
# the record file
# --------------------------------------------------------------------------


@dataclass
class Record:
    """What one run needs to remember for the next one.

    The dispatcher hook is a fresh process on every NM event, so without this
    file there is no storm control and no log latching at all.
    """

    last_attempt_at: dict[str, float] = field(default_factory=dict)
    last_change_at: float = 0.0
    last_decision: netpolicy.Decision | None = None

    def to_json(self) -> dict:
        return {
            "last_attempt_at": dict(self.last_attempt_at),
            "last_change_at": self.last_change_at,
            "last_decision": (self.last_decision.to_record() if self.last_decision else None),
        }


def record_path(statedir: Path) -> Path:
    return statedir / RECORD_FILENAME


def network_change_path(statedir: Path) -> Path:
    return statedir / NETWORK_CHANGE_FILENAME


def lock_path(statedir: Path) -> Path:
    return statedir / LOCK_FILENAME


@contextlib.contextmanager
def round_lock(statedir: Path, timeout: float | None = None):
    """Hold an exclusive ``flock`` on ``<state>/netfailover.lock``.

    Yields ``True`` when the lock was taken and ``False`` when *timeout*
    seconds elapsed without it — the caller then skips the round entirely
    rather than running it unserialised, because an unserialised round is
    precisely the bug the lock exists for.

    ``flock`` (not ``lockf``) on a dedicated file, because the lock has to
    hold between *processes*: the transient ``--once`` unit NM's dispatcher
    starts per event, and the long-lived ``--loop`` unit. It is released by
    the kernel if the process dies mid-round, so a killed activation can never
    wedge the next event's round.
    """
    budget = LOCK_TIMEOUT_S if timeout is None else float(timeout)
    path = lock_path(statedir)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        deadline = time.monotonic() + budget
        acquired = False
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(LOCK_POLL_S)
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:  # pragma: no cover - defensive
                    pass
    finally:
        handle.close()


def load_record(statedir: Path) -> Record:
    """Read the record, tolerating absence, corruption and the wrong shape.

    A corrupt record must never stop a failover — the worst case of ignoring
    it is one extra activation attempt, versus a robot that stays offline.
    """
    try:
        payload = json.loads(record_path(statedir).read_text())
    except (OSError, ValueError):
        return Record()
    if not isinstance(payload, dict):
        return Record()

    attempts: dict[str, float] = {}
    raw_attempts = payload.get("last_attempt_at")
    if isinstance(raw_attempts, dict):
        for key, value in raw_attempts.items():
            try:
                attempts[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    try:
        last_change = float(payload.get("last_change_at") or 0.0)
    except (TypeError, ValueError):
        last_change = 0.0

    decision = None
    raw_decision = payload.get("last_decision")
    if isinstance(raw_decision, dict):
        try:
            decision = netpolicy.Decision(
                action=raw_decision.get("action", "none"),
                target=raw_decision.get("target"),
                reason=str(raw_decision.get("reason", "")),
                next_check_s=float(raw_decision.get("next_check_s") or 0.0),
                ts=float(raw_decision.get("ts") or 0.0),
            )
        except (TypeError, ValueError):
            decision = None

    return Record(last_attempt_at=attempts, last_change_at=last_change, last_decision=decision)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write *payload* to *path* atomically (temp file in the same dir + rename).

    A half-written ``network-change`` would be read by the harness mid-write;
    a half-written record would be a corrupt record on the next boot.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Explicit 0644, not whatever umask root happens to carry: these files
        # are written by the ROOT dispatcher hook and read by the harness
        # running as pollen. A root umask of 077 would otherwise make
        # network-change invisible to the very process it exists to notify.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def save_record(statedir: Path, record: Record) -> None:
    _atomic_write_json(record_path(statedir), record.to_json())


def write_network_change(statedir: Path, ssid: str | None, ip: str | None, ts: float) -> None:
    """Atomically write ``<state>/network-change`` — the harness's t4 trigger."""
    _atomic_write_json(network_change_path(statedir), {"ssid": ssid, "ip": ip, "ts": ts})


def read_network_change(statedir: Path) -> dict | None:
    """The current ``network-change`` payload, or ``None`` if absent/unreadable.

    A corrupt file reads as ``None``, which makes the next round rewrite it —
    the same "prefer acting to being stuck" bias as :func:`load_record`.
    """
    try:
        payload = json.loads(network_change_path(statedir).read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def sync_network_change(
    statedir: Path,
    ssid: str | None,
    ip: str | None,
    ts: float,
) -> bool:
    """Refresh ``network-change`` iff the observed ``(ssid, ip)`` differs.

    Returns whether it wrote. Called on EVERY connected round, not only after
    one of our own activations: NetworkManager auto-connects and manual
    ``nmcli`` joins are invisible to this driver, and ``NetworkUnit`` has no
    other source for the SSID — left alone, it reports the previous network's
    name for as long as the robot stays on the new one.

    Rewriting only on a real difference is what keeps the file's mtime and
    ``ts`` meaningful ("when the network last changed", not "when we last
    looked").
    """
    current = read_network_change(statedir)
    if current is not None and current.get("ssid") == ssid and current.get("ip") == ip:
        return False
    write_network_change(statedir, ssid, ip, ts)
    return True


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


def _log_decision(decision: netpolicy.Decision, active: str | None, visible: dict[str, int]) -> None:
    """One ``[SENSE stage=supervise source=nova event=network]`` line per decision.

    Uses :mod:`reachy_nova.sensory_log` when it imports (the same grep-able
    convention as every other degradation in this repo) and falls back to a
    hand-formatted line otherwise — the dispatcher hook runs as root against
    the venv, and a missing optional import must not cost us the log.
    """
    detail = (
        f"action={decision.action} target={decision.target!r} reason={decision.reason} "
        f"active={active!r} visible={sorted(visible)} next_check_s={decision.next_check_s:g}"
    )
    try:
        from . import sensory_log

        sensory_log.stage("supervise", "nova", "network", detail)
    except Exception:  # pragma: no cover - defensive
        logger.info("[SENSE stage=supervise source=nova event=network] %s", detail)


# --------------------------------------------------------------------------
# the one round
# --------------------------------------------------------------------------


def run_once(
    now: float | None = None,
    nmcli: NmcliFn | None = None,
    rescan: bool = False,
    dry_run: bool = False,
    statedir: Path | None = None,
    policy: netpolicy.Policy | None = None,
    route_path: Path | None = None,
    lock_timeout_s: float | None = None,
) -> netpolicy.Decision:
    """Observe, decide, act, record — one full round of the failover driver.

    Returns the :class:`netpolicy.Decision` (which says what it *would* do
    even under *dry_run*, so ``--self-test`` and ``--dry-run`` are honest).

    The whole read->decide->activate->write transaction is serialised by
    :func:`round_lock`, so the per-event ``--once`` units and the ``--loop``
    unit can never interleave. A round that cannot take the lock within
    *lock_timeout_s* (default :data:`LOCK_TIMEOUT_S`) logs one line and
    returns a ``lock-busy`` no-op decision: skipping is safe (another round is
    right now doing this work), running unserialised is not.

    Under *dry_run* nothing at all is executed and **nothing is written** —
    not the record, not ``network-change``, not even the lock file, and no
    lock is taken. That is what makes the installer's self-test safe to run
    against the live network on a robot that is currently working fine, even
    while a real round holds the lock.
    """
    now = time.time() if now is None else now
    statedir = default_statedir() if statedir is None else statedir
    policy = policy_from_env() if policy is None else policy

    if dry_run:
        return _round(
            now=now,
            call=nmcli or run_nmcli,
            rescan=rescan,
            dry_run=True,
            statedir=statedir,
            policy=policy,
            route_path=route_path,
        )

    with round_lock(statedir, timeout=lock_timeout_s) as acquired:
        if not acquired:
            logger.warning(
                "netfailover: another round still holds %s after %gs — skipping this round",
                lock_path(statedir),
                LOCK_TIMEOUT_S if lock_timeout_s is None else lock_timeout_s,
            )
            return netpolicy.Decision(
                action="none",
                target=None,
                reason="lock-busy",
                next_check_s=MIN_INTERVAL_S,
                ts=now,
            )
        return _round(
            now=now,
            call=nmcli or run_nmcli,
            rescan=rescan,
            dry_run=False,
            statedir=statedir,
            policy=policy,
            route_path=route_path,
        )


def _round(
    now: float,
    call: NmcliFn,
    rescan: bool,
    dry_run: bool,
    statedir: Path,
    policy: netpolicy.Policy,
    route_path: Path | None,
) -> netpolicy.Decision:
    """One round's body. Always called with the round lock held (or dry-run)."""
    record = load_record(statedir)
    visible = scan(nmcli=call, rescan=rescan)
    active = active_ssid(nmcli=call)
    routed = has_route(route_path=route_path)

    observation = netpolicy.Observation(
        now=now,
        active_ssid=active,
        has_route=routed,
        visible=visible,
        last_attempt_at=dict(record.last_attempt_at),
        last_change_at=record.last_change_at,
    )
    decision = policy.decide(observation)

    if decision.should_log(record.last_decision):
        _log_decision(decision, active, visible)

    if dry_run:
        return decision

    changed = False
    if decision.action == "activate" and decision.target:
        # The attempt is recorded whether or not it succeeds: a FAILED attempt
        # is precisely the one that must be gapped (probe s15 — the hotspot
        # that stopped beaconing), or the hook storms nmcli on every NM event.
        record.last_attempt_at[decision.target] = now
        ok = False
        try:
            ok = activate(decision.target, nmcli=call)
        except OSError:
            raise
        except Exception:
            logger.warning("activation of %r raised", decision.target, exc_info=True)
        if ok:
            record.last_change_at = now
            write_network_change(statedir, decision.target, current_ip(nmcli=call), now)
            changed = True

    if not changed and routed:
        # Reconcile with the network we actually observe. NM auto-connects and
        # manual `nmcli connection up` never pass through this driver, and the
        # harness has no other SSID source (see sync_network_change). Only
        # while routed: `dropped` is NetworkUnit's own route-derived verdict,
        # and re-announcing an SSID we have no route to would contradict it.
        if sync_network_change(statedir, active, current_ip(nmcli=call), now):
            record.last_change_at = now

    record.last_decision = decision
    save_record(statedir, record)
    return decision


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def bounded_interval(seconds: float) -> float:
    """Clamp a policy's ``next_check_s`` into ``[MIN_INTERVAL_S, MAX_INTERVAL_S]``."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return MIN_INTERVAL_S
    return max(MIN_INTERVAL_S, min(MAX_INTERVAL_S, value))


def run_loop(
    nmcli: NmcliFn | None = None,
    rescan: bool = False,
    dry_run: bool = False,
    statedir: Path | None = None,
    policy: netpolicy.Policy | None = None,
    route_path: Path | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Re-run :func:`run_once` every ``Decision.next_check_s`` until settled.

    Exits on its own once the robot has been continuously on the preferred
    network for :data:`PREFERRED_SETTLE_S` — that self-termination is what
    lets the dispatcher hook fire-and-forget a transient
    ``reachy-netfailover-loop`` systemd unit without ever having to stop it.
    Also exits on SIGTERM (wired by :func:`main`) via *stop_event* —
    *promptly*: the production wait is ``stop_event.wait(timeout)``, never a
    blind sleep. On the preferred network ``next_check_s`` is 300 s, so a
    plain ``time.sleep`` would have made a ``systemctl stop`` (or the shutdown
    SIGTERM) hang for up to five minutes before systemd escalated to SIGKILL.
    *sleep* stays injectable purely as a test seam for the settle logic.
    """
    clock = time.time if clock is None else clock
    stop_event = threading.Event() if stop_event is None else stop_event
    if sleep is None:
        sleep = stop_event.wait

    preferred_since: float | None = None
    while not stop_event.is_set():
        now = clock()
        decision = run_once(
            now=now,
            nmcli=nmcli,
            rescan=rescan,
            dry_run=dry_run,
            statedir=statedir,
            policy=policy,
            route_path=route_path,
        )
        if decision.reason == "on-preferred":
            if preferred_since is None:
                preferred_since = now
            elif (now - preferred_since) >= PREFERRED_SETTLE_S:
                logger.debug("settled on the preferred network; loop exiting")
                return
        else:
            preferred_since = None
        sleep(bounded_interval(decision.next_check_s))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reachy_nova.netfailover",
        description="Dual-network failover driver (nmcli side of reachy_nova.netpolicy).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="one decision round, then exit (default)")
    mode.add_argument(
        "--loop",
        action="store_true",
        help="re-decide every Decision.next_check_s until settled on the preferred network",
    )
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="dry-run one decision against the LIVE scan; exit 0 if it works, 1 if it does not",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="decide and log, but execute nothing and write nothing",
    )
    return parser


def main(
    argv: list[str] | None = None,
    nmcli: NmcliFn | None = None,
    route_path: Path | None = None,
) -> int:
    """CLI entry. Returns an exit code and never lets an exception escape.

    The NM dispatcher hook runs this as root on every wlan0 event; a traceback
    there is noise in journald and (with ``set -e`` upstream) a failed
    dispatcher run, so every failure becomes a warning and rc 1.
    """
    args = _parser().parse_args([] if argv is None else argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    rescan = rescan_from_env()
    statedir = default_statedir()
    policy = policy_from_env()

    try:
        if args.self_test:
            decision = run_once(
                nmcli=nmcli,
                rescan=rescan,
                dry_run=True,
                statedir=statedir,
                policy=policy,
                route_path=route_path,
            )
            logger.info("self-test ok: %s", decision.to_record())
            return 0

        if args.loop:
            stop_event = threading.Event()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    signal.signal(sig, lambda *_: stop_event.set())
                except (ValueError, OSError):  # not the main thread / unsupported
                    pass
            run_loop(
                nmcli=nmcli,
                rescan=rescan,
                dry_run=args.dry_run,
                statedir=statedir,
                policy=policy,
                route_path=route_path,
                stop_event=stop_event,
            )
            return 0

        run_once(
            nmcli=nmcli,
            rescan=rescan,
            dry_run=args.dry_run,
            statedir=statedir,
            policy=policy,
            route_path=route_path,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - deliberate: never traceback into NM
        logger.warning("netfailover failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
