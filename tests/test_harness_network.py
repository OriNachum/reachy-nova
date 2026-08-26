"""Tests for reachy_nova.harness.network (task t4).

The unit polls: a default route (injected ``route_reader``), a wlan address
(injected ``addr_reader``) and the ``<statedir>/network-change`` drop file for
the SSID. It must log exactly one transition line per state change (latched —
no repeats while stable), fire ``on_change`` callbacks once per transition on
its own thread, never let a raising callback kill it, and stop promptly.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pytest

from reachy_nova.harness import network


def _write_change_file(path: Path, ssid: str, ip: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ssid": ssid, "ip": ip, "ts": time.time()}
    path.write_text(json.dumps(payload), encoding="utf-8")


class FakeReaders:
    """Mutable stand-ins for route_reader/addr_reader, driven by the test."""

    def __init__(self, *, route: bool = True, addr: str | None = "192.168.1.162") -> None:
        self.route = route
        self.addr = addr
        self.route_calls = 0
        self.addr_calls = 0

    def route_reader(self) -> bool:
        self.route_calls += 1
        return self.route

    def addr_reader(self) -> str | None:
        self.addr_calls += 1
        return self.addr


def _make_unit(
    tmp_path: Path,
    readers: FakeReaders,
    *,
    poll_interval: float = 0.02,
    change_file: Path | None = None,
) -> network.NetworkUnit:
    return network.NetworkUnit(
        route_reader=readers.route_reader,
        addr_reader=readers.addr_reader,
        change_file=change_file if change_file is not None else tmp_path / "network-change",
        poll_interval=poll_interval,
    )


def _sense_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records]


@pytest.fixture(autouse=True)
def _caplog_at_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="nova.sensory")


# --------------------------------------------------------------------------- #
# Contract: start/stop/is_alive/status                                        #
# --------------------------------------------------------------------------- #


def test_unit_contract_start_stop_is_alive(tmp_path: Path) -> None:
    readers = FakeReaders()
    unit = _make_unit(tmp_path, readers)
    stop_event = threading.Event()

    assert not unit.is_alive()
    unit.start(stop_event)
    try:
        assert unit.is_alive()
    finally:
        unit.stop()
    assert not unit.is_alive()


def test_stop_joins_the_thread_promptly(tmp_path: Path) -> None:
    readers = FakeReaders()
    unit = _make_unit(tmp_path, readers, poll_interval=5.0)
    unit.start(threading.Event())
    time.sleep(0.05)

    started = time.monotonic()
    unit.stop(timeout=2.0)
    elapsed = time.monotonic() - started

    assert not unit.is_alive()
    assert elapsed < 1.0


def test_status_reports_online_ip_ssid(tmp_path: Path) -> None:
    change_file = tmp_path / "network-change"
    _write_change_file(change_file, "bar-nachum")
    readers = FakeReaders(route=True, addr="192.168.1.162")
    unit = _make_unit(tmp_path, readers, change_file=change_file)
    unit.start(threading.Event())
    try:
        _wait_until(lambda: unit.status()["online"] is True)
        status = unit.status()
        assert status["online"] is True
        assert status["ip"] == "192.168.1.162"
        assert status["ssid"] == "bar-nachum"
    finally:
        unit.stop()


# --------------------------------------------------------------------------- #
# Transitions + latching                                                      #
# --------------------------------------------------------------------------- #


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met before timeout")


def test_startup_baseline_online_logs_one_joined_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    change_file = tmp_path / "network-change"
    _write_change_file(change_file, "bar-nachum")
    readers = FakeReaders(route=True, addr="192.168.1.162")
    unit = _make_unit(tmp_path, readers, change_file=change_file)
    unit.start(threading.Event())
    try:
        _wait_until(lambda: unit.status()["online"] is True)
        time.sleep(0.1)  # let a few extra polls happen — must not repeat
    finally:
        unit.stop()

    lines = [ln for ln in _sense_lines(caplog) if "event=network" in ln]
    joined = [ln for ln in lines if "joined=" in ln]
    assert len(joined) == 1
    assert "joined=bar-nachum ip=192.168.1.162" in joined[0]


def test_startup_baseline_offline_logs_one_dropped_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    readers = FakeReaders(route=False, addr=None)
    unit = _make_unit(tmp_path, readers)
    unit.start(threading.Event())
    try:
        time.sleep(0.15)
    finally:
        unit.stop()

    lines = [ln for ln in _sense_lines(caplog) if "event=network" in ln]
    dropped = [ln for ln in lines if "dropped" in ln]
    assert len(dropped) == 1
    assert "dropped reason=no-route" in dropped[0]


def test_one_joined_and_one_dropped_line_per_transition_no_repeats(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    readers = FakeReaders(route=True, addr="192.168.1.162")
    unit = _make_unit(tmp_path, readers)
    unit.start(threading.Event())
    try:
        _wait_until(lambda: unit.status()["online"] is True)
        time.sleep(0.1)  # stable online for a while — no repeats

        readers.route = False
        readers.addr = None
        _wait_until(lambda: unit.status()["online"] is False)
        time.sleep(0.1)  # stable offline for a while — no repeats

        readers.route = True
        readers.addr = "192.168.1.162"
        _wait_until(lambda: unit.status()["online"] is True)
        time.sleep(0.1)
    finally:
        unit.stop()

    lines = [ln for ln in _sense_lines(caplog) if "event=network" in ln]
    joined = [ln for ln in lines if "joined=" in ln and "moved" not in ln]
    dropped = [ln for ln in lines if ln.count("dropped") and "no-route" in ln]
    assert len(joined) == 2, lines
    assert len(dropped) == 1, lines


def test_ip_change_while_online_is_a_single_transition_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A changed ip/ssid while online is documented+tested as ONE 'moved' line."""
    readers = FakeReaders(route=True, addr="172.20.10.2")
    unit = _make_unit(tmp_path, readers)
    unit.start(threading.Event())
    try:
        _wait_until(lambda: unit.status()["ip"] == "172.20.10.2")
        time.sleep(0.05)

        readers.addr = "192.168.1.162"
        _wait_until(lambda: unit.status()["ip"] == "192.168.1.162")
        time.sleep(0.1)
    finally:
        unit.stop()

    lines = [ln for ln in _sense_lines(caplog) if "event=network" in ln]
    moved = [ln for ln in lines if "moved" in ln]
    dropped = [ln for ln in lines if "dropped" in ln]
    assert len(moved) == 1, lines
    assert not dropped, lines
    assert "192.168.1.162" in moved[0]


def test_network_change_file_supplies_ssid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    change_file = tmp_path / "network-change"
    _write_change_file(change_file, "iPhone (5)")
    readers = FakeReaders(route=True, addr="172.20.10.2")
    unit = _make_unit(tmp_path, readers, change_file=change_file)
    unit.start(threading.Event())
    try:
        _wait_until(lambda: unit.status()["online"] is True)
    finally:
        unit.stop()

    assert unit.status()["ssid"] == "iPhone (5)"
    lines = [ln for ln in _sense_lines(caplog) if "event=network" in ln]
    assert any("joined=iPhone (5)" in ln for ln in lines)


def test_missing_change_file_is_fine_ssid_unknown(tmp_path: Path) -> None:
    readers = FakeReaders(route=True, addr="192.168.1.162")
    unit = _make_unit(tmp_path, readers, change_file=tmp_path / "does-not-exist")
    unit.start(threading.Event())
    try:
        _wait_until(lambda: unit.status()["online"] is True)
    finally:
        unit.stop()
    assert unit.status()["ssid"] is None


# --------------------------------------------------------------------------- #
# on_change callbacks                                                         #
# --------------------------------------------------------------------------- #


def test_callback_invoked_once_per_transition_with_payload(tmp_path: Path) -> None:
    change_file = tmp_path / "network-change"
    _write_change_file(change_file, "bar-nachum")
    readers = FakeReaders(route=True, addr="192.168.1.162")
    unit = _make_unit(tmp_path, readers, change_file=change_file)

    events: list[tuple[str, dict]] = []
    lock = threading.Lock()

    def on_change(event: str, info: dict) -> None:
        with lock:
            events.append((event, dict(info)))

    unit.on_change(on_change)
    unit.start(threading.Event())
    try:
        _wait_until(lambda: unit.status()["online"] is True)
        time.sleep(0.1)

        readers.route = False
        readers.addr = None
        _wait_until(lambda: len(events) >= 2)
        time.sleep(0.1)
    finally:
        unit.stop()

    with lock:
        seen = list(events)
    assert len(seen) == 2
    assert seen[0][0] == "joined"
    assert seen[0][1] == {"ip": "192.168.1.162", "ssid": "bar-nachum"}
    assert seen[1][0] == "dropped"
    assert seen[1][1] == {"reason": "no-route"}


def test_callback_fires_on_the_units_own_thread(tmp_path: Path) -> None:
    readers = FakeReaders(route=True, addr="192.168.1.162")
    unit = _make_unit(tmp_path, readers)
    seen_threads: list[int] = []

    def on_change(event: str, info: dict) -> None:
        seen_threads.append(threading.get_ident())

    unit.on_change(on_change)
    unit.start(threading.Event())
    try:
        _wait_until(lambda: len(seen_threads) >= 1)
    finally:
        unit.stop()

    assert seen_threads[0] != threading.get_ident()
    assert seen_threads[0] == unit._thread.ident  # type: ignore[attr-defined]


def test_callback_exception_does_not_stop_subsequent_transitions(tmp_path: Path) -> None:
    readers = FakeReaders(route=True, addr="192.168.1.162")
    unit = _make_unit(tmp_path, readers)
    calls: list[str] = []

    def bad_callback(event: str, info: dict) -> None:
        calls.append(event)
        raise RuntimeError("boom")

    unit.on_change(bad_callback)
    unit.start(threading.Event())
    try:
        _wait_until(lambda: len(calls) >= 1)
        time.sleep(0.05)

        readers.route = False
        readers.addr = None
        _wait_until(lambda: len(calls) >= 2)
        # still alive AFTER two callback exceptions — the loop was not killed.
        assert unit.is_alive()
    finally:
        unit.stop()

    assert calls == ["joined", "dropped"]


# --------------------------------------------------------------------------- #
# Poll interval / env                                                         #
# --------------------------------------------------------------------------- #


def test_default_poll_interval_is_two_seconds() -> None:
    assert network.DEFAULT_POLL_S == 2.0


def test_poll_interval_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOVA_NET_POLL_S", "7.5")
    assert network._resolve_poll_interval() == 7.5


def test_poll_interval_env_bad_value_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOVA_NET_POLL_S", "not-a-number")
    assert network._resolve_poll_interval() == network.DEFAULT_POLL_S


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def test_default_route_present_parses_proc_net_route(tmp_path: Path) -> None:
    route_file = tmp_path / "route"
    route_file.write_text(
        "Iface\tDestination\tGateway\tFlags\n"
        "wlan0\t0002A8C0\t0102A8C0\t0003\n",
        encoding="utf-8",
    )
    assert network._parse_proc_net_route(route_file.read_text()) is False

    route_file.write_text(
        "Iface\tDestination\tGateway\tFlags\n"
        "wlan0\t00000000\t0102A8C0\t0003\n",
        encoding="utf-8",
    )
    assert network._parse_proc_net_route(route_file.read_text()) is True


def test_default_route_present_missing_file_returns_false(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-route-file"
    assert network._read_default_route(missing) is False


# --------------------------------------------------------------------------- #
# Boundary gate — must still pass with network.py added.                      #
# --------------------------------------------------------------------------- #


def test_boundary_gate_still_passes() -> None:
    """The AST boundary gate (t4's guardrail) must still pass with network.py added.

    ``tests/`` carries no ``__init__.py``, so the sibling module is loaded by
    path rather than imported as ``tests.test_harness_boundary`` — this is
    also, quite literally, "run it explicitly" per the task brief.
    """
    import importlib.util

    module_path = Path(__file__).parent / "test_harness_boundary.py"
    spec = importlib.util.spec_from_file_location("_t4_boundary_gate", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    violations = module.boundary_violations(module._HARNESS_ROOT)
    assert not violations, violations
