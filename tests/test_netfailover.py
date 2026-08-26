"""Tests for reachy_nova.netfailover — the thin I/O driver around netpolicy (task t3).

netpolicy (t2) is the pure brain; this module is the only place nmcli is ever
invoked. Everything here is driven through injected fakes — no subprocess is
spawned against a real nmcli, no real /proc is read for the parsing tests, and
no real NetworkManager is touched.

Covers:
  - terse nmcli scan parsing: escaped colons, escaped backslashes, duplicate
    SSIDs (max signal wins), hidden/empty SSIDs dropped, junk lines ignored
  - active_ssid parsing (wlan0 row preferred, loopback ignored)
  - has_route parsing of /proc/net/route (default route present/absent)
  - run_once: executes activate exactly per the policy's decision, writes both
    JSON files (the attempts record and <state>/network-change) atomically
  - the attempt gap is enforced ACROSS runs via the record file (storm control
    survives the dispatcher hook being a fresh process every time)
  - a corrupt/missing record file is tolerated
  - --dry-run executes nothing and writes nothing
  - the CLI: --self-test exit codes, --once, --dry-run
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachy_nova import netfailover, netpolicy

# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeNmcli:
    """Records every argv it is handed and replays canned (rc, stdout) pairs.

    Keys are matched on a distinguishing token so tests stay readable:
    ``"wifi list"``, ``"connection show"``, ``"device show"``, ``"up"``,
    ``"rescan"``.
    """

    def __init__(self, scan="", active="", ip="", up_rc=0, rescan_rc=0):
        self.scan_out = scan
        self.active_out = active
        self.ip_out = ip
        self.up_rc = up_rc
        self.rescan_rc = rescan_rc
        self.calls: list[list[str]] = []

    def __call__(self, args, timeout=None):
        self.calls.append(list(args))
        if "rescan" in args:
            return (self.rescan_rc, "")
        if "list" in args:
            return (0, self.scan_out)
        if "--active" in args:
            return (0, self.active_out)
        if "show" in args and "device" in args:
            return (0, self.ip_out)
        if "up" in args:
            return (self.up_rc, "" if self.up_rc == 0 else "Error: unknown connection")
        return (0, "")

    @property
    def activations(self) -> list[str]:
        return [a[-1] for a in self.calls if "up" in a]


SCAN_BOTH = "iPhone (5):72:no\nbar-nachum:58:yes\n"
ACTIVE_FALLBACK = "bar-nachum:wlan0\nlo:lo\n"
ACTIVE_PREFERRED = "iPhone (5):wlan0\n"
IP_OUT = "IP4.ADDRESS[1]:198.51.100.4/28\n"

ROUTE_WITH_DEFAULT = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\n"
    "wlan0\t00000000\t0102A8C0\t0003\t0\t0\t600\t00000000\n"
    "wlan0\t0002A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\n"
)
ROUTE_NO_DEFAULT = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\n"
    "wlan0\t0002A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\n"
)


@pytest.fixture()
def statedir(tmp_path: Path) -> Path:
    d = tmp_path / "state"
    d.mkdir()
    return d


# --------------------------------------------------------------------------
# scan parsing
# --------------------------------------------------------------------------


def test_parse_scan_basic():
    assert netfailover.parse_scan(SCAN_BOTH) == {"iPhone (5)": 72, "bar-nachum": 58}


def test_parse_scan_unescapes_colons_and_backslashes():
    # nmcli -t escapes ':' as '\:' and '\' as '\\' inside field values.
    out = "my\\:net:44:no\nback\\\\slash:31:no\n"
    parsed = netfailover.parse_scan(out)
    assert parsed == {"my:net": 44, "back\\slash": 31}


def test_parse_scan_duplicates_keep_max_signal():
    out = "bar-nachum:31:no\nbar-nachum:67:yes\nbar-nachum:12:no\n"
    assert netfailover.parse_scan(out) == {"bar-nachum": 67}


def test_parse_scan_drops_hidden_and_junk_rows():
    out = ":40:no\n--:20:no\nreal:55:no\ngarbage-with-no-fields\nreal:notanint:no\n"
    assert netfailover.parse_scan(out) == {"real": 55}


def test_parse_scan_empty_is_empty_dict():
    assert netfailover.parse_scan("") == {}
    assert netfailover.parse_scan("\n\n") == {}


def test_scan_calls_nmcli_wifi_list_and_skips_rescan_by_default():
    fake = FakeNmcli(scan=SCAN_BOTH)
    assert netfailover.scan(nmcli=fake) == {"iPhone (5)": 72, "bar-nachum": 58}
    assert not any("rescan" in c for c in fake.calls)
    assert fake.calls[0] == ["-t", "-f", "SSID,SIGNAL,ACTIVE", "device", "wifi", "list"]


def test_scan_rescan_true_rescans_first_and_tolerates_failure():
    fake = FakeNmcli(scan=SCAN_BOTH, rescan_rc=1)
    assert netfailover.scan(nmcli=fake, rescan=True) == {
        "iPhone (5)": 72,
        "bar-nachum": 58,
    }
    assert fake.calls[0] == ["device", "wifi", "rescan"]


# --------------------------------------------------------------------------
# active ssid / route / ip
# --------------------------------------------------------------------------


def test_active_ssid_prefers_wlan0_row():
    fake = FakeNmcli(active=ACTIVE_FALLBACK)
    assert netfailover.active_ssid(nmcli=fake) == "bar-nachum"


def test_active_ssid_none_when_nothing_active():
    fake = FakeNmcli(active="lo:lo\n")
    assert netfailover.active_ssid(nmcli=fake) is None


def test_active_ssid_unescapes():
    fake = FakeNmcli(active="my\\:net:wlan0\n")
    assert netfailover.active_ssid(nmcli=fake) == "my:net"


def test_has_route_true_with_default_route(tmp_path: Path):
    p = tmp_path / "route"
    p.write_text(ROUTE_WITH_DEFAULT)
    assert netfailover.has_route(route_path=p) is True


def test_has_route_false_without_default_route(tmp_path: Path):
    p = tmp_path / "route"
    p.write_text(ROUTE_NO_DEFAULT)
    assert netfailover.has_route(route_path=p) is False


def test_has_route_false_when_file_missing(tmp_path: Path):
    assert netfailover.has_route(route_path=tmp_path / "nope") is False


def test_current_ip_parses_nmcli_device_show():
    fake = FakeNmcli(ip=IP_OUT)
    assert netfailover.current_ip(nmcli=fake) == "198.51.100.4"


def test_current_ip_none_when_absent():
    fake = FakeNmcli(ip="")
    assert netfailover.current_ip(nmcli=fake) is None


# --------------------------------------------------------------------------
# record file
# --------------------------------------------------------------------------


def test_load_record_missing_file_is_empty(statedir: Path):
    rec = netfailover.load_record(statedir)
    assert rec.last_attempt_at == {}
    assert rec.last_decision is None


def test_load_record_tolerates_corrupt_file(statedir: Path):
    (statedir / netfailover.RECORD_FILENAME).write_text("{not json at all")
    rec = netfailover.load_record(statedir)
    assert rec.last_attempt_at == {}


def test_load_record_tolerates_wrong_shape(statedir: Path):
    (statedir / netfailover.RECORD_FILENAME).write_text('["a list, not an object"]')
    rec = netfailover.load_record(statedir)
    assert rec.last_attempt_at == {}


def test_save_record_roundtrip_and_atomic(statedir: Path):
    rec = netfailover.Record(last_attempt_at={"iPhone (5)": 12.5}, last_change_at=9.0)
    netfailover.save_record(statedir, rec)
    path = statedir / netfailover.RECORD_FILENAME
    assert json.loads(path.read_text())["last_attempt_at"] == {"iPhone (5)": 12.5}
    # atomic write leaves no temp files behind
    assert [p.name for p in statedir.iterdir()] == [netfailover.RECORD_FILENAME]
    again = netfailover.load_record(statedir)
    assert again.last_attempt_at == {"iPhone (5)": 12.5}
    assert again.last_change_at == 9.0


# --------------------------------------------------------------------------
# run_once
# --------------------------------------------------------------------------


def _run(statedir, fake, now, route, **kw):
    return netfailover.run_once(
        now=now,
        nmcli=fake,
        statedir=statedir,
        route_path=route,
        policy=netpolicy.Policy(preferred="iPhone (5)", fallback="bar-nachum"),
        **kw,
    )


def test_run_once_activates_preferred_when_disconnected(statedir, tmp_path):
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT)

    decision = _run(statedir, fake, now=1000.0, route=route)

    assert decision.action == "activate"
    assert decision.target == "iPhone (5)"
    assert fake.activations == ["iPhone (5)"]


def test_run_once_activates_preferred_from_fallback(statedir, tmp_path):
    route = tmp_path / "route"
    route.write_text(ROUTE_WITH_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active=ACTIVE_FALLBACK, ip=IP_OUT)

    decision = _run(statedir, fake, now=1000.0, route=route)

    assert decision.action == "activate"
    assert decision.reason == "on-fallback-hotspot-visible"
    assert fake.activations == ["iPhone (5)"]


def test_run_once_does_nothing_on_preferred(statedir, tmp_path):
    route = tmp_path / "route"
    route.write_text(ROUTE_WITH_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active=ACTIVE_PREFERRED, ip=IP_OUT)

    decision = _run(statedir, fake, now=1000.0, route=route)

    assert decision.action == "none"
    assert decision.reason == "on-preferred"
    assert fake.activations == []
    assert not (statedir / netfailover.NETWORK_CHANGE_FILENAME).exists()


def test_run_once_writes_network_change_on_activation(statedir, tmp_path):
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT)

    _run(statedir, fake, now=1234.5, route=route)

    payload = json.loads((statedir / netfailover.NETWORK_CHANGE_FILENAME).read_text())
    assert payload["ssid"] == "iPhone (5)"
    assert payload["ip"] == "198.51.100.4"
    assert payload["ts"] == 1234.5
    # no leftover temp file from the atomic write
    assert not [p for p in statedir.iterdir() if p.name.startswith(".")]


def test_run_once_no_network_change_when_activation_fails(statedir, tmp_path):
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT, up_rc=4)

    decision = _run(statedir, fake, now=1000.0, route=route)

    assert decision.action == "activate"
    assert fake.activations == ["iPhone (5)"]
    assert not (statedir / netfailover.NETWORK_CHANGE_FILENAME).exists()
    # the attempt is still recorded — a failed attempt must still be gapped
    assert netfailover.load_record(statedir).last_attempt_at["iPhone (5)"] == 1000.0


def test_attempt_gap_enforced_across_separate_runs(statedir, tmp_path):
    """Storm control must survive the process exiting — the record file is the memory."""
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)

    first = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT, up_rc=4)
    _run(statedir, first, now=1000.0, route=route)
    assert first.activations == ["iPhone (5)"]

    # a brand-new process 5 s later (default min_attempt_gap_s is 30)
    second = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT, up_rc=4)
    decision = _run(statedir, second, now=1005.0, route=route)
    assert decision.action == "none"
    assert decision.reason == "attempt-gap"
    assert second.activations == []

    # ... and past the gap it tries again
    third = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT, up_rc=4)
    decision = _run(statedir, third, now=1040.0, route=route)
    assert decision.action == "activate"
    assert third.activations == ["iPhone (5)"]


def test_run_once_tolerates_corrupt_record(statedir, tmp_path):
    (statedir / netfailover.RECORD_FILENAME).write_text("}}} not json")
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT)

    decision = _run(statedir, fake, now=1000.0, route=route)
    assert decision.action == "activate"


def test_run_once_persists_last_decision_for_log_latching(statedir, tmp_path):
    route = tmp_path / "route"
    route.write_text(ROUTE_WITH_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active=ACTIVE_PREFERRED, ip=IP_OUT)

    _run(statedir, fake, now=1000.0, route=route)
    rec = netfailover.load_record(statedir)
    assert rec.last_decision is not None
    assert rec.last_decision.reason == "on-preferred"
    assert rec.last_decision.ts == 1000.0


def test_run_once_dry_run_executes_and_writes_nothing(statedir, tmp_path):
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT)

    decision = _run(statedir, fake, now=1000.0, route=route, dry_run=True)

    assert decision.action == "activate"  # it still says what it WOULD do
    assert fake.activations == []
    assert list(statedir.iterdir()) == []


def test_run_once_no_candidate_visible(statedir, tmp_path):
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan="some-cafe-wifi:40:no\n", active="", ip="")

    decision = _run(statedir, fake, now=1000.0, route=route)
    assert decision.action == "none"
    assert decision.reason == "waiting-no-candidate"
    assert fake.activations == []


def test_run_once_logs_only_when_should_log(statedir, tmp_path, caplog):
    route = tmp_path / "route"
    route.write_text(ROUTE_WITH_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active=ACTIVE_PREFERRED, ip=IP_OUT)

    with caplog.at_level("INFO"):
        _run(statedir, fake, now=1000.0, route=route)
        _run(statedir, fake, now=1001.0, route=route)

    lines = [r for r in caplog.records if "event=network" in r.getMessage()]
    assert len(lines) == 1
    assert "stage=supervise" in lines[0].getMessage()


# --------------------------------------------------------------------------
# policy construction from the environment
# --------------------------------------------------------------------------


def test_policy_from_env_defaults(monkeypatch):
    monkeypatch.delenv("REACHY_NET_PREFERRED", raising=False)
    monkeypatch.delenv("REACHY_NET_FALLBACK", raising=False)
    policy = netfailover.policy_from_env()
    assert policy.preferred == "iPhone (5)"
    assert policy.fallback == "bar-nachum"


def test_policy_from_env_overrides(monkeypatch):
    monkeypatch.setenv("REACHY_NET_PREFERRED", "hotspot-x")
    monkeypatch.setenv("REACHY_NET_FALLBACK", "home-y")
    policy = netfailover.policy_from_env()
    assert policy.preferred == "hotspot-x"
    assert policy.fallback == "home-y"


def test_nmcli_binary_override(monkeypatch):
    monkeypatch.setenv("REACHY_NMCLI", "/usr/local/bin/nmcli")
    assert netfailover.nmcli_binary() == "/usr/local/bin/nmcli"
    monkeypatch.delenv("REACHY_NMCLI")
    assert netfailover.nmcli_binary() == "nmcli"


def test_rescan_from_env(monkeypatch):
    monkeypatch.delenv("REACHY_NET_RESCAN", raising=False)
    assert netfailover.rescan_from_env() is False
    monkeypatch.setenv("REACHY_NET_RESCAN", "1")
    assert netfailover.rescan_from_env() is True
    monkeypatch.setenv("REACHY_NET_RESCAN", "0")
    assert netfailover.rescan_from_env() is False


# --------------------------------------------------------------------------
# loop bounding
# --------------------------------------------------------------------------


def test_bounded_interval_clamps():
    assert netfailover.bounded_interval(0.5) == netfailover.MIN_INTERVAL_S
    assert netfailover.bounded_interval(9999) == netfailover.MAX_INTERVAL_S
    assert netfailover.bounded_interval(30.0) == 30.0


def test_run_loop_exits_after_preferred_settle(statedir, tmp_path):
    """--loop is self-terminating: it exits once the robot has been on the
    preferred network for PREFERRED_SETTLE_S, so the transient systemd unit
    the dispatcher starts cleans itself up."""
    route = tmp_path / "route"
    route.write_text(ROUTE_WITH_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active=ACTIVE_PREFERRED, ip=IP_OUT)

    clock = {"t": 0.0}
    slept: list[float] = []

    def sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    netfailover.run_loop(
        nmcli=fake,
        statedir=statedir,
        route_path=route,
        policy=netpolicy.Policy(preferred="iPhone (5)", fallback="bar-nachum"),
        clock=lambda: clock["t"],
        sleep=sleep,
    )

    assert slept  # it looped at least once
    assert clock["t"] >= netfailover.PREFERRED_SETTLE_S
    assert all(netfailover.MIN_INTERVAL_S <= s <= netfailover.MAX_INTERVAL_S for s in slept)


def test_run_loop_stops_on_stop_event(statedir, tmp_path):
    import threading

    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan="", active="", ip="")
    stop = threading.Event()

    clock = {"t": 0.0}

    def sleep(seconds):
        clock["t"] += seconds
        stop.set()

    netfailover.run_loop(
        nmcli=fake,
        statedir=statedir,
        route_path=route,
        policy=netpolicy.Policy(preferred="iPhone (5)", fallback="bar-nachum"),
        clock=lambda: clock["t"],
        sleep=sleep,
        stop_event=stop,
    )
    # one pass, then the stop event ends it — never the settle timer (it is
    # disconnected the whole time)
    assert clock["t"] < netfailover.PREFERRED_SETTLE_S


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_self_test_exits_zero_and_changes_nothing(statedir, tmp_path, monkeypatch):
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT)
    monkeypatch.setenv("REACHY_STATE_DIR", str(statedir))

    rc = netfailover.main(
        ["--self-test"], nmcli=fake, route_path=route
    )
    assert rc == 0
    assert fake.activations == []
    assert list(statedir.iterdir()) == []


def test_cli_self_test_exits_one_when_nmcli_is_broken(statedir, tmp_path, monkeypatch):
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    monkeypatch.setenv("REACHY_STATE_DIR", str(statedir))

    def broken(args, timeout=None):
        raise OSError("nmcli: command not found")

    rc = netfailover.main(["--self-test"], nmcli=broken, route_path=route)
    assert rc == 1


def test_cli_once_activates(statedir, tmp_path, monkeypatch):
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT)
    monkeypatch.setenv("REACHY_STATE_DIR", str(statedir))
    monkeypatch.setenv("REACHY_NET_PREFERRED", "iPhone (5)")
    monkeypatch.setenv("REACHY_NET_FALLBACK", "bar-nachum")

    rc = netfailover.main(["--once"], nmcli=fake, route_path=route)
    assert rc == 0
    assert fake.activations == ["iPhone (5)"]


def test_cli_once_dry_run_activates_nothing(statedir, tmp_path, monkeypatch):
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active="", ip=IP_OUT)
    monkeypatch.setenv("REACHY_STATE_DIR", str(statedir))

    rc = netfailover.main(["--once", "--dry-run"], nmcli=fake, route_path=route)
    assert rc == 0
    assert fake.activations == []
    assert list(statedir.iterdir()) == []


def test_cli_defaults_to_once(statedir, tmp_path, monkeypatch):
    route = tmp_path / "route"
    route.write_text(ROUTE_WITH_DEFAULT)
    fake = FakeNmcli(scan=SCAN_BOTH, active=ACTIVE_PREFERRED, ip=IP_OUT)
    monkeypatch.setenv("REACHY_STATE_DIR", str(statedir))

    rc = netfailover.main([], nmcli=fake, route_path=route)
    assert rc == 0
    assert fake.activations == []


def test_cli_never_raises_on_nmcli_failure(statedir, tmp_path, monkeypatch):
    """The dispatcher hook must never see a traceback — a broken nmcli is a
    warning and a non-zero rc, not an exception."""
    route = tmp_path / "route"
    route.write_text(ROUTE_NO_DEFAULT)
    monkeypatch.setenv("REACHY_STATE_DIR", str(statedir))

    def broken(args, timeout=None):
        raise OSError("boom")

    assert netfailover.main(["--once"], nmcli=broken, route_path=route) == 1


def test_module_has_main_entry_point():
    """`python -m reachy_nova.netfailover` must work (the dispatcher hook uses it)."""
    src = (Path(netfailover.__file__).parent / "netfailover.py").read_text()
    assert '__name__ == "__main__"' in src


def test_state_dir_env_is_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path / "sd"))
    assert netfailover.default_statedir() == tmp_path / "sd"
