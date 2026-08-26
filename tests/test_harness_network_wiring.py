"""The network leg's consumers, and a network-less start of the whole harness (t5).

Two things are proven here, both of them failure modes observed live on the
robot on 2026-08-26:

1. **A network transition reaches the cloud legs.**
   :class:`~reachy_nova.harness.app.NetworkReactor` is the consumer half of
   :class:`~reachy_nova.harness.network.NetworkUnit`: a ``joined``/``moved``
   transition asks Sonic to restart its stream and the Kiro session unit to
   respawn, ONCE each; a ``dropped`` asks for neither (the units' own
   watchdogs own the offline state, and respawning into a dead network only
   guarantees a failed respawn). A raising Sonic never costs Kiro its restart.

2. **Nothing in composition or startup blocks on the network.** With no route,
   a Sonic that cannot connect and a Kiro factory that fails, the harness must
   still claim its PID file, log ``harness up ... components=N`` with named
   degraded/absent lines, and observe engine liveness — the spec's
   "a network-less start yields named absences and later self-heal, never a
   stuck unit" boundary (h14).

Every test runs against an isolated ``REACHY_STATE_DIR`` and never touches a
real network: the route/address readers are stubbed at module level BEFORE any
:class:`NetworkUnit` is constructed.
"""

from __future__ import annotations

import os
import threading

import pytest

from reachy_nova.harness import app, network, statedir, supervisor
from reachy_nova.harness.app import NetworkReactor
from reachy_nova.harness.kiro_session import KiroSessionUnit
from reachy_nova.harness.network import NetworkUnit
from reachy_nova.kiro_acp import KiroAcpError


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHY_STATE_DIR", str(tmp_path / "reachy-state"))
    monkeypatch.delenv("NOVA_ACT_ENABLED", raising=False)
    monkeypatch.delenv("NOVA_OMNI_MODEL_ID", raising=False)
    # No route, no address: a machine with Wi-Fi down.
    monkeypatch.setattr(network, "_default_route_present", lambda: False)
    monkeypatch.setattr(network, "_default_wlan_address", lambda: None)
    yield


# --------------------------------------------------------------------------- #
# Recording stubs                                                             #
# --------------------------------------------------------------------------- #


class RecordingSonic:
    """Sonic with the t7 seam: records every ``request_immediate_restart``."""

    def __init__(self, *, raises: bool = False) -> None:
        self.restart_reasons: list[str] = []
        self._raises = raises

    def request_immediate_restart(self, reason: str) -> None:
        self.restart_reasons.append(reason)
        if self._raises:
            raise RuntimeError("sonic restart exploded")


class LegacySonic:
    """Sonic WITHOUT the t7 seam — only the older stop()/restart() pair."""

    def __init__(self) -> None:
        self.stops = 0
        self.restarts: list[threading.Event] = []

    def stop(self) -> None:
        self.stops += 1

    def restart(self, stop_event: threading.Event) -> None:
        self.restarts.append(stop_event)


class RecordingKiro:
    name = "kiro_session"

    def __init__(self) -> None:
        self.restart_reasons: list[str] = []

    def request_restart(self, reason: str) -> None:
        self.restart_reasons.append(reason)


def _messages(caplog) -> str:
    return " | ".join(r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# 1. joined / moved restart both legs, exactly once                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("event", ["joined", "moved"])
def test_joined_or_moved_requests_one_sonic_and_one_kiro_restart(event) -> None:
    sonic, kiro = RecordingSonic(), RecordingKiro()
    reactor = NetworkReactor(sonic, kiro)
    reactor.start(threading.Event())

    reactor.on_network_change(event, {"ip": "172.20.10.2", "ssid": "iPhone (5)"})

    assert len(sonic.restart_reasons) == 1
    assert len(kiro.restart_reasons) == 1
    assert "iPhone (5)" in sonic.restart_reasons[0]
    assert "172.20.10.2" in kiro.restart_reasons[0]


def test_dropped_requests_no_restart_at_all(caplog) -> None:
    sonic, kiro = RecordingSonic(), RecordingKiro()
    reactor = NetworkReactor(sonic, kiro)
    reactor.start(threading.Event())

    with caplog.at_level("INFO"):
        reactor.on_network_change("dropped", {"reason": "no-route"})

    assert sonic.restart_reasons == []
    assert kiro.restart_reasons == []
    assert "network dropped" in _messages(caplog)


def test_a_raising_sonic_does_not_cost_kiro_its_restart(caplog) -> None:
    sonic, kiro = RecordingSonic(raises=True), RecordingKiro()
    reactor = NetworkReactor(sonic, kiro)
    reactor.start(threading.Event())

    with caplog.at_level("INFO"):
        reactor.on_network_change("joined", {"ip": "1.2.3.4", "ssid": "bar-nachum"})

    assert len(sonic.restart_reasons) == 1  # it was asked, it just blew up
    assert len(kiro.restart_reasons) == 1
    assert "sonic restart failed" in _messages(caplog)


def test_reactor_never_raises_into_the_network_unit() -> None:
    """A callback exception would kill nothing (NetworkUnit catches), but the
    reactor is the layer that must not produce one in the first place."""
    sonic, kiro = RecordingSonic(raises=True), RecordingKiro()
    kiro.request_restart = lambda reason: (_ for _ in ()).throw(RuntimeError("kiro boom"))
    reactor = NetworkReactor(sonic, kiro)
    reactor.start(threading.Event())

    reactor.on_network_change("joined", {"ip": "1.2.3.4", "ssid": "x"})  # must not raise


def test_sonic_restart_path_is_named_in_the_log(caplog) -> None:
    sonic, kiro = RecordingSonic(), RecordingKiro()
    reactor = NetworkReactor(sonic, kiro)
    reactor.start(threading.Event())
    with caplog.at_level("INFO"):
        reactor.on_network_change("joined", {"ip": "1.2.3.4", "ssid": "x"})
    assert "path=request_immediate_restart" in _messages(caplog)


def test_legacy_sonic_falls_back_to_stop_plus_restart(caplog) -> None:
    """Without the t7 seam the older stop()/restart() pair is used — and said so."""
    sonic, kiro = LegacySonic(), RecordingKiro()
    reactor = NetworkReactor(sonic, kiro)
    stop_event = threading.Event()
    reactor.start(stop_event)

    with caplog.at_level("INFO"):
        reactor.on_network_change("joined", {"ip": "1.2.3.4", "ssid": "x"})

    assert sonic.stops == 1
    assert sonic.restarts == [stop_event]
    assert "path=stop+restart" in _messages(caplog)
    assert len(kiro.restart_reasons) == 1


def test_reactor_without_a_kiro_unit_still_restarts_sonic() -> None:
    sonic = RecordingSonic()
    reactor = NetworkReactor(sonic, None)
    reactor.start(threading.Event())
    reactor.on_network_change("joined", {"ip": "1.2.3.4", "ssid": "x"})
    assert len(sonic.restart_reasons) == 1


# --------------------------------------------------------------------------- #
# 2. The real NetworkUnit -> reactor edge                                     #
# --------------------------------------------------------------------------- #


def test_one_network_unit_transition_fires_exactly_one_restart_each(tmp_path) -> None:
    sonic, kiro = RecordingSonic(), RecordingKiro()
    reactor = NetworkReactor(sonic, kiro)
    reactor.start(threading.Event())

    online = {"value": True}
    unit = NetworkUnit(
        route_reader=lambda: online["value"],
        addr_reader=lambda: "172.20.10.2" if online["value"] else None,
        change_file=tmp_path / "network-change",
        poll_interval=10.0,
    )
    unit.on_change(reactor.on_network_change)

    unit._observe()  # startup -> joined
    unit._observe()  # stable online — latched, no second callback

    assert len(sonic.restart_reasons) == 1
    assert len(kiro.restart_reasons) == 1

    online["value"] = False
    unit._observe()  # -> dropped: still exactly one restart each
    assert len(sonic.restart_reasons) == 1
    assert len(kiro.restart_reasons) == 1


def test_kiro_session_unit_request_restart_is_the_wired_surface() -> None:
    """The reactor calls the REAL method name the unit exposes (t5 contract)."""
    assert callable(KiroSessionUnit.request_restart)


# --------------------------------------------------------------------------- #
# 3. build_app wires the network leg                                          #
# --------------------------------------------------------------------------- #


def test_build_app_constructs_the_network_unit_and_the_reactor() -> None:
    components = app.build_app()
    names = [type(c).__name__ for c in components]
    assert "NetworkUnit" in names
    assert "NetworkReactor" in names
    # The reactor must be started (given the stop_event) BEFORE the poll thread
    # that can fire a transition at it.
    assert names.index("NetworkReactor") < names.index("NetworkUnit")


def test_build_app_registers_the_reactor_as_a_network_callback() -> None:
    components = app.build_app()
    reactor = next(c for c in components if type(c).__name__ == "NetworkReactor")
    unit = next(c for c in components if type(c).__name__ == "NetworkUnit")
    assert reactor.on_network_change in unit._callbacks


# --------------------------------------------------------------------------- #
# 4. A network-less start of the whole harness (h14)                          #
# --------------------------------------------------------------------------- #


class UnreachableSonic:
    """A Sonic whose cloud connection can never be made.

    Mirrors the real class's contract rather than its internals: ``start()``
    returns (the connection happens on its own retrying thread) and the stream
    is simply never live. ``build_app`` assigns callbacks onto it, so it must
    tolerate arbitrary attribute assignment.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.started = 0
        self.connect_attempts = 0

    def start(self, stop_event: threading.Event) -> None:
        self.started += 1
        self.connect_attempts += 1  # would raise ConnectionError inside the loop

    def stop(self) -> None:
        return None

    def inject_text(self, text: str) -> None:
        return None

    def feed_audio(self, samples) -> None:
        return None

    def send_tool_result(self, tool_use_id: str, result: str) -> None:
        return None

    def request_immediate_restart(self, reason: str) -> None:
        self.connect_attempts += 1


class DeadKiroSession:
    """A kiro-cli that exits the moment it is spawned — the cold-boot shape."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        raise KiroAcpError("kiro-cli process exited")

    def close(self) -> None:
        return None


def test_networkless_start_claims_the_pid_and_comes_up_degraded(monkeypatch, caplog):
    import reachy_nova.kiro_acp as kiro_acp
    import reachy_nova.nova_sonic as nova_sonic

    monkeypatch.setattr(nova_sonic, "NovaSonic", UnreachableSonic)
    monkeypatch.setattr(kiro_acp, "KiroAcpSession", DeadKiroSession)
    monkeypatch.setenv("FORGE_WRITER", "kiro")

    stop_event = threading.Event()
    with caplog.at_level("INFO"):
        assert supervisor.acquire_pid_file() is True
        components = supervisor._composed_components()
        try:
            supervisor.run(
                components,
                stop_event,
                poll_interval=0.01,
                tick_hook=lambda ticks: stop_event.set() if ticks >= 2 else None,
            )
        finally:
            supervisor.release_pid_file()

    text = _messages(caplog)
    # PID claimed and the harness reported itself up with a component count.
    assert "harness up pid=" in text
    assert "components=" in text
    # Sonic came up (its own loop owns the reconnect) and Kiro came up degraded
    # — never the 2026-08-26 'start failed name=kiro_session' line (h12).
    assert "start failed name=kiro_session" not in text
    assert "started degraded" in text
    assert "started name=kiro_session" in text
    # Engine liveness was observed (absent on this box, but NAMED).
    assert "engine absent" in text or "engine live" in text
    # The network is down, so the network leg says so by name.
    assert "event=network" in text


def test_networkless_start_never_leaves_the_pid_file_behind(monkeypatch):
    import reachy_nova.kiro_acp as kiro_acp
    import reachy_nova.nova_sonic as nova_sonic

    monkeypatch.setattr(nova_sonic, "NovaSonic", UnreachableSonic)
    monkeypatch.setattr(kiro_acp, "KiroAcpSession", DeadKiroSession)
    monkeypatch.setenv("FORGE_WRITER", "kiro")

    assert supervisor.acquire_pid_file() is True
    assert supervisor.read_pid() == os.getpid()
    supervisor.release_pid_file()
    assert not statedir.harness_pid_path().exists()


def test_build_app_does_not_block_on_the_network(monkeypatch):
    """Composition itself must be pure wiring — no route, no address, no wait."""
    import reachy_nova.nova_sonic as nova_sonic

    monkeypatch.setattr(nova_sonic, "NovaSonic", UnreachableSonic)
    components = app.build_app()
    assert components  # a harness with something to run
    assert any(type(c).__name__ == "NetworkUnit" for c in components)
