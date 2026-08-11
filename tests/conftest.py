"""Shared pytest configuration for reachy_nova.

Several runtime dependencies are hardware-heavy or simply absent on a
plain dev machine: ``reachy_mini`` needs the physical robot SDK, ``nemo``
(``nemo_toolkit``) and ``cv2`` (``opencv-python``) are large/slow installs,
``boto3`` talks to AWS Bedrock, and ``paho`` (``paho-mqtt``) expects an MQTT
broker. None of that should be required to *collect and run* the unit test
suite.

The guard below stubs each one into ``sys.modules`` with a permissive
``types.SimpleNamespace`` ONLY when the real package fails to import. A
machine that already has the real dependency installed (e.g. the robot
itself, or a fully-provisioned dev box) keeps using the genuine module
untouched - this file never shadows a real, importable package.

Individual test modules that need a real submodule shape (e.g.
``paho.mqtt.client`` or ``nemo.collections.asr``) can extend this stub, or
fall back to ``pytest.importorskip("nemo.collections.asr")`` to skip
cleanly when the real dependency truly isn't present.
"""

from __future__ import annotations

import sys
import types

import pytest

# Hardware-heavy or often-absent-on-dev-machines top-level packages.
_HEAVY_OR_ABSENT_MODULES = ("reachy_mini", "nemo", "cv2", "boto3", "paho")

for _module_name in _HEAVY_OR_ABSENT_MODULES:
    if _module_name in sys.modules:
        continue
    try:
        __import__(_module_name)
    except ImportError:
        sys.modules[_module_name] = types.SimpleNamespace()  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _no_ambient_echo_gate_policy(monkeypatch):
    """Never inherit the developer's own ``NOVA_ECHO_GATE`` (see ``gate.py``).

    The hearing leg's suppression is env-selected, so a shell that exports the
    policy would otherwise flip the DEFAULT under every test that does not set
    it. Tests that care set it themselves — ``monkeypatch.setenv`` in the test
    body still wins over this deletion.
    """
    monkeypatch.delenv("NOVA_ECHO_GATE", raising=False)
