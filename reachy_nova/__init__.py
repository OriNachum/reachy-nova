"""Reachy Nova — AI brain for Reachy Mini, plus the wireless harness.

``ReachyNova`` (the direct-SDK ``ReachyMiniApp``) is exposed lazily (PEP 562)
so importing ``reachy_nova.harness.*`` or ``reachy_nova.config`` never drags in
``reachy_mini``/the robot SDK — the harness must run on a box where only the
AWS-side dependencies are installed, and the boundary test forbids the SDK
inside the harness package.
"""

__all__ = ["ReachyNova"]


def __getattr__(name: str):
    if name == "ReachyNova":
        from .main import ReachyNova

        return ReachyNova
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
