"""Chaos/degradation suite (task t12).

Cross-component failure sequences the per-module unit tests do not cover:
kill/restart of the runtime under a live harness, cloud loss mid-conversation,
harness death + rebirth, and wifi flaps. Every case asserts BOTH a named
``[SENSE ...]`` drop line and clean recovery in the same test, against local
fakes only — no network, no broker, no robot, no AWS.

The on-robot equivalents of each case are listed in ``ON_ROBOT.md`` next to
this file.
"""
