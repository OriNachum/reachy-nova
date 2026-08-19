"""Chaos: bad AWS creds / cloud loss mid-conversation — the mouth goes away.

On the robot this is a wrong ``AWS_ACCESS_KEY_ID`` in the env, a Bedrock
region outage, or the daemon's media route refusing connections while Nova is
mid-sentence. The speaker leg must resolve EVERY such failure to the
mouth-loss grace path — a named ``reason=playback-http-failed`` drop, the
echo gate cleared, every pending utterance purged (named), one
``on_playback_failure()`` call — and the SAME worker must speak again the
moment the transport heals. No restart, no rebuild, no real Bedrock call.

The unit tests (``tests/test_harness_speaking.py``) pin a single failure's
grace path; the chaos angle here is the ORDERING — a failure landing while a
further utterance is already queued behind it, and repeated loss/heal cycles
on one instance.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pytest

from reachy_nova.harness.gate import EchoGate
from reachy_nova.harness.speaking import SonicSpeaker

SAMPLE_RATE = 24000
#: 20 ms of audio per utterance — playback windows stay tiny.
UTTERANCE = np.full(480, 0.25, dtype=np.float32)


class ChaosPoster:
    """A scriptable HTTP transport: healthy, refusing, or held mid-flight.

    ``mode`` flips between ``"ok"`` and ``"refuse"`` (connection-refused
    style, what bad creds / a dead daemon look like). ``proceed`` lets a test
    HOLD the worker inside a post while it queues more utterances behind it,
    so the purge-on-failure ordering is real rather than lucky timing.
    """

    def __init__(self) -> None:
        self.mode = "ok"
        self.proceed = threading.Event()
        self.proceed.set()
        self.attempts: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, base_url: str, wav_bytes: bytes, filename: str) -> None:
        with self._lock:
            mode = self.mode
            self.attempts.append(mode)  # visible while held: the post is in flight
        assert self.proceed.wait(5.0), "test forgot to release the held poster"
        if mode == "refuse":
            raise ConnectionRefusedError(
                "[Errno 111] Connection refused (cloud/creds loss, simulated)"
            )

    @property
    def attempt_count(self) -> int:
        with self._lock:
            return len(self.attempts)


def speak(speaker: SonicSpeaker) -> None:
    """Drive one complete Sonic utterance through the callback wire."""
    speaker.on_state_change("speaking")
    speaker.on_audio_chunk(UTTERANCE)
    speaker.on_state_change("listening")


def wait_until(predicate, timeout: float = 5.0) -> bool:
    """Bounded poll: true as soon as predicate() is truthy, else re-check once
    more at the deadline (never a bare one-shot assert on unsynchronized
    cross-thread state — under pytest-xdist contention, the worker thread and
    the test thread run arbitrarily interleaved).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def speak_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "nova.sensory" and "stage=speak" in r.getMessage()
    ]


@pytest.fixture
def stop_event():
    ev = threading.Event()
    yield ev
    ev.set()


def test_cloud_loss_mid_conversation_purges_and_recovers(stop_event, caplog) -> None:
    caplog.set_level(logging.INFO, logger="nova.sensory")
    gate = EchoGate(margin_s=0.02)
    poster = ChaosPoster()
    failures: list[float] = []
    speaker = SonicSpeaker(
        gate=gate,
        sample_rate=SAMPLE_RATE,
        base_url="http://localhost:9",  # never dialled — the poster is fake
        poster=poster,
        on_playback_failure=lambda: failures.append(time.monotonic()),
    )
    speaker.start(stop_event)

    # Phase 1 — the conversation is healthy: utterance 1 plays.
    speak(speaker)
    assert wait_until(lambda: speaker.utterances_played == 1)
    assert poster.attempts == ["ok"]

    # Phase 2 — the cloud goes away mid-conversation. Hold the worker inside
    # utterance 2's post while utterance 3 queues up behind it.
    poster.proceed.clear()
    poster.mode = "refuse"
    speak(speaker)  # utterance 2 — will fail in flight
    assert wait_until(lambda: poster.attempt_count == 2), "worker never reached the post"
    speak(speaker)  # utterance 3 — queued behind the doomed one
    assert wait_until(lambda: not speaker.idle)
    poster.proceed.set()  # the post now fails, with a full queue behind it

    assert wait_until(lambda: speaker.playback_failures == 1)
    assert wait_until(lambda: len(failures) == 1), "on_playback_failure never fired"
    # Named drop, gate freed, queue purged — the full mouth-loss grace path.
    # `on_playback_failure` fires only after BOTH [SENSE] lines below are
    # written (same worker thread, strictly sequential — see speaking.py's
    # `_mouth_loss`), so the `failures` wait above already orders these; the
    # bounded polls here are belt-and-suspenders against caplog's own
    # cross-thread visibility rather than a real ordering assumption.
    assert wait_until(
        lambda: any("dropped reason=playback-http-failed" in m for m in speak_lines(caplog))
    ), "the cloud loss was not a named [SENSE] drop"
    assert wait_until(
        lambda: any(
            "dropped reason=preempted-after-failure pending=1" in m for m in speak_lines(caplog)
        )
    ), "the queued utterance was not purged (named) with the failure"
    assert gate.remaining() == 0.0, "the echo gate stayed armed after mouth loss"
    assert wait_until(lambda: speaker.idle), "purged work still counted as pending"
    assert speaker.worker_alive, "the worker died with the cloud"

    # Phase 3 — the transport heals: the NEXT utterance plays, same worker.
    # utt3 was purged (never posted, dropped above as preempted-after-failure)
    # — the LAST queued utterance at this point is the one enqueued below,
    # and recovery must play exactly that one.
    poster.mode = "ok"
    speak(speaker)
    assert wait_until(lambda: speaker.utterances_played == 2), "no recovery after heal"
    # ok, refused, ok — exactly three poster attempts; the worker calls the
    # poster strictly in queue order on a single thread, so this ordering is
    # not itself timing-dependent once utterances_played has reached 2.
    assert poster.attempts == ["ok", "refuse", "ok"]
    assert len(failures) == 1, "recovery re-fired the failure callback"
    # `utterances_played` is incremented BEFORE its "played" [SENSE] line is
    # written (speaking.py's `_play_one`), and — unlike the failure path
    # above — nothing downstream depends on that line, so there is no
    # happens-before edge the test can ride: poll for the line explicitly
    # rather than assuming it exists the instant the counter does.
    assert wait_until(
        lambda: len([m for m in speak_lines(caplog) if "] played" in m]) == 2
    ), "recovery's playback was never logged as played (utterance 1 + the post-recovery replay)"
    played = [m for m in speak_lines(caplog) if "] played" in m]
    assert len(played) == 2

    speaker.stop()


def test_repeated_cloud_flaps_never_require_a_restart(stop_event, caplog) -> None:
    """Two full loss/heal cycles on ONE speaker instance, worker alive throughout."""
    caplog.set_level(logging.INFO, logger="nova.sensory")
    gate = EchoGate(margin_s=0.01)
    poster = ChaosPoster()
    speaker = SonicSpeaker(
        gate=gate, sample_rate=SAMPLE_RATE, base_url="http://localhost:9", poster=poster
    )
    speaker.start(stop_event)

    for cycle in (1, 2):
        poster.mode = "refuse"
        speak(speaker)
        assert wait_until(lambda: speaker.playback_failures == cycle), f"cycle {cycle}"
        assert speaker.worker_alive
        poster.mode = "ok"
        speak(speaker)
        assert wait_until(lambda: speaker.utterances_played == cycle), f"cycle {cycle}"

    # The worker processes queue items strictly one at a time, so by the time
    # cycle 2's "ok" playback has been observed above, cycle 2's failure
    # (and its [SENSE] drop line) must already be fully processed — but poll
    # explicitly anyway rather than leaning on that transitive ordering.
    assert wait_until(
        lambda: len(
            [m for m in speak_lines(caplog) if "dropped reason=playback-http-failed" in m]
        )
        == 2
    ), "expected exactly 2 named playback-http-failed drops, one per flap cycle"
    drops = [
        m for m in speak_lines(caplog) if "dropped reason=playback-http-failed" in m
    ]
    assert len(drops) == 2, drops
    assert speaker.worker_alive
    assert gate.remaining() >= 0.0  # sanity: gate usable, not wedged
    speaker.stop()
