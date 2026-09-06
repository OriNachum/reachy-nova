"""Mood: a small decaying state rendered as one context sentence (t6).

Ports the *idea* of ``config/emotions.yaml`` (baseline + per-second decay
toward baseline, small per-event delta table) without loading that file or
the legacy ``emotions.py`` module — see the plan's t6 instruction.
"""

from __future__ import annotations

import threading

from reachy_nova.harness.mood import Mood


def make_clock(start: float = 0.0):
    state = {"t": start}

    def clock() -> float:
        return state["t"]

    def advance(delta: float) -> None:
        state["t"] += delta

    return clock, advance


# --------------------------------------------------------------------------- #
# 1. note() adjusts bounded scalars per event class from the table.           #
# --------------------------------------------------------------------------- #


def test_fresh_mood_renders_the_neutral_sentence():
    clock, _advance = make_clock()
    mood = Mood(clock=clock)
    assert mood.render() == "You are calm and easy right now."


def test_note_pat_raises_cheeky_and_playful_and_lowers_lonely():
    clock, _advance = make_clock()
    mood = Mood(clock=clock)
    before = mood.state()
    mood.note("pat")
    after = mood.state()
    assert after["cheeky"] > before["cheeky"]
    assert after["playful"] > before["playful"]
    assert after["lonely"] < before["lonely"]


def test_note_face_lowers_lonely():
    clock, _advance = make_clock()
    mood = Mood(clock=clock)
    before = mood.state()["lonely"]
    mood.note("face")
    assert mood.state()["lonely"] < before


def test_note_user_turn_and_assistant_turn_lower_lonely():
    clock, _advance = make_clock()
    mood = Mood(clock=clock)
    baseline_lonely = mood.state()["lonely"]
    mood.note("user_turn")
    after_user = mood.state()["lonely"]
    assert after_user < baseline_lonely
    mood.note("assistant_turn")
    after_assistant = mood.state()["lonely"]
    assert after_assistant < after_user


def test_note_unknown_event_class_is_a_safe_noop():
    clock, _advance = make_clock()
    mood = Mood(clock=clock)
    before = mood.state()
    mood.note("some-event-nobody-registered")
    after = mood.state()
    assert before == after


def test_note_silence_event_applies_no_delta_and_does_not_reset_the_clock():
    clock, advance = make_clock()
    mood = Mood(clock=clock)
    mood.note("pat")
    advance(700.0)
    before = mood.state()["silence_s"]
    mood.note("silence")
    after = mood.state()["silence_s"]
    # silence is DERIVED from time since the last real note -- explicitly
    # noting it must not itself count as activity (that would make the
    # silence clock un-advanceable).
    assert after == before


def test_intensity_scales_the_delta():
    clock, _advance = make_clock()
    mood_low = Mood(clock=clock)
    mood_low.note("pat", intensity=0.1)
    mood_high = Mood(clock=make_clock()[0])
    mood_high.note("pat", intensity=1.0)
    assert mood_high.state()["cheeky"] > mood_low.state()["cheeky"]


# --------------------------------------------------------------------------- #
# 2. render() picks one sentence from thresholds on levels + silence.         #
#    At least three distinct states as events and time pass.                 #
# --------------------------------------------------------------------------- #


def test_three_distinct_states_neutral_then_cheeky_then_bored():
    clock, advance = make_clock()
    mood = Mood(clock=clock)

    neutral = mood.render()
    assert neutral == "You are calm and easy right now."

    mood.note("pat")
    cheeky = mood.render()
    assert cheeky == "You have just been petted and feel cheeky."
    assert cheeky != neutral

    advance(700.0)
    bored = mood.render()
    assert "bored" in bored
    assert "11 minutes" in bored
    assert bored not in (neutral, cheeky)


def test_playful_sentence_emerges_once_the_sharper_cheeky_spike_fades():
    # A pat spikes both cheeky and playful, but cheeky decays faster (it is
    # the sharper, more transient reaction) -- a little while later cheeky
    # has faded below its threshold while playful lingers above its own.
    clock, advance = make_clock()
    mood = Mood(clock=clock)
    mood.note("pat")
    advance(60.0)
    state = mood.state()
    assert state["cheeky"] < 0.5
    assert state["playful"] >= 0.35
    assert mood.render() == "You are in a playful mood."


def test_moderate_silence_renders_the_lonely_sentence():
    clock, advance = make_clock()
    mood = Mood(clock=clock)
    advance(200.0)  # past the lonely threshold, short of the bored one
    rendered = mood.render()
    assert "lonely" in rendered
    assert "bored" not in rendered


def test_bored_sentence_names_the_minutes_of_silence():
    clock, advance = make_clock()
    mood = Mood(clock=clock)
    advance(600.0)
    assert mood.render() == "Nobody has spoken to you for 10 minutes; you are a little bored."
    advance(120.0)
    assert mood.render() == "Nobody has spoken to you for 12 minutes; you are a little bored."


def test_render_accepts_an_explicit_now_argument():
    clock, _advance = make_clock()
    mood = Mood(clock=clock)
    mood.note("pat")
    # explicit `now` far in the future overrides the injected clock's value
    rendered = mood.render(now=clock() + 700.0)
    assert "bored" in rendered


# --------------------------------------------------------------------------- #
# 3. Levels decay toward baseline with elapsed time; bounds hold.             #
# --------------------------------------------------------------------------- #


def test_levels_decay_toward_baseline_over_elapsed_time():
    clock, advance = make_clock()
    mood = Mood(clock=clock)
    mood.note("pat")
    spiked = mood.state()["cheeky"]
    advance(90.0)
    decayed_once = mood.state()["cheeky"]
    advance(900.0)
    decayed_more = mood.state()["cheeky"]
    baseline = Mood(clock=make_clock()[0]).state()["cheeky"]
    assert spiked > decayed_once > decayed_more
    assert abs(decayed_more - baseline) < 1e-3


def test_calm_recovers_upward_toward_its_baseline_after_a_pat():
    clock, advance = make_clock()
    mood = Mood(clock=clock)
    fresh_calm = mood.state()["calm"]
    mood.note("pat")  # pat nudges calm down
    dipped = mood.state()["calm"]
    assert dipped < fresh_calm
    advance(3600.0)
    recovered = mood.state()["calm"]
    assert recovered > dipped
    assert abs(recovered - fresh_calm) < 1e-3


def test_levels_never_leave_the_unit_interval_under_repeated_intense_notes():
    clock, advance = make_clock()
    mood = Mood(clock=clock)
    for _ in range(50):
        mood.note("pat", intensity=5.0)
        advance(0.01)
    state = mood.state()
    for name in ("playful", "calm", "lonely", "cheeky"):
        assert 0.0 <= state[name] <= 1.0


def test_state_reports_all_dimensions_and_silence_seconds():
    clock, advance = make_clock()
    mood = Mood(clock=clock)
    advance(42.0)
    state = mood.state()
    for name in ("playful", "calm", "lonely", "cheeky"):
        assert name in state
        assert 0.0 <= state[name] <= 1.0
    assert state["silence_s"] == 42.0


# --------------------------------------------------------------------------- #
# 4. Thread-safety: one lock guards note()/state()/render().                 #
# --------------------------------------------------------------------------- #


def test_note_and_state_are_thread_safe_under_concurrent_use():
    mood = Mood()  # real monotonic clock: this test just wants no crashes/races

    errors: list[Exception] = []

    def worker(event: str) -> None:
        try:
            for _ in range(200):
                mood.note(event, intensity=0.3)
                mood.state()
                mood.render()
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(event,))
        for event in ("pat", "face", "user_turn", "assistant_turn")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    state = mood.state()
    for name in ("playful", "calm", "lonely", "cheeky"):
        assert 0.0 <= state[name] <= 1.0
