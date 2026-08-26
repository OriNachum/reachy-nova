"""Tests for reachy_nova.vocalize and the "vocalize" skill executor.

vocalize.py (t4) is pure-numpy additive harmonic synthesis: a fundamental
sine plus 2-4 harmonics, driven by a per-kind pitch envelope, shaped by an
amplitude envelope that fades to (near) zero at both ends. Coverage here
targets the two acceptance criteria for this task:

1. ``vocalize.synthesize(kind, ...)`` returns float32 mono in [-1, 1] at a
   requested sample rate for at least 3 expressive kinds (chirp_up, trill,
   purr_tone); shape/range/duration and click-free envelope boundaries.
2. The "vocalize" executor is registered in skill_executors.py, pushes
   synthesized audio through the same speaker buffer path Sonic audio rides
   (an optional ``ctx.audio_output`` callback, looked up defensively), and
   returns a short result string. It degrades gracefully when that path
   isn't wired up (stub ctx / no ``audio_output`` attribute).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from reachy_nova import vocalize
from reachy_nova.skill_executors import _vocalize_executor, register_all

REQUIRED_KINDS = ("chirp_up", "trill", "purr_tone")


class TestValidKinds:
    def test_required_kinds_are_all_valid(self):
        for kind in REQUIRED_KINDS:
            assert kind in vocalize.VALID_KINDS


class TestSynthesizeShapeAndRange:
    @pytest.mark.parametrize("kind", REQUIRED_KINDS)
    def test_returns_float32_mono_array(self, kind):
        samples = vocalize.synthesize(kind, sample_rate=24000, duration=0.5)
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert samples.ndim == 1

    @pytest.mark.parametrize("kind", REQUIRED_KINDS)
    def test_samples_stay_in_unit_range(self, kind):
        samples = vocalize.synthesize(kind, sample_rate=24000, duration=0.6, intensity=1.0)
        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)

    @pytest.mark.parametrize("kind", REQUIRED_KINDS)
    def test_duration_controls_sample_count(self, kind):
        sample_rate = 24000
        duration = 0.5
        samples = vocalize.synthesize(kind, sample_rate=sample_rate, duration=duration)
        expected = round(duration * sample_rate)
        assert abs(len(samples) - expected) <= 1

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 24000, 44100])
    def test_honors_requested_sample_rate(self, sample_rate):
        duration = 0.4
        samples = vocalize.synthesize("chirp_up", sample_rate=sample_rate, duration=duration)
        expected = round(duration * sample_rate)
        assert abs(len(samples) - expected) <= 1

    def test_duration_clamped_to_min(self):
        samples = vocalize.synthesize("chirp_up", sample_rate=24000, duration=0.01)
        expected = round(vocalize.MIN_DURATION * 24000)
        assert abs(len(samples) - expected) <= 1

    def test_duration_clamped_to_max(self):
        samples = vocalize.synthesize("chirp_up", sample_rate=24000, duration=5.0)
        expected = round(vocalize.MAX_DURATION * 24000)
        assert abs(len(samples) - expected) <= 1

    def test_default_duration_within_bounds(self):
        for kind in REQUIRED_KINDS:
            samples = vocalize.synthesize(kind, sample_rate=24000)
            duration_s = len(samples) / 24000
            assert vocalize.MIN_DURATION <= duration_s <= vocalize.MAX_DURATION

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            vocalize.synthesize("bark", sample_rate=24000, duration=0.5)

    def test_nonpositive_sample_rate_raises(self):
        with pytest.raises(ValueError):
            vocalize.synthesize("chirp_up", sample_rate=0, duration=0.5)


class TestClickFreeEnvelope:
    """Amplitude envelope must fade to (near) zero at both boundaries."""

    _NEAR_ZERO = 1e-6

    @pytest.mark.parametrize("kind", REQUIRED_KINDS)
    def test_first_and_last_samples_near_zero(self, kind):
        samples = vocalize.synthesize(kind, sample_rate=24000, duration=0.6, intensity=1.0)
        assert abs(float(samples[0])) <= self._NEAR_ZERO
        assert abs(float(samples[-1])) <= self._NEAR_ZERO

    @pytest.mark.parametrize("kind", REQUIRED_KINDS)
    def test_boundary_near_zero_at_low_intensity_too(self, kind):
        samples = vocalize.synthesize(kind, sample_rate=24000, duration=0.6, intensity=0.0)
        assert abs(float(samples[0])) <= self._NEAR_ZERO
        assert abs(float(samples[-1])) <= self._NEAR_ZERO

    @pytest.mark.parametrize("kind", REQUIRED_KINDS)
    def test_signal_is_not_silent_in_the_middle(self, kind):
        # A click-free envelope shouldn't mean an inaudible sound — the
        # middle of the clip should carry real energy.
        samples = vocalize.synthesize(kind, sample_rate=24000, duration=0.6, intensity=1.0)
        midpoint = len(samples) // 2
        window = samples[midpoint - 50: midpoint + 50]
        assert np.max(np.abs(window)) > 0.05


class TestAdditiveHarmonics:
    """fundamental + 2-4 harmonics, per the acceptance criterion wording."""

    def test_num_harmonics_changes_the_waveform(self):
        kwargs = dict(kind="trill", sample_rate=24000, duration=0.5, intensity=1.0)
        few = vocalize.synthesize(num_harmonics=2, **kwargs)
        many = vocalize.synthesize(num_harmonics=4, **kwargs)
        assert few.shape == many.shape
        assert not np.allclose(few, many)

    def test_num_harmonics_is_clamped_into_range(self):
        # Out-of-range requests shouldn't raise — they clamp into [2, 4].
        low = vocalize.synthesize("trill", sample_rate=24000, duration=0.5, num_harmonics=0)
        clamped_low = vocalize.synthesize("trill", sample_rate=24000, duration=0.5, num_harmonics=2)
        high = vocalize.synthesize("trill", sample_rate=24000, duration=0.5, num_harmonics=99)
        clamped_high = vocalize.synthesize("trill", sample_rate=24000, duration=0.5, num_harmonics=4)
        assert np.allclose(low, clamped_low)
        assert np.allclose(high, clamped_high)


class TestMasterGain:
    """``NOVA_VOCALIZE_GAIN`` (default 0.35) scales the final synthesized
    signal so vocalize output isn't louder than Nova Sonic's own speech.
    Parsed defensively like ``NovaSonic``'s ``_liveness_window``/backoff env
    knobs: missing/non-numeric/non-positive falls back to the default, and
    the value is capped at 1.0 (vocalize can only be made quieter, never
    boosted past raw synthesis)."""

    def test_default_peak_is_at_or_below_default_gain(self, monkeypatch):
        monkeypatch.delenv("NOVA_VOCALIZE_GAIN", raising=False)
        samples = vocalize.synthesize("chirp_up", sample_rate=24000, duration=0.5, intensity=1.0)
        assert np.max(np.abs(samples)) <= vocalize.DEFAULT_MASTER_GAIN + 1e-6

    def test_env_override_above_default_is_louder(self, monkeypatch):
        monkeypatch.setenv("NOVA_VOCALIZE_GAIN", "1.0")
        samples = vocalize.synthesize("chirp_up", sample_rate=24000, duration=0.5, intensity=1.0)
        assert np.max(np.abs(samples)) > vocalize.DEFAULT_MASTER_GAIN

    def test_env_override_below_default_is_respected(self, monkeypatch):
        monkeypatch.setenv("NOVA_VOCALIZE_GAIN", "0.1")
        samples = vocalize.synthesize("chirp_up", sample_rate=24000, duration=0.5, intensity=1.0)
        assert np.max(np.abs(samples)) <= 0.1 + 1e-6

    def test_gain_is_capped_at_one(self, monkeypatch):
        monkeypatch.setenv("NOVA_VOCALIZE_GAIN", "5.0")
        assert vocalize._master_gain() == 1.0

    def test_missing_env_defaults(self, monkeypatch):
        monkeypatch.delenv("NOVA_VOCALIZE_GAIN", raising=False)
        assert vocalize._master_gain() == vocalize.DEFAULT_MASTER_GAIN

    def test_bad_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("NOVA_VOCALIZE_GAIN", "not-a-number")
        assert vocalize._master_gain() == vocalize.DEFAULT_MASTER_GAIN
        monkeypatch.setenv("NOVA_VOCALIZE_GAIN", "0")
        assert vocalize._master_gain() == vocalize.DEFAULT_MASTER_GAIN
        monkeypatch.setenv("NOVA_VOCALIZE_GAIN", "-1")
        assert vocalize._master_gain() == vocalize.DEFAULT_MASTER_GAIN


class _FakeSkillManager:
    """Records register_executor calls, mirroring SkillManager's signature."""

    def __init__(self):
        self.registered: dict[str, dict] = {}

    def register_executor(self, name, executor, input_schema=None):
        self.registered[name] = {"executor": executor, "input_schema": input_schema}


def _stub_ctx(audio_output=None, has_attr=True):
    """A minimal stand-in for NovaContext sufficient for register_all()."""
    ns = types.SimpleNamespace(
        state=types.SimpleNamespace(update=lambda **kw: None, get=lambda k: None),
        sonic=None,
        vision=types.SimpleNamespace(analyze_latest=lambda q: ""),
        browser=types.SimpleNamespace(execute=lambda q, u: ""),
        memory=types.SimpleNamespace(store=lambda q: "", get_startup_context=lambda: "", query=lambda q: ""),
        feedback=types.SimpleNamespace(record=lambda **kw: ""),
        slack_bot=types.SimpleNamespace(execute=lambda p: ""),
        tracker=types.SimpleNamespace(),
        face_manager=types.SimpleNamespace(),
        face_recognition=types.SimpleNamespace(is_admin_authenticated=lambda: False),
        skill_manager=None,
        gesture_engine=types.SimpleNamespace(execute=lambda g: ""),
        sleep_orchestrator=None,
        mqtt=None,
        safety=None,
        session=None,
        emotional_state=types.SimpleNamespace(get_event_names=lambda: []),
        reachy_mini=None,
    )
    if has_attr:
        ns.audio_output = audio_output
    return ns


class TestVocalizeExecutorRegistration:
    """AC2: the executor is registered in skill_executors.py."""

    def test_register_all_registers_vocalize(self):
        manager = _FakeSkillManager()
        ctx = _stub_ctx(has_attr=False)
        register_all(manager, ctx)
        assert "vocalize" in manager.registered

    def test_registered_schema_mentions_kind_and_intensity(self):
        manager = _FakeSkillManager()
        ctx = _stub_ctx(has_attr=False)
        register_all(manager, ctx)
        schema = manager.registered["vocalize"]["input_schema"]
        properties = schema["properties"]
        assert "kind" in properties
        assert "intensity" in properties
        for required_kind in REQUIRED_KINDS:
            assert required_kind in properties["kind"]["enum"]

    def test_registered_executor_pushes_through_audio_output(self):
        manager = _FakeSkillManager()
        received = []
        ctx = _stub_ctx(audio_output=lambda samples: received.append(samples))
        register_all(manager, ctx)

        executor = manager.registered["vocalize"]["executor"]
        result = executor({"kind": "chirp_up"})

        assert len(received) == 1
        assert isinstance(received[0], np.ndarray)
        assert isinstance(result, str)
        assert "unavailable" not in result.lower()


class TestVocalizeExecutorDegradesGracefully:
    """The executor must never raise, even with a bare stub ctx."""

    def test_no_audio_output_attribute_reports_unavailable(self):
        ctx = _stub_ctx(has_attr=False)
        result = _vocalize_executor({"kind": "trill"}, ctx)
        assert isinstance(result, str)
        assert "unavailable" in result.lower()

    def test_audio_output_none_reports_unavailable(self):
        ctx = _stub_ctx(audio_output=None)
        result = _vocalize_executor({"kind": "purr_tone"}, ctx)
        assert "unavailable" in result.lower()

    def test_unknown_kind_reports_error_and_never_calls_audio_output(self):
        received = []
        ctx = _stub_ctx(audio_output=lambda samples: received.append(samples))
        result = _vocalize_executor({"kind": "bark"}, ctx)
        assert received == []
        assert "unknown" in result.lower()

    def test_missing_kind_defaults_and_still_returns_a_string(self):
        ctx = _stub_ctx(audio_output=lambda samples: None)
        result = _vocalize_executor({}, ctx)
        assert isinstance(result, str)

    def test_audio_output_receives_samples_matching_synthesize(self):
        received = []
        ctx = _stub_ctx(audio_output=lambda samples: received.append(samples))
        _vocalize_executor({"kind": "chirp_up", "intensity": 0.5}, ctx)

        assert len(received) == 1
        samples = received[0]
        assert samples.dtype == np.float32
        assert np.all(samples >= -1.0) and np.all(samples <= 1.0)

    def test_result_string_is_short(self):
        ctx = _stub_ctx(audio_output=lambda samples: None)
        result = _vocalize_executor({"kind": "trill"}, ctx)
        assert len(result) < 80
