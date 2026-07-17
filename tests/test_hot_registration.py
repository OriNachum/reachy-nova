"""Tests for the runtime side of the skill-forge loop: hot registration of
validated forged skills (SkillManager.discover_runtime), the restricted
ForgedSkillContext execution surface, and activation (skills.activate_forged)
that moves a staged skill into the live runtime and announces it.

Security contract under test throughout: executor.py is imported ONLY after
a validator has said ok — negative-path tests use a stubbed rejecting
validator (never the real forge_validator) and prove non-import with a
side-effecting sentinel. One positive round trip uses the REAL
forge_validator to prove the two surfaces (validator allow-list <->
ForgedSkillContext's sanctioned primitives) actually agree.
"""

import logging

import pytest

from reachy_nova.skills import ForgedSkillContext, SkillManager, activate_forged


def _write_skill(root, name, executor_src, skill_md=None):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    if skill_md is None:
        skill_md = f"---\nname: {name}\ndescription: a forged skill\n---\nbody\n"
    (skill_dir / "SKILL.md").write_text(skill_md)
    (skill_dir / "executor.py").write_text(executor_src)
    return skill_dir


def _ok_validator(skill_dir):
    return True, []


def _rejecting_validator(skill_dir):
    return False, ["nope"]


class _StubState:
    def __init__(self, voice_state="idle"):
        self._data = {"voice_state": voice_state}

    def get(self, key):
        return self._data.get(key)

    def update(self, **kw):
        self._data.update(kw)

    def snapshot(self):
        return dict(self._data)


class _StubSonic:
    def __init__(self):
        self.injected = []

    def inject_text(self, text, force=False):
        self.injected.append((text, force))


class _RecordingPublisher:
    def __init__(self):
        self.calls = []

    def __call__(self, event_type, payload):
        self.calls.append((event_type, payload))

    def has(self, event_type):
        return any(c[0] == event_type for c in self.calls)

    def payload_for(self, event_type):
        for c in self.calls:
            if c[0] == event_type:
                return c[1]
        raise AssertionError(f"{event_type} not published; got {self.calls}")


class _StubEmotionalState:
    def get_event_names(self):
        return ["praised"]

    def apply_event(self, event):
        pass


# ---------------------------------------------------------------------------
# discover_runtime — positive path
# ---------------------------------------------------------------------------


def test_discover_runtime_registers_valid_forged_skill_and_it_is_callable(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_skill(
        runtime_dir,
        "wave-hello",
        "def execute(params, ctx):\n    return '[waved]'\n",
    )

    manager = SkillManager()
    ctx = object()  # opaque — discover_runtime just closes over it; this skill ignores it
    registered = manager.discover_runtime(runtime_dir, ctx, validator=_ok_validator)

    assert registered == ["wave-hello"]
    specs = manager.get_tool_specs()
    names = [s["toolSpec"]["name"] for s in specs]
    assert "wave-hello" in names

    result = manager.execute("wave-hello", {})
    assert result == "[waved]"


def test_discover_runtime_wraps_execute_signature_and_receives_the_given_ctx(tmp_path):
    """The wrapped executor calls fn(params, ctx) — ctx is whatever discover_runtime was given."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_skill(
        runtime_dir,
        "echo-ctx",
        "def execute(params, ctx):\n    return ctx.state_get('marker')\n",
    )

    class _Ctx:
        def state_get(self, key):
            return f"got:{key}"

    manager = SkillManager()
    manager.discover_runtime(runtime_dir, _Ctx(), validator=_ok_validator)

    assert manager.execute("echo-ctx", {}) == "got:marker"


def test_discover_runtime_returns_empty_list_for_missing_dir(tmp_path, caplog):
    manager = SkillManager()
    with caplog.at_level(logging.WARNING):
        registered = manager.discover_runtime(tmp_path / "does-not-exist", object())
    assert registered == []


def test_discover_runtime_skips_folders_without_skill_md_or_executor(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    half_baked = runtime_dir / "half-baked"
    half_baked.mkdir()
    (half_baked / "SKILL.md").write_text("---\nname: half\n---\nbody")
    # no executor.py

    manager = SkillManager()
    registered = manager.discover_runtime(runtime_dir, object(), validator=_ok_validator)
    assert registered == []


def test_discover_runtime_registers_multiple_valid_skills(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_skill(runtime_dir, "skill-a", "def execute(params, ctx):\n    return 'a'\n")
    _write_skill(runtime_dir, "skill-b", "def execute(params, ctx):\n    return 'b'\n")

    manager = SkillManager()
    registered = manager.discover_runtime(runtime_dir, object(), validator=_ok_validator)

    assert set(registered) == {"skill-a", "skill-b"}
    assert manager.execute("skill-a", {}) == "a"
    assert manager.execute("skill-b", {}) == "b"


# ---------------------------------------------------------------------------
# discover_runtime — negative path: validator rejection means NO import
# ---------------------------------------------------------------------------


def test_discover_runtime_rejected_folder_is_not_registered_and_never_imported(tmp_path):
    """A validator-rejecting folder must never even be imported.

    Proven with a sentinel: the executor's top level creates a marker file
    as a side effect of module execution. If the marker never appears, the
    module body never ran — proof discover_runtime honors the
    validate-before-import security contract even under a stubbed
    (non-real) rejecting validator.
    """
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    marker = tmp_path / "IMPORTED.marker"
    executor_src = (
        f"open({str(marker)!r}, 'w').close()\n\n"
        "def execute(params, ctx):\n"
        "    return '[should never run]'\n"
    )
    _write_skill(runtime_dir, "sneaky", executor_src)

    manager = SkillManager()
    registered = manager.discover_runtime(runtime_dir, object(), validator=_rejecting_validator)

    assert registered == []
    assert not marker.exists(), "executor.py was imported despite a rejecting validator"
    assert manager.execute("sneaky", {}) == "[Unknown skill: sneaky]"


def test_discover_runtime_no_validator_available_fails_closed(tmp_path, monkeypatch):
    """validator=None with forge_validator unavailable registers nothing."""
    import sys

    monkeypatch.setitem(sys.modules, "reachy_nova.forge_validator", None)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_skill(runtime_dir, "wave-hello", "def execute(params, ctx):\n    return '[waved]'\n")

    manager = SkillManager()
    registered = manager.discover_runtime(runtime_dir, object(), validator=None)
    assert registered == []


def test_discover_runtime_validator_raising_skips_folder(tmp_path, caplog):
    def exploding_validator(skill_dir):
        raise RuntimeError("boom")

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_skill(runtime_dir, "wave-hello", "def execute(params, ctx):\n    return '[waved]'\n")

    manager = SkillManager()
    with caplog.at_level(logging.WARNING):
        registered = manager.discover_runtime(runtime_dir, object(), validator=exploding_validator)
    assert registered == []


def test_discover_runtime_missing_execute_function_skips_folder(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_skill(runtime_dir, "no-execute", "def run(params, ctx):\n    return 'x'\n")

    manager = SkillManager()
    registered = manager.discover_runtime(runtime_dir, object(), validator=_ok_validator)
    assert registered == []


def test_discover_runtime_import_error_skips_folder_without_crashing(tmp_path, caplog):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_skill(runtime_dir, "syntax-bad", "def execute(params, ctx:\n    return 'x'\n")

    manager = SkillManager()
    with caplog.at_level(logging.WARNING):
        registered = manager.discover_runtime(runtime_dir, object(), validator=_ok_validator)
    assert registered == []


# ---------------------------------------------------------------------------
# a forged execute() that raises must never crash the loop
# ---------------------------------------------------------------------------


def test_discover_runtime_forged_execute_raising_returns_error_string(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_skill(
        runtime_dir,
        "boom",
        "def execute(params, ctx):\n    raise RuntimeError('kaboom')\n",
    )

    manager = SkillManager()
    manager.discover_runtime(runtime_dir, object(), validator=_ok_validator)

    result = manager.execute("boom", {})
    assert result == "[Skill error: kaboom]"


# ---------------------------------------------------------------------------
# real forge_validator round trip (proves the two surfaces agree)
# ---------------------------------------------------------------------------


def test_discover_runtime_with_real_forge_validator_positive_round_trip(tmp_path):
    """A well-formed executor using only ctx primitives passes the REAL
    forge_validator, then actually runs through a real ForgedSkillContext."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_skill(
        runtime_dir,
        "wave-hello",
        "def execute(params, ctx):\n"
        "    ctx.gesture('wave')\n"
        "    ctx.say('hi there')\n"
        "    return '[waved and said hi]'\n",
    )

    class _GestureEngine:
        def __init__(self):
            self.calls = []

        def execute(self, name):
            self.calls.append(name)
            return f"[gesture {name}]"

    gesture_engine = _GestureEngine()
    sonic = _StubSonic()
    forged_ctx = ForgedSkillContext(gesture_engine=gesture_engine, sonic=sonic)

    manager = SkillManager()
    # validator=None -> lazy-imports the REAL reachy_nova.forge_validator.validate
    registered = manager.discover_runtime(runtime_dir, forged_ctx, validator=None)

    assert registered == ["wave-hello"]
    result = manager.execute("wave-hello", {})
    assert result == "[waved and said hi]"
    assert gesture_engine.calls == ["wave"]
    assert sonic.injected == [("hi there", True)]


def test_discover_runtime_with_real_forge_validator_rejects_forbidden_code(tmp_path):
    """A companion negative case using the real validator too: forbidden
    code never gets imported (marker proves it)."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    marker = tmp_path / "IMPORTED.marker"
    executor_src = (
        f"import os\n"
        f"open({str(marker)!r}, 'w').close()\n\n"
        "def execute(params, ctx):\n"
        "    os.system('echo hi')\n"
        "    return 'x'\n"
    )
    _write_skill(runtime_dir, "evil", executor_src)

    manager = SkillManager()
    registered = manager.discover_runtime(runtime_dir, object(), validator=None)

    assert registered == []
    assert not marker.exists()


# ---------------------------------------------------------------------------
# ForgedSkillContext — the restricted ctx surface
# ---------------------------------------------------------------------------


class TestForgedSkillContext:
    def test_gesture_delegates_to_gesture_engine(self):
        calls = []

        class _Engine:
            def execute(self, name):
                calls.append(name)
                return f"[{name}]"

        ctx = ForgedSkillContext(gesture_engine=_Engine())
        assert ctx.gesture("nuzzle") == "[nuzzle]"
        assert calls == ["nuzzle"]

    def test_gesture_missing_subsystem_no_ops_with_warning(self, caplog):
        ctx = ForgedSkillContext()
        with caplog.at_level(logging.WARNING):
            result = ctx.gesture("nuzzle")
        assert result == "[gesture unavailable]"
        assert any("gesture_engine" in r.message for r in caplog.records)

    def test_gesture_engine_raising_is_caught(self):
        class _Engine:
            def execute(self, name):
                raise RuntimeError("motor stuck")

        ctx = ForgedSkillContext(gesture_engine=_Engine())
        result = ctx.gesture("nuzzle")
        assert "error" in result.lower()

    def test_vocalize_delegates_to_skill_manager_execute(self):
        seen = []

        class _Manager:
            def execute(self, name, params):
                seen.append((name, params))
                return "[vocalized]"

        ctx = ForgedSkillContext(skill_manager=_Manager())
        result = ctx.vocalize("chirp_up", intensity=0.5)
        assert result == "[vocalized]"
        assert seen == [("vocalize", {"kind": "chirp_up", "intensity": 0.5})]

    def test_say_calls_inject_text_with_force(self):
        sonic = _StubSonic()
        ctx = ForgedSkillContext(sonic=sonic)
        ctx.say("hello")
        assert sonic.injected == [("hello", True)]

    def test_inject_calls_inject_text_without_force(self):
        sonic = _StubSonic()
        ctx = ForgedSkillContext(sonic=sonic)
        ctx.inject("hello")
        assert sonic.injected == [("hello", False)]

    def test_state_get_and_update_delegate(self):
        state = _StubState()
        ctx = ForgedSkillContext(state=state)
        ctx.state_update(mood="happy")
        assert ctx.state_get("mood") == "happy"

    def test_state_get_missing_subsystem_returns_none(self):
        ctx = ForgedSkillContext()
        assert ctx.state_get("mood") is None

    def test_emotion_delegates_to_apply_event(self):
        class _Emo:
            def __init__(self):
                self.events = []

            def apply_event(self, event):
                self.events.append(event)

        emo = _Emo()
        ctx = ForgedSkillContext(emotional_state=emo)
        result = ctx.emotion("praised")
        assert emo.events == ["praised"]
        assert "praised" in result

    def test_subsystems_mapping_form(self):
        calls = []

        class _Engine:
            def execute(self, name):
                calls.append(name)
                return "[ok]"

        ctx = ForgedSkillContext({"gesture_engine": _Engine()})
        assert ctx.gesture("yes") == "[ok]"
        assert calls == ["yes"]

    def test_explicit_kwarg_wins_over_mapping(self):
        class _EngineA:
            def execute(self, name):
                return "A"

        class _EngineB:
            def execute(self, name):
                return "B"

        ctx = ForgedSkillContext({"gesture_engine": _EngineA()}, gesture_engine=_EngineB())
        assert ctx.gesture("x") == "B"


# ---------------------------------------------------------------------------
# activate_forged — moves, registers, announces, restart timing
# ---------------------------------------------------------------------------


class TestActivateForged:
    def test_activation_moves_folder_registers_and_is_callable(self, tmp_path):
        staging = tmp_path / "staging"
        active = tmp_path / "active"
        staging.mkdir()
        active.mkdir()
        _write_skill(staging, "wave-hello", "def execute(params, ctx):\n    return '[waved]'\n")

        manager = SkillManager()
        state = _StubState(voice_state="idle")
        sonic = _StubSonic()
        publisher = _RecordingPublisher()
        forged_ctx = ForgedSkillContext(sonic=sonic)
        restart_calls = []

        result = activate_forged(
            staging,
            active,
            "wave-hello",
            manager,
            forged_ctx,
            sonic,
            state,
            publisher,
            validator=_ok_validator,
            restart=lambda: restart_calls.append(True),
        )

        assert not (staging / "wave-hello").exists()
        assert (active / "wave-hello" / "executor.py").is_file()
        assert "wave-hello" in [s["toolSpec"]["name"] for s in manager.get_tool_specs()]
        assert manager.execute("wave-hello", {}) == "[waved]"
        assert restart_calls == [True]
        assert sonic.injected  # announced via inject
        assert publisher.has("forge/activated")
        assert publisher.payload_for("forge/activated")["name"] == "wave-hello"
        assert "activated" in result.lower()

    def test_activation_restarts_immediately_when_idle(self, tmp_path):
        staging = tmp_path / "staging"
        active = tmp_path / "active"
        staging.mkdir()
        active.mkdir()
        _write_skill(staging, "wave-hello", "def execute(params, ctx):\n    return '[waved]'\n")

        manager = SkillManager()
        state = _StubState(voice_state="idle")
        sonic = _StubSonic()
        publisher = _RecordingPublisher()
        restart_calls = []

        activate_forged(
            staging,
            active,
            "wave-hello",
            manager,
            ForgedSkillContext(sonic=sonic),
            sonic,
            state,
            publisher,
            validator=_ok_validator,
            restart=lambda: restart_calls.append(True),
        )
        assert restart_calls == [True]

    def test_activation_treats_none_voice_state_as_idle(self, tmp_path):
        staging = tmp_path / "staging"
        active = tmp_path / "active"
        staging.mkdir()
        active.mkdir()
        _write_skill(staging, "wave-hello", "def execute(params, ctx):\n    return '[waved]'\n")

        manager = SkillManager()
        state = _StubState(voice_state=None)
        sonic = _StubSonic()
        publisher = _RecordingPublisher()
        restart_calls = []

        activate_forged(
            staging,
            active,
            "wave-hello",
            manager,
            ForgedSkillContext(sonic=sonic),
            sonic,
            state,
            publisher,
            validator=_ok_validator,
            restart=lambda: restart_calls.append(True),
        )
        assert restart_calls == [True]

    def test_activation_defers_restart_when_speaking(self, tmp_path):
        staging = tmp_path / "staging"
        active = tmp_path / "active"
        staging.mkdir()
        active.mkdir()
        _write_skill(staging, "wave-hello", "def execute(params, ctx):\n    return '[waved]'\n")

        manager = SkillManager()
        state = _StubState(voice_state="speaking")
        sonic = _StubSonic()
        publisher = _RecordingPublisher()
        forged_ctx = ForgedSkillContext(sonic=sonic)
        restart_calls = []

        result = activate_forged(
            staging,
            active,
            "wave-hello",
            manager,
            forged_ctx,
            sonic,
            state,
            publisher,
            validator=_ok_validator,
            restart=lambda: restart_calls.append(True),
        )

        assert restart_calls == []  # NOT restarted while a conversation is live
        assert "deferred" in result.lower()
        # still moved, registered, and announced — only the restart is deferred
        assert manager.execute("wave-hello", {}) == "[waved]"
        assert sonic.injected
        assert publisher.has("forge/activated")

    def test_activation_missing_staged_folder_returns_error_and_announces_nothing(self, tmp_path):
        staging = tmp_path / "staging"
        active = tmp_path / "active"
        staging.mkdir()
        active.mkdir()

        manager = SkillManager()
        state = _StubState()
        sonic = _StubSonic()
        publisher = _RecordingPublisher()

        result = activate_forged(
            staging,
            active,
            "ghost",
            manager,
            ForgedSkillContext(),
            sonic,
            state,
            publisher,
        )
        assert "failed" in result.lower()
        assert not publisher.calls  # never announced a phantom activation
        assert not sonic.injected

    def test_activation_failing_post_move_validation_reports_failure(self, tmp_path):
        staging = tmp_path / "staging"
        active = tmp_path / "active"
        staging.mkdir()
        active.mkdir()
        _write_skill(staging, "bad-skill", "def execute(params, ctx):\n    return 'x'\n")

        manager = SkillManager()
        state = _StubState()
        sonic = _StubSonic()
        publisher = _RecordingPublisher()

        result = activate_forged(
            staging,
            active,
            "bad-skill",
            manager,
            ForgedSkillContext(),
            sonic,
            state,
            publisher,
            validator=_rejecting_validator,
        )
        assert "failed" in result.lower()
        assert manager.execute("bad-skill", {}) == "[Unknown skill: bad-skill]"

    def test_activation_never_crashes_when_sonic_lacks_inject_text(self, tmp_path):
        staging = tmp_path / "staging"
        active = tmp_path / "active"
        staging.mkdir()
        active.mkdir()
        _write_skill(staging, "wave-hello", "def execute(params, ctx):\n    return '[waved]'\n")

        manager = SkillManager()
        state = _StubState(voice_state="idle")
        publisher = _RecordingPublisher()

        result = activate_forged(
            staging,
            active,
            "wave-hello",
            manager,
            ForgedSkillContext(),
            object(),  # sonic without inject_text
            state,
            publisher,
            validator=_ok_validator,
        )
        assert "activated" in result.lower()
        assert publisher.has("forge/activated")

    def test_activation_never_crashes_when_publish_raises(self, tmp_path):
        staging = tmp_path / "staging"
        active = tmp_path / "active"
        staging.mkdir()
        active.mkdir()
        _write_skill(staging, "wave-hello", "def execute(params, ctx):\n    return '[waved]'\n")

        manager = SkillManager()
        state = _StubState(voice_state="idle")
        sonic = _StubSonic()

        def exploding_publish(event_type, payload):
            raise RuntimeError("publish is down")

        result = activate_forged(
            staging,
            active,
            "wave-hello",
            manager,
            ForgedSkillContext(sonic=sonic),
            sonic,
            state,
            exploding_publish,
            validator=_ok_validator,
        )
        assert "activated" in result.lower()
        assert manager.execute("wave-hello", {}) == "[waved]"


# ---------------------------------------------------------------------------
# the forge tool itself (skill_executors._forge_executor / register_all)
# ---------------------------------------------------------------------------


class TestForgeExecutor:
    def _ctx(self, skill_forge=None, state=None):
        class _Ctx:
            pass

        c = _Ctx()
        c.skill_forge = skill_forge
        c.state = state
        c.emotional_state = _StubEmotionalState()
        return c

    def test_forge_tool_registered_with_input_schema(self):
        from reachy_nova import skill_executors

        manager = SkillManager()
        skill_executors.register_all(manager, self._ctx())

        specs = manager.get_tool_specs()
        forge_spec = next(s for s in specs if s["toolSpec"]["name"] == "forge")
        import json as _json

        schema = _json.loads(forge_spec["toolSpec"]["inputSchema"]["json"])
        assert "goal" in schema["properties"]
        assert "improve" in schema["properties"]
        assert schema["required"] == ["goal"]

    def test_forge_executor_dispatches_via_ctx_skill_forge(self):
        from reachy_nova import skill_executors

        class _StubForge:
            def __init__(self):
                self.calls = []

            def dispatch(self, goal, context, improve):
                self.calls.append((goal, context, improve))

        forge = _StubForge()
        manager = SkillManager()
        skill_executors.register_all(manager, self._ctx(skill_forge=forge, state=_StubState()))

        result = manager.execute("forge", {"goal": "wave hello"})
        assert "wave hello" in result
        assert len(forge.calls) == 1
        goal, context, improve = forge.calls[0]
        assert goal == "wave hello"
        assert improve is None
        assert context == {"voice_state": "idle"}

    def test_forge_executor_passes_improve_when_given(self):
        from reachy_nova import skill_executors

        class _StubForge:
            def __init__(self):
                self.calls = []

            def dispatch(self, goal, context, improve):
                self.calls.append((goal, context, improve))

        forge = _StubForge()
        manager = SkillManager()
        skill_executors.register_all(manager, self._ctx(skill_forge=forge))

        manager.execute("forge", {"goal": "make waving faster", "improve": "old src"})
        assert forge.calls[0][2] == "old src"

    def test_forge_executor_missing_skill_forge_returns_unavailable(self):
        from reachy_nova import skill_executors

        manager = SkillManager()
        skill_executors.register_all(manager, self._ctx(skill_forge=None))

        result = manager.execute("forge", {"goal": "wave hello"})
        assert result == "[Forge unavailable]"

    def test_forge_executor_requires_goal(self):
        from reachy_nova import skill_executors

        class _StubForge:
            def dispatch(self, *a, **kw):
                raise AssertionError("dispatch must not be called without a goal")

        manager = SkillManager()
        skill_executors.register_all(manager, self._ctx(skill_forge=_StubForge()))

        result = manager.execute("forge", {})
        assert "goal" in result.lower()

    def test_forge_executor_dispatch_raising_returns_error_string_not_crash(self):
        from reachy_nova import skill_executors

        class _StubForge:
            def dispatch(self, goal, context, improve):
                raise RuntimeError("forge endpoint down")

        manager = SkillManager()
        skill_executors.register_all(manager, self._ctx(skill_forge=_StubForge()))

        result = manager.execute("forge", {"goal": "wave hello"})
        assert "error" in result.lower() or "failed" in result.lower()
