"""Tests for forge_validator — the negative tests ARE the deliverable.

The validator must reject dangerous generated code by static analysis alone,
without ever importing or executing it (covers spec targets c15, h13).
"""

import sys

import pytest

from reachy_nova.forge_validator import MAX_EXECUTOR_LINES, validate


def make_skill(tmp_path, executor_src, skill_md="---\nname: test-skill\ndescription: t\n---\nbody"):
    skill_dir = tmp_path / "forged-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(skill_md)
    (skill_dir / "executor.py").write_text(executor_src)
    return skill_dir


GOOD_SKILL = """\
import math

def execute(params, ctx):
    kind = params.get("kind", "chirp_up")
    pitch = math.sin(0.5)
    ctx.vocalize(kind)
    ctx.gesture("nuzzle")
    ctx.say(f"chirp at {pitch:.2f}!")
    return f"vocalized {kind}"
"""


class TestPositive:
    def test_well_formed_vocalize_skill_passes(self, tmp_path):
        ok, reasons = validate(make_skill(tmp_path, GOOD_SKILL))
        assert ok, f"expected pass, got reasons: {reasons}"
        assert reasons == []

    def test_allowed_imports_pass(self, tmp_path):
        src = (
            "import numpy\nimport math\nimport time\n"
            "from typing import Any\nfrom dataclasses import dataclass\n\n"
            "def execute(params, ctx):\n    ctx.inject('hi')\n    return 'ok'\n"
        )
        ok, reasons = validate(make_skill(tmp_path, src))
        assert ok, reasons


class TestForbiddenImports:
    @pytest.mark.parametrize(
        "stmt",
        [
            "import subprocess",
            "import socket",
            "import os",
            "import urllib.request",
            "import requests",
            "from pathlib import Path",
            "import shutil",
            "import sys",
        ],
    )
    def test_forbidden_import_rejected(self, tmp_path, stmt):
        src = f"{stmt}\n\ndef execute(params, ctx):\n    return 'x'\n"
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok
        assert reasons, "rejection must carry a reason"

    def test_os_system_rejected(self, tmp_path):
        src = "import os\n\ndef execute(params, ctx):\n    os.system('rm -rf /')\n    return 'x'\n"
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok

    def test_socket_open_rejected(self, tmp_path):
        src = (
            "import socket\n\ndef execute(params, ctx):\n"
            "    s = socket.socket()\n    s.connect(('evil', 80))\n    return 'x'\n"
        )
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok

    def test_path_write_text_rejected(self, tmp_path):
        src = (
            "from pathlib import Path\n\ndef execute(params, ctx):\n"
            "    Path('x').write_text('y')\n    return 'x'\n"
        )
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok


class TestForbiddenBuiltins:
    @pytest.mark.parametrize(
        "call",
        [
            "exec('x = 1')",
            "eval('1+1')",
            "compile('1', '<s>', 'eval')",
            "__import__('os')",
            "open('/etc/passwd')",
            "getattr(ctx, 'gesture')",
            "setattr(ctx, 'x', 1)",
            "globals()",
            "vars(ctx)",
        ],
    )
    def test_forbidden_builtin_rejected(self, tmp_path, call):
        src = f"def execute(params, ctx):\n    {call}\n    return 'x'\n"
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok
        assert reasons


class TestCtxSurface:
    def test_ctx_call_outside_surface_rejected(self, tmp_path):
        src = "def execute(params, ctx):\n    ctx.delete_everything()\n    return 'x'\n"
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok

    def test_ctx_dunder_rejected(self, tmp_path):
        src = "def execute(params, ctx):\n    ctx.__dict__['x'] = 1\n    return 'x'\n"
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok

    @pytest.mark.parametrize(
        "attr", ["gesture", "vocalize", "say", "inject", "state_get", "state_update", "emotion"]
    )
    def test_each_sanctioned_primitive_passes(self, tmp_path, attr):
        src = f"def execute(params, ctx):\n    ctx.{attr}('a')\n    return 'x'\n"
        ok, reasons = validate(make_skill(tmp_path, src))
        assert ok, reasons


class TestShape:
    def test_missing_execute_rejected(self, tmp_path):
        src = "def run(params, ctx):\n    return 'x'\n"
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok

    def test_syntax_error_rejected(self, tmp_path):
        ok, reasons = validate(make_skill(tmp_path, "def execute(params, ctx:\n"))
        assert not ok

    def test_oversized_executor_rejected(self, tmp_path):
        filler = "\n".join(f"x{i} = {i}" for i in range(MAX_EXECUTOR_LINES + 1))
        src = f"{filler}\n\ndef execute(params, ctx):\n    return 'x'\n"
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok

    def test_missing_skill_md_rejected(self, tmp_path):
        skill_dir = tmp_path / "forged-skill"
        skill_dir.mkdir()
        (skill_dir / "executor.py").write_text(GOOD_SKILL)
        ok, reasons = validate(skill_dir)
        assert not ok

    def test_missing_executor_rejected(self, tmp_path):
        skill_dir = tmp_path / "forged-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: x\ndescription: d\n---\n")
        ok, reasons = validate(skill_dir)
        assert not ok


class TestNeverExecutes:
    def test_validation_never_imports_the_skill(self, tmp_path):
        """A skill whose module body would leave a trace if executed leaves none.

        The executor's top level mutates a sentinel it smuggles via builtins —
        if validate() ever imported or exec'd the code, the sentinel would flip.
        The code is also fully allow-list-clean, so it validates OK purely
        statically.
        """
        src = (
            "SENTINEL_FLIPPED = True\n"
            "SIDE_EFFECT = [1] * 3\n\n"
            "def execute(params, ctx):\n    ctx.say('hi')\n    return 'x'\n"
        )
        skill_dir = make_skill(tmp_path, src)
        before = set(sys.modules)
        ok, reasons = validate(skill_dir)
        assert ok, reasons
        assert set(sys.modules) == before, "validate() must not import anything new"

    def test_top_level_danger_is_caught_statically(self, tmp_path):
        src = "import subprocess\nsubprocess.run(['ls'])\n\ndef execute(params, ctx):\n    return 'x'\n"
        before = set(sys.modules)
        ok, reasons = validate(make_skill(tmp_path, src))
        assert not ok
        assert set(sys.modules) == before
