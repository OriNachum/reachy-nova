"""Machine-check the harness boundary (task t4).

``reachy_nova/harness/`` is documented (see its ``__init__.py``) as attaching
to the reachy-mini-cli symbolic runtime ONLY through its filesystem/MQTT/HTTP
seams — never by importing the robot SDK package (``reachy_mini``) and never
by reaching into the runtime's own head-tracking API (``set_target``, the
method the tracking/motion code uses to steer the head). That boundary is a
comment today; this module turns it into something a machine enforces, so a
future edit cannot silently reintroduce a direct SDK/motion dependency into
what is supposed to be a thin peripheral.

Two forbidden shapes, found with :mod:`ast` rather than a text grep so that
aliasing, submodule imports and attribute-vs-call forms are all caught the
same way a text search would miss:

(a) any import whose root module is ``reachy_mini`` — ``import reachy_mini``,
    ``import reachy_mini.foo``, ``import reachy_mini as rm``,
    ``from reachy_mini import x``, ``from reachy_mini.foo import bar``, with
    or without ``as`` aliasing;
(b) any reference to the name ``set_target`` — a bare name (``set_target(...)``
    after a hypothetical ``from x import set_target``) or an attribute access
    (``robot.set_target(...)``).

:func:`boundary_violations` is the pure checker: point it at a directory and
it returns a list of human-readable offence strings (empty means clean). The
real gate below runs it over ``reachy_nova/harness/``, which must pass clean.
The injected-violation tests below point it at synthetic files under
``tmp_path`` to prove each forbidden shape is actually caught, independent of
whatever the harness package currently contains.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HARNESS_ROOT = _REPO_ROOT / "reachy_nova" / "harness"

#: The robot SDK package the harness must never import directly.
_FORBIDDEN_IMPORT_ROOT = "reachy_mini"

#: The runtime's head-steering API; the harness must never reference it,
#: imported or not — even a locally-defined ``def set_target(...)`` or a
#: dotted access like ``robot.set_target(...)`` is a boundary violation.
_FORBIDDEN_ATTRIBUTE = "set_target"


def boundary_violations(root: Path) -> list[str]:
    """AST-walk every ``.py`` file under *root* for the two forbidden shapes.

    Returns a list of ``"<path>:<lineno>: <reason>"`` strings; empty means the
    tree is clean. Nested (function-local, class-body, ``TYPE_CHECKING``-guarded)
    imports and references are covered by using :func:`ast.walk` rather than
    only looking at module-level statements.
    """
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module == _FORBIDDEN_IMPORT_ROOT:
                        violations.append(
                            f"{path}:{node.lineno}: forbidden import of "
                            f"'{alias.name}' (root package '{_FORBIDDEN_IMPORT_ROOT}' "
                            "must never be imported by the harness)"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    if root_module == _FORBIDDEN_IMPORT_ROOT:
                        violations.append(
                            f"{path}:{node.lineno}: forbidden import from "
                            f"'{node.module}' (root package "
                            f"'{_FORBIDDEN_IMPORT_ROOT}' must never be imported "
                            "by the harness)"
                        )
            elif isinstance(node, ast.Name):
                if node.id == _FORBIDDEN_ATTRIBUTE:
                    violations.append(
                        f"{path}:{node.lineno}: forbidden reference to "
                        f"'{_FORBIDDEN_ATTRIBUTE}' (the runtime's head-steering "
                        "API must not be reached from the harness)"
                    )
            elif isinstance(node, ast.Attribute):
                if node.attr == _FORBIDDEN_ATTRIBUTE:
                    violations.append(
                        f"{path}:{node.lineno}: forbidden reference to "
                        f"'.{_FORBIDDEN_ATTRIBUTE}' (the runtime's head-steering "
                        "API must not be reached from the harness)"
                    )
    return violations


# --------------------------------------------------------------------------- #
# 0. Vacuity guard — a clean scan over an empty directory proves nothing.     #
# --------------------------------------------------------------------------- #


def test_the_harness_directory_actually_has_modules_to_scan() -> None:
    modules = sorted(_HARNESS_ROOT.rglob("*.py"))
    assert modules, f"no .py files found under {_HARNESS_ROOT} — the scan below is vacuous"


# --------------------------------------------------------------------------- #
# 1. The real gate: reachy_nova/harness/ must pass clean today.               #
# --------------------------------------------------------------------------- #


def test_harness_package_has_no_boundary_violations() -> None:
    violations = boundary_violations(_HARNESS_ROOT)
    assert not violations, (
        "reachy_nova/harness/ crossed its documented boundary:\n  "
        + "\n  ".join(violations)
        + "\nThe harness attaches to reachy-mini-cli only through MQTT, the "
        "audio-tee socket, the intents spool and the daemon HTTP media route "
        "(see reachy_nova/harness/__init__.py) — never by importing "
        "reachy_mini or calling set_target directly."
    )


# --------------------------------------------------------------------------- #
# 2. Injected violations: prove each forbidden shape is actually caught.      #
# --------------------------------------------------------------------------- #

_IMPORT_VIOLATIONS = {
    "plain import": "import reachy_mini\n",
    "plain import, submodule": "import reachy_mini.robot\n",
    "plain import, aliased": "import reachy_mini as rm\n",
    "from-import, package": "from reachy_mini import Robot\n",
    "from-import, submodule": "from reachy_mini.foo import bar\n",
    "from-import, submodule aliased": "from reachy_mini.foo import bar as baz\n",
    "from-import, aliased name": "from reachy_mini import Robot as R\n",
    "function-local import": "def use():\n    import reachy_mini\n    return reachy_mini\n",
    "TYPE_CHECKING-guarded import": (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from reachy_mini import Robot\n"
    ),
}

_ATTRIBUTE_VIOLATIONS = {
    "attribute call": "def use(robot):\n    robot.set_target(1, 2, 3)\n",
    "nested attribute call": "def use(app):\n    app.robot.head.set_target(1)\n",
    "bare name call": "def use(set_target):\n    set_target(1, 2, 3)\n",
    "attribute reference, no call": "def use(robot):\n    fn = robot.set_target\n    return fn\n",
}

_CLEAN_SOURCE = (
    '"""A harness module with no boundary violations."""\n'
    "\n"
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def greet(name: str) -> str:\n"
    "    return json.dumps({'hello': name})\n"
    "\n"
    "\n"
    "def resolve(path: Path) -> Path:\n"
    "    return path.resolve()\n"
)


@pytest.mark.parametrize("source", _IMPORT_VIOLATIONS.values(), ids=_IMPORT_VIOLATIONS.keys())
def test_forbidden_reachy_mini_import_is_caught(tmp_path: Path, source: str) -> None:
    (tmp_path / "offender.py").write_text(source, encoding="utf-8")
    violations = boundary_violations(tmp_path)
    assert violations, f"expected a violation for source:\n{source}"
    assert any("reachy_mini" in v for v in violations)


@pytest.mark.parametrize(
    "source", _ATTRIBUTE_VIOLATIONS.values(), ids=_ATTRIBUTE_VIOLATIONS.keys()
)
def test_forbidden_set_target_reference_is_caught(tmp_path: Path, source: str) -> None:
    (tmp_path / "offender.py").write_text(source, encoding="utf-8")
    violations = boundary_violations(tmp_path)
    assert violations, f"expected a violation for source:\n{source}"
    assert any("set_target" in v for v in violations)


def test_a_clean_file_produces_no_violations(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text(_CLEAN_SOURCE, encoding="utf-8")
    assert boundary_violations(tmp_path) == []


def test_violations_report_file_and_line(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("import reachy_mini\n", encoding="utf-8")
    violations = boundary_violations(tmp_path)
    assert len(violations) == 1
    assert "offender.py:1" in violations[0]


def test_scan_covers_nested_subdirectories(tmp_path: Path) -> None:
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "offender.py").write_text("import reachy_mini\n", encoding="utf-8")
    (tmp_path / "clean.py").write_text(_CLEAN_SOURCE, encoding="utf-8")
    violations = boundary_violations(tmp_path)
    assert len(violations) == 1
    assert "deeper" in violations[0] and "offender.py" in violations[0]


def test_a_file_can_carry_both_violations_at_once(tmp_path: Path) -> None:
    source = "import reachy_mini\n\n\ndef use(robot):\n    robot.set_target(1)\n"
    (tmp_path / "offender.py").write_text(source, encoding="utf-8")
    violations = boundary_violations(tmp_path)
    assert len(violations) == 2
    assert any("reachy_mini" in v for v in violations)
    assert any("set_target" in v for v in violations)
