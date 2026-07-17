"""Static validator for forged skills — AST-only, never imports.

A forged skill (generated at runtime by the skill-forge) may only compose the
sanctioned reaction primitives exposed on the injected ``ctx`` object. This
module is the gate in front of activation: it parses ``executor.py`` with
:mod:`ast` and rejects anything outside the allow-list — it never imports,
compiles-to-exec, or otherwise runs generated code (spec targets c15/h13).

Rejection is fail-closed: a skill folder that cannot be positively verified
(missing files, syntax error, oversized, unknown constructs) is rejected with
reasons, never waved through.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Top-level module names generated code may import.
ALLOWED_IMPORTS = {"numpy", "math", "time", "typing", "dataclasses"}

#: The sanctioned reaction surface on the injected ``ctx`` object.
ALLOWED_CTX_ATTRS = {
    "gesture",
    "vocalize",
    "say",
    "inject",
    "state_get",
    "state_update",
    "emotion",
}

#: Names whose mere appearance is a rejection — dangerous builtins and the
#: dangerous stdlib roots, so an aliased or indirect use is still caught.
FORBIDDEN_NAMES = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "exit",
    "quit",
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "urllib",
    "requests",
    "http",
    "importlib",
    "ctypes",
    "pickle",
    "marshal",
}

#: Builtin callables plain enough to allow in generated code.
SAFE_BUILTIN_CALLS = {
    "abs",
    "bool",
    "dict",
    "enumerate",
    "float",
    "format",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}

MAX_EXECUTOR_LINES = 200


def validate(skill_dir: Path | str) -> tuple[bool, list[str]]:
    """Statically validate a staged skill folder.

    Args:
        skill_dir: folder expected to contain ``SKILL.md`` and ``executor.py``.

    Returns:
        ``(ok, reasons)`` — ``ok`` is True only when every check passes;
        ``reasons`` lists every violation found (empty when ok).
    """
    skill_dir = Path(skill_dir)
    reasons: list[str] = []

    skill_md = skill_dir / "SKILL.md"
    executor = skill_dir / "executor.py"
    if not skill_md.is_file() or not skill_md.read_text().strip():
        reasons.append("SKILL.md missing or empty")
    if not executor.is_file():
        reasons.append("executor.py missing")
        return False, reasons

    source = executor.read_text()
    if len(source.splitlines()) > MAX_EXECUTOR_LINES:
        reasons.append(
            f"executor.py exceeds {MAX_EXECUTOR_LINES} lines — too large to trust"
        )
        return False, reasons

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        reasons.append(f"executor.py has a syntax error: {e.msg} (line {e.lineno})")
        return False, reasons

    reasons.extend(_walk(tree))

    if not _has_execute(tree):
        reasons.append("executor.py must define execute(params, ctx)")

    return (not reasons), reasons


def _walk(tree: ast.AST) -> list[str]:
    """Collect every allow-list violation in the parsed executor."""
    reasons: list[str] = []
    local_funcs = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    import_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    reasons.append(f"import '{alias.name}' is not allowed (line {node.lineno})")
                else:
                    import_aliases.add(alias.asname or root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                reasons.append(
                    f"import from '{node.module}' is not allowed (line {node.lineno})"
                )
            else:
                for alias in node.names:
                    import_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                reasons.append(f"use of '{node.id}' is forbidden (line {node.lineno})")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                reasons.append(
                    f"dunder attribute access '.{node.attr}' is forbidden (line {node.lineno})"
                )
            base = _attribute_base(node)
            if base == "ctx" and node.attr not in ALLOWED_CTX_ATTRS:
                reasons.append(
                    f"ctx.{node.attr} is outside the sanctioned primitive surface "
                    f"(line {node.lineno})"
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
                allowed = (
                    name in SAFE_BUILTIN_CALLS
                    or name in local_funcs
                    or name in import_aliases
                )
                if not allowed and name not in FORBIDDEN_NAMES:
                    # FORBIDDEN_NAMES already flagged via the Name branch.
                    reasons.append(
                        f"call to '{name}' is outside the sanctioned surface (line {node.lineno})"
                    )

    return reasons


def _attribute_base(node: ast.Attribute) -> str | None:
    """Resolve the root Name of an attribute chain (``a.b.c`` -> ``a``)."""
    value = node.value
    while isinstance(value, ast.Attribute):
        value = value.value
    if isinstance(value, ast.Name):
        return value.id
    return None


def _has_execute(tree: ast.AST) -> bool:
    """True when the module defines a top-level ``execute`` taking two args."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute":
            args = node.args
            n = len(args.args) + len(args.posonlyargs)
            if n == 2 and not args.vararg and not args.kwarg:
                return True
    return False
