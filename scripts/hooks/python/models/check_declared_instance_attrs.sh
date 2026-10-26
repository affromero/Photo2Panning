#!/usr/bin/env bash
# Pre-commit hook: base-less classes must declare their attributes (BLOCKING).
#
# An instance attribute first assigned as `self.x = ...` inside a method is
# invisible state: you cannot know the class's shape without reading every
# method, and a typo (`self._catalgo = ...`) silently becomes a NEW forgotten
# attribute instead of an error. Declaring every attribute at class level
# (a PydanticBaseModel field, or a plain `name: T` annotation for stateful
# classes) makes the class's full purpose visible from its header , the same
# guarantee `__slots__`, dataclasses, and Pydantic `extra="forbid"` give.
#
# A ClassDef is flagged when ALL hold:
#   * zero base classes and no class keywords (metaclass) , a base-less class,
#     so every `self.x` is genuinely local and declarable (classes that
#     inherit unknowable attributes, e.g. nn.Module / Exception / Pydantic /
#     Enum, are auto-exempt because they have a base), AND
#   * it assigns some `self.<attr>` that is NOT declared at class level.
#
# "Declared at class level" = a class-body annotation (`x: T`) or assignment
# (`x = ...`, including names listed in a `__slots__` tuple), or a
# `@property` / `@x.setter` / `@cached_property` of that name.
#
# Skipped:
#   * test files (test_*.py / *_test.py) and anything under a tests/ dir.
#   * classes nested inside a function body (locals, not public shapes).
#   * a class with a `# loose-attr-exempt:` comment on the header line or the
#     two lines directly above it (genuinely dynamic / external-API shims).

set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" - "$@" <<'PY'
from __future__ import annotations

import ast
import os
import sys

MARKER = "# loose-attr-exempt:"


def _is_test_path(path: str) -> bool:
    norm = path.replace("\\", "/")
    base = os.path.basename(norm)
    if base.startswith("test_") or base.endswith("_test.py"):
        return True
    return "/tests/" in norm or norm.startswith("tests/")


def _has_exempt_comment(source_lines: list[str], lineno: int) -> bool:
    start = max(0, lineno - 3)
    for line in source_lines[start:lineno]:
        if MARKER in line:
            return True
    return False


def _nested_class_ids(tree: ast.Module) -> set[int]:
    """ids of ClassDefs that live inside a function body (locals)."""
    nested: set[int] = set()

    def _walk(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if in_function:
                    nested.add(id(child))
                _walk(child, in_function)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                _walk(child, True)
            else:
                _walk(child, in_function)

    _walk(tree, False)
    return nested


def _decorator_defines_attr(dec: ast.expr) -> bool:
    if isinstance(dec, ast.Name):
        return dec.id in ("property", "cached_property")
    if isinstance(dec, ast.Attribute):
        return dec.attr in (
            "property",
            "cached_property",
            "setter",
            "deleter",
            "getter",
        )
    return False


def _slots_names(value: ast.expr) -> set[str]:
    if not isinstance(value, ast.List | ast.Tuple | ast.Set):
        return set()
    return {
        e.value
        for e in value.elts
        if isinstance(e, ast.Constant) and isinstance(e.value, str)
    }


def _declared_names(cls: ast.ClassDef) -> set[str]:
    declared: set[str] = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(
            stmt.target, ast.Name
        ):
            declared.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    declared.add(target.id)
                    if target.id == "__slots__" and stmt.value is not None:
                        declared |= _slots_names(stmt.value)
        elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            if any(_decorator_defines_attr(d) for d in stmt.decorator_list):
                declared.add(stmt.name)
    return declared


def _iter_self_targets(target: ast.expr):
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        yield target.attr
    elif isinstance(target, ast.Tuple | ast.List):
        for element in target.elts:
            yield from _iter_self_targets(element)
    elif isinstance(target, ast.Starred):
        yield from _iter_self_targets(target.value)


def _self_assigned_names(cls: ast.ClassDef) -> set[str]:
    """`self.x` assignment targets in the class's own methods.

    Descends into nested functions (closures share ``self``) but not into
    nested classes (their ``self`` is a different instance).
    """
    names: set[str] = set()

    def _walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    names.update(_iter_self_targets(target))
            elif isinstance(child, ast.AnnAssign | ast.AugAssign):
                names.update(_iter_self_targets(child.target))
            _walk(child)

    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            _walk(stmt)
    return names


def _check_file(path: str) -> list[str]:
    if _is_test_path(path):
        return []

    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [f"{path}: could not parse Python AST: {exc}"]

    nested = _nested_class_ids(tree)

    errors: list[str] = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        if id(cls) in nested:
            continue
        if cls.bases or cls.keywords:
            continue
        undeclared = _self_assigned_names(cls) - _declared_names(cls)
        if not undeclared:
            continue
        if _has_exempt_comment(source_lines, cls.lineno):
            continue
        shown = ", ".join(sorted(undeclared))
        errors.append(
            f"{path}:{cls.lineno}: base-less class `{cls.name}` assigns "
            f"undeclared attribute(s) in a method: {shown}. Declare every "
            "instance attribute at class level (a PydanticBaseModel field, "
            "or `name: T` for stateful classes) so the class shape is known "
            "upfront, or add a `# loose-attr-exempt: <reason>` comment."
        )
    return errors


all_errors: list[str] = []
for filename in sys.argv[1:]:
    if not filename.endswith(".py"):
        continue
    try:
        with open(filename, encoding="utf-8"):
            pass
    except FileNotFoundError:
        continue
    all_errors.extend(_check_file(filename))

if all_errors:
    sys.stdout.write("\n".join(all_errors))
    sys.stdout.write("\n")
    raise SystemExit(1)
PY
