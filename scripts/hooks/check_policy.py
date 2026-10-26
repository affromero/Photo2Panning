"""Enforce project rules that lint and type checking do not express."""

import ast
import io
import sys
import tokenize
from collections.abc import Sequence

from klogr.path import path_basename, path_is_file, path_read_text


def comment_violations(filename: str, source: str) -> list[str]:
    """Reject unfinished markers and inline quality suppressions."""
    errors: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        if any(
            marker in token.string
            for marker in (
                "noqa",
                "type: ignore",
                "mypy: ignore",
                "fmt: off",
                "TODO",
                "FIXME",
                "HACK",
                "XXX",
            )
        ):
            errors.append(
                f"{filename}:{token.start[0]}: remove suppression or unfinished marker"
            )
    return errors


def import_violation(node: ast.ImportFrom, filename: str) -> str:
    """Enforce absolute imports and validated data models."""
    if node.level:
        return "Use absolute imports"
    if node.module in {"dataclasses", "pydantic.dataclasses"}:
        return "Use the project Pydantic base for data models"
    if (
        node.module == "pydantic"
        and any(item.name == "BaseModel" for item in node.names)
        and filename != "pic2panning/schemas/base.py"
    ):
        return "Import the project Pydantic base"
    return ""


def node_violation(node: ast.AST, filename: str) -> str:
    """Describe one forbidden syntax pattern."""
    match node:
        case ast.Name(id="Any"):
            return "Use a concrete type or a typed external boundary"
        case ast.ImportFrom():
            return import_violation(node, filename)
        case ast.Call(func=ast.Name(id="print")):
            return "Use klogr or the explicit CLI output stream"
        case ast.Global() | ast.Nonlocal():
            return "Keep state explicit"
        case ast.Assign(targets=targets) if path_basename(
            filename
        ) != "__init__.py" and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            return "Declare package exports in __init__.py"
    return ""


def violations(filename: str, source: str) -> list[str]:
    """Report forbidden imports, dynamic state escapes, and suppressions."""
    tree = ast.parse(source, filename=filename)
    errors = comment_violations(filename, source)
    for node in ast.walk(tree):
        message = node_violation(node, filename)
        if message:
            errors.append(f"{filename}: {message}")
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            errors.extend(
                f"{filename}:{child.lineno}: keep imports at module scope"
                for child in ast.walk(node)
                if isinstance(child, ast.Import | ast.ImportFrom)
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (
                isinstance(node.func.value, ast.Attribute)
                and ast.unparse(node.func.value) == "sys.path"
            )
        ):
            errors.append(
                f"{filename}:{node.lineno}: do not mutate the import path"
            )
    return errors


def main(filenames: Sequence[str]) -> int:
    """Validate existing Python files passed by pre-commit or CI."""
    errors = [
        error
        for filename in filenames
        if filename.endswith(".py") and path_is_file(filename)
        for error in violations(filename, path_read_text(filename))
    ]
    if errors:
        sys.stderr.write("\n".join(errors) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
