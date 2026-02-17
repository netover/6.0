from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    message: str


def _is_any_annotation(node: ast.AST | None) -> bool:
    if node is None:
        return False
    # Any
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    # typing.Any
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    # Annotated[..., Any] etc.
    if isinstance(node, ast.Subscript):
        return _is_any_annotation(node.value) or _is_any_annotation(node.slice)
    if isinstance(node, ast.Tuple):
        return any(_is_any_annotation(elt) for elt in node.elts)
    return False


def _iter_target_files(project_root: str) -> Iterable[str]:
    # Canonical public dependency providers:
    # - resync/core/wiring.py
    # - any resync/api/**/dependencies*.py
    yield os.path.join(project_root, "resync", "core", "wiring.py")

    api_root = os.path.join(project_root, "resync", "api")
    for root, _, files in os.walk(api_root):
        for name in files:
            if not name.endswith(".py"):
                continue
            if name.startswith("dependencies") or name.endswith("_dependencies.py") or name == "dependencies.py":
                yield os.path.join(root, name)


def find_violations(project_root: str) -> List[Violation]:
    violations: List[Violation] = []
    for path in _iter_target_files(project_root):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=path)

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Public dependency providers convention: get_*
            if not node.name.startswith("get_"):
                continue

            # Enforce explicit return annotation and no Any
            if node.returns is None:
                violations.append(Violation(path, node.lineno, f"{node.name}: missing return annotation"))
            elif _is_any_annotation(node.returns):
                violations.append(Violation(path, node.lineno, f"{node.name}: return type must not be Any"))

            # Enforce param annotations (except self/cls)
            args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
            for arg in args:
                if arg.arg in ("self", "cls"):
                    continue
                if arg.annotation is None:
                    violations.append(Violation(path, arg.lineno, f"{node.name}: param '{arg.arg}' missing annotation"))
                elif _is_any_annotation(arg.annotation):
                    violations.append(Violation(path, arg.lineno, f"{node.name}: param '{arg.arg}' must not be Any"))

    return violations


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    violations = find_violations(root)
    if not violations:
        return 0

    for v in violations:
        print(f"{v.file}:{v.line}: {v.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
