from __future__ import annotations

import ast
from pathlib import Path


def _async_functions_with_sleep(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id == "time" and func.attr == "sleep":
                        return True
    return False


def main() -> int:
    bad: list[Path] = []
    for p in Path("resync").rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        try:
            has_blocking_sleep = _async_functions_with_sleep(txt)
        except SyntaxError:
            continue
        if has_blocking_sleep:
            bad.append(p)
    if bad:
        for p in bad:
            print(f"Async blocking pattern time.sleep() in async function at {p}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
