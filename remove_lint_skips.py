#!/usr/bin/env python3
"""remove_lint_skips.py

Utility script to remove or normalize ad-hoc lint-skip comments across the repo.

It searches for common one-line "skip" markers and replaces them with a single
standardized marker (or removes them), to keep the codebase consistent.

This script is intentionally dependency-free and safe to run with Python 3.14+.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Exact line matches (after .strip()) we want to normalize.
_REPLACEMENTS: dict[str, str] = {
    "# pylint: skip-file": "# ruff: noqa",
    "# pylint: skip_file": "# ruff: noqa",
    "# pylint: disable=all": "# ruff: noqa",
    "# mypy: ignore-errors": "# ruff: noqa",
    "# mypy: ignore-errors  # noqa": "# ruff: noqa",
}

def process_file(path: Path, *, dry_run: bool, verbose: bool) -> bool | None:
    """Process a single file.

    Returns:
        True if modified, False if not modified, None if unreadable.
    """
    try:
        original = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"⚠️  Could not read {path}: {exc}")
        return None

    lines = original.splitlines(keepends=True)
    modified = False
    out: list[str] = []

    for line in lines:
        key = line.strip()
        if key in _REPLACEMENTS:
            out.append(_REPLACEMENTS[key] + "\n")
            modified = True
            if verbose:
                print(f"🔧 {path}: replaced '{key}'")
        else:
            out.append(line)

    if not modified:
        return False

    if not dry_run:
        path.write_text("".join(out), encoding="utf-8")
    return True

def iter_python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if p.is_file()]

def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize/remove lint-skip markers.")
    parser.add_argument("path", nargs="?", default=".", help="Root path to scan")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    parser.add_argument("--verbose", action="store_true", help="Print replacements")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    py_files = iter_python_files(root)

    changed = 0
    unreadable = 0
    for f in py_files:
        res = process_file(f, dry_run=args.dry_run, verbose=args.verbose)
        if res is True:
            changed += 1
        elif res is None:
            unreadable += 1

    print(f"Done. files_changed={changed} unreadable={unreadable} dry_run={args.dry_run}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
