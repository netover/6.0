#!/usr/bin/env python3
"""Conservative validation for Valkey naming migration.

Runs:
1) compileall (syntax/import-level)
2) sweep script check (no pending import rewrites)
3) pytest on the migration-specific test only (doesn't require full optional deps)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))

def main() -> int:
    run([sys.executable, "-m", "compileall", "-q", "resync"])
    run([sys.executable, "scripts/sweep_valkey_imports.py", "--check"])
    # Run only the migration test to avoid requiring optional deps
    run([sys.executable, "-m", "pytest", "-q", "tests/test_valkey_naming_migration.py"])
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
