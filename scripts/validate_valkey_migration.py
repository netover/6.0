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
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = ROOT / "scripts" / "sweep_valkey_imports.py"
MIGRATION_TEST = ROOT / "tests" / "test_valkey_naming_migration.py"

def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))

def main() -> int:
    run([sys.executable, "-m", "compileall", "-q", "resync"])

    # Backward-compatible: run sweep script only when present in this checkout.
    if SWEEP_SCRIPT.exists():
        run([sys.executable, str(SWEEP_SCRIPT.relative_to(ROOT)), "--check"])
    else:
        print(
            "! skipped: scripts/sweep_valkey_imports.py not found; "
            "continuing with conservative import scan"
        )
        redis_hits: list[str] = []
        import_pattern = re.compile(r"^\s*(import|from)\s+redis\b")
        for path in ROOT.rglob("*.py"):
            if ".git" in path.parts or ".venv" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(import_pattern.match(line) for line in text.splitlines()):
                redis_hits.append(str(path.relative_to(ROOT)))
        if redis_hits:
            print("Found potential redis imports (expected valkey):")
            for hit in redis_hits:
                print(" -", hit)
            return 1

    # Run migration-specific tests only when file exists in current branch.
    if MIGRATION_TEST.exists():
        run([sys.executable, "-m", "pytest", "-q", str(MIGRATION_TEST.relative_to(ROOT))])
    else:
        print("! skipped: tests/test_valkey_naming_migration.py not found")

    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
