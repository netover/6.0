#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

BASELINE_FILE = Path("config/quality/semgrep_baseline.json")
CURRENT_FILE = Path("semgrep-report.json")

def to_key(item: dict) -> str:
    path = item.get("path", "")
    check_id = item.get("check_id", "")
    line = item.get("start", {}).get("line", 0)
    return f"{path}:{line}:{check_id}"

def main() -> int:
    baseline = json.loads(BASELINE_FILE.read_text())
    current = json.loads(CURRENT_FILE.read_text())

    baseline_set = {k for k in baseline.get("findings", [])}
    current_set = {to_key(item) for item in current.get("results", [])}

    if not baseline_set:
        print(
            "Semgrep baseline is empty; skipping regression gate. "
            "Run update_semgrep_baseline.py to initialize."
        )
        return 0

    new_findings = sorted(current_set - baseline_set)
    print(
        "baseline="
        f"{len(baseline_set)} current={len(current_set)} "
        f"new={len(new_findings)}"
    )
    if new_findings:
        print("Semgrep regression detected:")
        for item in new_findings[:50]:
            print(f"- {item}")
        return 1

    print("Semgrep regression gate passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
