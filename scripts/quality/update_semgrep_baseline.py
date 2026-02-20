#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

CURRENT_FILE = Path("semgrep-report.json")
BASELINE_FILE = Path("config/quality/semgrep_baseline.json")


def to_key(item: dict) -> str:
    path = item.get("path", "")
    check_id = item.get("check_id", "")
    line = item.get("start", {}).get("line", 0)
    return f"{path}:{line}:{check_id}"


def main() -> int:
    current = json.loads(CURRENT_FILE.read_text())
    findings = sorted({to_key(item) for item in current.get("results", [])})
    baseline = {"generated_at": "manual-update", "findings": findings}
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Baseline updated with {len(findings)} findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
