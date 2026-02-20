#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

BASELINE_FILE = Path("config/quality/mypy_baseline.json")
ERROR_RE = re.compile(r": error:")


def _run(paths: list[str]) -> tuple[int, str]:
    cmd = ["python", "-m", "mypy", "--strict", *paths]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return len(ERROR_RE.findall(output)), output


def main() -> int:
    data = json.loads(BASELINE_FILE.read_text())
    require_reduction = os.getenv("MYPY_REQUIRE_REDUCTION", "0") == "1"

    failures: list[str] = []
    reductions = 0

    for name, batch in data["batches"].items():
        baseline = int(batch["error_count"])
        current, _ = _run(batch["paths"])
        delta = current - baseline
        print(f"[{name}] baseline={baseline} current={current} delta={delta:+d}")
        if current > baseline:
            failures.append(
                f"{name}: current errors ({current}) exceeded baseline ({baseline})"
            )
        if current < baseline:
            reductions += 1

    if require_reduction and reductions == 0:
        failures.append(
            "No mypy batch reduced error count while MYPY_REQUIRE_REDUCTION=1"
        )

    if failures:
        print("\nMypy regression gate failed:")
        for line in failures:
            print(f"- {line}")
        return 1

    print("\nMypy regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
