#!/usr/bin/env python3
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

RAMP_FILE = Path("config/quality/coverage_ramp.json")
COVERAGE_XML = Path("coverage.xml")


def get_target(ramp: dict) -> float:
    today = date.today()
    target = float(ramp.get("default_min", 0.0))
    for milestone in ramp.get("milestones", []):
        start = date.fromisoformat(milestone["from"])
        if today >= start:
            target = max(target, float(milestone["min"]))
    return max(target, float(ramp.get("non_regression_floor", 0.0)))


def main() -> int:
    ramp = json.loads(RAMP_FILE.read_text())
    target = get_target(ramp)

    root = ET.fromstring(COVERAGE_XML.read_text())
    line_rate = float(root.attrib["line-rate"]) * 100.0
    print(f"coverage={line_rate:.2f}% target={target:.2f}%")

    if line_rate < target:
        print("Coverage ramp gate failed.")
        return 1

    print("Coverage ramp gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
