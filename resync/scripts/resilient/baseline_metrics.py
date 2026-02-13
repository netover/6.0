#!/usr/bin/env python3
"""
Baseline Testing Script

Establishes baseline metrics before refactoring:
- Test results
- Coverage
- Performance

Run before starting migration to have comparison point.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_tests() -> dict:
    """Run test suite and capture results"""
    print("🧪 Running test suite...")

    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short", "-q"], capture_output=True, text=True
    )

    # Parse output for summary
    lines = result.stdout.strip().split("\n")
    summary_line = [l for l in lines if "passed" in l or "failed" in l]

    return {
        "exit_code": result.returncode,
        "summary": summary_line[-1] if summary_line else "Unknown",
        "passed": result.returncode == 0,
    }


def run_coverage() -> dict:
    """Run coverage analysis"""
    print("\n📊 Running coverage analysis...")

    subprocess.run(
        ["pytest", "tests/", "--cov=resync", "--cov-report=json", "-q"],
        capture_output=True,
        text=True,
    )

    coverage_data = {}
    if Path("coverage.json").exists():
        with open("coverage.json") as f:
            data = json.load(f)
            coverage_data = {
                "percent_covered": data["totals"]["percent_covered"],
                "num_statements": data["totals"]["num_statements"],
                "missing_lines": data["totals"]["missing_lines"],
            }

    return coverage_data


def count_core_files() -> dict:
    """Count files in core/"""
    print("\n📁 Counting core files...")

    core_path = Path("resync/core")

    total_files = 0
    root_files = 0
    total_lines = 0

    for py_file in core_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        total_files += 1

        if py_file.parent == core_path:
            root_files += 1

        with open(py_file, errors="ignore") as f:
            total_lines += len(f.readlines())

    return {
        "total_files": total_files,
        "root_files": root_files,
        "total_lines": total_lines,
        "directories": len(set(f.parent for f in core_path.rglob("*.py"))),
    }


def save_baseline(data: dict):
    """Save baseline data"""
    output_file = Path("docs/BASELINE_METRICS.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    # Also create markdown report
    md_file = Path("docs/BASELINE_METRICS.md")

    report = f"""# Baseline Metrics (Before Refactoring)

**Date:** {data["timestamp"]}

---

## 📁 Core Structure

| Metric | Value |
|--------|-------|
| Total Files | {data["files"]["total_files"]} |
| Root Files | {data["files"]["root_files"]} |
| Total Lines | {data["files"]["total_lines"]:,} |
| Directories | {data["files"]["directories"]} |

---

## 🧪 Test Results

| Metric | Value |
|--------|-------|
| Status | {"✅ PASSED" if data["tests"]["passed"] else "❌ FAILED"} |
| Summary | {data["tests"]["summary"]} |

---

## 📊 Coverage

| Metric | Value |
|--------|-------|
| Coverage | {data["coverage"].get("percent_covered", "N/A"):.1f}% |
| Statements | {data["coverage"].get("num_statements", "N/A"):,} |
| Missing | {data["coverage"].get("missing_lines", "N/A"):,} |

---

## ✅ Refactoring Rules

After refactoring:
1. **Tests:** Must remain 100% passing
2. **Coverage:** Must be >= {data["coverage"].get("percent_covered", 0):.1f}%
3. **Root Files:** Must be 0 (all organized)
4. **Performance:** < 5% degradation

---

**IMPORTANT:** Do not proceed with migration if tests are failing!
"""

    with open(md_file, "w") as f:
        f.write(report)

    print(f"\n✅ Baseline saved to {output_file}")
    print(f"✅ Report saved to {md_file}")


def main():
    print("=" * 60)
    print("Baseline Metrics Collection")
    print("=" * 60)

    baseline = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "coverage": {},
        "files": {},
    }

    # 1. Count files
    baseline["files"] = count_core_files()

    # 2. Run tests
    baseline["tests"] = run_tests()

    # 3. Run coverage (if tests pass)
    if baseline["tests"]["passed"]:
        baseline["coverage"] = run_coverage()
    else:
        print("\n⚠️  Skipping coverage - tests failed!")
        baseline["coverage"] = {"error": "Tests failed"}

    # 4. Save baseline
    save_baseline(baseline)

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"📁 Files: {baseline['files']['total_files']}")
    print(f"📁 Root Files: {baseline['files']['root_files']}")
    print("📝 Lines: {baseline['files']['total_lines']:,}")
    print(f"🧪 Tests: {'✅ PASSED' if baseline['tests']['passed'] else '❌ FAILED'}")

    if baseline["tests"]["passed"]:
        print("📊 Coverage: {baseline['coverage'].get('percent_covered', 'N/A'):.1f}%")

    print("\n" + "=" * 60)

    # Exit code
    if baseline["tests"]["passed"]:
        print("✅ Ready for refactoring!")
        return 0
    print("❌ Fix failing tests before refactoring!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
