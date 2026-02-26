"""
Core Domain Mypy Analyzer.

Analyzes MyPy reports specifically for the resync/core domain.
Groups errors by subdomain for targeted remediation planning.
"""

import sys
from collections import defaultdict
from pathlib import Path

def _parse_mypy_line(line: str, prefix: str = "resync/core/") -> str | None:
    """
    Extract domain from a mypy error line.

    Expected format: filepath:line:col: severity: message
    Returns None for lines that are not errors or outside the prefix.

    Args:
        line: Single line from mypy output.
        prefix: File path prefix to filter by.

    Returns:
        Domain string (e.g., 'resync/core/cache') or None.
    """
    if "error:" not in line:
        return None

    parts = line.split(":")

    # Need at least: filepath:line:col:severity (4+ parts)
    if len(parts) < 4:
        return None

    # Normalize path separators before use (handles Windows CI paths)
    filepath = parts[0].replace("\\", "/")
    if not filepath.startswith(prefix):
        return None

    path_parts = filepath.split("/")
    if len(path_parts) >= 3:
        return f"{path_parts[0]}/{path_parts[1]}/{path_parts[2]}"

    return f"{prefix.rstrip('/')} (root files)"

def _count_errors_by_domain(
    report_file: Path, prefix: str = "resync/core/"
) -> dict[str, int]:  # P2-01: dict instead of deprecated typing.Dict
    """
    Read mypy report and count errors grouped by domain.

    Args:
        report_file: Path to mypy output file.
        prefix: File prefix to filter errors by.

    Returns:
        Mapping of domain path to error count.
    """
    domain_counts: dict[str, int] = defaultdict(int)

    try:
        content = report_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            domain = _parse_mypy_line(line, prefix)
            if domain:
                domain_counts[domain] += 1
    except OSError as e:
        print(
            f"ERROR: Failed to read report {report_file.name}: {e}", file=sys.stderr
        )

    return domain_counts

def _format_plan(domain_counts: dict[str, int]) -> str:
    """
    Generate structured Markdown remediation plan.

    Args:
        domain_counts: Mapping of domain to error count.

    Returns:
        Markdown-formatted plan string.
    """
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)

    plan_lines = [
        "# Mypy Core Remediation Task",
        "",
        "## Sub-tasks for `resync/core/`",
        "",
    ]

    if not sorted_domains:
        plan_lines.append("No errors found! Core is 100% compliant.")
    else:
        for domain, count in sorted_domains:
            plan_lines.append(f"- [ ] `mypy` for `{domain}/` ({count} errors)")

    return "\n".join(plan_lines)

def generate_core_plan() -> None:
    """Main orchestration function."""
    report_file = Path("mypy_core_report.txt")

    if not report_file.exists():
        print(f"Report '{report_file.name}' not found.")
        return

    domain_counts = _count_errors_by_domain(report_file)
    if not domain_counts:
        print("No errors processed or file was empty/unreadable.")
        return

    plan = _format_plan(domain_counts)
    print(plan)

if __name__ == "__main__":
    generate_core_plan()
