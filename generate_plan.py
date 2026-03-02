"""
Global Mypy Plan Generator.

Generates the overall remediation plan for full mypy compliance
across the entire Resync project, writing output atomically.

Usage::

    python generate_plan.py

Reads ``mypy_global_report.txt`` (run ``mypy . 2>&1 | tee mypy_global_report.txt``
first) and writes ``MYPY_REMEDIATION_PLAN.md``.
"""

# All standard-library imports at module level (never inside functions)
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

def _parse_global_mypy_line(line: str) -> tuple[str, str] | None:
    """
    Extract the global domain and file path from a mypy error line.

    Expected format: ``filepath:line:col: error: message``

    Args:
        line: Single line from mypy output.

    Returns:
        ``(domain, filepath)`` tuple, or ``None`` if line is not a
        relevant error inside the ``resync/`` tree.
    """
    if "error:" not in line:
        return None

    parts = line.split(":", maxsplit=3)  # P2-07: safe for messages with ":"

    # Need at least: filepath:line:col:severity (4+ parts)
    if len(parts) < 4:
        return None

    # Normalize path separators (handles Windows CI output)
    filepath = parts[0].replace("\\", "/")
    if not filepath.startswith("resync/"):
        return None

    path_parts = filepath.split("/")
    if len(path_parts) >= 2:
        return (f"{path_parts[0]}/{path_parts[1]}", filepath)

    return None

def _aggregate_errors(
    report_file: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Aggregate mypy errors by domain and file, handling I/O failures.

    Args:
        report_file: Path to mypy report file.

    Returns:
        ``(domain_counts, file_counts)`` — both default to zero for
        unseen keys.
    """
    domain_counts: dict[str, int] = defaultdict(int)
    file_counts: dict[str, int] = defaultdict(int)

    try:
        content = report_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            result = _parse_global_mypy_line(line)
            if result:
                domain, filepath = result
                domain_counts[domain] += 1
                file_counts[filepath] += 1
    except OSError as e:
        print(f"ERROR: Failed to read {report_file.name}: {e}", file=sys.stderr)

    return domain_counts, file_counts

def _write_plan(domain_counts: dict[str, int], output_file: Path) -> bool:
    """
    Write the Markdown remediation plan atomically.

    Uses ``tempfile`` + ``os.fsync`` + ``Path.replace`` to guarantee that
    the destination is either the old version or the fully-written new
    version — never a partial write.

    Args:
        domain_counts: Mapping of domain path → error count.
        output_file:   Destination ``.md`` file.

    Returns:
        ``True`` on success, ``False`` if an I/O error occurred.
    """
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)

    plan_lines = [
        "# MYPY REMEDIATION PLAN",
        "",
        "Strict tracking file for achieving 100% mypy compliance.",
        "",
        "## Domain Groups",
        "",
    ]

    if not sorted_domains:
        plan_lines.append("No errors found — project is 100% compliant.")
    else:
        for i, (domain, count) in enumerate(sorted_domains, 1):
            plan_lines.append(f"- [ ] STEP {i}: Fix `{domain}/` ({count} errors)")

    content = "\n".join(plan_lines)

    tmp_path: Path | None = None  # pre-initialize so except block is safe
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_file.parent,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())  # Flush to OS before atomic rename

        tmp_path.replace(output_file)  # POSIX-atomic on same filesystem
        return True

    except OSError as e:
        print(f"ERROR: Failed to save {output_file.name}: {e}", file=sys.stderr)
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False

def generate_plan() -> None:
    """Main orchestration entry point."""
    report_file = Path("mypy_global_report.txt")
    output_file = Path("MYPY_REMEDIATION_PLAN.md")

    if not report_file.exists():
        print(f"Report '{report_file.name}' not found.")
        print("Run:  mypy . 2>&1 | tee mypy_global_report.txt")
        return

    domain_counts, _ = _aggregate_errors(report_file)

    if _write_plan(domain_counts, output_file):
        print(
            f"Plan generated → '{output_file.absolute()}' "
            f"({len(domain_counts)} domain(s))."
        )

if __name__ == "__main__":
    generate_plan()
