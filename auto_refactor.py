"""
auto_refactor.py — Run analysis tools and package results.

WARNING: subprocess calls without timeout can hang CI indefinitely.
This module provides tooling for code analysis and packaging.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

_SUBPROCESS_TIMEOUT = 300  # 5 minutes max per tool


def run_tool(command: list[str]) -> str:
    """
    Run an external command and return combined stdout+stderr.

    Args:
        command: Command and arguments as list.

    Returns:
        Combined stdout and stderr output.
    """
    print(f"Running: {' '.join(command)}")
   (
        command,
 result = subprocess.run        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,  # P1-01: prevent CI hang
    )
    return result.stdout + "\n" + result.stderr


def run_ruff_fix(target_dir: str) -> None:
    """
    Apply ruff safe fixes and format to target directory.

    Args:
        target_dir: Directory to apply fixes.
    """
    print("Running Ruff safe fixes...")
    subprocess.run(
        ["python", "-m", "ruff", "check", "--fix", target_dir],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    print("Running Ruff format...")
    subprocess.run(
        ["python", "-m", "ruff", "format", target_dir],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )


def create_zip(source_dir: str, zip_filename: str) -> None:
    """
    Package root .py files and allowed dirs into a zip.

    Args:
        source_dir: Source directory to package.
        zip_filename: Output zip file path.
    """
    print(f"Creating zip file {zip_filename}...")
    allowed_dirs = ["resync", "tests"]
    root = Path(source_dir)

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add root .py files
        for f in root.iterdir():
            if f.is_file() and f.suffix == ".py":
                zipf.write(f, f.name)

        # Add allowed directories - P1-03: use Path consistently
        for d in allowed_dirs:
            dir_path = root / d
            if dir_path.is_dir():
                for file_path in dir_path.rglob("*"):
                    if "__pycache__" in file_path.parts:
                        continue
                    if file_path.suffix == ".pyc" or not file_path.is_file():
                        continue
                    arcname = file_path.relative_to(root)
                    zipf.write(file_path, arcname)


def main() -> int:  # P1-02: return exit code
    """
    Main entry point for the auto-refactor script.

    Returns:
        Exit code (0 for success).
    """
    target_dir = "resync"
    report_file = "analysis_report.txt"
    zip_file = "resync_fixed.zip"

    # 1. Run tools analysis and save to txt
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE ANÁLISE DO CÓDIGO\n")
        f.write("=" * 50 + "\n\n")

        f.write("=== 1. RUFF (Linting & Bugs) ===\n")
        f.write(run_tool(["python", "-m", "ruff", "check", target_dir]))
        f.write("\n" + "=" * 50 + "\n\n")

        f.write("=== 2. MYPY (Typing) ===\n")
        f.write(run_tool(["python", "-m", "mypy", target_dir]))
        f.write("\n" + "=" * 50 + "\n\n")

        f.write("=== 3. PYLINT (Code Smells) ===\n")
        f.write(run_tool(["python", "-m", "pylint", target_dir, "--exit-zero"]))
        f.write("\n" + "=" * 50 + "\n\n")

        f.write("=== 4. RADON (Cyclomatic Complexity - Grades C or worse) ===\n")
        f.write(run_tool(["python", "-m", "radon", "cc", target_dir, "-nc"]))
        f.write("\n" + "=" * 50 + "\n\n")

    print(f"Report saved to {report_file}")

    # 2. Run autofixes and format (Ruff is the only reliable auto-fixer here)
    run_ruff_fix(target_dir)

    # 3. Create zip of the project
    create_zip(".", zip_file)
    print(f"Project zipped to {zip_file}")
    print("Done!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
