"""
Clean lint comments from Python source files.

Removes everything after 'pylint' or 'mypy' on the same line if they appear in a comment.
"""

import argparse
import re
from pathlib import Path


def clean_comment_line(line: str) -> tuple[str, bool]:
    """
    Remove everything after 'pylint' or 'mypy' on the same line if they appear in a comment.

    Examples:
        '# pylint: some comment' -> '# pylint'
        '# mypy: ignore-errors' -> '# mypy'
        'x = 1 # some comment' -> 'x = 1 # some comment' (no change)
    """
    # Regex to match a comment start followed by optional spaces,
    # then 'pylint' or 'mypy', followed by anything else on the line.
    # We want to keep up to 'pylint' or 'mypy'.
    pattern = re.compile(r'(#\s*(?:pylint|mypy)).*')

    new_line = pattern.sub(r'\1', line)

    return new_line, new_line != line


def process_file(file_path: Path, dry_run: bool = False, verbose: bool = False) -> bool:
    """
    Process a single Python file to clean lint comments.

    Args:
        file_path: Path to the Python file.
        dry_run: If True, only report changes without writing.
        verbose: If True, print detailed output.

    Returns:
        True if file was modified, False otherwise.
    """
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as e:  # P1-04: specific exception, not generic Exception
        if verbose:
            print(f"Error reading {file_path}: {e}")
        return False

    modified = False
    new_lines: list[str] = []

    for i, line in enumerate(lines):
        processed_line, changed = clean_comment_line(line)

        if changed:
            # P2-03: splitlines(keepends=True) already preserves line endings
            # Code below is simplified - no need for redundant CRLF handling
            new_lines.append(processed_line)
            modified = True
            if verbose:
                print(f"  [{file_path.name}:{i+1}] {line.strip()} -> {processed_line.strip()}")
        else:
            new_lines.append(line)

    if modified:
        if dry_run:
            print(f"Would modify: {file_path}")
        else:
            try:
                # P2-02: use Path.write_text instead of open()
                file_path.write_text("".join(new_lines), encoding="utf-8")
                if verbose:
                    print(f"Modified: {file_path}")
            except OSError as e:
                print(f"Error writing {file_path}: {e}")
                return False
    return modified


def main() -> None:
    """Main entry point for the lint comment cleaner."""
    parser = argparse.ArgumentParser(
        description="Remove content after 'pylint' and 'mypy' in comments."
    )
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()
    root = Path(args.path).resolve()

    exclude_dirs = {'.git', 'venv', '.venv', '__pycache__', '.mypy_cache', '.pytest_cache'}

    count = 0
    modified_count = 0

    if root.is_file():
        files = [root]
    else:
        files = root.rglob("*.py")

    for py_file in files:
        # Skip excluded directories
        if any(part in exclude_dirs for part in py_file.parts):
            continue

        count += 1
        if process_file(py_file, args.dry_run, args.verbose):
            modified_count += 1

    print(f"\nFinished processing {count} files.")
    print(f"Modified {modified_count} files.")


if __name__ == "__main__":
    main()
