#!/usr/bin/env python3
"""
Fix B904 errors: raise without from inside except clause.

This script automatically adds 'from e' or 'from None' to raise statements
inside except blocks.
"""

import json
import re
import subprocess
from pathlib import Path


def get_b904_locations():
    """Get all B904 error locations from ruff."""
    result = subprocess.run(
        ["ruff", "check", ".", "--select=B904", "--output-format=json"],
        capture_output=True,
        text=True,
    )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("No B904 errors found or error parsing output")
        return []


def fix_file(filepath: str, line_numbers: list[int]):
    """Fix B904 errors in a single file."""
    path = Path(filepath)
    content = path.read_text()
    lines = content.splitlines()

    modified = False

    for line_num in sorted(line_numbers, reverse=True):
        # Line numbers are 1-indexed
        idx = line_num - 1
        if idx >= len(lines):
            continue

        line = lines[idx]

        # Skip if already has 'from'
        if " from " in line and "raise " in line:
            continue

        # Find the except block this raise is in to get the exception variable
        exception_var = None
        for i in range(idx, -1, -1):
            except_match = re.match(r"\s*except\s+[^:]+\s+as\s+(\w+)\s*:", lines[i])
            if except_match:
                exception_var = except_match.group(1)
                break
            except_match2 = re.match(r"\s*except\s*:", lines[i])
            if except_match2:
                exception_var = None
                break
            except_match3 = re.match(r"\s*except\s+\w+\s*:", lines[i])
            if except_match3:
                exception_var = None
                break
            # Check for multi-line raise statement start
            if "raise " in lines[i] and i < idx:
                break

        # Handle multi-line raise statements
        # Find the end of the raise statement (closing parenthesis)
        raise_end = idx
        paren_count = 0
        started = False

        for i in range(idx, len(lines)):
            line_content = lines[i]
            for char in line_content:
                if char == "(":
                    paren_count += 1
                    started = True
                elif char == ")":
                    paren_count -= 1

            if started and paren_count == 0:
                raise_end = i
                break

        # Modify the last line of the raise statement
        end_line = lines[raise_end]

        # Skip if already has from
        if " from " in end_line:
            continue

        # Find the closing parenthesis and add 'from' before it
        from_clause = f" from {exception_var}" if exception_var else " from None"

        # Handle various patterns
        if end_line.rstrip().endswith(")"):
            # Find the last closing paren
            last_paren_idx = end_line.rstrip().rfind(")")
            indent_after = end_line[last_paren_idx + 1 :]
            modified_line = end_line[:last_paren_idx] + ")" + from_clause + indent_after.rstrip()
            lines[raise_end] = modified_line
            modified = True
        elif re.search(r"raise\s+\w+\s*$", end_line.strip()):
            # Simple raise statement without parens: raise SomeException
            lines[raise_end] = end_line.rstrip() + from_clause
            modified = True

    if modified:
        path.write_text("\n".join(lines) + "\n")
        return True
    return False


def main():
    """Main function."""
    errors = get_b904_locations()

    if not errors:
        print("No B904 errors to fix")
        return

    # Group by file
    by_file = {}
    for err in errors:
        f = err["filename"]
        line = err["location"]["row"]
        if f not in by_file:
            by_file[f] = []
        by_file[f].append(line)

    print(f"Found {len(errors)} B904 errors in {len(by_file)} files")

    fixed_count = 0
    for filepath, lines in by_file.items():
        try:
            if fix_file(filepath, lines):
                fixed_count += 1
                print(f"  Fixed: {filepath}")
        except Exception as e:
            print(f"  Error fixing {filepath}: {e}")

    print(f"\nFixed {fixed_count} files")

    # Re-run ruff to check
    result = subprocess.run(
        ["ruff", "check", ".", "--select=B904", "--output-format=json"],
        capture_output=True,
        text=True,
    )

    try:
        remaining = json.loads(result.stdout)
        print(f"Remaining B904 errors: {len(remaining)}")
    except Exception:
        print("Check complete")


if __name__ == "__main__":
    main()
