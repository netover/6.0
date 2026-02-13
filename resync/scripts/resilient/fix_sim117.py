#!/usr/bin/env python3
"""
Fix SIM117 errors: Combine nested with statements.

Example:
    with A():
        with B():
            code()

Becomes:
    with A(), B():
        code()
"""

import json
import re
import subprocess
from pathlib import Path


def get_sim117_locations():
    """Get all SIM117 error locations from ruff."""
    result = subprocess.run(
        ["ruff", "check", ".", "--select=SIM117", "--output-format=json"],
        capture_output=True,
        text=True,
    )

    # Filter out warnings
    stdout = result.stdout
    if stdout.startswith("warning:"):
        lines = stdout.split("\n")
        json_lines = [l for l in lines if l.strip().startswith("[") or l.strip().startswith("{")]
        stdout = "\n".join(json_lines)

    try:
        return json.loads(stdout) if stdout.strip() else []
    except json.JSONDecodeError:
        return []


def fix_file(filepath: str, line_numbers: list[int]) -> bool:
    """Fix SIM117 errors in a file."""
    path = Path(filepath)
    content = path.read_text()
    lines = content.split("\n")

    modified = False
    skip_lines = set()

    # Process in reverse to avoid line number shifts
    for line_num in sorted(line_numbers, reverse=True):
        idx = line_num - 1
        if idx in skip_lines or idx >= len(lines):
            continue

        line = lines[idx]

        # Check if this is a 'with' statement
        with_match = re.match(r"^(\s*)with\s+(.+):\s*$", line)
        if not with_match:
            continue

        indent = with_match.group(1)
        context1 = with_match.group(2)

        # Check next line for another 'with' with one more indent level
        if idx + 1 >= len(lines):
            continue

        next_line = lines[idx + 1]
        expected_indent = indent + "    "  # 4 spaces more

        next_with_match = re.match(
            r"^" + re.escape(expected_indent) + r"with\s+(.+):\s*$", next_line
        )
        if not next_with_match:
            continue

        context2 = next_with_match.group(1)

        # Combine the two with statements
        combined = f"{indent}with {context1}, {context2}:"

        # Replace lines
        lines[idx] = combined

        # Get the body - everything after the second with, with reduced indent
        # Find the body lines
        body_start = idx + 2
        if body_start >= len(lines):
            continue

        # The body should have double indent (indent + 8 spaces)
        body_indent = expected_indent + "    "

        # Reduce indent of all body lines by 4 spaces
        i = body_start
        while i < len(lines):
            body_line = lines[i]
            if body_line.strip() == "":
                i += 1
                continue
            if body_line.startswith(body_indent):
                lines[i] = expected_indent + body_line[len(body_indent) :]
                i += 1
            elif body_line.startswith(expected_indent) and body_line.strip():
                # Less indented, end of body
                break
            else:
                # Back to original indent or less
                break

        # Remove the second 'with' line
        lines.pop(idx + 1)
        skip_lines.add(idx + 1)

        modified = True

    if modified:
        path.write_text("\n".join(lines))
        return True
    return False


def main():
    """Main function."""
    errors = get_sim117_locations()

    if not errors:
        print("No SIM117 errors found or could not parse errors")
        return

    # Group by file
    by_file = {}
    for err in errors:
        f = err["filename"]
        line = err["location"]["row"]
        if f not in by_file:
            by_file[f] = []
        by_file[f].append(line)

    print(f"Found {len(errors)} SIM117 errors in {len(by_file)} files")

    fixed_count = 0
    for filepath, line_nums in by_file.items():
        try:
            if fix_file(filepath, line_nums):
                fixed_count += 1
                print(f"  Fixed: {filepath}")
        except Exception as e:
            print(f"  Error fixing {filepath}: {e}")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
