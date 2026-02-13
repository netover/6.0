#!/usr/bin/env python3
"""
Fix deprecated typing imports (UP035).

Replaces typing.Dict, typing.List, typing.Tuple, typing.Set with built-in types.
"""

import json
import re
import subprocess
from pathlib import Path


def get_up035_files():
    """Get all files with UP035 errors."""
    result = subprocess.run(
        ["ruff", "check", ".", "--select=UP035", "--output-format=json"],
        capture_output=True,
        text=True,
    )

    try:
        errors = json.loads(result.stdout)
        files = set(e["filename"] for e in errors)
        return list(files)
    except json.JSONDecodeError:
        return []


def fix_file(filepath: str) -> bool:
    """Fix deprecated typing imports in a file."""
    path = Path(filepath)
    content = path.read_text()
    original = content

    # Pattern to find typing imports
    # Match: from typing import X, Y, Z
    typing_import_pattern = r"^from typing import (.+)$"

    lines = content.split("\n")
    modified_lines = []

    for line in lines:
        match = re.match(typing_import_pattern, line)
        if match:
            imports_str = match.group(1)
            # Parse imports, handling multi-line and parentheses
            imports = [i.strip() for i in imports_str.replace("(", "").replace(")", "").split(",")]

            # Deprecated types to remove
            deprecated = {"Dict", "List", "Tuple", "Set", "FrozenSet"}

            # Filter out deprecated imports
            new_imports = [i for i in imports if i and i not in deprecated]

            if new_imports:
                modified_lines.append(f"from typing import {', '.join(new_imports)}")
            # If all imports were deprecated, remove the line entirely
            continue

        modified_lines.append(line)

    new_content = "\n".join(modified_lines)

    # Now replace usages in type hints
    # Dict[x, y] -> dict[x, y]
    new_content = re.sub(r"\bDict\[", "dict[", new_content)
    new_content = re.sub(r"\bList\[", "list[", new_content)
    new_content = re.sub(r"\bTuple\[", "tuple[", new_content)
    new_content = re.sub(r"\bSet\[", "set[", new_content)
    new_content = re.sub(r"\bFrozenSet\[", "frozenset[", new_content)

    # Also handle standalone type annotations without brackets
    # e.g., -> Dict, -> List (less common but possible)
    # Be careful not to replace in strings or comments

    if new_content != original:
        path.write_text(new_content)
        return True
    return False


def main():
    """Main function."""
    files = get_up035_files()

    if not files:
        print("No UP035 errors found")
        return

    print(f"Found {len(files)} files with deprecated typing imports")

    fixed = 0
    for filepath in files:
        try:
            if fix_file(filepath):
                fixed += 1
                print(f"  Fixed: {filepath}")
        except Exception as e:
            print(f"  Error fixing {filepath}: {e}")

    print(f"\nFixed {fixed} files")


if __name__ == "__main__":
    main()
