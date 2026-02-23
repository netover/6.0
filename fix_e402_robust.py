import sys
import re

def fix_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    future_imports = []
    docstrings = []
    imports = []
    code = []

    # State machine
    state = "START" # START, FUTURE, DOCSTRING, IMPORTS, CODE

    # Check for future import
    if lines and lines[0].startswith("from __future__"):
        future_imports.append(lines[0])
        lines = lines[1:]

    # Helper to clean up empty lines at start of list
    def strip_leading_empty(l):
        while l and not l[0].strip():
            l.pop(0)
        return l

    lines = strip_leading_empty(lines)

    # Check for module docstring
    if lines and (lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''")):
        quote_char = lines[0].strip()[:3]
        docstrings.append(lines[0])
        lines = lines[1:]
        if not docstrings[0].strip().endswith(quote_char) or len(docstrings[0].strip()) == 3:
            # Multi-line docstring
            while lines:
                line = lines.pop(0)
                docstrings.append(line)
                if line.strip().endswith(quote_char):
                    break

    # Process remaining lines
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if imports: # If we have imports, keep empty lines in imports section? No, discard or move to code.
                pass
            elif code:
                code.append(line)
            continue

        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
        else:
            code.append(line)

    # Reassemble
    final_lines = []
    final_lines.extend(future_imports)
    # Add empty line after future if present
    # if future_imports: final_lines.append("\n")

    final_lines.extend(docstrings)
    # Add empty line after docstring if present
    if docstrings: final_lines.append("\n")

    final_lines.extend(imports)
    # Add empty line after imports
    if imports: final_lines.append("\n")

    final_lines.extend(code)

    with open(filename, 'w') as f:
        f.writelines(final_lines)

fix_file('resync/core/specialists/tools.py')
fix_file('resync/core/structured_logger.py')
