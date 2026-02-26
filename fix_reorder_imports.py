"""
Shared utility: reorder Python file sections into canonical order.

Canonical section order:
  1. ``from __future__ import ...``
  2. Module docstring
  3. Stdlib / third-party imports
  4. Application code

This module is imported by ``fix_logger_final.py`` and
``fix_tools_final.py`` to avoid code duplication.

P0-02: Uses atomic write (temp-file + fsync + rename) so the target
file is never left in a partially-written state.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

def reorder_python_file(filename: str) -> None:
    """
    Reorder a Python source file into canonical section order.

    Sections detected and reordered:
    - ``from __future__ import`` lines (must be first)
    - Module-level docstring (triple-quoted string immediately after future imports)
    - Import statements (``import …`` / ``from … import …``)
    - Remaining code

    The function performs an **atomic write** using a temporary file and
    ``os.fsync``: either the old file survives intact or the new file is
    fully written — never a partial result.

    A ``.bak`` backup is created before writing and removed only after a
    successful rename, leaving a recovery path on failure.

    Args:
        filename: Path (relative to CWD) of the Python source file.

    Raises:
        FileNotFoundError: If *filename* does not exist.
        RuntimeError: If *filename* is empty.
        OSError: If the atomic write fails (backup is preserved).
    """
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Target file not found: {filename!r}. "
            "Run from the project root directory."
        )

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    if not lines:
        raise RuntimeError(f"File is empty: {filename!r}")

    # ------------------------------------------------------------------ #
    # Partition lines into four buckets                                   #
    # ------------------------------------------------------------------ #
    future_imports: list[str] = []
    start_idx = 0

    # Collect consecutive ``from __future__`` lines at the top
    for i, line in enumerate(lines):
        if line.startswith("from __future__"):
            future_imports.append(line)
            start_idx = i + 1
        else:
            break

    docstring: list[str] = []
    imports: list[str] = []
    code: list[str] = []

    in_docstring = False
    docstring_char = ""

    for i in range(start_idx, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Opening triple-quote — enter docstring mode
                in_docstring = True
                docstring_char = stripped[:3]
                docstring.append(line)
                # Check if it closes on the same line (single-line docstring)
                rest = stripped[3:]
                if len(rest) >= 3 and rest.endswith(docstring_char):
                    in_docstring = False
            elif stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(line)
            else:
                code.append(line)
        else:
            docstring.append(line)
            # P2-05: Check for closing triple-quote (must be at end of stripped line)
            if stripped.endswith(docstring_char):
                # Line contains the closing delimiter - exit docstring mode
                # This handles both single-line docstrings and multi-line ones
                in_docstring = False

    final_lines = future_imports + docstring + imports + code

    # ------------------------------------------------------------------ #
    # Atomic write                                                         #
    # ------------------------------------------------------------------ #
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)

    tmp_path: Path | None = None  # pre-initialize for safe use in except
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.writelines(final_lines)
            tmp.flush()
            os.fsync(tmp.fileno())  # Flush to disk before rename

        # POSIX-atomic rename: either old or new file exists, never both half-written
        tmp_path.replace(path)

        # Cleanup backup only after confirmed success
        backup.unlink(missing_ok=True)
        print(f"Reordered: {filename}")

    except Exception:
        # Restore from backup on any failure
        if backup.exists():
            shutil.copy2(backup, path)
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
