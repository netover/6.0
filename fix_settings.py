"""
Fix: insert SMTP configuration block inside the Settings class.

Target file: ``resync/settings.py``

This script:
1. Detects a trailing SMTP field block (starting at ``smtp_enabled: bool = Field``).
2. Finds the ``# VALIDADORES`` marker inside the class body.
3. Inserts the SMTP fields (properly indented) just before the marker.
4. Writes the result **atomically** (temp-file + fsync + rename).

Raises:
    FileNotFoundError: If ``resync/settings.py`` does not exist.
    RuntimeError: If the ``# VALIDADORES`` marker is absent — fails loudly
                  instead of silently no-oping.

WARNING: Operates on source files.  Run from a clean git working tree.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

_TARGET = Path("resync/settings.py")
_MARKER = "# VALIDADORES"

def _apply_smtp_fix() -> None:
    """
    Insert SMTP config fields inside the Settings class atomically.

    Raises:
        FileNotFoundError: Target file missing.
        RuntimeError: Insertion marker not found.
        OSError: Atomic write failed.
    """
    if not _TARGET.exists():
        raise FileNotFoundError(
            f"Settings file not found: {_TARGET}. "
            "Ensure you run from the project root."
        )

    lines = _TARGET.read_text(encoding="utf-8").splitlines(keepends=True)

    # ------------------------------------------------------------------ #
    # Step 1: Separate out any trailing SMTP block                        #
    # ------------------------------------------------------------------ #
    clean_lines: list[str] = []
    smtp_lines: list[str] = []
    capturing_smtp = False

    for line in lines:
        if "smtp_enabled: bool = Field" in line:
            capturing_smtp = True
        if capturing_smtp:
            smtp_lines.append(line)
        else:
            clean_lines.append(line)

    # ------------------------------------------------------------------ #
    # Step 2: Find insertion point                                        #
    # ------------------------------------------------------------------ #
    insert_idx: int = -1
    for i, line in enumerate(clean_lines):
        if _MARKER in line:
            insert_idx = i  # insert BEFORE this marker line
            break

    if insert_idx == -1:
        raise RuntimeError(
            f"Insertion marker {_MARKER!r} not found in {_TARGET}. "
            "The file may have been refactored — update fix_settings.py."
        )

    # ------------------------------------------------------------------ #
    # Step 3: Ensure 4-space class-body indentation                      #
    # ------------------------------------------------------------------ #
    indented_smtp: list[str] = []
    for sline in smtp_lines:
        if not sline.strip():
            indented_smtp.append(sline)
        elif sline.startswith("    "):
            indented_smtp.append(sline)
        else:
            indented_smtp.append("    " + sline)

    final_lines = clean_lines[:insert_idx] + indented_smtp + clean_lines[insert_idx:]

    # ------------------------------------------------------------------ #
    # Step 4: Atomic write                                                #
    # ------------------------------------------------------------------ #
    backup = _TARGET.with_suffix(_TARGET.suffix + ".bak")
    shutil.copy2(_TARGET, backup)

    tmp_path: Path | None = None  # pre-initialize for safe use in except
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=_TARGET.parent,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.writelines(final_lines)
            tmp.flush()
            os.fsync(tmp.fileno())

        tmp_path.replace(_TARGET)
        backup.unlink(missing_ok=True)  # remove backup only after success
        print("Settings updated successfully.")

    except Exception:
        if backup.exists():
            shutil.copy2(backup, _TARGET)
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

if __name__ == "__main__":
    try:
        _apply_smtp_fix()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
