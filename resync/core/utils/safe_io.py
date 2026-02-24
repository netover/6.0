"""
Safe I/O Utilities for Resync.

Provides atomic file write operations to prevent data corruption
during system crashes or power failures.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Union


def safe_write_file(
    filepath: Union[str, Path],
    content: Union[str, bytes],
    encoding: str = "utf-8",
) -> None:
    """
    Writes content to a file atomically using a temporary file.

    This ensures that the target file is either fully written or
    remains unchanged in case of a crash during the write process.

    Args:
        filepath: Path to the target file
        content: String or bytes to write
        encoding: Encoding for string content (default: utf-8)
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create a temporary file in the same directory to ensure
    # it is on the same filesystem for an atomic rename operation.
    fd, temp_path_str = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp-",
    )
    temp_path = Path(temp_path_str)

    try:
        if isinstance(content, str):
            with os.fdopen(fd, "w", encoding=encoding) as f:
                f.write(content)
        else:
            with os.fdopen(fd, "wb") as f:
                f.write(content)

        # Ensure data is flushed to disk
        # (Though rename is atomic, we want the content to be there)
        # os.fsync(fd) is handled by os.fdopen(fd, "w").close() mostly,
        # but for maximum safety:
        with open(temp_path, "ab") as f:
            os.fsync(f.fileno())

        # Atomic replace
        os.replace(temp_path, path)

    except Exception as e:
        # Cleanup temp file on failure
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise e
    finally:
        # mkstemp keeps the file descriptor open, but os.fdopen closes it.
        # If it wasn't closed by fdopen, we'd need to close it here.
        pass


def backup_and_replace(
    filepath: Union[str, Path],
    new_content: Union[str, bytes],
    backup_ext: str = ".bak",
) -> None:
    """
    Creates a backup of the existing file before replacing it atomically.
    """
    path = Path(filepath)
    if path.exists():
        backup_path = path.with_suffix(path.suffix + backup_ext)
        shutil.copy2(path, backup_path)

    safe_write_file(path, new_content)
