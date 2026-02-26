"""
Fix: reorder sections in ``resync/core/structured_logger.py``.

Delegates to :mod:`fix_reorder_imports` for the shared reorder logic.

WARNING: Operates on source files.  Run from a clean git working tree.
"""

from fix_reorder_imports import reorder_python_file

if __name__ == "__main__":
    reorder_python_file("resync/core/structured_logger.py")
