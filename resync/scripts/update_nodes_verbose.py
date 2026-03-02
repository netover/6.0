# ruff: noqa: E501
"""Utility script to patch job execution history query limit."""

from pathlib import Path

TARGET = Path("resync/workflows/nodes_verbose.py")
OLD = """                    ORDER BY timestamp DESC
                    LIMIT 1000"""
NEW = """                    ORDER BY timestamp DESC
                    LIMIT {_JOB_EXECUTION_HISTORY_LIMIT}"""

def main() -> None:
    content = TARGET.read_text(encoding="utf-8")
    updated = content.replace(OLD, NEW)
    TARGET.write_text(updated, encoding="utf-8")

if __name__ == "__main__":
    main()
