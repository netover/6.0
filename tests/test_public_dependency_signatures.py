from __future__ import annotations

import os

from resync.tools.check_public_dependencies import find_violations


def test_no_any_in_public_dependency_signatures() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    violations = find_violations(project_root)
    assert violations == [], "\n".join(f"{v.file}:{v.line}: {v.message}" for v in violations)
