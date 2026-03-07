from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from resync.api.routes.admin import v2 as admin_v2


@pytest.mark.asyncio
async def test_stream_logs_rejects_path_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_module_file = tmp_path / "resync" / "api" / "routes" / "admin" / "v2.py"
    fake_module_file.parent.mkdir(parents=True)
    fake_module_file.write_text("# test\n", encoding="utf-8")

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "app.log").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(admin_v2, "__file__", str(fake_module_file))

    with pytest.raises(HTTPException) as exc_info:
        await admin_v2.stream_logs("../secrets.txt")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid log file path"
