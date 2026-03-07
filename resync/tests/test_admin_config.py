from __future__ import annotations

import pytest

from resync.api.routes.admin import config as admin_config
import resync.core.database.repositories as repositories


class _FailingAuditRepo:
    async def get_all(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("database offline: internal detail")


@pytest.mark.asyncio
async def test_get_audit_logs_masks_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repositories, "AuditEntryRepository", _FailingAuditRepo)

    result = await admin_config.get_audit_logs()

    assert result["records"] == []
    assert result["error"] == "audit_query_failed"
