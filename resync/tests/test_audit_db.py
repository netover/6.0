from __future__ import annotations

import pytest

from resync.core import audit_db


@pytest.mark.asyncio
async def test_add_audit_records_batch_rejects_async_context() -> None:
    with pytest.raises(RuntimeError, match="called from async context"):
        audit_db.add_audit_records_batch([{"action": "login"}])
