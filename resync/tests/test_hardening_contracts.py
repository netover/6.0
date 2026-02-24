# pylint: disable=all
from __future__ import annotations

import inspect
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from resync.core import distributed_tracing as dt
from resync.workflows import statistical_analysis as sa


class _FakeSpan:
    def __init__(self) -> None:
        self.recorded_exception = None
        self.status = None

    def record_exception(self, exc: Exception) -> None:
        self.recorded_exception = exc

    def set_status(self, status) -> None:
        self.status = status


class _FakeManager:
    def __init__(self, span: _FakeSpan) -> None:
        self._span = span

    @contextmanager
    def trace_context(self, operation_name: str, **attributes):
        try:
            yield self._span
        except Exception as exc:
            self._span.record_exception(exc)
            self._span.set_status(str(exc))
            raise


@pytest.mark.asyncio
async def test_fetch_job_history_from_db_has_async_contract() -> None:
    assert inspect.iscoroutinefunction(sa.fetch_job_history_from_db)


@pytest.mark.asyncio
async def test_traced_sync_wrapper_records_exception_and_status(monkeypatch) -> None:
    span = _FakeSpan()
    monkeypatch.setattr(dt, "_get_tracing_manager", lambda: _FakeManager(span))

    @dt.traced("sync-operation")
    def _boom() -> None:
        raise RuntimeError("sync fail")

    with pytest.raises(RuntimeError, match="sync fail"):
        _boom()

    assert isinstance(span.recorded_exception, RuntimeError)
    assert span.status is not None


@pytest.mark.asyncio
async def test_traced_async_wrapper_records_exception_and_status(monkeypatch) -> None:
    span = _FakeSpan()

    async def _get_manager() -> _FakeManager:
        return _FakeManager(span)

    monkeypatch.setattr(dt, "get_distributed_tracing_manager", _get_manager)

    @dt.traced("async-operation")
    async def _boom_async() -> None:
        raise ValueError("async fail")

    with pytest.raises(ValueError, match="async fail"):
        await _boom_async()

    assert isinstance(span.recorded_exception, ValueError)
    assert span.status is not None


@pytest.mark.asyncio
async def test_fetch_job_history_from_db_returns_empty_on_failure() -> None:
    class _FailingDB:
        async def execute(self, _stmt):
            raise RuntimeError("db down")

    out = await sa.fetch_job_history_from_db(
        _FailingDB(),
        "JOB_A",
        datetime.now(timezone.utc),
        10,
    )

    assert out == []
