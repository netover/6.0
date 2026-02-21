from __future__ import annotations

import pytest
from sqlalchemy import literal, or_, select
from sqlalchemy.dialects import postgresql

from resync.core.database.repositories.stores import FeedbackRepository


@pytest.mark.parametrize(
    ("is_positive", "rating", "expected_in_negative"),
    [
        (True, 5, False),
        (False, 5, True),
        (None, 5, True),
        (True, 2, True),
    ],
)
def test_negative_predicate_tri_state_semantics(
    is_positive: bool | None,
    rating: int,
    expected_in_negative: bool,
) -> None:
    # Mirrors repository predicate:
    # or_(Feedback.is_positive.is_not(True), Feedback.rating <= 2)
    predicate = or_(literal(is_positive).is_not(True), literal(rating) <= 2)

    compiled = str(
        select(literal(1)).where(predicate).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )

    # Evaluate truth table in Python equivalent for tri-state business semantics
    in_negative = (is_positive is not True) or (rating <= 2)
    assert in_negative is expected_in_negative
    assert "IS NOT true" in compiled


@pytest.mark.asyncio
async def test_feedback_repository_query_uses_is_not_true() -> None:
    repo = FeedbackRepository(session_factory=None)

    captured: dict[str, object] = {}

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return []

    class _Session:
        async def execute(self, stmt):
            captured["stmt"] = stmt
            return _Result()

    class _SessionCtx:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    repo._get_session = lambda: _SessionCtx()  # type: ignore[method-assign]

    rows = await repo.get_negative_examples(limit=10)
    assert rows == []

    stmt = captured.get("stmt")
    assert stmt is not None
    sql = str(
        stmt.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert "is_positive IS NOT true" in sql
