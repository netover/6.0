from __future__ import annotations

import pytest

from resync.knowledge.retrieval.hybrid import QueryClassifier, QueryIntent


def test_query_classifier_does_not_create_lock_eagerly() -> None:
    classifier = QueryClassifier(use_llm_fallback=False)

    assert classifier._cache_lock is None


@pytest.mark.asyncio
async def test_query_classifier_cache_lock_is_created_lazily() -> None:
    classifier = QueryClassifier(use_llm_fallback=False)

    await classifier._add_to_cache("k", QueryIntent.GENERAL)

    assert classifier._cache_lock is not None
