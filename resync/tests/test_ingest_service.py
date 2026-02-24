"""
Unit tests for IngestService — batch SHA-256 dedup (N+1 fix) and
path traversal guard in the batch ingestion API.

Requires: pytest-asyncio, pytest
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from resync.knowledge.ingestion.ingest import IngestService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(*, existing_shas: set[str] | None = None) -> MagicMock:
    """Return a mock VectorStore with exists_batch_by_sha256 configured."""
    store = MagicMock()
    store.exists_batch_by_sha256 = AsyncMock(return_value=existing_shas or set())
    store.upsert_batch = AsyncMock(return_value=None)
    return store


def _make_embedder(dim: int = 4) -> MagicMock:
    """Return a mock Embedder that returns zero vectors."""
    embedder = MagicMock()

    async def _embed_batch(texts, **_):  # noqa: ANN001
        return [[0.0] * dim for _ in texts]

    embedder.embed_batch = _embed_batch
    return embedder


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# IngestService — basic path (ingest_document)
# ---------------------------------------------------------------------------


class TestIngestDocumentBatchDedup:
    """ingest_document uses exists_batch_by_sha256 once, never per-chunk."""

    @pytest.mark.asyncio
    async def test_calls_batch_sha_exactly_once(self) -> None:
        """exists_batch_by_sha256 is called exactly ONE time for N chunks."""
        store = _make_store()
        svc = IngestService(_make_embedder(), store, batch_size=128)

        await svc.ingest_document(
            tenant="t1",
            doc_id="doc1",
            source="file.txt",
            text="chunk one. " * 30 + "\n\n" + "chunk two. " * 30,
            ts_iso="2026-01-01T00:00:00Z",
        )

        assert store.exists_batch_by_sha256.call_count == 1, (
            "Expected exactly 1 batch call, got N per-chunk calls (N+1 not fixed)"
        )

    @pytest.mark.asyncio
    async def test_all_duplicates_skips_embedder(self) -> None:
        """When all chunks are duplicates, the embedder is never called."""
        text = "hello world " * 20
        # Pre-compute what the SHA will be after strip
        normalized = text.strip()
        sha = _sha(normalized)

        store = _make_store(existing_shas={sha})
        embedder = _make_embedder()
        embedder.embed_batch = AsyncMock()  # track calls

        svc = IngestService(embedder, store, batch_size=128)
        result = await svc.ingest_document(
            tenant="t1",
            doc_id="doc1",
            source="file.txt",
            text=text,
            ts_iso="2026-01-01T00:00:00Z",
        )

        assert result == 0
        embedder.embed_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_duplicates_embeds_only_new(self) -> None:
        """Only fresh chunks reach the embedder; duplicates are filtered out."""
        # Produce two distinct chunks by using very different content
        chunk_a = "aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll mmm " * 10
        chunk_b = "zzz yyy xxx www vvv uuu ttt sss rrr qqq ppp ooo nnn " * 10
        text = chunk_a + "\n\n\n" + chunk_b

        sha_a = _sha(chunk_a.strip())

        store = _make_store(existing_shas={sha_a})  # chunk_a is a dupe
        embedder = _make_embedder()
        call_lengths: list[int] = []

        async def _capture_embed(texts, **_):  # noqa: ANN001
            call_lengths.append(len(texts))
            return [[0.0] * 4 for _ in texts]

        embedder.embed_batch = _capture_embed

        svc = IngestService(embedder, store, batch_size=128)
        result = await svc.ingest_document(
            tenant="t1",
            doc_id="doc1",
            source="file.txt",
            text=text,
            ts_iso="2026-01-01T00:00:00Z",
        )

        # At least 1 chunk was embedded (chunk_b is new)
        assert result >= 1
        # Total texts embedded must not include chunk_a
        total_embedded = sum(call_lengths)
        assert total_embedded < 2, (
            "Duplicate chunk was not filtered — should have embedded only 1 chunk"
        )


# ---------------------------------------------------------------------------
# IngestService — advanced path (ingest_document_advanced)
# ---------------------------------------------------------------------------


class TestIngestDocumentAdvancedBatchDedup:
    """ingest_document_advanced also uses exists_batch_by_sha256 once."""

    @pytest.mark.asyncio
    async def test_calls_batch_sha_exactly_once(self) -> None:
        store = _make_store()
        svc = IngestService(_make_embedder(), store, batch_size=128)

        await svc.ingest_document_advanced(
            tenant="t1",
            doc_id="doc1",
            source="manual.md",
            text="# Section\n\nSome content here.\n\n## Subsection\n\nMore text.\n",
            ts_iso="2026-01-01T00:00:00Z",
            chunking_strategy="structure_aware",
            enable_multi_view=False,
        )

        assert store.exists_batch_by_sha256.call_count == 1, (
            "Expected exactly 1 batch call for main chunks (N+1 not fixed in advanced path)"
        )

    @pytest.mark.asyncio
    async def test_multi_view_calls_batch_sha_separately(self) -> None:
        """Multi-view deduplication also uses a single batch call."""
        store = _make_store()
        svc = IngestService(_make_embedder(), store, batch_size=128)

        await svc.ingest_document_advanced(
            tenant="t1",
            doc_id="doc1",
            source="manual.md",
            text=(
                "# Error Codes\n\n"
                "## AWS001E\n\nCause: connection failed.\nSolution: check network.\n\n"
                "## AWS002W\n\nCause: low memory.\nSolution: free memory.\n"
            ),
            ts_iso="2026-01-01T00:00:00Z",
            chunking_strategy="structure_aware",
            enable_multi_view=True,
        )

        # Call 1 = main chunks dedup, Call 2 = multi-view dedup (if any views generated)
        assert store.exists_batch_by_sha256.call_count >= 1, (
            "exists_batch_by_sha256 should have been called at least once"
        )
        # Crucially no per-chunk single-SHA call should have been made
        assert not hasattr(store, "exists_by_sha256") or store.exists_by_sha256.call_count == 0


# ---------------------------------------------------------------------------
# Path traversal guard — ingest_api.py batch endpoint
# ---------------------------------------------------------------------------


class TestPathTraversalGuard:
    """The batch ingest endpoint rejects paths outside KNOWLEDGE_DOCS_ROOT."""

    def _make_request(self, file_paths: list[str]) -> Any:
        from resync.api.routes.knowledge.ingest_api import BatchIngestRequest

        return BatchIngestRequest(
            file_paths=file_paths,
            tenant="default",
        )

    @pytest.mark.asyncio
    async def test_path_outside_root_rejected(self, tmp_path: Path) -> None:
        """Paths outside docs_root are rejected with a path traversal error."""
        from resync.api.routes.knowledge.ingest_api import ingest_batch

        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        # A path clearly outside the docs root
        evil_path = str(tmp_path / "etc" / "passwd")

        request = self._make_request([evil_path])

        with (
            patch("resync.api.routes.knowledge.ingest_api._get_pipeline") as mock_pipe,
            patch(
                "resync.api.routes.knowledge.ingest_api._get_settings"
            ) as mock_settings,
        ):
            mock_settings.return_value.KNOWLEDGE_DOCS_ROOT = docs_root
            mock_pipe.return_value = MagicMock()

            response = await ingest_batch(request)

        assert response.failed == 1
        assert response.succeeded == 0
        assert "denied" in (response.results[0].error or "").lower()

    @pytest.mark.asyncio
    async def test_path_inside_root_accepted(self, tmp_path: Path) -> None:
        """Paths inside docs_root are not rejected by the path traversal guard."""
        from resync.api.routes.knowledge.ingest_api import ingest_batch

        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        # Create a real file inside the root
        good_file = docs_root / "manual.pdf"
        good_file.write_bytes(b"%PDF fake content")

        request = self._make_request([str(good_file)])

        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.doc_id = "doc1"
        mock_result.source = "manual.pdf"
        mock_result.format = "pdf"
        mock_result.pages = 1
        mock_result.tables_extracted = 0
        mock_result.chunks_stored = 3
        mock_result.conversion_time_s = 0.5
        mock_result.total_time_s = 1.0
        mock_result.error = None

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file = AsyncMock(return_value=mock_result)

        # Patch DoclingConverter.is_supported to return True for .pdf
        with (
            patch("resync.api.routes.knowledge.ingest_api._get_pipeline", return_value=mock_pipeline),
            patch(
                "resync.api.routes.knowledge.ingest_api._get_settings"
            ) as mock_settings,
            patch(
                "resync.api.routes.knowledge.ingest_api.DoclingConverter.is_supported",
                return_value=True,
            ),
        ):
            mock_settings.return_value.KNOWLEDGE_DOCS_ROOT = docs_root
            response = await ingest_batch(request)

        assert response.failed == 0
        assert response.succeeded == 1

    @pytest.mark.asyncio
    async def test_traversal_via_dotdot_rejected(self, tmp_path: Path) -> None:
        """Path with ../.. components that escapes root is rejected after resolve()."""
        from resync.api.routes.knowledge.ingest_api import ingest_batch

        docs_root = tmp_path / "docs"
        docs_root.mkdir()

        # Construct a path that traverses out: docs/../../../etc/passwd
        traversal = str(docs_root / ".." / ".." / "etc" / "passwd")
        request = self._make_request([traversal])

        with (
            patch("resync.api.routes.knowledge.ingest_api._get_pipeline") as mock_pipe,
            patch(
                "resync.api.routes.knowledge.ingest_api._get_settings"
            ) as mock_settings,
        ):
            mock_settings.return_value.KNOWLEDGE_DOCS_ROOT = docs_root
            mock_pipe.return_value = MagicMock()

            response = await ingest_batch(request)

        assert response.failed == 1
        assert "denied" in (response.results[0].error or "").lower()
