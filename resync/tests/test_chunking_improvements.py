"""
Tests for v6.0 RAG Chunking Improvements.

Tests the 9 RAG Chunking Decisions implementation:
- Decision #3: Structure-aware overlap
- Decision #7: Citation-friendly IDs
- Decision #8: Multi-view chunk indexing
- Decision #9: Failure-slice eval pipeline
"""

from __future__ import annotations

import pytest

from resync.knowledge.ingestion.advanced_chunking import (
    AdvancedChunker,
    ChunkingConfig,
    ChunkingStrategy,
    ChunkMetadata,
    ChunkType,
    ChunkViewType,
    MultiViewChunk,
    OverlapStrategy,
    chunk_document,
    generate_stable_id,
    generate_snippet_preview,
    get_structure_aware_overlap,
    create_multi_view_chunk,
    generate_entities_view,
    generate_faq_view,
)
from resync.knowledge.ingestion.chunking_eval import (
    ChunkingEvalPipeline,
    EvalReport,
    EvalResult,
    FailureSlice,
    FailureSeverity,
    RetrievedChunk,
    detect_failure_slice,
    generate_rule_suggestions,
    create_eval_query,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_text() -> str:
    """Sample document text for testing."""
    return """
# Troubleshooting Guide

## Error Codes

### AWS001E - Connection Failed

**Cause:** The system could not establish a connection to the server.

**Solution:** Check your network settings and try again.

**System Action:** The connection attempt was terminated after 30 seconds.

### AWS002W - Low Memory Warning

**Cause:** Available memory is below 10%.

**Solution:** Close unnecessary applications or increase system memory.

## Procedures

### How to Restart the Service

1. Stop the current service using `conman` command.
2. Wait for all jobs to complete.
3. Start the service again.
4. Verify the service is running.

## Configuration Table

| Parameter | Default | Description |
|-----------|---------|-------------|
| timeout   | 30      | Connection timeout in seconds |
| retries   | 3       | Number of retry attempts |
| buffer    | 1024    | Buffer size in bytes |
"""


@pytest.fixture
def sample_error_doc() -> str:
    """Sample error documentation for testing."""
    return """
AWS001E - Connection Failed

Cause: The system could not establish a connection to the server.
This may be due to network issues or firewall restrictions.

Solution: Check your network settings and try again.
If the problem persists, contact your system administrator.

User Response: Verify network connectivity and firewall rules.
"""


@pytest.fixture
def sample_procedure() -> str:
    """Sample procedure for testing."""
    return """
How to Configure the Service

1. Open the configuration file.
2. Set the required parameters.
3. Save the file.
4. Restart the service.
"""


# =============================================================================
# DECISION #3: STRUCTURE-AWARE OVERLAP TESTS
# =============================================================================


class TestStructureAwareOverlap:
    """Tests for structure-aware overlap (Decision #3)."""

    def test_overlap_strategy_none(self):
        """Test that NONE strategy returns empty string."""
        text = "This is a test paragraph.\n\nThis is another paragraph."
        result = get_structure_aware_overlap(text, 50, OverlapStrategy.NONE)
        assert result == ""

    def test_overlap_strategy_constant(self):
        """Test that CONSTANT strategy returns last sentences."""
        text = "First sentence. Second sentence. Third sentence."
        result = get_structure_aware_overlap(text, 10, OverlapStrategy.CONSTANT)
        # Should return some text
        assert len(result) > 0

    def test_overlap_strategy_structure(self):
        """Test that STRUCTURE strategy returns paragraph-level overlap."""
        text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
        result = get_structure_aware_overlap(text, 50, OverlapStrategy.STRUCTURE)
        # Should return at least one paragraph
        assert len(result) > 0

    def test_chunker_uses_overlap_strategy(self, sample_text):
        """Test that AdvancedChunker respects overlap strategy."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            max_tokens=200,
            overlap_tokens=50,
            overlap_strategy=OverlapStrategy.STRUCTURE,
        )
        chunker = AdvancedChunker(config)
        chunks = chunker.chunk_document(sample_text, source="test.md")

        assert len(chunks) > 0
        # Verify chunks have proper structure
        for chunk in chunks:
            assert chunk.content.strip() != ""


# =============================================================================
# DECISION #7: CITATION-FRIENDLY IDS TESTS
# =============================================================================


class TestCitationFriendlyIDs:
    """Tests for citation-friendly IDs (Decision #7)."""

    def test_generate_stable_id(self):
        """Test stable ID generation."""
        stable_id = generate_stable_id(
            doc_id="manual_v2",
            section_path="Troubleshooting > Error Codes",
            chunk_index=1,
        )
        assert "manual_v2" in stable_id
        assert "troubleshooting_error-codes" in stable_id
        assert "000001" in stable_id

    def test_generate_stable_id_empty_section(self):
        """Test stable ID with empty section."""
        stable_id = generate_stable_id(
            doc_id="doc123",
            section_path="",
            chunk_index=0,
        )
        assert "doc123" in stable_id
        assert "root" in stable_id

    def test_generate_snippet_preview(self):
        """Test snippet preview generation."""
        content = "This is a long piece of content that should be truncated at a word boundary."
        preview = generate_snippet_preview(content, max_chars=30)
        assert len(preview) <= 33  # max_chars + "..."
        assert preview.endswith("...")

    def test_snippet_preview_short_content(self):
        """Test snippet preview with short content."""
        content = "Short content."
        preview = generate_snippet_preview(content, max_chars=100)
        assert preview == content

    def test_chunk_has_stable_id(self, sample_text):
        """Test that chunks get stable IDs."""
        config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE_AWARE)
        chunker = AdvancedChunker(config)
        chunks = chunker.chunk_document(
            sample_text,
            source="test.md",
            doc_id="test_doc",
        )

        for chunk in chunks:
            assert chunk.metadata.stable_id != ""
            assert "test_doc" in chunk.metadata.stable_id

    def test_chunk_has_snippet_preview(self, sample_text):
        """Test that chunks get snippet previews."""
        config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE_AWARE)
        chunker = AdvancedChunker(config)
        chunks = chunker.chunk_document(sample_text, source="test.md")

        for chunk in chunks:
            assert chunk.metadata.snippet_preview != ""


# =============================================================================
# DECISION #8: MULTI-VIEW CHUNK INDEXING TESTS
# =============================================================================


class TestMultiViewChunking:
    """Tests for multi-view chunk indexing (Decision #8)."""

    def test_multi_view_chunk_creation(self, sample_error_doc):
        """Test creating a multi-view chunk."""
        # First create a regular chunk
        config = ChunkingConfig()
        chunker = AdvancedChunker(config)
        chunks = chunker.chunk_document(sample_error_doc, source="error.md")

        if chunks:
            mvc = create_multi_view_chunk(chunks[0], "doc123")
            assert mvc.chunk_id != ""
            assert mvc.raw_content != ""
            assert mvc.summary_view != ""
            assert mvc.entities_view != ""

    def test_entities_view_extraction(self, sample_error_doc):
        """Test entities view extraction."""
        config = ChunkingConfig(extract_error_codes=True)
        chunker = AdvancedChunker(config)
        chunks = chunker.chunk_document(sample_error_doc, source="error.md")

        if chunks and chunks[0].metadata.error_codes:
            entities_view = generate_entities_view(
                chunks[0].content,
                chunks[0].metadata,
            )
            assert "AWS001E" in entities_view or "Error codes" in entities_view

    def test_faq_view_generation_error_doc(self, sample_error_doc):
        """Test FAQ view generation for error documentation."""
        config = ChunkingConfig()
        chunker = AdvancedChunker(config)
        chunks = chunker.chunk_document(sample_error_doc, source="error.md")

        if chunks:
            faq_view = generate_faq_view(
                chunks[0].content,
                chunks[0].metadata.section_path,
                chunks[0].metadata.chunk_type,
            )
            # Should generate FAQ for error docs
            if chunks[0].metadata.chunk_type == ChunkType.ERROR_DOC:
                assert faq_view is not None
                assert "Q:" in faq_view
                assert "A:" in faq_view

    def test_faq_view_generation_procedure(self, sample_procedure):
        """Test FAQ view generation for procedures."""
        faq_view = generate_faq_view(
            sample_procedure,
            "Configuration",
            ChunkType.PROCEDURE,
        )
        assert faq_view is not None
        assert "Q:" in faq_view
        assert "A:" in faq_view

    def test_multi_view_disabled(self, sample_text):
        """Test that multi-view can be disabled."""
        config = ChunkingConfig(enable_multi_view=False)
        chunker = AdvancedChunker(config)
        # Should still work for regular chunking
        chunks = chunker.chunk_document(sample_text, source="test.md")
        assert len(chunks) > 0

    def test_get_view_method(self):
        """Test MultiViewChunk.get_view() method."""
        mvc = MultiViewChunk(
            chunk_id="test#001",
            doc_id="test",
            raw_content="Raw content",
            summary_view="Summary content",
            entities_view="Entity1, Entity2",
            faq_view="Q: Test?\nA: Answer.",
        )

        assert mvc.get_view(ChunkViewType.RAW) == "Raw content"
        assert mvc.get_view(ChunkViewType.SUMMARY) == "Summary content"
        assert mvc.get_view(ChunkViewType.ENTITIES) == "Entity1, Entity2"
        assert mvc.get_view(ChunkViewType.FAQ) == "Q: Test?\nA: Answer."


# =============================================================================
# DECISION #9: FAILURE-SLICE EVAL PIPELINE TESTS
# =============================================================================


class TestFailureSliceEval:
    """Tests for failure-slice eval pipeline (Decision #9)."""

    def test_detect_missing_exception(self):
        """Test detection of missing exception failure."""
        query = "What is the timeout setting?"
        retrieved = [
            RetrievedChunk(
                chunk_id="c1",
                content="The timeout is 30 seconds.",
                score=0.9,
                rank=1,
            )
        ]
        expected = "The timeout is 30 seconds unless overridden in config."
        relevant_ids = ["c2"]  # The chunk with exception is not retrieved

        failure_slice, _ = detect_failure_slice(
            query, retrieved, expected, relevant_ids
        )
        # Should detect some failure since relevant chunk wasn't retrieved
        assert failure_slice != FailureSlice.UNKNOWN or not relevant_ids

    def test_detect_lost_table_header(self):
        """Test detection of lost table header failure."""
        query = "What are the configuration options?"
        retrieved = [
            RetrievedChunk(
                chunk_id="c1",
                content="| timeout | 30 | seconds |\n| retries | 3 | count |",
                score=0.9,
                rank=1,
            )
        ]
        expected = "Configuration table with headers"
        relevant_ids = ["c_header"]

        failure_slice, _ = detect_failure_slice(
            query, retrieved, expected, relevant_ids
        )
        # Table without header should trigger LOST_TABLE_HEADER
        assert failure_slice in [FailureSlice.LOST_TABLE_HEADER, FailureSlice.UNKNOWN]

    def test_detect_redundant_overlaps(self):
        """Test detection of redundant overlaps failure."""
        query = "What is the timeout?"
        # Two chunks with high overlap
        retrieved = [
            RetrievedChunk(
                chunk_id="c1",
                content="The timeout setting controls how long to wait. The default is 30 seconds.",
                score=0.95,
                rank=1,
            ),
            RetrievedChunk(
                chunk_id="c2",
                content="The timeout setting controls how long to wait. The default is 30 seconds for most cases.",
                score=0.93,
                rank=2,
            ),
            RetrievedChunk(
                chunk_id="c3",
                content="The timeout setting controls how long to wait. The default is 30 seconds for all connections.",
                score=0.91,
                rank=3,
            ),
        ]
        expected = "Timeout is 30 seconds"
        relevant_ids = ["c1"]

        failure_slice, _ = detect_failure_slice(
            query, retrieved, expected, relevant_ids
        )
        # High overlap should trigger REDUNDANT_OVERLAPS
        assert failure_slice in [FailureSlice.REDUNDANT_OVERLAPS, FailureSlice.UNKNOWN]

    def test_generate_rule_suggestions(self):
        """Test rule suggestion generation."""
        results = [
            EvalResult(
                query_id="q1",
                query_text="Test query",
                expected_answer="Expected answer",
                failure_slice=FailureSlice.MISSING_EXCEPTION,
                failure_description="Missing exception",
                relevant_chunk_ids=["c1"],
                retrieved_relevant=False,
            ),
            EvalResult(
                query_id="q2",
                query_text="Another query",
                expected_answer="Another expected",
                failure_slice=FailureSlice.MISSING_EXCEPTION,
                failure_description="Missing exception again",
                relevant_chunk_ids=["c2"],
                retrieved_relevant=False,
            ),
        ]

        suggestions = generate_rule_suggestions(results)
        assert len(suggestions) > 0
        assert any(s.failure_slice == FailureSlice.MISSING_EXCEPTION for s in suggestions)

    def test_eval_pipeline_creation(self):
        """Test eval pipeline creation."""
        pipeline = ChunkingEvalPipeline(top_k=5)
        assert pipeline.top_k == 5

    def test_create_eval_query(self):
        """Test eval query creation helper."""
        query = create_eval_query(
            query_id="q1",
            query_text="What is the error?",
            expected_answer="The error means X",
            relevant_chunk_ids=["c1", "c2"],
        )
        assert query["id"] == "q1"
        assert query["text"] == "What is the error?"
        assert query["expected"] == "The error means X"
        assert query["relevant_ids"] == ["c1", "c2"]

    def test_eval_result_to_dict(self):
        """Test EvalResult serialization."""
        result = EvalResult(
            query_id="q1",
            query_text="Test",
            expected_answer="Expected",
            failure_slice=FailureSlice.MISSING_EXCEPTION,
            failure_severity=FailureSeverity.HIGH,
        )
        d = result.to_dict()
        assert d["query_id"] == "q1"
        assert d["failure_slice"] == "missing_exception"
        assert d["failure_severity"] == "high"

    def test_eval_report_to_dict(self):
        """Test EvalReport serialization."""
        report = EvalReport(
            total_queries=10,
            successful_queries=7,
            failed_queries=3,
            avg_recall=0.7,
            avg_mrr=0.5,
        )
        d = report.to_dict()
        assert d["total_queries"] == 10
        assert d["successful_queries"] == 7
        assert d["failed_queries"] == 3


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for all chunking improvements."""

    def test_full_chunking_pipeline(self, sample_text):
        """Test the full chunking pipeline with all features."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.TWS_OPTIMIZED,
            max_tokens=500,
            overlap_tokens=75,
            overlap_strategy=OverlapStrategy.STRUCTURE,
            enable_multi_view=True,
        )
        chunker = AdvancedChunker(config)

        # Chunk document
        chunks = chunker.chunk_document(
            sample_text,
            source="test.md",
            document_title="Test Document",
            doc_id="test_doc_001",
        )

        assert len(chunks) > 0

        # Verify all chunks have citation-friendly fields
        for chunk in chunks:
            assert chunk.metadata.stable_id != ""
            assert chunk.metadata.snippet_preview != ""
            assert chunk.metadata.chunk_type in [
                ChunkType.TEXT,
                ChunkType.CODE,
                ChunkType.TABLE,
                ChunkType.ERROR_DOC,
                ChunkType.PROCEDURE,
                ChunkType.HEADER,
                ChunkType.DEFINITION,
            ]

    def test_multi_view_pipeline(self, sample_text):
        """Test multi-view chunk generation."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            enable_multi_view=True,
        )
        chunker = AdvancedChunker(config)

        multi_view_chunks = chunker.chunk_document_multi_view(
            sample_text,
            source="test.md",
            document_title="Test",
            doc_id="test_doc",
        )

        assert len(multi_view_chunks) > 0

        for mvc in multi_view_chunks:
            assert mvc.raw_content != ""
            assert mvc.summary_view != ""
            assert mvc.entities_view != ""


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_chunk_document_function(self, sample_text):
        """Test that chunk_document convenience function still works."""
        chunks = chunk_document(
            sample_text,
            source="test.md",
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
        )
        assert len(chunks) > 0

    def test_default_config(self):
        """Test that default config is valid."""
        config = ChunkingConfig()
        assert config.strategy == ChunkingStrategy.TWS_OPTIMIZED
        assert config.overlap_strategy == OverlapStrategy.STRUCTURE
        assert config.enable_multi_view is True

    def test_metadata_to_dict_includes_new_fields(self):
        """Test that ChunkMetadata.to_dict includes new fields."""
        metadata = ChunkMetadata(
            source_file="test.md",
            stable_id="test::section::000001",
            snippet_preview="This is a preview...",
        )
        d = metadata.to_dict()
        assert "stable_id" in d
        assert "snippet_preview" in d
        assert d["stable_id"] == "test::section::000001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
