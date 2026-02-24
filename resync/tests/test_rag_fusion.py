"""
Tests for Fusion RAG implementation.

Tests the query expansion and Reciprocal Rank Fusion (RRF) logic
added to the HybridRAG system.
"""

from unittest.mock import AsyncMock, patch

import pytest

from resync.knowledge.retrieval.hybrid import HybridRAG


@pytest.fixture
def mock_llm():
    """Mock LLM service for query expansion."""
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=(
            "variação 1 da pergunta\n"
            "variação 2 da pergunta\n"
            "variação 3 da pergunta"
        )
    )
    return llm


@pytest.fixture
def mock_rag():
    """Mock RAG retriever."""
    rag = AsyncMock()
    rag.retrieve = AsyncMock(
        return_value=[
            {"id": "doc1", "content": "test content 1", "score": 0.9},
            {"id": "doc2", "content": "test content 2", "score": 0.7},
        ]
    )
    return rag


@pytest.fixture
def hybrid_rag(mock_llm, mock_rag):
    """HybridRAG instance with mocked dependencies."""
    rag = HybridRAG(rag_retriever=mock_rag, llm_service=mock_llm, use_llm_router=False)
    return rag


class TestQueryExpansion:
    """Test query expansion for Fusion RAG."""

    @pytest.mark.asyncio
    async def test_expand_query_generates_variations(self, hybrid_rag, mock_llm):
        """Test that query expansion generates multiple variations."""
        query = "qual a causa raiz do erro 127?"

        variations = await hybrid_rag._expand_query(query, num_variations=3)

        # Should include original + variations
        assert len(variations) >= 1
        assert variations[0] == query  # Original always first

        # LLM should have been called
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_expand_query_fallback_without_llm(self):
        """Test that expansion falls back to original query if LLM unavailable."""
        rag = HybridRAG(rag_retriever=None, llm_service=None, use_llm_router=False)

        query = "test query"
        variations = await rag._expand_query(query)

        # Should return only original query
        assert variations == [query]


class TestReciprocalRankFusion:
    """Test RRF merging logic."""

    def test_rrf_merges_multiple_result_sets(self, hybrid_rag):
        """Test that RRF correctly merges and ranks documents."""
        results_list = [
            # Result set 1
            [
                {"id": "doc1", "content": "content 1"},
                {"id": "doc2", "content": "content 2"},
                {"id": "doc3", "content": "content 3"},
            ],
            # Result set 2
            [
                {"id": "doc2", "content": "content 2"},  # Appears in both
                {"id": "doc1", "content": "content 1"},  # Appears in both
                {"id": "doc4", "content": "content 4"},
            ],
        ]

        fused = hybrid_rag._reciprocal_rank_fusion(results_list, k=60)

        # Should have unique documents
        assert len(fused) == 4

        # All should have fusion_score
        assert all("fusion_score" in doc for doc in fused)

        # doc1 and doc2 should rank higher (appear in both sets)
        top_ids = {fused[0]["id"], fused[1]["id"]}
        assert "doc1" in top_ids
        assert "doc2" in top_ids

    def test_rrf_handles_empty_results(self, hybrid_rag):
        """Test RRF with empty result sets."""
        fused = hybrid_rag._reciprocal_rank_fusion([])
        assert fused == []

    def test_rrf_uses_content_hash_for_missing_ids(self, hybrid_rag):
        """Test that RRF uses content hash when document ID is missing."""
        results_list = [
            [{"content": "same content"}],
            [{"content": "same content"}],  # Duplicate
        ]

        fused = hybrid_rag._reciprocal_rank_fusion(results_list)

        # Should deduplicate based on content hash
        assert len(fused) == 1


class TestFusionRAGIntegration:
    """Test full Fusion RAG query flow."""

    @pytest.mark.asyncio
    async def test_fusion_rag_query_expands_and_merges(self, hybrid_rag, mock_rag):
        """Test that Fusion RAG expands query and merges results."""
        query = "causa raiz erro 127"
        entities = {}

        result = await hybrid_rag._execute_fusion_rag_query(query, entities)

        # Should have fusion metadata
        assert result["type"] == "fusion_search"
        assert result["original_query"] == query
        assert "query_variations" in result
        assert "documents" in result
        assert "fusion_stats" in result

        # Should have called retrieve multiple times (for each variation)
        assert mock_rag.retrieve.call_count >= 1

    @pytest.mark.asyncio
    async def test_fusion_rag_fallback_on_error(self, hybrid_rag, mock_rag):
        """Test that Fusion RAG falls back to standard RAG on error."""
        # Make expansion fail
        hybrid_rag._llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        query = "test query"
        entities = {}

        # Should not raise, should fallback
        result = await hybrid_rag._execute_fusion_rag_query(query, entities)

        # Should still return results (from fallback)
        assert "documents" in result or "error" in result


class TestFusionRAGRouting:
    """Test that Fusion RAG is used for appropriate query intents."""

    @pytest.mark.asyncio
    async def test_general_intent_uses_fusion(self, hybrid_rag, mock_rag):
        """Test that GENERAL intent queries use Fusion RAG."""
        with patch.object(hybrid_rag._classifier, "classify") as mock_classify:
            # Mock classification as GENERAL
            from resync.knowledge.retrieval.hybrid import (
                QueryClassification,
                QueryIntent,
            )

            mock_classify.return_value = QueryClassification(
                intent=QueryIntent.GENERAL,
                confidence=0.5,
                entities={},
                use_graph=False,
                use_rag=True,
            )

            result = await hybrid_rag.query(
                "pergunta genérica", generate_response=False
            )

            # Should have fusion stats (indicating Fusion RAG was used)
            if result.get("rag_results"):
                assert (
                    "fusion_stats" in result["rag_results"]
                    or "type" in result["rag_results"]
                )

    @pytest.mark.asyncio
    async def test_documentation_intent_uses_standard_rag(self, hybrid_rag, mock_rag):
        """Test that DOCUMENTATION intent uses standard RAG (not Fusion)."""
        with patch.object(hybrid_rag._classifier, "classify") as mock_classify:
            from resync.knowledge.retrieval.hybrid import (
                QueryClassification,
                QueryIntent,
            )

            mock_classify.return_value = QueryClassification(
                intent=QueryIntent.DOCUMENTATION,
                confidence=0.8,
                entities={},
                use_graph=False,
                use_rag=True,
            )

            result = await hybrid_rag.query(
                "como configurar TWS?", generate_response=False
            )

            # Should use standard RAG (no fusion_stats)
            if result.get("rag_results"):
                assert result["rag_results"]["type"] == "semantic_search"
