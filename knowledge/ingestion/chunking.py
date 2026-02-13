"""
Token-aware text chunking utility for RAG systems.

v5.4.2: Now wraps AdvancedChunker for improved accuracy.
Maintains backward compatibility with original API.

For new code, use advanced_chunking module directly:
    from resync.knowledge.ingestion.advanced_chunking import (
        AdvancedChunker,
        ChunkingConfig,
        ChunkingStrategy,
    )
"""

from __future__ import annotations

import re
from collections.abc import Iterator

# Import from advanced chunking
from resync.knowledge.ingestion.advanced_chunking import (
    AdvancedChunker,
    ChunkingConfig,
    ChunkingStrategy,
    EnrichedChunk,
    chunk_text_simple,
    count_tokens,
)

# Optional import for direct token operations
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False
    _ENC = None


def _tokens_len(s: str) -> int:
    """Estimate token count for a string."""
    return count_tokens(s)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences based on punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.split(r"(?<=[.!?])\s+", text) if text else []


def chunk_text(text: str, max_tokens: int = 512, overlap_tokens: int = 64) -> Iterator[str]:
    """
    Chunk text in a token-aware way, respecting sentence boundaries and applying overlap.

    v5.4.2: Now uses AdvancedChunker with recursive strategy for improved quality.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks

    Yields:
        Chunk strings
    """
    if not text:
        return iter(())

    # Use the new advanced chunker with recursive strategy
    yield from chunk_text_simple(text, max_tokens, overlap_tokens)


def chunk_text_advanced(
    text: str,
    source: str = "",
    document_title: str = "",
    strategy: str = "tws_optimized",
    max_tokens: int = 500,
    overlap_tokens: int = 75,
) -> list[EnrichedChunk]:
    """
    Advanced chunking with rich metadata and contextual enrichment.

    Args:
        text: Document text
        source: Source filename
        document_title: Document title
        strategy: One of 'fixed_size', 'recursive', 'semantic',
                  'structure_aware', 'hierarchical', 'tws_optimized'
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap

    Returns:
        List of EnrichedChunk objects with content and metadata
    """
    strategy_map = {
        "fixed_size": ChunkingStrategy.FIXED_SIZE,
        "recursive": ChunkingStrategy.RECURSIVE,
        "semantic": ChunkingStrategy.SEMANTIC,
        "structure_aware": ChunkingStrategy.STRUCTURE_AWARE,
        "hierarchical": ChunkingStrategy.HIERARCHICAL,
        "tws_optimized": ChunkingStrategy.TWS_OPTIMIZED,
    }

    config = ChunkingConfig(
        strategy=strategy_map.get(strategy, ChunkingStrategy.TWS_OPTIMIZED),
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    chunker = AdvancedChunker(config)
    return chunker.chunk_document(text, source, document_title)


# Export for backward compatibility
__all__ = [
    "chunk_text",
    "chunk_text_advanced",
    "count_tokens",
    "AdvancedChunker",
    "ChunkingConfig",
    "ChunkingStrategy",
    "EnrichedChunk",
]
