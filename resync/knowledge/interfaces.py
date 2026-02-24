"""
Protocols for RAG system components.

Defines async interfaces for Embedder, VectorStore, and Retriever
to enable dependency injection, testing, and proper async/await usage
with FastAPI, asyncpg, and the async stack.

All protocols use async methods to prevent event loop blocking.
Compatible with Python 3.14 async improvements (PEP 701, 688, 692).

Version: 6.0.0 - Converted to async protocols
"""

from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


# Type aliases for clarity
EmbeddingVector = list[float]
EmbeddingBatch = list[list[float]]
DocumentPayload = Mapping[str, Any]
SearchFilters = Mapping[str, Any]


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol for async embedding text into vectors.

    All methods are async to prevent event loop blocking when calling
    external embedding APIs (OpenAI, Cohere, local models via HTTP).

    Implementations must support:
    - Single text embedding with timeout
    - Batch embedding with automatic batching and rate limiting
    - Proper error handling and retries
    """

    async def embed(
        self,
        text: str,
        *,
        timeout: float = 30.0,
    ) -> EmbeddingVector:
        """
        Embed single text into vector.

        Args:
            text: Input text to embed
            timeout: Maximum time to wait for embedding (seconds)

        Returns:
            Embedding vector as list of floats

        Raises:
            asyncio.TimeoutError: If embedding exceeds timeout
            ValueError: If text is empty or too long
        """
        ...

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> EmbeddingBatch:
        """
        Embed multiple texts into vectors with automatic batching.

        Args:
            texts: Sequence of texts to embed
            batch_size: Maximum texts to embed per API call
            timeout: Maximum time to wait for entire batch (seconds)

        Returns:
            Embedding vectors as list of lists (batch_size, embedding_dim)

        Raises:
            asyncio.TimeoutError: If embedding exceeds timeout
            ValueError: If any text is empty or too long
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """
    Protocol for async storing and retrieving vector embeddings with metadata.

    All methods are async for integration with asyncpg/psycopg3 async drivers.
    Supports pgvector HNSW index operations with proper connection pooling.
    """

    async def upsert_batch(
        self,
        ids: Sequence[str],
        vectors: EmbeddingBatch,
        payloads: Sequence[DocumentPayload],
        *,
        collection: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Upsert vectors and metadata into vector store.

        Args:
            ids: Unique identifiers for vectors
            vectors: Embedding vectors (batch_size, embedding_dim)
            payloads: Metadata for each vector
            collection: Target collection name (uses default if None)
            timeout: Maximum time for batch upsert (seconds)

        Raises:
            asyncio.TimeoutError: If upsert exceeds timeout
            ValueError: If ids/vectors/payloads lengths don't match
            asyncpg.PostgresError: On database errors
        """
        ...

    async def query(
        self,
        vector: EmbeddingVector,
        top_k: int,
        *,
        collection: str | None = None,
        filters: SearchFilters | None = None,
        ef_search: int | None = None,
        with_vectors: bool = False,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        """
        Query vector store for similar vectors.

        Args:
            vector: Query embedding vector
            top_k: Number of results to return
            collection: Target collection name (uses default if None)
            filters: Metadata filters for results
            ef_search: HNSW ef_search parameter (uses default if None)
            with_vectors: Include embedding vectors in results
            timeout: Maximum time for query (seconds)

        Returns:
            List of results with scores and metadata

        Raises:
            asyncio.TimeoutError: If query exceeds timeout
            ValueError: If top_k < 1 or > max_top_k
            asyncpg.PostgresError: On database errors
        """
        ...

    async def count(
        self,
        collection: str | None = None,
        *,
        timeout: float = 5.0,
    ) -> int:
        """
        Count vectors in collection.

        Args:
            collection: Target collection name (uses default if None)
            timeout: Maximum time for count (seconds)

        Returns:
            Number of vectors in collection

        Raises:
            asyncio.TimeoutError: If count exceeds timeout
        """
        ...

    async def exists_by_sha256(
        self,
        sha256: str,
        collection: str | None = None,
        *,
        timeout: float = 5.0,
    ) -> bool:
        """
        Check if document with SHA256 exists in collection.

        Args:
            sha256: Document SHA256 hash
            collection: Target collection name (uses default if None)
            timeout: Maximum time for check (seconds)

        Returns:
            True if document exists, False otherwise

        Raises:
            asyncio.TimeoutError: If check exceeds timeout
        """
        ...

    async def get_all_documents(
        self,
        collection: str | None = None,
        limit: int = 10000,
        *,
        offset: int = 0,
        timeout: float = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Get all documents from collection (for BM25 index building).

        Use pagination (limit/offset) to avoid memory exhaustion.

        Args:
            collection: Target collection name (uses default if None)
            limit: Maximum documents to return
            offset: Number of documents to skip
            timeout: Maximum time for query (seconds)

        Returns:
            List of documents with metadata

        Raises:
            asyncio.TimeoutError: If query exceeds timeout
            ValueError: If limit > 100000
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """
    Protocol for async retrieving relevant documents based on query.

    High-level interface that combines embedding, vector search, and reranking.
    All methods are async for integration with FastAPI and LangChain async tools.
    """

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: SearchFilters | None = None,
        timeout: float = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for query.

        Pipeline:
        1. Embed query text
        2. Vector similarity search
        3. Optional reranking (if gating conditions met)
        4. Return top-k results

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters for results
            timeout: Maximum time for entire retrieval pipeline (seconds)

        Returns:
            List of results with scores and metadata

        Raises:
            asyncio.TimeoutError: If retrieval exceeds timeout
            ValueError: If query empty or top_k invalid
        """
        ...
