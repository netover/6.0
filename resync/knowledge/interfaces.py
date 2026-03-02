"""
Protocols for RAG system components.

Defines async interfaces for Embedder, VectorStore, and Retriever
to enable dependency injection, testing, and proper async/await usage
with FastAPI, asyncpg, and the async stack.

All protocols use async methods to prevent event loop blocking.
Compatible with Python 3.14 (PEP 649 lazy annotations are the default).

Version: 6.1.0 — Added TypeAlias annotations, runtime async validator.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable
import inspect

# ---------------------------------------------------------------------------
# Type aliases — explicit TypeAlias annotation for mypy/pyright clarity
# ---------------------------------------------------------------------------

#: A single embedding vector (one row returned by an Embedder).
EmbeddingVector: TypeAlias = list[float]

#: A batch of embedding vectors (shape: batch_size × embedding_dim).
EmbeddingBatch: TypeAlias = list[list[float]]

#: Arbitrary key-value metadata payload attached to a stored vector.
DocumentPayload: TypeAlias = Mapping[str, Any]

#: Filter expression for vector store queries (field → value).
SearchFilters: TypeAlias = Mapping[str, Any]

# ---------------------------------------------------------------------------
# Runtime async-protocol validator
# ---------------------------------------------------------------------------

def verify_async_protocol(obj: object, protocol_cls: type) -> bool:
    """
    Verify that *obj* implements *protocol_cls* **and** that all protocol
    methods are coroutine functions on *obj*.

    ``@runtime_checkable`` only checks method *presence*, not whether the
    method is async.  This helper performs the additional async check.

    Args:
        obj:          Object to verify.
        protocol_cls: A ``@runtime_checkable`` Protocol class.

    Returns:
        ``True`` if *obj* passes both ``isinstance`` and the async check.

    Example::

        class MyEmbedder:
            async def embed(self, text: str, *, timeout: float = 30.0) -> list[float]:
                ...

        assert verify_async_protocol(MyEmbedder(), Embedder)
    """
    if not isinstance(obj, protocol_cls):
        return False

    for name in vars(protocol_cls):
        if name.startswith("_"):
            continue
        proto_method = getattr(protocol_cls, name, None)
        if proto_method is None or not callable(proto_method):
            continue
        impl_method = getattr(obj, name, None)
        if impl_method is None:
            return False
        if not inspect.iscoroutinefunction(impl_method):
            return False

    return True

# ---------------------------------------------------------------------------
# Embedder Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    """
    Async protocol for embedding text into dense vectors.

    All methods are coroutines to prevent event-loop blocking when
    calling CPU-bound local models (via ``asyncio.to_thread``) or
    remote embedding APIs (OpenAI, Cohere, etc.).

    Implementations must provide:
    - Single-text embedding with configurable timeout.
    - Batch embedding with automatic sub-batching and rate-limiting.
    - Proper ``asyncio.TimeoutError`` propagation.
    """

    async def embed(
        self,
        text: str,
        *,
        timeout: float = 30.0,
    ) -> EmbeddingVector:
        """
        Embed a single text string into a dense vector.

        Args:
            text:    Input text to embed.  Must be non-empty.
            timeout: Maximum wall-clock time in seconds.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            asyncio.TimeoutError: Embedding exceeded *timeout*.
            ValueError: *text* is empty or exceeds model token limit.
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
        Embed multiple texts with automatic sub-batching.

        Args:
            texts:      Texts to embed.  Each must be non-empty.
            batch_size: Maximum texts per API call.
            timeout:    Maximum wall-clock time for the entire batch.

        Returns:
            List of embedding vectors, one per input text,
            in the same order as *texts*.

        Raises:
            asyncio.TimeoutError: Batch exceeded *timeout*.
            ValueError: Any text is empty or exceeds model token limit.
        """
        ...

# ---------------------------------------------------------------------------
# VectorStore Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorStore(Protocol):
    """
    Async protocol for storing and querying dense vector embeddings.

    All methods are coroutines for integration with asyncpg / psycopg3
    async connection pools and pgvector HNSW index operations.
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
        Insert or update vectors and their metadata.

        Args:
            ids:        Unique string identifiers (one per vector).
            vectors:    Embedding vectors (must match len(ids)).
            payloads:   Metadata dicts (must match len(ids)).
            collection: Collection name; uses implementation default if ``None``.
            timeout:    Maximum time for the entire batch operation.

        Raises:
            asyncio.TimeoutError: Upsert exceeded *timeout*.
            ValueError: ``len(ids) != len(vectors)`` or ``!= len(payloads)``.
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
        Find the *top_k* most similar vectors.

        Args:
            vector:      Query embedding vector.
            top_k:       Number of results to return (must be ≥ 1).
            collection:  Collection to search; uses default if ``None``.
            filters:     Metadata filter predicates (implementation-defined).
            ef_search:   HNSW ``ef_search`` tuning parameter.
            with_vectors: Include the stored embedding in each result.
            timeout:     Maximum query time in seconds.

        Returns:
            List of result dicts with at least ``id``, ``score``,
            and ``payload`` keys.

        Raises:
            asyncio.TimeoutError: Query exceeded *timeout*.
            ValueError: ``top_k < 1``.
        """
        ...

    async def count(
        self,
        collection: str | None = None,
        *,
        timeout: float = 5.0,
    ) -> int:
        """
        Return the number of vectors in *collection*.

        Args:
            collection: Collection to count; uses default if ``None``.
            timeout:    Maximum time in seconds.

        Returns:
            Non-negative integer vector count.
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
        Check whether a document identified by SHA-256 hash exists.

        Args:
            sha256:     Hex-encoded SHA-256 digest of the source document.
            collection: Collection to check; uses default if ``None``.
            timeout:    Maximum time in seconds.

        Returns:
            ``True`` if the document exists, ``False`` otherwise.
        """
        ...

    async def get_all_documents(
        self,
        collection: str | None = None,
        limit: int = 1_000,
        *,
        offset: int = 0,
        timeout: float = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Paginate through all documents in a collection.

        Use ``limit`` / ``offset`` to avoid OOM on large collections.
        Maximum recommended ``limit`` is 10 000 per call.

        Args:
            collection: Collection name; uses default if ``None``.
            limit:      Maximum documents per page (must be ≤ 100 000).
            offset:     Number of documents to skip for pagination.
            timeout:    Maximum time for the request.

        Returns:
            List of document dicts with ``id``, ``payload``, and
            optionally ``vector`` keys.

        Raises:
            ValueError: ``limit > 100_000``.
        """
        ...

    async def delete_by_doc_id(
        self,
        doc_id: str,
        collection: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> int:
        """
        Delete all vectors whose payload contains *doc_id*.

        Args:
            doc_id:     Source document identifier.
            collection: Target collection; uses default if ``None``.
            timeout:    Maximum time for deletion.

        Returns:
            Number of vectors deleted (0 if none matched).
        """
        ...

    async def exists_batch_by_sha256(
        self,
        sha256_list: Sequence[str],
        collection: str | None = None,
        *,
        timeout: float = 10.0,
    ) -> set[str]:
        """
        Batch check which SHA-256 hashes exist in *collection*.

        More efficient than *N* sequential :meth:`exists_by_sha256` calls.

        Args:
            sha256_list: SHA-256 digests to check.
            collection:  Collection to check; uses default if ``None``.
            timeout:     Maximum time for the batch check.

        Returns:
            Set of digests that exist in the collection.
        """
        ...

# ---------------------------------------------------------------------------
# Retriever Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Retriever(Protocol):
    """
    Async protocol for high-level document retrieval.

    Combines embedding, vector search, and optional reranking into a
    single interface.  All methods are coroutines for integration with
    FastAPI dependency injection and LangChain async tools.
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
        Retrieve the most relevant documents for *query*.

        Pipeline:
        1. Embed *query* text.
        2. Vector similarity search.
        3. Optional reranking (implementation-defined gating).
        4. Return top-*k* results.

        Args:
            query:   Search query string (must be non-empty).
            top_k:   Number of results to return (must be ≥ 1).
            filters: Metadata filter predicates.
            timeout: Maximum wall-clock time for the full pipeline.

        Returns:
            List of result dicts with ``score``, ``payload``, and
            source metadata.

        Raises:
            asyncio.TimeoutError: Retrieval exceeded *timeout*.
            ValueError: *query* is empty, or ``top_k < 1``.
        """
        ...
