"""
Protocols for RAG system components.

Defines interfaces for Embedder, VectorStore, and Retriever to enable dependency injection and testing.
"""

from __future__ import annotations

from typing import Any, Protocol


# pylint: disable=too-few-public-methods
class Embedder(Protocol):
    """
    Protocol for embedding text into vectors.
    """

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# pylint: disable=too-few-public-methods
class VectorStore(Protocol):
    """
    Protocol for storing and retrieving vector embeddings with metadata.
    """

    def upsert_batch(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        collection: str | None = None,
    ) -> None: ...
    def query(
        self,
        vector: list[float],
        top_k: int,
        collection: str | None = None,
        filters: dict[str, Any] | None = None,
        ef_search: int | None = None,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]: ...
    def count(self, collection: str | None = None) -> int: ...
    def exists_by_sha256(self, sha256: str, collection: str | None = None) -> bool: ...
    # v5.4.0: Added for hybrid retrieval BM25 index building
    def get_all_documents(
        self, collection: str | None = None, limit: int = 10000
    ) -> list[dict[str, Any]]: ...


# pylint: disable=too-few-public-methods
class Retriever(Protocol):
    """
    Protocol for retrieving relevant documents based on a query.
    """

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...
