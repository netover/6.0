# pylint: disable=missing-class-docstring
# mypy: ignore-errors
"""
Embedding Provider - Generate embeddings for vector search.

Provides a unified interface for generating text embeddings using various
providers (OpenAI, NVIDIA, local models) via LiteLLM.

Usage:
    from resync.knowledge.ingestion.embeddings import get_embedding_provider

    provider = get_embedding_provider()
    embedding = await provider.embed("Hello world")
    embeddings = await provider.embed_batch(["Hello", "World"])

CORRIGIDO (P1): Todos os logger calls migrados de extra={} para kwargs diretos structlog.
"""

from __future__ import annotations

import asyncio
import os
import structlog
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

logger = structlog.get_logger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider."""

    model: str = "text-embedding-ada-002"
    dimension: int = 1536
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers — ASYNC interface."""

    @abstractmethod
    async def embed(self, text: str, *, timeout: float = 30.0) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            timeout: Maximum time to wait for embedding (seconds)

        Returns:
            Embedding vector as list of floats

        Raises:
            asyncio.TimeoutError: If embedding exceeds timeout
        """

    @abstractmethod
    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed
            batch_size: Maximum texts per API call
            timeout: Maximum time to wait for entire batch (seconds)

        Returns:
            List of embedding vectors
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""

class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using LiteLLM.

    Supports multiple backends:
    - OpenAI: text-embedding-ada-002, text-embedding-3-small/large
    - NVIDIA: NV-Embed, snowflake-arctic-embed
    - Azure OpenAI: azure/text-embedding-ada-002
    - Local: ollama/nomic-embed-text
    """

    MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "nvidia/NV-Embed-QA": 1024,
        "nvidia/snowflake-arctic-embed": 1024,
        "ollama/nomic-embed-text": 768,
        "ollama/mxbai-embed-large": 1024,
    }

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """
        Initialize LiteLLM embedding provider.

        Args:
            config: Embedding configuration
        """
        self._config = config or EmbeddingConfig()
        self._dimension = self.MODEL_DIMENSIONS.get(
            self._config.model, self._config.dimension
        )

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    async def embed(self, text: str, *, timeout: float = 30.0) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            timeout: Maximum wait time in seconds

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_batch([text], timeout=timeout)
        return embeddings[0]

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts using batching.

        Args:
            texts: Texts to embed
            batch_size: Maximum texts to embed per API call
            timeout: Maximum time to wait for entire batch (seconds)

        Returns:
            List of embedding vectors

        Raises:
            asyncio.TimeoutError: If embedding exceeds timeout
        """
        import litellm

        text_list = list(texts)
        all_embeddings: list[list[float]] = []

        for i in range(0, len(text_list), self._config.batch_size):
            batch = text_list[i : i + self._config.batch_size]
            for attempt in range(self._config.max_retries):
                try:
                    response = await asyncio.wait_for(
                        litellm.aembedding(model=self._config.model, input=batch),
                        timeout=self._config.timeout,
                    )
                    batch_embeddings = [item["embedding"] for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except asyncio.TimeoutError:
                    # CORRIGIDO P1: extra={} → kwargs diretos
                    logger.warning(
                        "embedding_timeout",
                        attempt=attempt + 1,
                        batch_size=len(batch),
                        model=self._config.model,
                    )
                    if attempt == self._config.max_retries - 1:
                        raise
                    await asyncio.sleep(2**attempt)
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                    # CORRIGIDO P1: extra={} → kwargs diretos
                    logger.error(
                        "embedding_error",
                        error=str(e),
                        attempt=attempt + 1,
                        model=self._config.model,
                    )
                    if attempt == self._config.max_retries - 1:
                        raise
                    await asyncio.sleep(2**attempt)

        # CORRIGIDO P1: extra={} → kwargs diretos
        logger.debug(
            "embeddings_generated",
            count=len(all_embeddings),
            model=self._config.model,
        )
        return all_embeddings

class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing.

    Generates deterministic pseudo-embeddings based on text hash.
    """

    def __init__(self, dimension: int = 1536) -> None:
        """
        Initialize mock provider.

        Args:
            dimension: Embedding dimension
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    async def embed(self, text: str, *, timeout: float = 30.0) -> list[float]:
        """Generate mock embedding (deterministic via SHA-256 hash)."""
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding: list[float] = []
        idx = 0
        while len(embedding) < self._dimension:
            if idx >= len(hash_bytes):
                hash_bytes = hashlib.sha256(hash_bytes).digest()
                idx = 0
            value = hash_bytes[idx] / 127.5 - 1.0
            embedding.append(value)
            idx += 1
        return embedding[: self._dimension]

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> list[list[float]]:
        """Generate mock embeddings for batch."""
        return [await self.embed(text) for text in texts]

# Singleton state — thread-safe via threading.Lock
_embedding_provider: EmbeddingProvider | None = None
_embedding_provider_lock = __import__("threading").Lock()

def get_embedding_provider() -> EmbeddingProvider:
    """
    Get singleton embedding provider instance (thread-safe).

    Configures provider based on environment variables:
    - EMBEDDING_MODEL: Model to use (default: text-embedding-ada-002)
    - EMBEDDING_DIMENSION: Override dimension
    - EMBEDDING_MOCK: Use mock provider for testing

    Returns:
        Configured embedding provider
    """
    global _embedding_provider

    if _embedding_provider is not None:
        return _embedding_provider

    with _embedding_provider_lock:
        if _embedding_provider is not None:
            return _embedding_provider

        if os.getenv("EMBEDDING_MOCK", "").lower() in ("true", "1", "yes"):
            dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
            _embedding_provider = MockEmbeddingProvider(dimension=dimension)
            # CORRIGIDO P1: extra={} → kwargs diretos
            logger.info("mock_embedding_provider_initialized", dimension=dimension)
            return _embedding_provider

        config = EmbeddingConfig(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
        )
        _embedding_provider = LiteLLMEmbeddingProvider(config)
        # CORRIGIDO P1: extra={} → kwargs diretos
        logger.info(
            "litellm_embedding_provider_initialized",
            model=config.model,
            dimension=_embedding_provider.dimension,
        )
        return _embedding_provider

def set_embedding_provider(provider: EmbeddingProvider) -> None:
    """
    Set custom embedding provider (for testing).

    Args:
        provider: Custom embedding provider instance
    """
    global _embedding_provider
    _embedding_provider = provider

# Alias pointing to pgvector store
def get_vector_service():
    """Return the vector store service (alias for get_vector_store)."""
    from resync.knowledge.store.pgvector import get_vector_store
    return get_vector_store()
