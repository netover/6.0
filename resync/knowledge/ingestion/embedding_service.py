# pylint: skip-file
# mypy: ignore-errors
"""
Multi-Provider Embedding Service using LiteLLM.

Supports multiple embedding providers through LiteLLM's unified interface:
- OpenAI (text-embedding-3-small, text-embedding-ada-002)
- Azure OpenAI
- Cohere (embed-english-v3.0, embed-multilingual-v3.0)
- HuggingFace
- Ollama (local models)
- Voyage AI
- AWS Bedrock (Titan)
- And many more via LiteLLM

Falls back to deterministic SHA-256 hash-based vectors for development/testing.
"""

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from resync.knowledge.config import CFG
from resync.knowledge.interfaces import Embedder

logger = logging.getLogger(__name__)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    AZURE = "azure"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    VOYAGE = "voyage"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    MISTRAL = "mistral"
    JINA = "jina"
    AUTO = "auto"  # Auto-detect from model name


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""

    model: str
    provider: EmbeddingProvider = EmbeddingProvider.AUTO
    dimension: int = 1536
    api_key: str | None = None
    api_base: str | None = None
    batch_size: int = 128
    timeout: float = 60.0
    retry_attempts: int = 3

    # Provider-specific options
    extra_params: dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


# Default configurations for common providers
DEFAULT_CONFIGS = {
    EmbeddingProvider.OPENAI: EmbeddingConfig(
        model="text-embedding-3-small",
        dimension=1536,
    ),
    EmbeddingProvider.COHERE: EmbeddingConfig(
        model="embed-english-v3.0",
        dimension=1024,
        extra_params={"input_type": "search_document"},
    ),
    EmbeddingProvider.OLLAMA: EmbeddingConfig(
        model="ollama/nomic-embed-text",
        dimension=768,
        api_base="http://localhost:11434",
    ),
    EmbeddingProvider.VOYAGE: EmbeddingConfig(
        model="voyage/voyage-2",
        dimension=1024,
    ),
}


class MultiProviderEmbeddingService(Embedder):
    """
    Multi-provider embedding service using LiteLLM.

    Supports automatic provider detection, batching, retry logic,
    and graceful fallback to hash-based embeddings.

    Example:
        # Using OpenAI
        service = MultiProviderEmbeddingService(
            model="text-embedding-3-small",
            provider=EmbeddingProvider.OPENAI
        )

        # Using Cohere
        service = MultiProviderEmbeddingService(
            model="embed-english-v3.0",
            provider=EmbeddingProvider.COHERE
        )

        # Using Ollama (local)
        service = MultiProviderEmbeddingService(
            model="ollama/nomic-embed-text",
            api_base="http://localhost:11434"
        )

        # Auto-detect provider from model name
        service = MultiProviderEmbeddingService(model="cohere/embed-english-v3.0")
    """

    # Model name prefixes for auto-detection
    PROVIDER_PREFIXES = {
        "text-embedding-": EmbeddingProvider.OPENAI,
        "openai/": EmbeddingProvider.OPENAI,
        "azure/": EmbeddingProvider.AZURE,
        "cohere/": EmbeddingProvider.COHERE,
        "embed-": EmbeddingProvider.COHERE,  # Cohere model names
        "huggingface/": EmbeddingProvider.HUGGINGFACE,
        "ollama/": EmbeddingProvider.OLLAMA,
        "voyage/": EmbeddingProvider.VOYAGE,
        "bedrock/": EmbeddingProvider.BEDROCK,
        "vertex_ai/": EmbeddingProvider.VERTEX,
        "mistral/": EmbeddingProvider.MISTRAL,
        "jina/": EmbeddingProvider.JINA,
    }

    def __init__(
        self,
        model: str | None = None,
        provider: EmbeddingProvider = EmbeddingProvider.AUTO,
        dimension: int | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        batch_size: int = 128,
        timeout: float = 60.0,
        retry_attempts: int = 3,
        **extra_params,
    ) -> None:
        """
        Initialize the multi-provider embedding service.

        Args:
            model: Model name (e.g., "text-embedding-3-small", "cohere/embed-english-v3.0")
            provider: Explicit provider selection (auto-detected if not specified)
            dimension: Embedding dimension (auto-detected from model if not specified)
            api_key: API key (falls back to environment variables)
            api_base: API base URL (for self-hosted models)
            batch_size: Maximum batch size for embedding requests
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
            **extra_params: Provider-specific parameters
        """
        # Use config or defaults
        self._model = model or os.getenv("EMBED_MODEL", CFG.embed_model)
        self._provider = (
            self._detect_provider(self._model)
            if provider == EmbeddingProvider.AUTO
            else provider
        )
        self._dimension = (
            dimension or self._infer_dimension(self._model) or CFG.embed_dim
        )
        self._api_key = api_key
        self._api_base = api_base
        self._batch_size = batch_size
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._extra_params = extra_params

        # Check LiteLLM availability
        self._litellm_available = self._check_litellm()

        # Initialize statistics
        self._stats = {
            "total_requests": 0,
            "total_texts": 0,
            "litellm_calls": 0,
            "fallback_calls": 0,
            "errors": 0,
        }

        logger.info(
            "MultiProviderEmbeddingService initialized",
            extra={
                "model": self._model,
                "provider": self._provider.value,
                "dimension": self._dimension,
                "litellm_available": self._litellm_available,
            },
        )

    def _check_litellm(self) -> bool:
        """Check if LiteLLM is available and properly configured."""
        try:
            import litellm  # noqa: F401

            # Check for API keys based on provider
            if self._provider == EmbeddingProvider.OPENAI:
                if self._api_key or os.getenv("OPENAI_API_KEY"):
                    return True
            elif self._provider == EmbeddingProvider.AZURE:
                if os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"):
                    return True
            elif self._provider == EmbeddingProvider.COHERE:
                if self._api_key or os.getenv("COHERE_API_KEY"):
                    return True
            elif self._provider == EmbeddingProvider.VOYAGE:
                if self._api_key or os.getenv("VOYAGE_API_KEY"):
                    return True
            elif self._provider == EmbeddingProvider.OLLAMA:
                # Ollama doesn't need API key
                return True
            elif self._provider == EmbeddingProvider.HUGGINGFACE:
                if self._api_key or os.getenv("HUGGINGFACE_API_KEY"):
                    return True
            elif self._provider == EmbeddingProvider.BEDROCK:
                # AWS Bedrock uses AWS credentials
                if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv(
                    "AWS_SECRET_ACCESS_KEY"
                ):
                    return True
            elif self._provider == EmbeddingProvider.VERTEX:
                # Google Vertex uses service account
                if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    return True
            else:
                # Try to use any available key
                if any(
                    os.getenv(k)
                    for k in [
                        "OPENAI_API_KEY",
                        "COHERE_API_KEY",
                        "VOYAGE_API_KEY",
                        "HUGGINGFACE_API_KEY",
                        "AZURE_API_KEY",
                    ]
                ):
                    return True

            logger.warning(
                f"No API key found for provider {self._provider.value}. "
                "Falling back to hash-based embeddings."
            )
            return False

        except ImportError:
            logger.warning("LiteLLM not installed. Using hash-based fallback.")
            return False

    def _detect_provider(self, model: str) -> EmbeddingProvider:
        """Auto-detect provider from model name."""
        model_lower = model.lower()

        for prefix, provider in self.PROVIDER_PREFIXES.items():
            if model_lower.startswith(prefix.lower()):
                logger.debug(
                    "Auto-detected provider %s from model %s", provider.value, model
                )
                return provider

        # Default to OpenAI for unknown models
        logger.debug(
            "Could not detect provider for model %s, defaulting to OpenAI", model
        )
        return EmbeddingProvider.OPENAI

    def _infer_dimension(self, model: str) -> int | None:
        """Infer embedding dimension from model name."""
        model_lower = model.lower()

        # OpenAI models
        if "text-embedding-3-small" in model_lower:
            return 1536
        if "text-embedding-3-large" in model_lower:
            return 3072
        if "text-embedding-ada-002" in model_lower:
            return 1536

        # Cohere models
        if "embed-english-v3" in model_lower or "embed-multilingual-v3" in model_lower:
            return 1024
        if "embed-english-light-v3" in model_lower:
            return 384

        # Voyage models
        if "voyage-2" in model_lower or "voyage-large-2" in model_lower:
            return 1024
        if "voyage-code-2" in model_lower:
            return 1536

        # Ollama/local models
        if "nomic-embed-text" in model_lower:
            return 768
        if "all-minilm" in model_lower:
            return 384
        if "bge-" in model_lower or "mistral-embed" in model_lower:
            return 1024

        return None

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of text strings.

        Uses LiteLLM if available, otherwise falls back to hash-based embeddings.
        Automatically handles batching for large inputs.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self._stats["total_requests"] += 1
        self._stats["total_texts"] += len(texts)

        if self._litellm_available:
            try:
                return await self._embed_with_litellm(texts)
            except Exception as e:
                logger.warning(
                    f"LiteLLM embedding failed, falling back to hash: {e}",
                    exc_info=True,
                )
                self._stats["errors"] += 1

        # Fallback to hash-based embeddings
        self._stats["fallback_calls"] += 1
        return [self._hash_vec(t) for t in texts]

    async def _embed_with_litellm(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using LiteLLM."""
        import litellm  # noqa: F401

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            # Build request parameters
            params: dict[str, Any] = {
                "model": self._model,
                "input": batch,
                "timeout": self._timeout,
            }

            # Add API key if provided
            if self._api_key:
                params["api_key"] = self._api_key

            # Add API base if provided
            if self._api_base:
                params["api_base"] = self._api_base

            # Add provider-specific parameters
            if self._provider == EmbeddingProvider.COHERE:
                params["input_type"] = self._extra_params.get(
                    "input_type", "search_document"
                )

            # Add any extra parameters
            params.update(self._extra_params)

            # Make the embedding call with retries
            for attempt in range(self._retry_attempts):
                try:
                    response = await asyncio.to_thread(litellm.embedding, **params)

                    # Extract embeddings from response
                    batch_embeddings = [item["embedding"] for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    self._stats["litellm_calls"] += 1
                    break

                except Exception as e:
                    if attempt < self._retry_attempts - 1:
                        wait_time = 2**attempt  # Exponential backoff
                        logger.warning(
                            f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise

        return all_embeddings

    def _hash_vec(self, text: str) -> list[float]:
        """
        Generate deterministic embedding vector from text using SHA-256 hash.

        This is a fallback for development/CI environments.
        The hash is spread across the embedding dimension.

        Args:
            text: Input text to hash

        Returns:
            Deterministic embedding vector
        """
        dim = self._dimension
        buf = [0.0] * dim
        h = hashlib.sha256(text.encode("utf-8")).digest()

        # Spread 32 bytes across the vector dimension
        for i, b in enumerate(h):
            buf[(i * 64) % dim] = b / 255.0

        return buf

    def get_stats(self) -> dict[str, Any]:
        """Get embedding service statistics."""
        return {
            **self._stats,
            "model": self._model,
            "provider": self._provider.value,
            "dimension": self._dimension,
            "litellm_available": self._litellm_available,
        }

    @property
    def model(self) -> str:
        """Get current model name."""
        return self._model

    @property
    def provider(self) -> EmbeddingProvider:
        """Get current provider."""
        return self._provider

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


# Backwards compatibility alias
class EmbeddingService(MultiProviderEmbeddingService):
    """
    Backwards-compatible embedding service.

    Maintains the original API while using the new multi-provider implementation.
    """

    def __init__(self) -> None:
        """Initialize with defaults from environment/config."""
        super().__init__(
            model=os.getenv("EMBED_MODEL", CFG.embed_model),
            dimension=int(os.getenv("EMBED_DIM", str(CFG.embed_dim))),
        )


# Factory function for creating embedding services
def create_embedding_service(
    provider: str | EmbeddingProvider = EmbeddingProvider.AUTO,
    model: str | None = None,
    **kwargs,
) -> MultiProviderEmbeddingService:
    """
    Factory function to create an embedding service.

    Args:
        provider: Provider name or enum (auto-detected if not specified)
        model: Model name (uses provider default if not specified)
        **kwargs: Additional configuration options

    Returns:
        Configured MultiProviderEmbeddingService

    Example:
        # OpenAI
        service = create_embedding_service("openai")

        # Cohere with custom model
        service = create_embedding_service("cohere", model="embed-multilingual-v3.0")

        # Ollama local
        service = create_embedding_service("ollama", api_base="http://localhost:11434")
    """
    if isinstance(provider, str):
        try:
            provider = EmbeddingProvider(provider.lower())
        except ValueError:
            logger.warning("Unknown provider '%s', using auto-detection", provider)
            provider = EmbeddingProvider.AUTO

    # Get default config for provider if available
    if provider in DEFAULT_CONFIGS and model is None:
        default_config = DEFAULT_CONFIGS[provider]
        model = model or default_config.model
        if "dimension" not in kwargs:
            kwargs["dimension"] = default_config.dimension
        if "api_base" not in kwargs and default_config.api_base:
            kwargs["api_base"] = default_config.api_base
        if default_config.extra_params:
            kwargs.update(default_config.extra_params)

    return MultiProviderEmbeddingService(
        model=model,
        provider=provider,
        **kwargs,
    )
