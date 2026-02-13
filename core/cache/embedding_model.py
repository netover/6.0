"""
Local Embedding Model for Semantic Cache.

v5.3.16 - Local embedding generation with:
- Sentence-Transformers for high-quality embeddings
- Lazy model loading (download on first use)
- Memory-efficient singleton pattern
- Fallback to hash-based vectors if model unavailable

Model choice rationale (after 30 years, I've learned to start simple):
- all-MiniLM-L6-v2: 384 dimensions, ~80MB, very fast on CPU
- Good enough for cache similarity (not perfect, but cost-effective)
- Can upgrade to larger model if hit rate is too low

Performance characteristics:
- First call: 2-5s (model download + load)
- Subsequent calls: ~10ms per embedding
- Memory: ~200MB RAM
"""

import hashlib
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# Singleton lock and instance
_model_lock = threading.Lock()
_embedding_model: "SentenceTransformer | None" = None
_model_load_attempted: bool = False


# Model configuration
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
FALLBACK_EMBEDDING_DIM = 384  # Same dim for compatibility


class EmbeddingModelError(Exception):
    """Raised when embedding model fails to load or generate."""


def _get_model() -> "SentenceTransformer | None":
    """
    Get or load the embedding model (singleton with double-checked locking).

    Why double-checked locking?
    - Thread-safe without holding lock on every access
    - Model loads only once even with concurrent requests

    Returns:
        SentenceTransformer model or None if unavailable
    """
    global _embedding_model, _model_load_attempted

    # Fast path: model already loaded
    if _embedding_model is not None:
        return _embedding_model

    # Slow path: need to load
    with _model_lock:
        # Double-check inside lock
        if _embedding_model is not None:
            return _embedding_model

        # Don't retry if we already failed
        if _model_load_attempted:
            return None

        _model_load_attempted = True

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", DEFAULT_MODEL_NAME)
            _embedding_model = SentenceTransformer(DEFAULT_MODEL_NAME)
            logger.info(
                f"Embedding model loaded successfully. "
                f"Dimension: {_embedding_model.get_sentence_embedding_dimension()}"
            )
            return _embedding_model

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Semantic cache will use hash-based fallback. "
                "Install with: pip install sentence-transformers"
            )
            return None

        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            return None


def _hash_to_vector(text: str, dim: int = FALLBACK_EMBEDDING_DIM) -> list[float]:
    """
    Generate deterministic pseudo-embedding from text hash.

    This is a fallback when sentence-transformers is not available.

    WARNING: This provides only exact-match semantics, not true similarity!
    "restart job" and "reiniciar job" will have DIFFERENT vectors.

    Only use this for testing or when you can't install sentence-transformers.

    Args:
        text: Input text
        dim: Desired vector dimension

    Returns:
        List of floats representing the pseudo-embedding
    """
    # Use SHA-256 for determinism
    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()

    # Extend hash to fill dimension (256 bits = 32 bytes = 32 floats max)
    # For 384 dim, we need to hash multiple times
    vectors = []
    current_text = text
    while len(vectors) < dim:
        hash_bytes = hashlib.sha256(current_text.encode("utf-8")).digest()
        # Convert each byte to float in [-1, 1] range
        for b in hash_bytes:
            if len(vectors) >= dim:
                break
            vectors.append((b / 127.5) - 1.0)  # Map 0-255 to -1 to 1
        current_text = hash_bytes.hex()  # Use hash as next input

    # Normalize to unit vector (important for cosine similarity)
    arr = np.array(vectors[:dim])
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm

    return arr.tolist()


def generate_embedding(
    text: str,
    normalize: bool = True,
    use_fallback: bool = True,
) -> list[float]:
    """
    Generate embedding vector for text.

    Args:
        text: Input text to embed
        normalize: Whether to L2-normalize the output (recommended for cosine similarity)
        use_fallback: If True, use hash-based fallback when model unavailable

    Returns:
        List of floats representing the embedding

    Raises:
        EmbeddingModelError: If model unavailable and use_fallback=False
    """
    # Clean input
    text = text.strip()
    if not text:
        raise ValueError("Cannot generate embedding for empty text")

    model = _get_model()

    if model is not None:
        try:
            # Generate embedding using sentence-transformers
            embedding = model.encode(
                text,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return embedding.tolist()

        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            if not use_fallback:
                raise EmbeddingModelError(f"Failed to generate embedding: {e}") from e
            logger.warning("Falling back to hash-based embedding")

    # Fallback to hash-based
    if use_fallback:
        logger.debug("Using hash-based fallback for embedding")
        return _hash_to_vector(text, FALLBACK_EMBEDDING_DIM)

    raise EmbeddingModelError(
        "Embedding model not available and fallback disabled. "
        "Install sentence-transformers: pip install sentence-transformers"
    )


def generate_embeddings_batch(
    texts: list[str],
    normalize: bool = True,
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts (more efficient than one-by-one).

    Args:
        texts: List of input texts
        normalize: Whether to L2-normalize outputs
        batch_size: Batch size for model inference

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    model = _get_model()

    if model is not None:
        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=normalize,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            return embeddings.tolist()

        except Exception as e:
            logger.error("Batch embedding failed: %s", e)

    # Fallback: generate individually
    return [generate_embedding(text, normalize, use_fallback=True) for text in texts]


def get_embedding_dimension() -> int:
    """
    Get the dimension of embeddings produced by current model.

    Returns:
        Embedding dimension (384 for default model)
    """
    model = _get_model()
    if model is not None:
        return model.get_sentence_embedding_dimension()
    return FALLBACK_EMBEDDING_DIM


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Similarity score in range [-1, 1], where 1 is identical

    Note: If vectors are normalized, this is just the dot product.
    """
    a = np.array(vec1)
    b = np.array(vec2)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def cosine_distance(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine distance between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Distance in range [0, 2], where 0 is identical

    Note: distance = 1 - similarity
    """
    return 1.0 - cosine_similarity(vec1, vec2)


def is_model_loaded() -> bool:
    """Check if the embedding model is currently loaded in memory."""
    return _embedding_model is not None


def get_model_info() -> dict:
    """
    Get information about the current embedding model.

    Returns:
        Dict with model name, dimension, and status
    """
    model = _get_model()

    if model is not None:
        return {
            "model_name": DEFAULT_MODEL_NAME,
            "dimension": model.get_sentence_embedding_dimension(),
            "status": "loaded",
            "type": "sentence_transformers",
            "device": str(model.device),
        }

    return {
        "model_name": "hash_fallback",
        "dimension": FALLBACK_EMBEDDING_DIM,
        "status": "fallback",
        "type": "deterministic_hash",
        "device": "cpu",
    }


def preload_model() -> bool:
    """
    Preload the embedding model into memory.

    Call this at application startup to avoid cold-start latency.

    Returns:
        True if model loaded successfully, False otherwise
    """
    model = _get_model()
    return model is not None


def unload_model() -> None:
    """
    Unload the embedding model from memory.

    Use this if you need to free up RAM.
    """
    global _embedding_model, _model_load_attempted

    with _model_lock:
        _embedding_model = None
        _model_load_attempted = False
        logger.info("Embedding model unloaded")


__all__ = [
    "EmbeddingModelError",
    "generate_embedding",
    "generate_embeddings_batch",
    "get_embedding_dimension",
    "cosine_similarity",
    "cosine_distance",
    "is_model_loaded",
    "get_model_info",
    "preload_model",
    "unload_model",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_EMBEDDING_DIM",
]
