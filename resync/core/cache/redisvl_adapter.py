# pylint: skip-file
# mypy: ignore-errors
"""
RedisVL Adapter for Resync.

This module provides the bridge between RedisVL and our existing
embedding models and rerankers.
"""

import logging
from typing import Any, Callable, List, Optional

try:
    from redisvl.utils.vectorize import BaseVectorizer
except Exception:
    # Support environments where redisvl is unavailable/incompatible (e.g. pydantic v1 on py3.14)
    BaseVectorizer = object

from resync.core.cache.embedding_model import (
    generate_embedding,
    generate_embeddings_batch,
    get_embedding_dimension,
)

logger = logging.getLogger(__name__)


class ResyncVectorizer(BaseVectorizer):
    """
    Adapter that allows RedisVL to use our local embedding model.

    This avoids redundant model loading as it uses the singleton
    instance from embedding_model.py.
    """

    def __init__(self, model_name: str = "local-resync"):
        """Initialize the vectorizer using our existing model geometry."""
        dims = get_embedding_dimension()
        # Note: In newer RedisVL, dims might be passed differently or inferred
        super().__init__(model=model_name, dims=dims)
        logger.info(f"ResyncVectorizer initialized with {dims} dimensions")

    def embed(
        self,
        text: str,
        preprocess: Optional[Callable[..., Any]] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """Generate embedding for a single text."""
        if preprocess:
            text = preprocess(text)

        embedding = generate_embedding(text)

        if as_buffer:
            import struct

            return struct.pack(f"{len(embedding)}f", *embedding)  # type: ignore[return-value]
        return embedding

    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable[..., Any]] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if preprocess:
            texts = [preprocess(t) for t in texts]

        embeddings = generate_embeddings_batch(texts, batch_size=batch_size)

        if as_buffer:
            import struct

            return [struct.pack(f"{len(e)}f", *e) for e in embeddings]  # type: ignore[misc]
        return embeddings

    async def aembed(self, *args, **kwargs):
        """Async version of embed - delegated to sync for now as local model is sync."""
        return self.embed(*args, **kwargs)

    async def aembed_many(self, *args, **kwargs):
        """Async version of embed_many - delegated to sync."""
        return self.embed_many(*args, **kwargs)
