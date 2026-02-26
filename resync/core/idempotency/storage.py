"""
Abstração de armazenamento para o sistema de idempotency.
"""

from redis.asyncio import Redis

from resync.core.idempotency.exceptions import IdempotencyStorageError
from resync.core.idempotency.models import IdempotencyRecord

class IdempotencyStorage:
    """Abstração de armazenamento para o sistema de idempotency"""

    def __init__(self, redis_client: Redis | None):
        self.redis = redis_client

    def _ensure_redis(self) -> Redis:
        """Return the Redis client or raise a descriptive error.

        When Redis is disabled/unavailable, wiring may create the manager with
        ``redis_client=None``. In that case, raise a clear IdempotencyStorageError
        instead of letting a NoneType AttributeError leak out.
        """
        if self.redis is None:
            raise IdempotencyStorageError(
                "Redis client not available (idempotency disabled)"
            )
        return self.redis

    async def get(self, key: str) -> IdempotencyRecord | None:
        """Recupera registro de idempotency"""
        try:
            import json

            redis = self._ensure_redis()
            cached_data = await redis.get(key)
            if not cached_data:
                return None

            record_dict = json.loads(cached_data)
            return IdempotencyRecord.from_dict(record_dict)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            raise IdempotencyStorageError(
                f"Failed to get idempotency record: {str(e)}"
            ) from e

    async def set(self, key: str, record: IdempotencyRecord, ttl_seconds: int) -> bool:
        """Armazena registro de idempotency"""
        try:
            import json

            data = record.to_dict()
            serialized_data = json.dumps(data, default=str)
            redis = self._ensure_redis()
            success = await redis.setex(key, ttl_seconds, serialized_data)
            return bool(success)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            raise IdempotencyStorageError(
                f"Failed to set idempotency record: {str(e)}"
            ) from e

    async def set_nx(
        self, key: str, record: IdempotencyRecord, ttl_seconds: int
    ) -> bool:
        """Armazena registro SOMENTE se a chave NÃO existir (atômico).

        Usa ``SET key value NX EX ttl`` do Redis — operação atômica que evita
        race conditions entre requests concorrentes com a mesma chave.

        Returns:
            True se a chave foi criada (não existia), False se já existia.
        """
        try:
            import json

            data = record.to_dict()
            serialized_data = json.dumps(data, default=str)
            redis = self._ensure_redis()
            result = await redis.set(key, serialized_data, nx=True, ex=ttl_seconds)
            return result is not None and bool(result)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            raise IdempotencyStorageError(
                f"Failed to set_nx idempotency record: {str(e)}"
            ) from e

    async def exists(self, key: str) -> bool:
        """Verifica se chave existe"""
        try:
            redis = self._ensure_redis()
            return bool(await redis.exists(key))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            raise IdempotencyStorageError(f"Failed to check existence: {str(e)}") from e

    async def delete(self, key: str) -> bool:
        """Remove chave"""
        try:
            redis = self._ensure_redis()
            deleted = await redis.delete(key)
            return deleted > 0
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            raise IdempotencyStorageError(f"Failed to delete key: {str(e)}") from e
