from __future__ import annotations

"""
Abstração de armazenamento para o sistema de idempotency.
"""

from typing import Any

try:
    from valkey.asyncio import Valkey
except ImportError:  # pragma: no cover - optional dependency
    Valkey = Any  # type: ignore[assignment]

from resync.core.idempotency.exceptions import IdempotencyStorageError
from resync.core.idempotency.models import IdempotencyRecord

class IdempotencyStorage:
    """Abstração de armazenamento para o sistema de idempotency"""

    def __init__(self, valkey_client: Valkey | None) -> None:
        self.valkey = valkey_client

    def _ensure_valkey(self) -> Valkey:
        """Return the Valkey client or raise a descriptive error.

        When Valkey is disabled/unavailable, wiring may create the manager with
        ``valkey_client=None``. In that case, raise a clear IdempotencyStorageError
        instead of letting a NoneType AttributeError leak out.
        """
        if self.valkey is None:
            raise IdempotencyStorageError(
                "Valkey client not available (idempotency disabled)"
            )
        return self.valkey

    async def get(self, key: str) -> IdempotencyRecord | None:
        """Recupera registro de idempotency"""
        try:
            import json

            valkey = self._ensure_valkey()
            cached_data = await valkey.get(key)
            if not cached_data:
                return None

            record_dict = json.loads(cached_data)
            return IdempotencyRecord.from_dict(record_dict)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

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
            valkey = self._ensure_valkey()
            success = await valkey.set(key, serialized_data, ex=ttl_seconds, nx=True)
            return bool(success)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

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

        Usa ``SET key value NX EX ttl`` do Valkey — operação atômica que evita
        race conditions entre requests concorrentes com a mesma chave.

        Returns:
            True se a chave foi criada (não existia), False se já existia.
        """
        try:
            import json

            data = record.to_dict()
            serialized_data = json.dumps(data, default=str)
            valkey = self._ensure_valkey()
            result = await valkey.set(key, serialized_data, nx=True, ex=ttl_seconds)
            return result is not None and bool(result)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            raise IdempotencyStorageError(
                f"Failed to set_nx idempotency record: {str(e)}"
            ) from e

    async def exists(self, key: str) -> bool:
        """Verifica se chave existe"""
        try:
            valkey = self._ensure_valkey()
            return bool(await valkey.exists(key))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            raise IdempotencyStorageError(f"Failed to check existence: {str(e)}") from e

    async def delete(self, key: str) -> bool:
        """Remove chave"""
        try:
            valkey = self._ensure_valkey()
            deleted = await valkey.delete(key)
            return deleted > 0
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            raise IdempotencyStorageError(f"Failed to delete key: {str(e)}") from e
