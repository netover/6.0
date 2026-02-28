"""
Cache Transaction Mixin.

Provides transaction and rollback functionality for cache implementations.
"""

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

class CacheTransactionMixinProtocol(Protocol):
    """Protocol defining methods expected by CacheTransactionMixin."""

    async def delete(self, key: str) -> None: ...
    async def set(self, key: str, value: Any, **kwargs: Any) -> None: ...

class CacheTransactionMixin:
    """
    Mixin providing transaction capabilities for cache.

    Supports recording operations and rolling them back if needed.
    """

    _transaction_log: list[dict[str, Any]]
    _in_transaction: bool

    def __init__(self) -> None:
        self._transaction_log = []
        self._in_transaction = False

    def begin_transaction(self):
        """Begin a new transaction."""
        self._transaction_log = []
        self._in_transaction = True
        logger.debug("Cache transaction started")

    def _log_operation(self, operation: str, key: str, old_value: Any = None):
        """Log an operation for potential rollback."""
        if self._in_transaction:
            self._transaction_log.append(
                {
                    "operation": operation,
                    "key": key,
                    "old_value": old_value,
                }
            )

    def commit_transaction(self):
        """Commit the current transaction."""
        self._transaction_log = []
        self._in_transaction = False
        logger.debug("Cache transaction committed")

    async def rollback_transaction(
        self, operations: list[dict[str, Any]] | None = None
    ) -> bool:
        """
        Rollback operations from the transaction log.

        Args:
            operations: Specific operations to rollback, or None for all

        Returns:
            True if rollback was successful
        """
        ops_to_rollback = operations or self._transaction_log

        if not ops_to_rollback:
            return True

        try:
            # Process in reverse order
            for op in reversed(ops_to_rollback):
                operation = op.get("operation")
                key = op.get("key")
                old_value = op.get("old_value")

                if operation == "set":
                    if old_value is None:
                        await self.delete(key)  # type: ignore[attr-defined]
                    else:
                        await self.set(key, old_value)  # type: ignore[attr-defined]
                elif operation == "delete" and old_value is not None:
                    await self.set(key, old_value)  # type: ignore[attr-defined]

            logger.info("Rolled back %s operations", len(ops_to_rollback))

            self._transaction_log = []
            self._in_transaction = False
            return True

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("Rollback failed: %s", e)
            return False

    def get_transaction_log(self) -> list[dict[str, Any]]:
        """Get current transaction log."""
        return self._transaction_log.copy()

    def is_in_transaction(self) -> bool:
        """Check if currently in a transaction."""
        return self._in_transaction
