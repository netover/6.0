"""
Transaction Manager for Cache Operations

This module provides transaction management functionality for cache operations,
allowing atomic multi-key operations with rollback capabilities.
"""

import logging
import uuid
from dataclasses import dataclass, field
from time import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TransactionState:
    """Represents the state of a cache transaction."""

    transaction_id: str
    key: str
    start_time: float
    operations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "active"  # active, committed, rolled_back
    committed: bool = False
    rolled_back: bool = False


class CacheTransactionManager:
    """
    Manages cache transactions for atomic multi-key operations.

    This class provides transaction management functionality that can be used
    with the existing cache implementation to ensure atomicity across multiple
    cache operations.

    Features:
    - Transaction lifecycle management (begin, commit, rollback)
    - Operation tracking and state management
    - Automatic cleanup of expired transactions
    - Thread-safe transaction state management

    Example:
        transaction_manager = CacheTransactionManager()

        # Begin a transaction
        tx_id = await transaction_manager.begin_transaction("user:123")

        # Perform operations (tracked automatically)
        await cache.set("user:123:name", "John")
        await cache.set("user:123:email", "john@example.com")

        # Commit or rollback
        success = await transaction_manager.commit_transaction(tx_id)
        # OR
        success = await transaction_manager.rollback_transaction(tx_id)
    """

    def __init__(
        self,
        max_transactions: int = 1000,
        transaction_timeout: int = 300,  # 5 minutes default
        cleanup_interval: int = 60,  # 1 minute cleanup interval
    ):
        """
        Initialize the transaction manager.

        Args:
            max_transactions: Maximum number of active transactions to track
            transaction_timeout: Transaction timeout in seconds
            cleanup_interval: How often to cleanup expired transactions in seconds
        """
        self.max_transactions = max_transactions
        self.transaction_timeout = transaction_timeout
        self.cleanup_interval = cleanup_interval

        # In-memory storage for transaction states
        self._transactions: dict[str, TransactionState] = {}
        self._active_transactions: set[str] = set()

        # Cleanup task management
        self._cleanup_task = None
        self._is_running = False

        logger.info(
            "CacheTransactionManager initialized",
            max_transactions=max_transactions,
            transaction_timeout=transaction_timeout,
            cleanup_interval=cleanup_interval,
        )

    async def begin_transaction(self, key: str) -> str:
        """
        Begin a new cache transaction for the specified key.

        Args:
            key: The primary key this transaction is operating on

        Returns:
            str: Unique transaction ID for this transaction

        Raises:
            ValueError: If transaction limit exceeded or key is invalid
        """
        # Validate input
        if not key or not isinstance(key, str):
            raise ValueError(f"Invalid key for transaction: {key}")

        # Check transaction limits
        if len(self._active_transactions) >= self.max_transactions:
            raise ValueError(
                f"Maximum number of active transactions ({self.max_transactions}) exceeded"
            )

        # Generate unique transaction ID
        transaction_id = str(uuid.uuid4())

        # Create transaction state
        current_time = time()
        transaction_state = TransactionState(
            transaction_id=transaction_id,
            key=key,
            start_time=current_time,
            status="active",
        )

        # Store transaction state
        self._transactions[transaction_id] = transaction_state
        self._active_transactions.add(transaction_id)

        logger.debug(
            "Transaction started",
            transaction_id=transaction_id,
            key=key,
            total_active=len(self._active_transactions),
        )

        return transaction_id

    async def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a cache transaction.

        Args:
            transaction_id: The transaction ID to commit

        Returns:
            bool: True if transaction was committed successfully, False otherwise
        """
        try:
            # Validate transaction ID
            if not transaction_id or not isinstance(transaction_id, str):
                logger.warning(f"Invalid transaction ID for commit: {transaction_id}")
                return False

            # Get transaction state
            if transaction_id not in self._transactions:
                logger.warning(f"Transaction not found for commit: {transaction_id}")
                return False

            transaction_state = self._transactions[transaction_id]

            # Check if transaction is already committed or rolled back
            if transaction_state.committed:
                logger.warning(f"Transaction already committed: {transaction_id}")
                return True

            if transaction_state.rolled_back:
                logger.warning(f"Transaction already rolled back: {transaction_id}")
                return False

            # Mark transaction as committed
            transaction_state.status = "committed"
            transaction_state.committed = True

            # Remove from active transactions
            self._active_transactions.discard(transaction_id)

            logger.info(
                "Transaction committed",
                transaction_id=transaction_id,
                key=transaction_state.key,
                operation_count=len(transaction_state.operations),
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to commit transaction",
                transaction_id=transaction_id,
                error=str(e),
            )
            return False

    async def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a cache transaction.

        Args:
            transaction_id: The transaction ID to rollback

        Returns:
            bool: True if transaction was rolled back successfully, False otherwise
        """
        try:
            # Validate transaction ID
            if not transaction_id or not isinstance(transaction_id, str):
                logger.warning(f"Invalid transaction ID for rollback: {transaction_id}")
                return False

            # Get transaction state
            if transaction_id not in self._transactions:
                logger.warning(f"Transaction not found for rollback: {transaction_id}")
                return False

            transaction_state = self._transactions[transaction_id]

            # Check if transaction is already committed or rolled back
            if transaction_state.committed:
                logger.warning(f"Cannot rollback committed transaction: {transaction_id}")
                return False

            if transaction_state.rolled_back:
                logger.warning(f"Transaction already rolled back: {transaction_id}")
                return True

            # Mark transaction as rolled back
            transaction_state.status = "rolled_back"
            transaction_state.rolled_back = True

            # Remove from active transactions
            self._active_transactions.discard(transaction_id)

            logger.info(
                "Transaction rolled back",
                transaction_id=transaction_id,
                key=transaction_state.key,
                operation_count=len(transaction_state.operations),
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to rollback transaction",
                transaction_id=transaction_id,
                error=str(e),
            )
            return False

    def get_transaction_state(self, transaction_id: str) -> TransactionState | None:
        """
        Get the current state of a transaction.

        Args:
            transaction_id: The transaction ID to query

        Returns:
            TransactionState or None if transaction not found
        """
        return self._transactions.get(transaction_id)

    def get_active_transaction_count(self) -> int:
        """
        Get the number of currently active transactions.

        Returns:
            int: Number of active transactions
        """
        return len(self._active_transactions)

    def get_transaction_info(self, transaction_id: str) -> dict[str, Any] | None:
        """
        Get detailed information about a transaction.

        Args:
            transaction_id: The transaction ID to query

        Returns:
            Dict with transaction information or None if not found
        """
        transaction_state = self.get_transaction_state(transaction_id)
        if not transaction_state:
            return None

        current_time = time()
        age_seconds = current_time - transaction_state.start_time

        return {
            "transaction_id": transaction_state.transaction_id,
            "key": transaction_state.key,
            "status": transaction_state.status,
            "start_time": transaction_state.start_time,
            "age_seconds": age_seconds,
            "operation_count": len(transaction_state.operations),
            "committed": transaction_state.committed,
            "rolled_back": transaction_state.rolled_back,
        }

    def cleanup_expired_transactions(self) -> int:
        """
        Clean up expired transactions.

        Returns:
            int: Number of transactions cleaned up
        """
        current_time = time()
        expired_transactions = []
        cleaned_count = 0

        # Find expired transactions
        for transaction_id, transaction_state in self._transactions.items():
            if transaction_state.status == "active":
                age = current_time - transaction_state.start_time
                if age > self.transaction_timeout:
                    expired_transactions.append(transaction_id)

        # Clean up expired transactions
        for transaction_id in expired_transactions:
            transaction_state = self._transactions[transaction_id]
            transaction_state.status = "expired"
            self._active_transactions.discard(transaction_id)
            cleaned_count += 1

            logger.debug(
                "Cleaned up expired transaction",
                transaction_id=transaction_id,
                key=transaction_state.key,
                age_seconds=current_time - transaction_state.start_time,
            )

        if cleaned_count > 0:
            logger.info(
                "Cleaned up expired transactions",
                count=cleaned_count,
                remaining_active=len(self._active_transactions),
            )

        return cleaned_count

    def get_all_transaction_info(self) -> dict[str, dict[str, Any]]:
        """
        Get information about all transactions (for debugging/monitoring).

        Returns:
            Dict mapping transaction IDs to transaction information
        """
        return {tx_id: self.get_transaction_info(tx_id) for tx_id in self._transactions}

    def clear_all_transactions(self) -> int:
        """
        Clear all transaction states (use with caution).

        Returns:
            int: Number of transactions cleared
        """
        count = len(self._transactions)
        self._transactions.clear()
        self._active_transactions.clear()

        logger.warning("All transactions cleared", count=count)
        return count
