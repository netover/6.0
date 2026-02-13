"""
Base Repository Pattern for PostgreSQL.

Provides async CRUD operations using SQLAlchemy 2.0.
All stores will inherit from this base class.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Generic, TypeVar

from sqlalchemy import and_, delete, func, select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from ..engine import get_session

logger = logging.getLogger(__name__)

# Type variable for model classes
ModelT = TypeVar("ModelT", bound=DeclarativeBase)


class BaseRepository(Generic[ModelT]):
    """
    Base repository providing common CRUD operations.

    All store implementations should inherit from this class
    and specify their model type.

    Example:
        class TWSJobRepository(BaseRepository[TWSJobStatus]):
            def __init__(self, session_factory):
                super().__init__(TWSJobStatus, session_factory)
    """

    def __init__(self, model: type[ModelT], session_factory: async_sessionmaker | None = None):
        """
        Initialize repository.

        Args:
            model: SQLAlchemy model class
            session_factory: Optional session factory (uses default if not provided)
        """
        self.model = model
        self._session_factory = session_factory
        self._initialized = False

    @asynccontextmanager
    async def _get_session(self):
        """Get a database session."""
        if self._session_factory:
            async with self._session_factory() as session:
                yield session
        else:
            async with get_session() as session:
                yield session

    async def create(self, **kwargs) -> ModelT:
        """
        Create a new record.

        Args:
            **kwargs: Model field values

        Returns:
            Created model instance
        """
        async with self._get_session() as session:
            instance = self.model(**kwargs)
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            return instance

    async def create_many(self, items: list[dict[str, Any]]) -> list[ModelT]:
        """
        Create multiple records in a single transaction.

        Args:
            items: List of dictionaries with field values

        Returns:
            List of created model instances
        """
        async with self._get_session() as session:
            instances = [self.model(**item) for item in items]
            session.add_all(instances)
            await session.commit()
            for instance in instances:
                await session.refresh(instance)
            return instances

    async def get_by_id(self, id: int) -> ModelT | None:
        """
        Get a record by ID.

        Args:
            id: Record ID

        Returns:
            Model instance or None
        """
        async with self._get_session() as session:
            result = await session.execute(select(self.model).where(self.model.id == id))
            return result.scalar_one_or_none()

    async def get_all(
        self, limit: int = 100, offset: int = 0, order_by: str | None = None, desc: bool = True
    ) -> list[ModelT]:
        """
        Get all records with pagination.

        Args:
            limit: Maximum records to return
            offset: Number of records to skip
            order_by: Field name to order by
            desc: Order descending if True

        Returns:
            List of model instances
        """
        async with self._get_session() as session:
            query = select(self.model)

            if order_by and hasattr(self.model, order_by):
                order_field = getattr(self.model, order_by)
                query = query.order_by(order_field.desc() if desc else order_field.asc())

            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def find(
        self,
        filters: dict[str, Any],
        limit: int = 100,
        offset: int = 0,
        order_by: str | None = None,
        desc: bool = True,
    ) -> list[ModelT]:
        """
        Find records matching filters.

        Args:
            filters: Dictionary of field:value pairs
            limit: Maximum records to return
            offset: Number of records to skip
            order_by: Field name to order by
            desc: Order descending if True

        Returns:
            List of matching model instances
        """
        async with self._get_session() as session:
            query = select(self.model)

            # Apply filters
            conditions = []
            for field, value in filters.items():
                if hasattr(self.model, field):
                    conditions.append(getattr(self.model, field) == value)

            if conditions:
                query = query.where(and_(*conditions))

            if order_by and hasattr(self.model, order_by):
                order_field = getattr(self.model, order_by)
                query = query.order_by(order_field.desc() if desc else order_field.asc())

            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def find_one(self, filters: dict[str, Any]) -> ModelT | None:
        """
        Find a single record matching filters.

        Args:
            filters: Dictionary of field:value pairs

        Returns:
            Model instance or None
        """
        results = await self.find(filters, limit=1)
        return results[0] if results else None

    async def update(self, id: int, **kwargs) -> ModelT | None:
        """
        Update a record by ID.

        Args:
            id: Record ID
            **kwargs: Fields to update

        Returns:
            Updated model instance or None
        """
        async with self._get_session() as session:
            result = await session.execute(
                update(self.model).where(self.model.id == id).values(**kwargs).returning(self.model)
            )
            await session.commit()
            return result.scalar_one_or_none()

    async def update_many(self, filters: dict[str, Any], values: dict[str, Any]) -> int:
        """
        Update multiple records matching filters.

        Args:
            filters: Dictionary of field:value pairs for matching
            values: Dictionary of field:value pairs to update

        Returns:
            Number of updated records
        """
        async with self._get_session() as session:
            conditions = []
            for field, value in filters.items():
                if hasattr(self.model, field):
                    conditions.append(getattr(self.model, field) == value)

            if not conditions:
                return 0

            result = await session.execute(
                update(self.model).where(and_(*conditions)).values(**values)
            )
            await session.commit()
            return result.rowcount

    async def delete(self, id: int) -> bool:
        """
        Delete a record by ID.

        Args:
            id: Record ID

        Returns:
            True if deleted, False if not found
        """
        async with self._get_session() as session:
            result = await session.execute(delete(self.model).where(self.model.id == id))
            await session.commit()
            return result.rowcount > 0

    async def delete_many(self, filters: dict[str, Any]) -> int:
        """
        Delete multiple records matching filters.

        Args:
            filters: Dictionary of field:value pairs

        Returns:
            Number of deleted records
        """
        async with self._get_session() as session:
            conditions = []
            for field, value in filters.items():
                if hasattr(self.model, field):
                    conditions.append(getattr(self.model, field) == value)

            if not conditions:
                return 0

            result = await session.execute(delete(self.model).where(and_(*conditions)))
            await session.commit()
            return result.rowcount

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count records, optionally filtered.

        Args:
            filters: Optional dictionary of field:value pairs

        Returns:
            Number of matching records
        """
        async with self._get_session() as session:
            query = select(func.count()).select_from(self.model)

            if filters:
                conditions = []
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        conditions.append(getattr(self.model, field) == value)
                if conditions:
                    query = query.where(and_(*conditions))

            result = await session.execute(query)
            return result.scalar() or 0

    async def exists(self, filters: dict[str, Any]) -> bool:
        """
        Check if any record matches filters.

        Args:
            filters: Dictionary of field:value pairs

        Returns:
            True if exists, False otherwise
        """
        count = await self.count(filters)
        return count > 0

    async def execute_raw(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a raw SQL query.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            List of result dictionaries
        """
        async with self._get_session() as session:
            result = await session.execute(text(query), params or {})

            # Try to fetch results if it's a SELECT
            try:
                rows = result.fetchall()
                if rows and hasattr(result, "keys"):
                    keys = result.keys()
                    return [dict(zip(keys, row, strict=False)) for row in rows]
                return []
            except Exception:
                await session.commit()
                return []


class TimestampedRepository(BaseRepository[ModelT]):
    """
    Repository for models with timestamp fields.

    Adds convenience methods for time-based queries.
    """

    async def find_in_range(
        self,
        start: datetime,
        end: datetime,
        timestamp_field: str = "timestamp",
        filters: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[ModelT]:
        """
        Find records within a time range.

        Args:
            start: Start datetime
            end: End datetime
            timestamp_field: Name of timestamp field
            filters: Additional filters
            limit: Maximum records

        Returns:
            List of matching records
        """
        async with self._get_session() as session:
            query = select(self.model)

            if hasattr(self.model, timestamp_field):
                ts_field = getattr(self.model, timestamp_field)
                query = query.where(and_(ts_field >= start, ts_field <= end))

            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        query = query.where(getattr(self.model, field) == value)

            query = query.order_by(getattr(self.model, timestamp_field).desc())
            query = query.limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_latest(
        self, filters: dict[str, Any] | None = None, timestamp_field: str = "timestamp"
    ) -> ModelT | None:
        """
        Get the most recent record.

        Args:
            filters: Optional filters
            timestamp_field: Name of timestamp field

        Returns:
            Most recent record or None
        """
        async with self._get_session() as session:
            query = select(self.model)

            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        query = query.where(getattr(self.model, field) == value)

            if hasattr(self.model, timestamp_field):
                query = query.order_by(getattr(self.model, timestamp_field).desc())

            query = query.limit(1)
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def delete_older_than(self, cutoff: datetime, timestamp_field: str = "timestamp") -> int:
        """
        Delete records older than cutoff.

        Args:
            cutoff: Delete records before this time
            timestamp_field: Name of timestamp field

        Returns:
            Number of deleted records
        """
        async with self._get_session() as session:
            if not hasattr(self.model, timestamp_field):
                return 0

            ts_field = getattr(self.model, timestamp_field)
            result = await session.execute(delete(self.model).where(ts_field < cutoff))
            await session.commit()
            return result.rowcount
