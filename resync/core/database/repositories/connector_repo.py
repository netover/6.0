"""
Connector Repository.
"""

from typing import Any
from uuid import UUID

from sqlalchemy import select

from resync.core.database.models.connector import Connector
from resync.core.database.repositories.base import BaseRepository

class ConnectorRepository(BaseRepository[Connector]):
    """
    Repository for managing external connectors in PostgreSQL.
    """

    def __init__(self, session_factory=None) -> None:
        """
        Initialize the connector repository.
        """
        super().__init__(Connector, session_factory)

    async def get_by_name(self, name: str) -> Connector | None:
        """
        Get a connector by its unique name.
        
        Args:
            name: Connector name
            
        Returns:
            Connector instance or None
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(self.model).where(self.model.name == name)
            )
            return result.scalar_one_or_none()

    async def list_by_type(self, connector_type: str) -> list[Connector]:
        """
        List all connectors of a specific type.
        
        Args:
            connector_type: Type of connector (tws, database, etc.)
            
        Returns:
            List of matching connectors
        """
        return await self.find({"type": connector_type})

    async def update_status(self, connector_id: UUID, status: str, error_message: str | None = None) -> Connector | None:
        """
        Update the status of a connector.
        """
        from datetime import datetime, timezone
        return await self.update(
            connector_id, 
            status=status, 
            error_message=error_message,
            last_check=datetime.now(timezone.utc)
        )
