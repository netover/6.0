"""
Orchestration Configuration Repository

Provides data access methods for orchestration configurations.
"""
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database.models.orchestration import OrchestrationConfig


class OrchestrationConfigRepository:
    """
    Repository for orchestration configuration data access.
    
    Provides CRUD operations and queries for orchestration configurations.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async SQLAlchemy session
        """
        self._session = session
    
    async def create(
        self,
        name: str,
        strategy: str,
        steps: dict,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        owner_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        is_global: bool = False,
    ) -> OrchestrationConfig:
        """
        Create a new orchestration configuration.
        
        Args:
            name: Configuration name
            strategy: Execution strategy (sequential, parallel, consensus, fallback)
            steps: Steps definition (JSON)
            description: Optional description
            metadata: Optional metadata
            owner_id: Owner user ID
            tenant_id: Tenant ID for multi-tenancy
            is_global: Whether config is global
            
        Returns:
            Created configuration
        """
        config = OrchestrationConfig(
            name=name,
            strategy=strategy,
            steps=steps,
            description=description,
            meta_data=metadata or {},
            owner_id=owner_id,
            tenant_id=tenant_id,
            is_global=is_global,
        )
        self._session.add(config)
        await self._session.commit()
        await self._session.refresh(config)
        return config
    
    async def get_by_id(self, config_id: UUID) -> Optional[OrchestrationConfig]:
        """
        Get configuration by ID.
        
        Args:
            config_id: Configuration UUID
            
        Returns:
            Configuration if found, None otherwise
        """
        result = await self._session.execute(
            select(OrchestrationConfig).where(OrchestrationConfig.id == config_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_name(
        self, 
        name: str, 
        owner_id: Optional[str] = None
    ) -> Optional[OrchestrationConfig]:
        """
        Get configuration by name, optionally scoped to owner.
        
        Args:
            name: Configuration name
            owner_id: Optional owner filter
            
        Returns:
            Configuration if found, None otherwise
        """
        query = select(OrchestrationConfig).where(OrchestrationConfig.name == name)
        
        if owner_id:
            query = query.where(
                (OrchestrationConfig.owner_id == owner_id)
                | (OrchestrationConfig.is_global.is_(True))
            )
        
        result = await self._session.execute(query)
        return result.scalar_one_or_none()
    
    async def list_all(
        self,
        owner_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        strategy: Optional[str] = None,
        is_active: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[OrchestrationConfig]:
        """
        List orchestration configurations with filters.
        
        Args:
            owner_id: Filter by owner
            tenant_id: Filter by tenant
            strategy: Filter by strategy
            is_active: Filter by active status
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of configurations
        """
        query = select(OrchestrationConfig)
        
        if owner_id:
            query = query.where(OrchestrationConfig.owner_id == owner_id)
        
        if tenant_id:
            query = query.where(OrchestrationConfig.tenant_id == tenant_id)
        
        if strategy:
            query = query.where(OrchestrationConfig.strategy == strategy)
        
        if is_active is not None:
            query = query.where(OrchestrationConfig.is_active == is_active)
        
        query = (
            query
            .order_by(OrchestrationConfig.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await self._session.execute(query)
        return list(result.scalars().all())
    
    async def update(
        self,
        config_id: UUID,
        **kwargs,
    ) -> Optional[OrchestrationConfig]:
        """
        Update configuration fields.
        
        Args:
            config_id: Configuration ID
            **kwargs: Fields to update
            
        Returns:
            Updated configuration if found, None otherwise
        """
        await self._session.execute(
            update(OrchestrationConfig)
            .where(OrchestrationConfig.id == config_id)
            .values(**kwargs)
        )
        await self._session.commit()
        return await self.get_by_id(config_id)
    
    async def delete(self, config_id: UUID) -> bool:
        """
        Delete configuration (soft delete by setting is_active=False).
        
        Args:
            config_id: Configuration ID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self._session.execute(
            update(OrchestrationConfig)
            .where(OrchestrationConfig.id == config_id)
            .values(is_active=False)
        )
        await self._session.commit()
        return result.rowcount > 0
    
    async def count(
        self,
        owner_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        is_active: bool = True,
    ) -> int:
        """
        Count configurations with filters.
        
        Args:
            owner_id: Filter by owner
            tenant_id: Filter by tenant
            is_active: Filter by active status
            
        Returns:
            Count of matching configurations
        """
        query = select(func.count(OrchestrationConfig.id))
        
        if owner_id:
            query = query.where(OrchestrationConfig.owner_id == owner_id)
        
        if tenant_id:
            query = query.where(OrchestrationConfig.tenant_id == tenant_id)
        
        if is_active is not None:
            query = query.where(OrchestrationConfig.is_active == is_active)
        
        result = await self._session.execute(query)
        return result.scalar_one()
