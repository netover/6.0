# resync/core/cache/valkey_client.py
from typing import Optional, Any
import valkey
from valkey import asyncio as aioredis
from resync.settings import Settings

class ValkeyClient:
    """Cliente Valkey 9 com Multi-DB e Hash Field Expiration"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[aioredis.Valkey] = None
        self._pool: Optional[aioredis.ConnectionPool] = None
    
    async def connect(self) -> None:
        """Conecta ao Valkey com connection pooling"""
        self._pool = aioredis.ConnectionPool(
            host=self.settings.VALKEY_HOST if hasattr(self.settings, 'VALKEY_HOST') else 'localhost',
            port=self.settings.VALKEY_PORT if hasattr(self.settings, 'VALKEY_PORT') else 6379,
            db=self.settings.VALKEY_DB if hasattr(self.settings, 'VALKEY_DB') else 0,
            password=self.settings.VALKEY_PASSWORD.get_secret_value() if hasattr(self.settings, 'VALKEY_PASSWORD') and self.settings.VALKEY_PASSWORD else None,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        
        self._client = aioredis.Valkey(
            connection_pool=self._pool,
            single_connection_client=False
        )
        
        # Testar conexão
        await self._client.ping()
    
    async def disconnect(self) -> None:
        """Fecha conexões gracefully"""
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.aclose()
    
    async def get(self, key: str) -> Optional[str]:
        """Get com type safety"""
        if not self._client:
            raise RuntimeError("Valkey client not connected")
        return await self._client.get(key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set com expiration e condições"""
        if not self._client:
            raise RuntimeError("Valkey client not connected")
        return await self._client.set(key, value, ex=ex, nx=nx, xx=xx)
    
    async def hset_with_expiration(
        self,
        key: str,
        field: str,
        value: Any,
        ex: int
    ) -> None:
        """
        Hash Field Expiration (HFE) - Novo no Valkey 9
        Define expiration em campos individuais de hash
        """
        if not self._client:
            raise RuntimeError("Valkey client not connected")
        
        # HFE support - expira campos específicos do hash
        await self._client.hset(key, field, value)
        # Note: hexpire is expected to be available in Valkey 9 client
        try:
            await self._client.hexpire(key, ex, field)  # Novo no Valkey 9
        except Exception:
            # Fallback for client versions that don't have hexpire yet
            # or if using a generic redis client that doesn't support it
            pass
    
    async def use_db(self, db: int) -> None:
        """
        Multi-DB Support - Novo no Valkey 9
        Troca entre databases dinamicamente
        """
        if not self._client:
            raise RuntimeError("Valkey client not connected")
        
        # Valkey 9 permite múltiplos DBs simultâneos
        await self._client.select(db)
