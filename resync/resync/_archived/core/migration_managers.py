"""
Migration Managers para Migração Gradual de Componentes.

Este módulo implementa gerenciadores que permitem migração gradual
de componentes antigos para novos, mantendo backward compatibility
e possibilidade de rollback.
"""

import asyncio
import logging
from typing import Any

from resync.core.cache import AsyncTTLCache, ImprovedAsyncCache
from resync.core.metrics_compat import Counter, Histogram
from resync.services.tws_service import OptimizedTWSClient

# Rate limiter imports - commented out as classes may not exist yet
# from resync.core.rate_limiter import RateLimiter
# from resync.core.rate_limiter_improved import TokenBucketRateLimiter
from resync.settings import settings

logger = logging.getLogger(__name__)

# Métricas de migração
migration_legacy_hits = Counter(
    "migration_legacy_hits_total", "Total hits no sistema legado", ["component"]
)

migration_new_hits = Counter(
    "migration_new_hits_total", "Total hits no novo sistema", ["component"]
)

migration_fallbacks = Counter(
    "migration_fallbacks_total",
    "Total fallbacks para sistema legado",
    ["component", "reason"],
)

migration_errors = Counter(
    "migration_errors_total",
    "Total erros durante migração",
    ["component", "error_type"],
)

migration_latency = Histogram(
    "migration_operation_latency_seconds",
    "Latência de operações durante migração",
    ["component", "operation"],
)


class CacheMigrationManager:
    """
    Gerenciador de migração para sistema de cache.

    Permite transição gradual entre AsyncTTLCache (legado)
    e ImprovedAsyncCache (novo), com fallback automático.
    """

    def __init__(self):
        self.legacy_cache: AsyncTTLCache | None = None
        self.new_cache: ImprovedAsyncCache | None = None
        self.use_new_cache = settings.MIGRATION_USE_NEW_CACHE
        self.enable_metrics = settings.MIGRATION_ENABLE_METRICS

    async def initialize(self):
        """Inicializar ambos os sistemas de cache."""
        try:
            # Inicializar cache legado
            self.legacy_cache = AsyncTTLCache(
                ttl_seconds=getattr(settings, "ASYNC_CACHE_TTL", 3600),
                num_shards=getattr(settings, "ASYNC_CACHE_NUM_SHARDS", 16),
            )

            # Inicializar novo cache
            self.new_cache = ImprovedAsyncCache(
                default_ttl=getattr(settings, "ASYNC_CACHE_TTL", 3600),
                enable_metrics=self.enable_metrics,
            )
            await self.new_cache.initialize()

            logger.info("CacheMigrationManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CacheMigrationManager: {e}")
            raise

    async def shutdown(self):
        """Desligar ambos os sistemas de cache."""
        if self.new_cache:
            await self.new_cache.shutdown()
        # Legacy cache não precisa de shutdown específico

    async def get(self, key: str) -> Any:
        """Obter valor do cache com migração gradual."""
        start_time = asyncio.get_event_loop().time()

        if self.use_new_cache and self.new_cache:
            try:
                result = await self.new_cache.get(key)
                if result is not None:
                    if self.enable_metrics:
                        migration_new_hits.labels(component="cache").inc()
                    migration_latency.labels(component="cache", operation="get").observe(
                        asyncio.get_event_loop().time() - start_time
                    )
                    return result

                # Se novo cache não tem o valor, tentar legacy como fallback
                if self.legacy_cache:
                    legacy_result = await self.legacy_cache.get(key)
                    if legacy_result is not None:
                        if self.enable_metrics:
                            migration_fallbacks.labels(
                                component="cache", reason="cache_miss_new"
                            ).inc()
                        # Opcional: popular novo cache com valor encontrado
                        await self.new_cache.set(key, legacy_result)
                        return legacy_result

            except Exception as e:
                logger.warning(f"Error in new cache, falling back to legacy: {e}")
                if self.enable_metrics:
                    migration_errors.labels(component="cache", error_type="new_cache_error").inc()
                    migration_fallbacks.labels(component="cache", reason="error").inc()

                # Fallback para cache legado
                if self.legacy_cache:
                    try:
                        result = await self.legacy_cache.get(key)
                        migration_latency.labels(component="cache", operation="get").observe(
                            asyncio.get_event_loop().time() - start_time
                        )
                        return result
                    except Exception as e2:
                        logger.error(f"Error in legacy cache fallback: {e2}")
                        if self.enable_metrics:
                            migration_errors.labels(
                                component="cache", error_type="legacy_cache_error"
                            ).inc()

        # Usar apenas cache legado
        if self.legacy_cache:
            try:
                result = await self.legacy_cache.get(key)
                if self.enable_metrics:
                    migration_legacy_hits.labels(component="cache").inc()
                migration_latency.labels(component="cache", operation="get").observe(
                    asyncio.get_event_loop().time() - start_time
                )
                return result
            except Exception as e:
                logger.error(f"Error in legacy cache: {e}")
                if self.enable_metrics:
                    migration_errors.labels(
                        component="cache", error_type="legacy_cache_error"
                    ).inc()

        return None

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Definir valor no cache."""
        start_time = asyncio.get_event_loop().time()

        # Sempre escrever nos dois sistemas durante migração
        tasks = []

        if self.new_cache:
            tasks.append(self.new_cache.set(key, value, ttl))

        if self.legacy_cache:
            tasks.append(self.legacy_cache.set(key, value, ttl))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        migration_latency.labels(component="cache", operation="set").observe(
            asyncio.get_event_loop().time() - start_time
        )

    async def delete(self, key: str) -> bool:
        """Remover valor do cache."""
        start_time = asyncio.get_event_loop().time()

        # Remover dos dois sistemas
        tasks = []

        if self.new_cache:
            tasks.append(self.new_cache.delete(key))

        if self.legacy_cache:
            tasks.append(self.legacy_cache.delete(key))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success = any(isinstance(r, bool) and r for r in results)
        else:
            success = False

        migration_latency.labels(component="cache", operation="delete").observe(
            asyncio.get_event_loop().time() - start_time
        )

        return success

    async def clear(self) -> None:
        """Limpar cache completamente."""
        tasks = []

        if self.new_cache:
            tasks.append(self.new_cache.clear())

        if self.legacy_cache:
            tasks.append(self.legacy_cache.clear())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def get_stats(self) -> dict[str, Any]:
        """Obter estatísticas de ambos os sistemas."""
        stats = {
            "migration_mode": "new" if self.use_new_cache else "legacy",
            "dual_write_enabled": True,
            "legacy_stats": {},
            "new_stats": {},
        }

        if self.legacy_cache and hasattr(self.legacy_cache, "get_stats"):
            try:
                stats["legacy_stats"] = self.legacy_cache.get_stats()
            except Exception as e:
                logger.warning(f"Error getting legacy stats: {e}")

        if self.new_cache:
            try:
                stats["new_stats"] = await self.new_cache.get_stats()
            except Exception as e:
                logger.warning(f"Error getting new stats: {e}")

        return stats


class TWSMigrationManager:
    """
    Gerenciador de migração para cliente TWS.

    Permite transição gradual entre diferentes configurações
    de cliente TWS (legado vs novo).
    """

    def __init__(self):
        self.legacy_client: OptimizedTWSClient | None = None
        self.new_client: Any | None = None
        self.use_new_client = settings.MIGRATION_USE_NEW_TWS
        self.enable_metrics = settings.MIGRATION_ENABLE_METRICS

    async def initialize(self):
        """Inicializar ambos os clientes TWS."""
        try:
            # Cliente legado
            hostname = settings.TWS_HOST or "localhost"
            port = settings.TWS_PORT or 31111
            base_url = f"http://{hostname}:{port}"
            self.legacy_client = OptimizedTWSClient(
                base_url=base_url,
                username=settings.TWS_USER,
                password=settings.TWS_PASSWORD,
                engine_name=getattr(settings, "TWS_ENGINE_NAME", ""),
                engine_owner=getattr(settings, "TWS_ENGINE_OWNER", ""),
            )

            # Cliente novo (mesma implementação, pode ser configurado diferentemente)
            self.new_client = OptimizedTWSClient(
                base_url=base_url,
                username=settings.TWS_USER,
                password=settings.TWS_PASSWORD,
                engine_name=getattr(settings, "TWS_ENGINE_NAME", ""),
                engine_owner=getattr(settings, "TWS_ENGINE_OWNER", ""),
            )

            logger.info("TWSMigrationManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TWSMigrationManager: {e}")
            raise

    async def connect(self) -> bool:
        """Conectar usando cliente apropriado."""
        if self.use_new_client and self.new_client:
            try:
                result = await self.new_client.connect()
                if self.enable_metrics:
                    migration_new_hits.labels(component="tws").inc()
                return result
            except Exception as e:
                logger.warning(f"Error in new TWS client, falling back: {e}")
                if self.enable_metrics:
                    migration_errors.labels(component="tws", error_type="new_client_error").inc()
                    migration_fallbacks.labels(component="tws", reason="error").inc()

        # Fallback para cliente legado
        if self.legacy_client:
            try:
                # Assumindo que o cliente legado tem método connect
                result = await self.legacy_client.connect()
                if self.enable_metrics:
                    migration_legacy_hits.labels(component="tws").inc()
                return result
            except Exception as e:
                logger.error(f"Error in legacy TWS client: {e}")
                if self.enable_metrics:
                    migration_errors.labels(component="tws", error_type="legacy_client_error").inc()

        return False

    async def execute_command(self, command: str) -> str:
        """Executar comando usando cliente apropriado."""
        start_time = asyncio.get_event_loop().time()

        if self.use_new_client and self.new_client:
            try:
                result = await self.new_client.execute_command(command)
                migration_latency.labels(component="tws", operation="execute_command").observe(
                    asyncio.get_event_loop().time() - start_time
                )
                return result
            except Exception as e:
                logger.warning(f"Error in new TWS client, falling back: {e}")
                if self.enable_metrics:
                    migration_fallbacks.labels(component="tws", reason="error").inc()

        # Fallback para cliente legado
        if self.legacy_client:
            try:
                result = await self.legacy_client.execute_command(command)
                migration_latency.labels(component="tws", operation="execute_command").observe(
                    asyncio.get_event_loop().time() - start_time
                )
                return result
            except Exception as e:
                logger.error(f"Error in legacy TWS client: {e}")

        raise RuntimeError("No TWS client available") from None

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Obter status de job."""
        start_time = asyncio.get_event_loop().time()

        if self.use_new_client and self.new_client:
            try:
                result = await self.new_client.get_job_status(job_id)
                migration_latency.labels(component="tws", operation="get_job_status").observe(
                    asyncio.get_event_loop().time() - start_time
                )
                return result
            except Exception as e:
                logger.warning(f"Error in new TWS client, falling back: {e}")
                if self.enable_metrics:
                    migration_fallbacks.labels(component="tws", reason="error").inc()

        # Fallback para cliente legado
        if self.legacy_client:
            try:
                result = await self.legacy_client.get_job_status(job_id)
                migration_latency.labels(component="tws", operation="get_job_status").observe(
                    asyncio.get_event_loop().time() - start_time
                )
                return result
            except Exception as e:
                logger.error(f"Error in legacy TWS client: {e}")

        raise RuntimeError("No TWS client available") from None


class RateLimitMigrationManager:
    """
    Gerenciador de migração para rate limiting.

    Placeholder - rate limiting classes not yet implemented.
    """

    def __init__(self):
        self.use_new_limiter = settings.MIGRATION_USE_NEW_RATE_LIMIT
        self.enable_metrics = settings.MIGRATION_ENABLE_METRICS

    async def initialize(self):
        """Inicializar sistemas de rate limiting."""
        # Placeholder - implement when rate limiter classes are available
        logger.info("RateLimitMigrationManager initialized (placeholder)")

    async def acquire(self, tokens: int = 1) -> bool:
        """Adquirir permissão de rate limiting."""
        # Placeholder - always allow for now
        if self.enable_metrics:
            migration_legacy_hits.labels(component="rate_limit").inc()
        return True

    def get_stats(self) -> dict[str, Any]:
        """Obter estatísticas de rate limiting."""
        return {
            "migration_mode": "placeholder",
            "status": "rate_limiter_not_implemented",
        }


# Instâncias globais dos migration managers
cache_migration_manager = CacheMigrationManager()
tws_migration_manager = TWSMigrationManager()
rate_limit_migration_manager = RateLimitMigrationManager()  # Placeholder


async def initialize_migration_managers():
    """Inicializar todos os migration managers."""
    await cache_migration_manager.initialize()
    await tws_migration_manager.initialize()
    await rate_limit_migration_manager.initialize()
    logger.info("All migration managers initialized")


async def shutdown_migration_managers():
    """Desligar todos os migration managers."""
    await cache_migration_manager.shutdown()
    logger.info("All migration managers shutdown")


def get_migration_stats() -> dict[str, Any]:
    """Obter estatísticas de todos os migration managers."""
    # Synchronous wrapper. If this is called within an active event loop, it
    # must be awaited via `get_migration_stats_async()`.
    from resync.core.utils.async_bridge import run_sync

    return run_sync(get_migration_stats_async())


async def get_migration_stats_async() -> dict[str, Any]:
    """Obter estatísticas de todos os migration managers (async)."""
    return {
        "cache": await cache_migration_manager.get_stats(),
        "tws": {},  # TWS manager não tem stats ainda
        "rate_limit": rate_limit_migration_manager.get_stats(),
    }
