"""
Cache Utilities - Utilitários para gerenciamento inteligente de cache.

Este módulo fornece funcionalidades para:
- Cache warming (pré-aquecimento) no startup
- Invalidação inteligente por padrões
- Métricas detalhadas de cache

Para o volume do Resync (14k jobs/dia, ~10 req/min), Redis direto é suficiente.
Não há necessidade de L1 (memory) cache - seria over-engineering.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# MODELS
# =============================================================================


@dataclass
class CacheWarmingConfig:
    """Configuração de cache warming."""

    warmers: dict[str, Callable] = field(default_factory=dict)
    enabled: bool = True
    warmup_on_startup: bool = True
    warmup_interval_minutes: int = 60  # Re-warm a cada hora


@dataclass
class CacheStats:
    """Estatísticas de cache."""

    total_gets: int = 0
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    invalidations: int = 0
    last_warmup: datetime | None = None

    @property
    def hit_rate(self) -> float:
        """Calcula hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


# =============================================================================
# CACHE MANAGER
# =============================================================================


class EnhancedCacheManager:
    """
    Gerenciador de cache melhorado com warming e invalidação inteligente.

    Para o Resync:
    - Volume: 14k jobs/dia (~10 req/min)
    - Conclusão: Redis direto é suficiente
    - NÃO usar L1 (memory) - over-engineering para esse volume
    """

    def __init__(self, redis_client=None):
        """
        Inicializa o cache manager.

        Args:
            redis_client: Cliente Redis (se None, obtém do redis_init)
        """
        if redis_client is None:
            from resync.core.redis_init import get_redis_client

            self.redis = get_redis_client()
        else:
            self.redis = redis_client

        self.stats = CacheStats()
        self.warming_config = CacheWarmingConfig()

    async def warm_cache(self, warmers: dict[str, Callable] | None = None):
        """
        Pré-aquece cache com dados críticos.

        Args:
            warmers: Dict[key, fetcher] para pré-carregar
                     Se None, usa self.warming_config.warmers

        Exemplo:
            await cache.warm_cache({
                "tws:critical_jobs": lambda: fetch_critical_jobs(),
                "tws:system_status": lambda: fetch_system_status(),
            })
        """
        if not self.warming_config.enabled:
            logger.info("Cache warming disabled")
            return

        warmers_to_use = warmers or self.warming_config.warmers

        if not warmers_to_use:
            logger.warning("No warmers configured")
            return

        logger.info("Warming cache with %s entries...", len(warmers_to_use))

        tasks = []
        for key, fetcher in warmers_to_use.items():
            tasks.append(self._warm_single_key(key, fetcher))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = len(results) - successes

        self.stats.last_warmup = datetime.now(timezone.utc)

        logger.info(
            f"Cache warming complete: {successes}/{len(warmers_to_use)} successful, {failures} failed"
        )

    async def _warm_single_key(self, key: str, fetcher: Callable):
        """Aquece uma única chave."""
        try:
            # Verificar se já está em cache
            exists = await self.redis.exists(key)
            if exists:
                logger.debug("Cache key already exists, skipping: %s", key)
                return

            # Buscar dados
            if inspect.iscoroutinefunction(fetcher):
                data = await fetcher()
            else:
                data = await asyncio.to_thread(fetcher)
                if inspect.isawaitable(data):
                    data = await data

            # Armazenar no cache
            import json

            await self.redis.setex(key, 3600, json.dumps(data))  # TTL: 1 hora

            logger.debug("Cache warmed: %s", key)

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to warm cache key %s: %s", key, e, exc_info=True)
            raise

    async def invalidate_pattern(self, pattern: str):
        """
        Invalida cache por pattern.

        Args:
            pattern: Pattern Redis (ex: "tws:job:*")

        Exemplo:
            # Invalidar todos os jobs
            await cache.invalidate_pattern("tws:job:*")

            # Invalidar job específico
            await cache.invalidate_pattern("tws:job:PAYROLL_*")
        """
        logger.info("Invalidating cache pattern: %s", pattern)

        try:
            # Scan e delete por pattern
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)

                if keys:
                    await self.redis.delete(*keys)
                    deleted_count += len(keys)

                if cursor == 0:
                    break

            self.stats.invalidations += 1

            logger.info(
                f"Cache invalidation complete: {deleted_count} keys deleted for pattern {pattern}"
            )

        except Exception as e:
            logger.error(
                "Failed to invalidate pattern %s: %s", pattern, e, exc_info=True
            )
            raise

    async def invalidate_job_cache(self, job_name: str):
        """
        Invalida cache relacionado a um job específico.

        Args:
            job_name: Nome do job
        """
        patterns = [
            f"tws:job_status:{job_name}",
            f"tws:job_logs:{job_name}",
            f"tws:job_deps:{job_name}",
        ]

        try:
            await self.redis.delete(*patterns)
        except Exception as e:
            logger.warning("Failed to batch-delete cache keys: %s", e)

        logger.info("Invalidated job cache for: %s", job_name)

    async def get_stats(self) -> dict[str, Any]:
        """
        Retorna estatísticas de cache.

        Returns:
            Dict com estatísticas
        """
        # Obter info do Redis
        try:
            redis_info = await self.redis.info("stats")
            keyspace_info = await self.redis.info("keyspace")
        except Exception as e:
            logger.error("Failed to get Redis info: %s", e)
            redis_info = {}
            keyspace_info = {}

        return {
            "hit_rate": self.stats.hit_rate,
            "total_gets": self.stats.total_gets,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "invalidations": self.stats.invalidations,
            "last_warmup": (
                self.stats.last_warmup.isoformat() if self.stats.last_warmup else None
            ),
            "redis_stats": redis_info,
            "keyspace": keyspace_info,
        }

    def register_warmer(self, key: str, fetcher: Callable):
        """
        Registra um warmer para execução automática.

        Args:
            key: Chave de cache
            fetcher: Função para buscar dados

        Exemplo:
            cache.register_warmer(
                "tws:system_status",
                lambda: tws_client.get_engine_info()
            )
        """
        self.warming_config.warmers[key] = fetcher
        logger.debug("Registered cache warmer: %s", key)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_cache_manager: EnhancedCacheManager | None = None


def get_cache_manager() -> EnhancedCacheManager:
    """Obtém instância global do cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = EnhancedCacheManager()
    return _cache_manager


# =============================================================================
# STARTUP HOOK
# =============================================================================


async def warmup_cache_on_startup():
    """
    Hook de startup para aquecer cache automaticamente.

    Deve ser chamado no lifespan do FastAPI.
    """
    cache_manager = get_cache_manager()

    if not cache_manager.warming_config.warmup_on_startup:
        logger.info("Cache warmup on startup is disabled")
        return

    # Registrar warmers padrão
    try:
        from resync.services.tws_service import get_tws_client

        tws_client = await get_tws_client()

        cache_manager.register_warmer(
            "tws:system_status", lambda: tws_client.get_engine_info()
        )

        # Aquecer cache
        await cache_manager.warm_cache()

    except Exception as e:
        logger.error("Failed to warm cache on startup: %s", e, exc_info=True)
        # Não falhar o startup por causa do cache warming


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnhancedCacheManager",
    "CacheWarmingConfig",
    "CacheStats",
    "get_cache_manager",
    "warmup_cache_on_startup",
]
