"""
Métricas do sistema de idempotency.
"""

from dataclasses import dataclass


@dataclass
class IdempotencyMetrics:
    """Métricas do sistema de idempotency"""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    concurrent_blocks: int = 0
    storage_errors: int = 0
    expired_cleanups: int = 0

    @property
    def hit_rate(self) -> float:
        """Taxa de acertos do cache"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
