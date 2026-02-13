"""
Configuração do sistema de idempotency.
"""

from dataclasses import dataclass


@dataclass
class IdempotencyConfig:
    """Configuração do sistema de idempotency"""

    ttl_hours: int = 24
    redis_db: int = 1
    key_prefix: str = "idempotency"
    processing_prefix: str = "processing"
    max_response_size_kb: int = 64  # 64KB máximo por resposta


# Instância global de configuração
config = IdempotencyConfig()
