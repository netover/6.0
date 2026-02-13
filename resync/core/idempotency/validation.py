"""
Validação de chaves de idempotency.
"""

import re
import uuid


class IdempotencyKeyValidator:
    """Validador de chaves de idempotency"""

    UUID_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    @classmethod
    def is_valid(cls, key: str) -> bool:
        """Verifica se a chave é um UUID v4 válido"""
        return bool(cls.UUID_PATTERN.match(key))

    @classmethod
    def validate(cls, key: str) -> str:
        """Valida e retorna a chave se válida"""
        if not key:
            raise ValueError("Idempotency key cannot be empty")
        if not cls.is_valid(key):
            raise ValueError(f"Invalid idempotency key format: {key}")
        return key


def generate_idempotency_key() -> str:
    """Gera uma nova chave de idempotency (UUID v4)"""
    return str(uuid.uuid4())


def validate_idempotency_key(key: str) -> str:
    """Função de conveniência para validar chaves de idempotency"""
    return IdempotencyKeyValidator.validate(key)
