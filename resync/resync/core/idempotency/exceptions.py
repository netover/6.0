"""
Exceções específicas do sistema de idempotency.
"""


class IdempotencyError(Exception):
    """Exceção base para o sistema de idempotency"""


class IdempotencyKeyError(IdempotencyError):
    """Exceção para erros relacionados à chave de idempotency"""


class IdempotencyStorageError(IdempotencyError):
    """Exceção para erros de armazenamento"""


class IdempotencyConflictError(IdempotencyError):
    """Exceção para conflitos de idempotency"""
