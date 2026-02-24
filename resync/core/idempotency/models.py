# ruff: noqa: E501
"""
Modelos de dados para o sistema de idempotency.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class IdempotencyRecord:
    """Registro de idempotency armazenado"""

    idempotency_key: str
    request_hash: str
    response_data: dict[str, Any]
    status_code: int
    created_at: datetime
    expires_at: datetime
    request_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converte para dicionário para serialização"""
        return {
            "idempotency_key": self.idempotency_key,
            "request_hash": self.request_hash,
            "response_data": self.response_data,
            "status_code": self.status_code,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "request_metadata": self.request_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdempotencyRecord":
        """Cria instância a partir de dicionário"""
        return cls(
            idempotency_key=data["idempotency_key"],
            request_hash=data["request_hash"],
            response_data=data["response_data"],
            status_code=data["status_code"],
            # Use more robust parsing that handles potential format variations
            # If fromisoformat fails or format is slightly different, we might need dateutil
            # For now, sticking to robust stdlib based on expected ISO format
            created_at=datetime.fromisoformat(
                str(data["created_at"]).replace("Z", "+00:00")
            ),
            expires_at=datetime.fromisoformat(
                str(data["expires_at"]).replace("Z", "+00:00")
            ),
            request_metadata=data.get("request_metadata", {}),
        )


@dataclass
class RequestContext:
    """Contexto de uma requisição para idempotency"""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    idempotency_key: str | None = None

    def get_request_hash(self) -> str:
        """Gera hash único da requisição"""
        import hashlib
        import json

        # Criar representação canônica da requisição
        request_data = {
            "method": self.method,
            "url": self.url,
            "headers": dict(sorted(self.headers.items())),
            "body": self.body.decode("utf-8") if self.body else None,
        }

        # Gerar hash
        request_json = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(request_json.encode()).hexdigest()
