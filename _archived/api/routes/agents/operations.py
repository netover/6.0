"""Endpoints de exemplo para operações críticas com idempotency.

Este módulo demonstra o uso de idempotency keys em operações que não devem
ser duplicadas, como criação de recursos, transações, etc.
"""

from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from resync.api.dependencies import (
    get_correlation_id,
    get_idempotency_manager,
    require_idempotency_key,
)
from resync.core.idempotency.manager import IdempotencyManager
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/operations", tags=["Critical Operations"])


# ============================================================================
# MODELS
# ============================================================================


class CreateResourceRequest(BaseModel):
    """Request para criar um recurso."""

    name: str = Field(..., description="Nome do recurso", min_length=1, max_length=100)
    description: str | None = Field(None, description="Descrição do recurso", max_length=500)
    metadata: dict | None = Field(default_factory=dict, description="Metadados adicionais")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "My Resource",
                "description": "A sample resource",
                "metadata": {"category": "test", "priority": "high"},
            }
        }
    )


class ResourceResponse(BaseModel):
    """Response com dados do recurso criado."""

    id: str = Field(..., description="ID único do recurso")
    name: str = Field(..., description="Nome do recurso")
    description: str | None = Field(None, description="Descrição do recurso")
    metadata: dict = Field(default_factory=dict, description="Metadados")
    created_at: str = Field(..., description="Timestamp de criação")
    idempotency_key: str = Field(..., description="Chave de idempotência usada")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "My Resource",
                "description": "A sample resource",
                "metadata": {"category": "test", "priority": "high"},
                "created_at": "2024-01-15T10:30:00Z",
                "idempotency_key": "660e8400-e29b-41d4-a716-446655440001",
            }
        }
    )


Currency = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]


class TransactionRequest(BaseModel):
    """Request para criar uma transação."""

    amount: float = Field(..., description="Valor da transação", gt=0)
    currency: Currency = Field(default="USD", description="Moeda")
    description: str = Field(..., description="Descrição da transação", min_length=1)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "amount": 100.50,
                "currency": "USD",
                "description": "Payment for services",
            }
        }
    )


class TransactionResponse(BaseModel):
    """Response com dados da transação."""

    id: str = Field(..., description="ID único da transação")
    amount: float = Field(..., description="Valor da transação")
    currency: str = Field(..., description="Moeda")
    description: str = Field(..., description="Descrição")
    status: str = Field(..., description="Status da transação")
    created_at: str = Field(..., description="Timestamp de criação")
    idempotency_key: str = Field(..., description="Chave de idempotência")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "txn_550e8400e29b41d4",
                "amount": 100.50,
                "currency": "USD",
                "description": "Payment for services",
                "status": "completed",
                "created_at": "2024-01-15T10:30:00Z",
                "idempotency_key": "660e8400-e29b-41d4-a716-446655440001",
            }
        }
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post(
    "/resources",
    response_model=ResourceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new resource",
    description="""
    Creates a new resource with idempotency support.

    **Idempotency**: This endpoint requires an `X-Idempotency-Key` header.
    Multiple requests with the same key will return the same result without
    creating duplicate resources.

    **Headers**:
    - `X-Idempotency-Key`: UUID v4 format (required)
    - `X-Correlation-ID`: For request tracing (optional)

    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/operations/resources" \\
      -H "X-Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000" \\
      -H "Content-Type: application/json" \\
      -d '{"name": "My Resource", "description": "Test resource"}'
    ```
    """,
)
async def create_resource(
    request: CreateResourceRequest,
    idempotency_key: str = Depends(require_idempotency_key),
    manager: IdempotencyManager = Depends(get_idempotency_manager),
    correlation_id: str = Depends(get_correlation_id),
) -> ResourceResponse:
    """Create a new resource with idempotency support.

    Args:
        request: Resource creation request
        idempotency_key: Unique idempotency key
        manager: Idempotency manager
        correlation_id: Correlation ID for tracing

    Returns:
        Created resource data

    Raises:
        ValidationError: If request data is invalid
        ResourceConflictError: If operation is already in progress
    """

    async def _create_resource() -> ResourceResponse:
        """Internal function to create resource."""
        logger.info(
            "Creating resource",
            resource_name=request.name,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
        )

        # Simulate resource creation
        # In production, this would interact with database
        resource_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat() + "Z"

        # Simulate some processing time
        import asyncio

        await asyncio.sleep(0.1)

        response = ResourceResponse(
            id=resource_id,
            name=request.name,
            description=request.description,
            metadata=request.metadata or {},
            created_at=created_at,
            idempotency_key=idempotency_key,
        )

        logger.info(
            "Resource created successfully",
            resource_id=resource_id,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
        )

        return response

    # Execute with idempotency
    return await manager.execute_idempotent(
        key=idempotency_key,
        func=_create_resource,
        ttl_seconds=86400,  # 24 hours
    )


@router.post(
    "/transactions",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new transaction",
    description="""
    Creates a new transaction with idempotency support.

    **Idempotency**: This endpoint requires an `X-Idempotency-Key` header.
    This is critical for financial operations to prevent duplicate charges.

    **Headers**:
    - `X-Idempotency-Key`: UUID v4 format (required)
    - `X-Correlation-ID`: For request tracing (optional)

    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/operations/transactions" \\
      -H "X-Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000" \\
      -H "Content-Type: application/json" \\
      -d '{"amount": 100.50, "currency": "USD", "description": "Payment"}'
    ```
    """,
)
async def create_transaction(
    request: TransactionRequest,
    idempotency_key: str = Depends(require_idempotency_key),
    manager: IdempotencyManager = Depends(get_idempotency_manager),
    correlation_id: str = Depends(get_correlation_id),
) -> TransactionResponse:
    """Create a new transaction with idempotency support.

    Args:
        request: Transaction creation request
        idempotency_key: Unique idempotency key
        manager: Idempotency manager
        correlation_id: Correlation ID for tracing

    Returns:
        Created transaction data

    Raises:
        ValidationError: If request data is invalid
        ResourceConflictError: If operation is already in progress
    """

    async def _create_transaction() -> TransactionResponse:
        """Internal function to create transaction."""
        logger.info(
            "Creating transaction",
            amount=request.amount,
            currency=request.currency,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
        )

        # Simulate transaction processing
        # In production, this would interact with payment gateway
        transaction_id = f"txn_{uuid4().hex[:16]}"
        created_at = datetime.now(timezone.utc).isoformat() + "Z"

        # Simulate processing time
        import asyncio

        await asyncio.sleep(0.2)

        response = TransactionResponse(
            id=transaction_id,
            amount=request.amount,
            currency=request.currency,
            description=request.description,
            status="completed",
            created_at=created_at,
            idempotency_key=idempotency_key,
        )

        logger.info(
            "Transaction completed successfully",
            transaction_id=transaction_id,
            amount=request.amount,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
        )

        return response

    # Execute with idempotency
    return await manager.execute_idempotent(
        key=idempotency_key,
        func=_create_transaction,
        ttl_seconds=86400,  # 24 hours
    )


@router.get(
    "/idempotency-example",
    summary="Get idempotency usage example",
    description="Returns documentation and examples for using idempotency keys",
)
async def get_idempotency_example():
    """Get examples and documentation for idempotency usage."""
    return {
        "description": "Idempotency keys prevent duplicate operations",
        "header": "X-Idempotency-Key",
        "format": "UUID v4",
        "ttl": "24 hours",
        "examples": {
            "curl": {
                "create_resource": """
curl -X POST "http://localhost:8000/api/v1/operations/resources" \\
  -H "X-Idempotency-Key: $(uuidgen)" \\
  -H "Content-Type: application/json" \\
  -d '{"name": "My Resource", "description": "Test"}'
                """.strip(),
                "create_transaction": """
curl -X POST "http://localhost:8000/api/v1/operations/transactions" \\
  -H "X-Idempotency-Key: $(uuidgen)" \\
  -H "Content-Type: application/json" \\
  -d '{"amount": 100.50, "currency": "USD", "description": "Payment"}'
                """.strip(),
            },
            "python": {
                "create_resource": """
import requests
import uuid

response = requests.post(
    "http://localhost:8000/api/v1/operations/resources",
    headers={
        "X-Idempotency-Key": str(uuid.uuid4()),
        "Content-Type": "application/json"
    },
    json={
        "name": "My Resource",
        "description": "Test resource"
    }
)
                """.strip()
            },
        },
        "behavior": {
            "first_request": "Creates the resource and returns 201 Created",
            "duplicate_request": "Returns cached result with same status code",
            "different_payload": "Returns 400 Bad Request (request mismatch)",
            "in_progress": "Returns 409 Conflict (operation already in progress)",
        },
    }
