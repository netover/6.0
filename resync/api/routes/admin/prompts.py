"""
Admin API Endpoints for Prompt Management.

Provides REST API endpoints for managing prompts through the admin interface.
These endpoints allow CRUD operations on prompts stored in LangFuse/YAML.

Endpoints:
- GET /admin/prompts - List all prompts
- GET /admin/prompts/{id} - Get a specific prompt
- POST /admin/prompts - Create a new prompt
- PUT /admin/prompts/{id} - Update a prompt
- DELETE /admin/prompts/{id} - Delete a prompt
- POST /admin/prompts/{id}/test - Test a prompt with variables
"""

from __future__ import annotations
# mypy

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from resync.api.auth import verify_admin_credentials
from resync.core.langfuse import (
    PromptConfig,
    PromptType,
    get_prompt_manager,
)

logger = logging.getLogger(__name__)

# Router for prompt management - v5.9.5: Added authentication
prompt_router = APIRouter(
    prefix="/admin/prompts",
    tags=["Admin - Prompts"],
    dependencies=[Depends(verify_admin_credentials)],
)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PromptListResponse(BaseModel):
    """Response for listing prompts."""

    prompts: list[dict[str, Any]]
    total: int

class PromptDetailResponse(BaseModel):
    """Response for a single prompt."""

    id: str
    name: str
    type: str
    version: str
    content: str
    description: str
    variables: list[str]
    default_values: dict[str, str]
    model_hint: str | None = None
    temperature_hint: float | None = None
    max_tokens_hint: int | None = None
    is_active: bool
    is_default: bool
    created_at: str
    updated_at: str

class PromptCreateRequest(BaseModel):
    """Request to create a new prompt."""

    id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    type: str = Field(default="system")
    version: str = Field(default="1.0.0")
    content: str = Field(..., min_length=1)
    description: str = Field(default="")
    variables: list[str] = Field(default_factory=list)
    default_values: dict[str, str] = Field(default_factory=dict)
    model_hint: str | None = None
    temperature_hint: float | None = Field(None, ge=0, le=2)
    max_tokens_hint: int | None = Field(None, gt=0)
    is_active: bool = True
    is_default: bool = False

class PromptUpdateRequest(BaseModel):
    """Request to update a prompt."""

    name: str | None = None
    content: str | None = None
    description: str | None = None
    variables: list[str] | None = None
    default_values: dict[str, str] | None = None
    model_hint: str | None = None
    temperature_hint: float | None = Field(None, ge=0, le=2)
    max_tokens_hint: int | None = Field(None, gt=0)
    is_active: bool | None = None
    is_default: bool | None = None

class PromptTestRequest(BaseModel):
    """Request to test a prompt."""

    variables: dict[str, str] = Field(default_factory=dict)

class PromptTestResponse(BaseModel):
    """Response from testing a prompt."""

    compiled: str
    variables_used: dict[str, str]
    missing_variables: list[str]

# =============================================================================
# ENDPOINTS
# =============================================================================

@prompt_router.get("", summary="List all prompts")
async def list_prompts(
    _: Annotated[str, Depends(verify_admin_credentials)],
    prompt_type: str | None = None,
    active_only: bool = True,
) -> PromptListResponse:
    """List all prompts."""
    try:
        prompt_manager = get_prompt_manager()

        type_filter = None
        if prompt_type:
            try:
                type_filter = PromptType(prompt_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid prompt type: {prompt_type}",
                ) from None

        prompts = await prompt_manager.list_prompts(
            prompt_type=type_filter, active_only=active_only
        )

        return PromptListResponse(
            prompts=[
                {
                    "id": p.id,
                    "name": p.name,
                    "type": p.type.value if hasattr(p.type, "value") else p.type,
                    "version": p.version,
                    "is_active": p.is_active,
                    "is_default": p.is_default,
                }
                for p in prompts
            ],
            total=len(prompts),
        )
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("list_prompts_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

@prompt_router.get("/{prompt_id}", summary="Get a prompt")
async def get_prompt(
    prompt_id: str,
    _: Annotated[str, Depends(verify_admin_credentials)],
) -> PromptDetailResponse:
    """Get a specific prompt by ID."""
    prompt_manager = get_prompt_manager()
    template = await prompt_manager.get_prompt(prompt_id)

    if not template:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    config = template.config
    return PromptDetailResponse(
        id=config.id,
        name=config.name,
        type=config.type.value if hasattr(config.type, "value") else config.type,
        version=config.version,
        content=config.content,
        description=config.description,
        variables=config.variables,
        default_values=config.default_values,
        model_hint=config.model_hint,
        temperature_hint=config.temperature_hint,
        max_tokens_hint=config.max_tokens_hint,
        is_active=config.is_active,
        is_default=config.is_default,
        created_at=config.created_at.isoformat() if config.created_at else "",
        updated_at=config.updated_at.isoformat() if config.updated_at else "",
    )

@prompt_router.post("", status_code=201, summary="Create a prompt")
async def create_prompt(
    request: PromptCreateRequest,
    _: Annotated[str, Depends(verify_admin_credentials)],
) -> PromptDetailResponse:
    """Create a new prompt."""
    try:
        prompt_type = PromptType(request.type)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid prompt type: {request.type}"
        ) from None

    prompt_manager = get_prompt_manager()
    config = PromptConfig(
        id=request.id,
        name=request.name,
        type=prompt_type,
        version=request.version,
        content=request.content,
        description=request.description,
        variables=request.variables,
        default_values=request.default_values,
        model_hint=request.model_hint,
        temperature_hint=request.temperature_hint,
        max_tokens_hint=request.max_tokens_hint,
        ab_test_group=None,
        is_active=request.is_active,
        is_default=request.is_default,
    )

    try:
        created = await prompt_manager.create_prompt(config)
    except ValueError as e:
        logger.error("request_failed: %s (%s)", str(e), type(e).__name__, exc_info=True)
        raise HTTPException(
            status_code=400, detail="Invalid request. Check server logs for details."
        ) from e

    return PromptDetailResponse(
        id=created.id,
        name=created.name,
        type=created.type.value if hasattr(created.type, "value") else created.type,
        version=created.version,
        content=created.content,
        description=created.description,
        variables=created.variables,
        default_values=created.default_values,
        model_hint=created.model_hint,
        temperature_hint=created.temperature_hint,
        max_tokens_hint=created.max_tokens_hint,
        is_active=created.is_active,
        is_default=created.is_default,
        created_at=created.created_at.isoformat() if created.created_at else "",
        updated_at=created.updated_at.isoformat() if created.updated_at else "",
    )

@prompt_router.put("/{prompt_id}", summary="Update a prompt")
async def update_prompt(
    prompt_id: str,
    request: PromptUpdateRequest,
    _: Annotated[str, Depends(verify_admin_credentials)],
) -> PromptDetailResponse:
    """Update an existing prompt."""
    prompt_manager = get_prompt_manager()
    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    updated = await prompt_manager.update_prompt(prompt_id, updates)

    if not updated:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    return PromptDetailResponse(
        id=updated.id,
        name=updated.name,
        type=updated.type.value if hasattr(updated.type, "value") else updated.type,
        version=updated.version,
        content=updated.content,
        description=updated.description,
        variables=updated.variables,
        default_values=updated.default_values,
        model_hint=updated.model_hint,
        temperature_hint=updated.temperature_hint,
        max_tokens_hint=updated.max_tokens_hint,
        is_active=updated.is_active,
        is_default=updated.is_default,
        created_at=updated.created_at.isoformat() if updated.created_at else "",
        updated_at=updated.updated_at.isoformat() if updated.updated_at else "",
    )

@prompt_router.delete("/{prompt_id}", status_code=204, summary="Delete a prompt")
async def delete_prompt(
    prompt_id: str,
    _: Annotated[str, Depends(verify_admin_credentials)],
):
    """Delete a prompt."""
    prompt_manager = get_prompt_manager()
    deleted = await prompt_manager.delete_prompt(prompt_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

@prompt_router.post("/{prompt_id}/test", summary="Test a prompt")
async def test_prompt(
    prompt_id: str,
    request: PromptTestRequest,
    _: Annotated[str, Depends(verify_admin_credentials)],
) -> PromptTestResponse:
    """Test a prompt by compiling it."""
    prompt_manager = get_prompt_manager()
    template = await prompt_manager.get_prompt(prompt_id)

    if not template:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    required = set(template.config.variables)
    provided = set(request.variables.keys())
    defaults = set(template.config.default_values.keys())
    missing = required - provided - defaults

    final_vars = {**template.config.default_values, **request.variables}

    try:
        compiled = template.compile(**final_vars)
    except ValueError as e:
        logger.error("request_failed: %s (%s)", str(e), type(e).__name__, exc_info=True)
        raise HTTPException(
            status_code=400, detail="Invalid request. Check server logs for details."
        ) from e

    return PromptTestResponse(
        compiled=compiled,
        variables_used=final_vars,
        missing_variables=list(missing),
    )

@prompt_router.get("/types", summary="Get prompt types")
async def get_prompt_types() -> dict[str, Any]:
    """Get valid prompt types."""
    return {"types": [{"value": t.value, "name": t.name} for t in PromptType]}
