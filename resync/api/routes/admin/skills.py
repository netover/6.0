"""
Admin API endpoints for Skill Management.

Provides read-only access to skills metadata and content,
plus admin-only reload capability.

v6.0 - Part of Skills Orchestration refactoring.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

from resync.core.wiring import get_skill_manager
from resync.core.skill_manager import SkillManager
from resync.api.dependencies_v2 import get_logger

router = APIRouter(prefix="/skills", tags=["Admin - Skills"])


class SkillSummary(BaseModel):
    """Summary of a skill for listing."""

    name: str
    description: str
    tags: List[str]
    intents: List[str]
    version: str


class SkillDetail(BaseModel):
    """Full skill details including content."""

    name: str
    description: str
    tags: List[str]
    intents: List[str]
    tools_expected: List[str]
    version: str
    content: Optional[str] = None
    content_length: int = 0


class SkillListResponse(BaseModel):
    """Response for skill list endpoint."""

    skills: List[SkillSummary]
    total: int
    skills_dir: str


class ReloadResponse(BaseModel):
    """Response for reload endpoint."""

    status: str
    skills_count: int
    skills: List[str]


@router.get("/", response_model=SkillListResponse)
async def list_skills(
    logger_instance=Depends(get_logger),
) -> SkillListResponse:
    """
    List all available skills.

    Returns metadata only (no content) for performance.
    Use GET /skills/{name} to get full content.
    """
    skill_manager = get_skill_manager.__wrapped__.__self__
    skills = skill_manager.list_skills()

    return SkillListResponse(
        skills=[
            SkillSummary(
                name=s["name"],
                description=s["description"],
                tags=s.get("tags", []),
                intents=s.get("intents", []),
                version=s.get("version", "1.0"),
            )
            for s in skills
        ],
        total=len(skills),
        skills_dir=str(skill_manager.skills_dir),
    )


@router.get("/{skill_name}", response_model=SkillDetail)
async def get_skill(
    skill_name: str,
    include_content: bool = True,
    logger_instance=Depends(get_logger),
) -> SkillDetail:
    """
    Get details for a specific skill.

    Args:
        skill_name: Name of the skill
        include_content: Whether to include full content (default: True)
    """

    # Get skill manager via DI
    # Note: Using direct instantiation as fallback

    skill_manager = SkillManager()

    metadata = skill_manager.get_skill_metadata(skill_name)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{skill_name}' not found",
        )

    content = None
    content_length = 0

    if include_content:
        content = skill_manager.get_skill_content(skill_name)
        content_length = len(content) if content else 0

    return SkillDetail(
        name=metadata.name,
        description=metadata.description,
        tags=metadata.tags,
        intents=metadata.intents,
        tools_expected=metadata.tools_expected,
        version=metadata.version,
        content=content,
        content_length=content_length,
    )


@router.post("/reload", response_model=ReloadResponse)
async def reload_skills(
    logger_instance=Depends(get_logger),
) -> ReloadResponse:
    """
    Reload all skills from disk.

    Admin-only endpoint. Useful after modifying SKILL.md files
    without restarting the server.
    """

    skill_manager = SkillManager()
    skill_manager.reload()

    skills = skill_manager.list_skills()

    logger_instance.info(
        "skills_reloaded_via_api",
        count=len(skills),
    )

    return ReloadResponse(
        status="success",
        skills_count=len(skills),
        skills=[s["name"] for s in skills],
    )


@router.get("/intents/{intent}/skills")
async def get_skills_for_intent(
    intent: str,
    logger_instance=Depends(get_logger),
) -> dict:
    """
    Get all skills that handle a specific intent.

    Useful for debugging intent-to-skill mapping.
    """

    skill_manager = SkillManager()

    matching_skills = []
    for skill in skill_manager.skills_metadata:
        if intent.lower() in [i.lower() for i in skill.intents]:
            matching_skills.append(
                {
                    "name": skill.name,
                    "description": skill.description,
                }
            )

    return {
        "intent": intent,
        "matching_skills": matching_skills,
        "count": len(matching_skills),
    }
