# pylint: skip-file
# mypy: ignore-errors
"""
Template Loader for LangGraph Synthesis Templates.

Loads and manages YAML-based templates for response synthesis.
Supports hot-reloading and versioning.

Usage:
    from resync.core.langgraph.templates import get_template, render_template

    template = get_template("status_success")
    response = render_template("status_success", job_name="BATCH001", status="SUCC")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# TEMPLATE PATHS
# =============================================================================

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "prompts"
SYNTHESIS_TEMPLATES_FILE = TEMPLATES_DIR / "synthesis_templates.yaml"


# =============================================================================
# TEMPLATE LOADING
# =============================================================================


@lru_cache(maxsize=1)
def _load_templates() -> dict[str, Any]:
    """Load templates from YAML file (cached)."""
    try:
        if SYNTHESIS_TEMPLATES_FILE.exists():
            with open(SYNTHESIS_TEMPLATES_FILE, encoding="utf-8") as f:
                templates = yaml.safe_load(f)
                logger.debug("templates_loaded", count=len(templates))
                return templates
        else:
            logger.warning(
                "templates_file_not_found", path=str(SYNTHESIS_TEMPLATES_FILE)
            )
            return {}
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("templates_load_error", error=str(e))
        return {}


def reload_templates() -> None:
    """Force reload of templates (clears cache)."""
    _load_templates.cache_clear()
    _load_templates()
    logger.info("templates_reloaded")


def get_template(name: str) -> dict[str, Any] | None:
    """
    Get a template by name.

    Args:
        name: Template name (e.g., "status_success")

    Returns:
        Template dict with 'template', 'version', 'description' or None
    """
    templates = _load_templates()
    return templates.get(name)


def get_template_string(name: str) -> str | None:
    """
    Get just the template string by name.

    Args:
        name: Template name

    Returns:
        Template string or None
    """
    template = get_template(name)
    if template and isinstance(template, dict):
        return template.get("template")
    elif template and isinstance(template, str):
        return template
    return None


# =============================================================================
# TEMPLATE RENDERING
# =============================================================================


def render_template(name: str, context: dict[str, Any] | None = None, **kwargs) -> str:
    """
    Render a template with given values.

    Args:
        name: Template name
        context: Optional dict with values to substitute
        **kwargs: Additional values to substitute in template

    Returns:
        Rendered template string

    Example:
        # Using kwargs
        response = render_template(
            "status_success",
            job_name="BATCH001",
            status="✅ Sucesso",
        )

        # Using context dict
        response = render_template("status_success", {"job_name": "BATCH001"})
    """
    # Merge context and kwargs
    values = {**(context or {}), **kwargs}

    template_str = get_template_string(name)

    if not template_str:
        logger.warning("template_not_found", name=name)
        return _fallback_render(name, values)

    try:
        # Handle missing keys gracefully
        return template_str.format(**{k: v or "N/A" for k, v in values.items()})
    except KeyError as e:
        logger.warning("template_missing_key", name=name, key=str(e))
        # Try partial rendering
        return _safe_format(template_str, values)
    except Exception as e:
        logger.error("template_render_error", name=name, error=str(e))
        return _fallback_render(name, values)


def _safe_format(template: str, values: dict) -> str:
    """Format template, replacing missing keys with placeholders."""
    import re

    def replace_placeholder(match):
        key = match.group(1)
        return str(values.get(key, f"[{key}]"))

    return re.sub(r"\{(\w+)\}", replace_placeholder, template)


def _fallback_render(name: str, values: dict) -> str:
    """Fallback rendering when template not found."""
    lines = [f"**{name.replace('_', ' ').title()}**", ""]
    for key, value in values.items():
        if value:
            lines.append(f"- **{key}:** {value}")
    return "\n".join(lines)


# =============================================================================
# STATUS TRANSLATION
# =============================================================================


def get_status_translation(status: str) -> str:
    """
    Get human-friendly translation for status codes.

    Args:
        status: Status code (e.g., "SUCC", "ABEND")

    Returns:
        Translated status with emoji
    """
    templates = _load_templates()
    translations = templates.get("status_translations", {})
    return translations.get(status.upper(), status)


# =============================================================================
# CLARIFICATION HELPERS
# =============================================================================


def get_clarification_question(
    entity_type: str, language: str = "pt", **format_args
) -> str:
    """
    Get clarification question for missing entity.

    Args:
        entity_type: Type of entity (job_name, workstation, etc.)
        language: Language code (pt/en)
        **format_args: Format arguments for the question

    Returns:
        Formatted question string
    """
    templates = _load_templates()
    questions = templates.get("clarification_questions", {})

    entity_questions = questions.get(entity_type, {})
    question_template = entity_questions.get(language, entity_questions.get("en", ""))

    if question_template and format_args:
        try:
            return question_template.format(**format_args)
        except KeyError:
            return question_template

    return question_template or f"Please provide the {entity_type}."


def get_action_verb(intent: str, language: str = "pt") -> str:
    """
    Get action verb for intent in specified language.

    Args:
        intent: Intent name (status, troubleshoot, action)
        language: Language code (pt/en)

    Returns:
        Action verb string
    """
    templates = _load_templates()
    verbs = templates.get("action_verbs", {})

    intent_verbs = verbs.get(intent, {})
    return intent_verbs.get(language, intent_verbs.get("en", intent))


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "get_template",
    "get_template_string",
    "render_template",
    "reload_templates",
    "get_status_translation",
    "get_clarification_question",
    "get_action_verb",
]
