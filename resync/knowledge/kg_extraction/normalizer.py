"""
Entity and relation normalizer for Knowledge Graph extraction.

Prevents duplicate KG nodes by normalizing entity names and relation
types to canonical forms before inserting extracted triplets into the
knowledge graph.

Version: 1.0.0 - Initial implementation (P1-02 fix: was empty file)
"""

import re
import unicodedata

def normalize_entity(name: str) -> str:
    """
    Normalize entity name to canonical lowercase form for KG deduplication.

    Transformations applied in order:
    1. Unicode NFKC normalization (handles accented chars, ligatures, fullwidth)
    2. Strip leading/trailing whitespace
    3. Collapse internal whitespace to a single space
    4. Lowercase for canonical comparison

    The original cased name should be preserved in node metadata.
    This normalized form is used only for lookup/deduplication.

    Args:
        name: Raw entity name from LLM extraction or user input.

    Returns:
        Normalized canonical name (empty string if input is empty/whitespace).

    Examples:
        >>> normalize_entity("IBM  Corp.")
        'ibm corp.'
        >>> normalize_entity("  PostgreSQL  ")
        'postgresql'
        >>> normalize_entity("FastAPI")
        'fastapi'
    """
    if not name or not name.strip():
        return ""

    # NFKC: Normalize unicode
    # - Converts fullwidth chars: A -> A
    # - Expands ligatures: fi -> fi
    # - Normalizes superscripts: 2 -> 2
    normalized = unicodedata.normalize("NFKC", name)

    # Collapse all whitespace variants (tabs, newlines, etc.) to single space
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Lowercase for canonical comparison
    return normalized.lower()

def normalize_relation_type(relation: str) -> str:
    """
    Normalize relation type to UPPER_SNAKE_CASE for consistent graph schema.

    Transformations:
    1. Unicode NFKC normalization
    2. Strip and collapse whitespace
    3. Uppercase
    4. Replace spaces/hyphens with underscores
    5. Remove non-alphanumeric/underscore characters

    Falls back to 'RELATED_TO' for empty or unrecognizable input.

    Args:
        relation: Raw relation type string from LLM extraction.

    Returns:
        Normalized UPPER_SNAKE_CASE relation type.

    Examples:
        >>> normalize_relation_type("depends on")
        'DEPENDS_ON'
        >>> normalize_relation_type("  RELATED-TO  ")
        'RELATED_TO'
        >>> normalize_relation_type("")
        'RELATED_TO'
    """
    if not relation or not relation.strip():
        return "RELATED_TO"

    normalized = unicodedata.normalize("NFKC", relation)

    # Collapse whitespace and hyphens to underscores
    normalized = re.sub(r"[\s\-]+", "_", normalized.strip())

    # Uppercase
    normalized = normalized.upper()

    # Remove any character that is not alphanumeric or underscore
    normalized = re.sub(r"[^A-Z0-9_]", "", normalized)

    # Remove leading/trailing underscores
    normalized = normalized.strip("_")

    return normalized or "RELATED_TO"

def are_same_entity(name_a: str, name_b: str) -> bool:
    """
    Check if two entity names refer to the same canonical entity.

    Uses normalize_entity() for comparison, so "IBM Corp." and "ibm corp"
    are considered the same entity.

    Args:
        name_a: First entity name.
        name_b: Second entity name.

    Returns:
        True if both names normalize to the same canonical form.
    """
    return normalize_entity(name_a) == normalize_entity(name_b)

# =============================================================================
# Graph utility helpers
# =============================================================================

def make_node_id(entity_type: str, name: str) -> str:
    """Create a canonical node ID from entity type and name."""
    return f"{entity_type.lower()}:{normalize_entity(name)}"


def dedup_concepts(concepts: list[dict]) -> list[dict]:
    """Remove duplicate concept nodes (same entity_type + normalized name)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for c in concepts:
        key = make_node_id(c.get("entity_type", "entity"), c.get("name", ""))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def dedup_edges(edges: list[dict]) -> list[dict]:
    """Remove duplicate edges (same source + relation_type + target)."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for e in edges:
        key = (
            e.get("source", ""),
            normalize_relation_type(e.get("relation_type", "")),
            e.get("target", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique
