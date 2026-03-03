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
    normalized_name = normalize_entity(name)
    normalized_name = re.sub(r"[^a-z0-9]+", "_", normalized_name).strip("_")
    return f"{entity_type}:{normalized_name}"


def _get_attr(item: object, key: str, default: str = "") -> str:
    """Read dict-like or model-like attributes safely."""
    if isinstance(item, dict):
        value = item.get(key, default)
    else:
        value = getattr(item, key, default)
    return value if isinstance(value, str) else default


def dedup_concepts(concepts: list[object]) -> list[object]:
    """Remove duplicate concept nodes and merge aliases/properties when possible."""
    by_key: dict[str, object] = {}
    for c in concepts:
        entity_type = _get_attr(c, "entity_type") or _get_attr(c, "node_type") or "entity"
        name = _get_attr(c, "name")
        key = make_node_id(entity_type, name)

        existing = by_key.get(key)
        if existing is None:
            by_key[key] = c
            continue

        # Merge aliases for pydantic Concept models
        if hasattr(existing, "aliases") and hasattr(c, "aliases"):
            current_aliases = list(getattr(existing, "aliases") or [])
            incoming_aliases = list(getattr(c, "aliases") or [])
            merged_aliases = list(dict.fromkeys(current_aliases + incoming_aliases))
            setattr(existing, "aliases", merged_aliases)

        # Merge properties maps if available
        if hasattr(existing, "properties") and hasattr(c, "properties"):
            props = dict(getattr(existing, "properties") or {})
            props.update(dict(getattr(c, "properties") or {}))
            setattr(existing, "properties", props)

    return list(by_key.values())


def dedup_edges(edges: list[object]) -> list[object]:
    """Remove duplicate edges (same source + relation_type + target), keeping max weight."""
    by_key: dict[tuple[str, str, str], object] = {}
    for e in edges:
        source = _get_attr(e, "source")
        relation = normalize_relation_type(_get_attr(e, "relation_type"))
        target = _get_attr(e, "target")
        key = (source, relation, target)

        current = by_key.get(key)
        if current is None:
            by_key[key] = e
            continue

        cur_weight = float(getattr(current, "weight", 0.0) or 0.0)
        new_weight = float(getattr(e, "weight", 0.0) or 0.0)
        if new_weight > cur_weight:
            by_key[key] = e

    return list(by_key.values())
