# pylint
# mypy
"""
TWS Ontology Manager v5.9.2

Provides ontology-driven extraction, validation, and entity management.
Implements concepts from "Ontology-Driven GraphRAG" adapted for TWS domain.

Features:
1. YAML Schema Loading - Human-readable ontology definition
2. Dynamic Prompt Generation - Ontology-aware extraction prompts
3. Entity Validation - SHACL-like validation rules
4. Alias Resolution - Normalize entity type variations

Author: Resync Team
Version: 5.9.2
"""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class ExtractionStrategy(str, Enum):
    """How to extract entities of this type."""

    LLM = "llm"  # Use LLM with ontology prompt
    REGEX = "regex"  # Use regex patterns only
    HYBRID = "hybrid"  # Combine LLM + regex


@dataclass
class ValidationRule:
    """Validation rule for a property."""

    pattern: str | None = None
    min_value: int | None = None
    max_value: int | None = None
    max_length: int | None = None
    allowed_values: list[str] | None = None


@dataclass
class PropertyDefinition:
    """Definition of an entity property."""

    name: str
    data_type: str
    required: bool = False
    description: str = ""
    validation_rules: ValidationRule | None = None
    extraction_patterns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "PropertyDefinition":
        """Create from dictionary."""
        validation_rules = None
        if "validation_rules" in data:
            validation_rules = ValidationRule(**data["validation_rules"])

        return cls(
            name=data["name"],
            data_type=data.get("data_type", "string"),
            required=data.get("required", False),
            description=data.get("description", ""),
            validation_rules=validation_rules,
            extraction_patterns=data.get("extraction_patterns", []),
        )


@dataclass
class EntityTypeDefinition:
    """Definition of an entity type in the ontology."""

    name: str
    description: str
    aliases: list[str] = field(default_factory=list)
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.HYBRID
    properties: list[PropertyDefinition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "EntityTypeDefinition":
        """Create from dictionary."""
        properties = [
            PropertyDefinition.from_dict(p) for p in data.get("properties", [])
        ]

        strategy = ExtractionStrategy(data.get("extraction_strategy", "hybrid"))

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            extraction_strategy=strategy,
            properties=properties,
        )

    @property
    def required_properties(self) -> list[str]:
        """Get list of required property names."""
        return [p.name for p in self.properties if p.required]

    @property
    def all_aliases(self) -> list[str]:
        """Get all variations of this entity type name."""
        return [self.name, self.name.lower(), self.name.upper()] + self.aliases


@dataclass
class RelationshipTypeDefinition:
    """Definition of a relationship type."""

    name: str
    description: str
    source_types: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)
    properties: list[PropertyDefinition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "RelationshipTypeDefinition":
        """Create from dictionary."""
        properties = [
            PropertyDefinition.from_dict(p) for p in data.get("properties", [])
        ]

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            source_types=data.get("source_types", []),
            target_types=data.get("target_types", []),
            properties=properties,
        )


@dataclass
class OntologyMetadata:
    """Metadata about the ontology."""

    name: str
    version: str
    domain: str
    description: str = ""
    author: str = ""
    last_updated: str = ""


@dataclass
class Ontology:
    """Complete ontology definition."""

    metadata: OntologyMetadata
    entity_types: list[EntityTypeDefinition]
    relationship_types: list[RelationshipTypeDefinition]
    validation_rules: dict[str, Any] = field(default_factory=dict)
    extraction_prompts: dict[str, str] = field(default_factory=dict)

    def get_entity_type(self, name: str) -> EntityTypeDefinition | None:
        """Get entity type by name or alias."""
        name_lower = name.lower()
        for et in self.entity_types:
            if et.name.lower() == name_lower:
                return et
            for alias in et.aliases:
                if alias.lower() == name_lower:
                    return et
        return None

    def get_relationship_type(self, name: str) -> RelationshipTypeDefinition | None:
        """Get relationship type by name."""
        for rt in self.relationship_types:
            if rt.name == name:
                return rt
        return None


# =============================================================================
# VALIDATION RESULT
# =============================================================================


@dataclass
class ValidationError:
    """A single validation error."""

    property_name: str
    error_type: str
    message: str
    value: Any = None


@dataclass
class ValidationResult:
    """Result of entity validation."""

    is_valid: bool
    entity_type: str
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def add_error(
        self, property_name: str, error_type: str, message: str, value: Any = None
    ):
        """Add a validation error."""
        self.errors.append(ValidationError(property_name, error_type, message, value))
        self.is_valid = False

    def add_warning(
        self, property_name: str, error_type: str, message: str, value: Any = None
    ):
        """Add a validation warning."""
        self.warnings.append(ValidationError(property_name, error_type, message, value))


# =============================================================================
# ONTOLOGY MANAGER
# =============================================================================


class OntologyManager:
    """
    Manages TWS ontology for extraction, validation, and entity resolution.

    Features:
    - Load ontology from YAML
    - Generate extraction prompts
    - Validate extracted entities
    - Resolve entity type aliases
    - Provide required entities for intents
    """

    def __init__(self, ontology_path: str | Path | None = None):
        """
        Initialize OntologyManager synchronously (Warning: blocking I/O).
        Use `create_async` in production async contexts.
        """
        if ontology_path is None:
            ontology_path = Path(__file__).parent / "tws_schema.yaml"

        self.ontology_path = Path(ontology_path)
        self.ontology: Ontology | None = None
        self._alias_map: dict[str, str] = {}  # alias -> canonical name
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}

        self._load_ontology()
        self._build_alias_map()
        self._compile_patterns()

        logger.info(
            "ontology_manager_initialized_sync",
            ontology=self.ontology.metadata.name if self.ontology else "none",
        )

    @classmethod
    async def create_async(cls, ontology_path: str | Path | None = None) -> "OntologyManager":
        """
        Asynchronously creates an OntologyManager instance without blocking the event loop.
        """
        instance = cls.__new__(cls)
        instance.ontology_path = Path(ontology_path) if ontology_path else Path(__file__).parent / "tws_schema.yaml"
        instance.ontology = None
        instance._alias_map = {}
        instance._compiled_patterns = {}

        await asyncio.to_thread(instance._load_ontology)
        instance._build_alias_map()
        instance._compile_patterns()

        logger.info(
            "ontology_manager_initialized_async",
            ontology=instance.ontology.metadata.name if instance.ontology else "none",
            version=instance.ontology.metadata.version if instance.ontology else "none",
            entity_types=len(instance.ontology.entity_types) if instance.ontology else 0,
        )
        return instance

    def _load_ontology(self):
        """Load ontology from YAML file."""
        if not self.ontology_path.exists():
            logger.warning(
                "ontology_file_not_found",
                path=str(self.ontology_path),
            )
            return

        try:
            with open(self.ontology_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Parse metadata
            metadata = OntologyMetadata(**data.get("metadata", {}))

            # Parse entity types
            entity_types = [
                EntityTypeDefinition.from_dict(et)
                for et in data.get("entity_types", [])
            ]

            # Parse relationship types
            relationship_types = [
                RelationshipTypeDefinition.from_dict(rt)
                for rt in data.get("relationship_types", [])
            ]

            self.ontology = Ontology(
                metadata=metadata,
                entity_types=entity_types,
                relationship_types=relationship_types,
                validation_rules=data.get("validation_rules", {}),
                extraction_prompts=data.get("extraction_prompts", {}),
            )

            logger.info(
                "ontology_loaded",
                name=metadata.name,
                version=metadata.version,
                entity_types=len(entity_types),
                relationship_types=len(relationship_types),
            )

        except Exception as e:
            logger.error("ontology_load_error", error=str(e))
            raise

    def _build_alias_map(self):
        """Build mapping from aliases to canonical entity type names."""
        if not self.ontology:
            return

        for et in self.ontology.entity_types:
            canonical = et.name
            for alias in et.all_aliases:
                self._alias_map[alias.lower()] = canonical

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        if not self.ontology:
            return

        for et in self.ontology.entity_types:
            patterns = []
            for prop in et.properties:
                for pattern_str in prop.extraction_patterns:
                    try:
                        patterns.append(re.compile(pattern_str, re.IGNORECASE))
                    except re.error as e:
                        logger.warning(
                            "pattern_compile_error",
                            entity_type=et.name,
                            property=prop.name,
                            pattern=pattern_str,
                            error=str(e),
                        )
            if patterns:
                self._compiled_patterns[et.name] = patterns

    # =========================================================================
    # PROMPT GENERATION
    # =========================================================================

    def generate_extraction_prompt(
        self,
        entity_type: str,
        text: str,
        additional_context: str = "",
    ) -> str:
        """
        Generate an ontology-aware extraction prompt.

        Args:
            entity_type: Type of entity to extract (e.g., "Job", "ErrorCode")
            text: Text to extract from
            additional_context: Optional additional context

        Returns:
            Formatted prompt for LLM extraction
        """
        if not self.ontology:
            return f"Extract {entity_type} entities from: {text}"

        et = self.ontology.get_entity_type(entity_type)
        if not et:
            return f"Extract {entity_type} entities from: {text}"

        # Check for pre-defined prompt template
        prompt_key = entity_type.lower().replace(" ", "_")
        if prompt_key in self.ontology.extraction_prompts:
            template = self.ontology.extraction_prompts[prompt_key]
            return template.format(text=text, context=additional_context)

        # Generate dynamic prompt from entity definition
        prompt = f"""Extract {et.name} entities from the following text.

Entity Type: {et.name}
Description: {et.description}

Required Properties:
"""
        for prop in et.properties:
            if prop.required:
                prompt += f"- {prop.name} ({prop.data_type}): {prop.description}\n"
                if prop.validation_rules:
                    if prop.validation_rules.pattern:
                        prompt += f"  Pattern: {prop.validation_rules.pattern}\n"
                    if prop.validation_rules.allowed_values:
                        allowed_values = ", ".join(
                            prop.validation_rules.allowed_values
                        )
                        prompt += f"  Allowed values: {allowed_values}\n"

        prompt += "\nOptional Properties:\n"
        for prop in et.properties:
            if not prop.required:
                prompt += f"- {prop.name} ({prop.data_type}): {prop.description}\n"

        if et.aliases:
            prompt += f"\nAlso known as: {', '.join(et.aliases)}\n"

        prompt += f"""
{additional_context}

Return a JSON array of {et.name} objects with the properties listed above.
Only include entities you find in the text. Do not invent data.

Text:
{text}

JSON:"""

        return prompt

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_entity(
        self,
        entity_type: str,
        entity_data: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate an extracted entity against ontology rules.

        Args:
            entity_type: Type of entity
            entity_data: Extracted entity data

        Returns:
            ValidationResult with is_valid flag and errors/warnings
        """
        result = ValidationResult(is_valid=True, entity_type=entity_type)

        if not self.ontology:
            return result  # No validation without ontology

        et = self.ontology.get_entity_type(entity_type)
        if not et:
            result.add_warning(
                "_entity_type", "unknown", f"Unknown entity type: {entity_type}"
            )
            return result

        # Check required properties
        for prop in et.properties:
            if prop.required and prop.name not in entity_data:
                result.add_error(
                    prop.name,
                    "missing_required",
                    f"Missing required property: {prop.name}",
                )

        # Validate each property
        for prop_name, value in entity_data.items():
            prop_def = next((p for p in et.properties if p.name == prop_name), None)
            if not prop_def:
                result.add_warning(
                    prop_name,
                    "unknown_property",
                    f"Property not in ontology: {prop_name}",
                )
                continue

            self._validate_property(prop_def, value, result)

        return result

    def _validate_property(
        self,
        prop_def: PropertyDefinition,
        value: Any,
        result: ValidationResult,
    ):
        """Validate a single property value."""
        if value is None:
            return

        rules = prop_def.validation_rules
        if not rules:
            return

        # Type checking
        if prop_def.data_type == "integer":
            if not isinstance(value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    result.add_error(
                        prop_def.name,
                        "invalid_type",
                        f"Expected integer, got {type(value).__name__}",
                        value,
                    )
                    return

            # Range validation
            if rules.min_value is not None and value < rules.min_value:
                result.add_error(
                    prop_def.name,
                    "below_minimum",
                    f"Value {value} below minimum {rules.min_value}",
                    value,
                )

            if rules.max_value is not None and value > rules.max_value:
                result.add_error(
                    prop_def.name,
                    "above_maximum",
                    f"Value {value} above maximum {rules.max_value}",
                    value,
                )

        elif prop_def.data_type == "string":
            if not isinstance(value, str):
                value = str(value)

            # Pattern validation
            if rules.pattern and not re.match(rules.pattern, value):
                result.add_error(
                    prop_def.name,
                    "pattern_mismatch",
                    f"Value '{value}' does not match pattern {rules.pattern}",
                    value,
                )

            # Length validation
            if rules.max_length and len(value) > rules.max_length:
                result.add_error(
                    prop_def.name,
                    "too_long",
                    f"Value length {len(value)} exceeds maximum {rules.max_length}",
                    value,
                )

            # Allowed values validation
            if rules.allowed_values and value not in rules.allowed_values:
                result.add_error(
                    prop_def.name,
                    "invalid_value",
                    f"Value '{value}' not in allowed values: {rules.allowed_values}",
                    value,
                )

        elif prop_def.data_type == "boolean":
            if not isinstance(value, bool):
                if str(value).lower() not in ("true", "false", "1", "0", "yes", "no"):
                    result.add_error(
                        prop_def.name,
                        "invalid_type",
                        f"Expected boolean, got {type(value).__name__}",
                        value,
                    )

    def validate_entities(
        self,
        entities: list[dict[str, Any]],
        entity_type: str,
    ) -> tuple[list[dict], list[ValidationResult]]:
        """
        Validate multiple entities, returning valid ones and all results.

        Args:
            entities: List of entity data dicts
            entity_type: Type of entities

        Returns:
            Tuple of (valid_entities, validation_results)
        """
        valid = []
        results = []

        for entity in entities:
            result = self.validate_entity(entity_type, entity)
            results.append(result)
            if result.is_valid:
                valid.append(entity)
            else:
                logger.debug(
                    "entity_validation_failed",
                    entity_type=entity_type,
                    errors=[e.message for e in result.errors],
                )

        logger.info(
            "entities_validated",
            entity_type=entity_type,
            total=len(entities),
            valid=len(valid),
            rejected=len(entities) - len(valid),
        )

        return valid, results

    # =========================================================================
    # REGEX EXTRACTION
    # =========================================================================

    def extract_with_regex(
        self,
        entity_type: str,
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Extract entities using pre-compiled regex patterns.

        Args:
            entity_type: Type of entity to extract
            text: Text to extract from

        Returns:
            List of extracted entity dicts
        """
        if entity_type not in self._compiled_patterns:
            return []

        entities = []
        seen = set()

        for pattern in self._compiled_patterns[entity_type]:
            for match in pattern.finditer(text):
                value = match.group(1) if match.groups() else match.group(0)
                if value and value not in seen:
                    seen.add(value)
                    entities.append(
                        {"_extracted_value": value, "_pattern": pattern.pattern}
                    )

        return entities

    # =========================================================================
    # ALIAS RESOLUTION
    # =========================================================================

    def resolve_entity_type(self, name: str) -> str:
        """
        Resolve an entity type alias to its canonical name.

        Args:
            name: Entity type name or alias

        Returns:
            Canonical entity type name
        """
        return self._alias_map.get(name.lower(), name)

    def normalize_status(self, status: str) -> str:
        """
        Normalize job status to canonical form.

        Args:
            status: Raw status string

        Returns:
            Normalized status (uppercase)
        """
        status_map = {
            "success": "SUCC",
            "successful": "SUCC",
            "ok": "SUCC",
            "completed": "SUCC",
            "abend": "ABEND",
            "abended": "ABEND",
            "failed": "ABEND",
            "failure": "ABEND",
            "error": "ABEND",
            "running": "EXEC",
            "executing": "EXEC",
            "in progress": "EXEC",
            "ready": "READY",
            "waiting": "WAIT",
            "held": "HOLD",
            "hold": "HOLD",
            "cancelled": "CANCEL",
            "canceled": "CANCEL",
        }
        return status_map.get(status.lower(), status.upper())

    # =========================================================================
    # REQUIRED ENTITIES (for Clarification Loop integration)
    # =========================================================================

    def get_required_entities_for_intent(self, intent: str) -> list[str]:
        """
        Get required entity properties for a given intent.

        This integrates with the clarification_node to determine
        what information is needed.

        Args:
            intent: User intent (STATUS, ACTION, TROUBLESHOOT, etc.)

        Returns:
            List of required property names
        """
        intent_requirements = {
            "STATUS": ["job_name"],
            "TROUBLESHOOT": ["job_name"],
            "ACTION": ["job_name", "action_type"],
            "QUERY": [],
            "GENERAL": [],
        }

        return intent_requirements.get(intent.upper(), [])

    def get_entity_type_for_property(self, property_name: str) -> str | None:
        """
        Find which entity type defines a property.

        Args:
            property_name: Name of the property

        Returns:
            Entity type name or None
        """
        if not self.ontology:
            return None

        for et in self.ontology.entity_types:
            for prop in et.properties:
                if prop.name == property_name:
                    return et.name

        return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_all_entity_types(self) -> list[str]:
        """Get list of all entity type names."""
        if not self.ontology:
            return []
        return [et.name for et in self.ontology.entity_types]

    def get_all_relationship_types(self) -> list[str]:
        """Get list of all relationship type names."""
        if not self.ontology:
            return []
        return [rt.name for rt in self.ontology.relationship_types]

    def get_extraction_strategy(self, entity_type: str) -> ExtractionStrategy:
        """Get extraction strategy for an entity type."""
        if not self.ontology:
            return ExtractionStrategy.HYBRID

        et = self.ontology.get_entity_type(entity_type)
        if et:
            return et.extraction_strategy

        return ExtractionStrategy.HYBRID

    def get_property_allowed_values(
        self, entity_type: str, property_name: str
    ) -> list[str] | None:
        """Get allowed values for a property."""
        if not self.ontology:
            return None

        et = self.ontology.get_entity_type(entity_type)
        if not et:
            return None

        for prop in et.properties:
            if prop.name == property_name and prop.validation_rules:
                return prop.validation_rules.allowed_values

        return None


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================


_ontology_manager: OntologyManager | None = None


def get_ontology_manager() -> OntologyManager:
    """Get or create the singleton OntologyManager instance."""
    global _ontology_manager
    if _ontology_manager is None:
        _ontology_manager = OntologyManager()
    return _ontology_manager


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_job(job_data: dict[str, Any]) -> ValidationResult:
    """Convenience function to validate a Job entity."""
    return get_ontology_manager().validate_entity("Job", job_data)


def validate_error_code(error_data: dict[str, Any]) -> ValidationResult:
    """Convenience function to validate an ErrorCode entity."""
    return get_ontology_manager().validate_entity("ErrorCode", error_data)


def generate_job_extraction_prompt(text: str) -> str:
    """Convenience function to generate Job extraction prompt."""
    return get_ontology_manager().generate_extraction_prompt("Job", text)


def normalize_status(status: str) -> str:
    """Convenience function to normalize job status."""
    return get_ontology_manager().normalize_status(status)
