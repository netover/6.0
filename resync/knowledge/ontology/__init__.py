"""
TWS Ontology Module v5.9.2

Provides ontology-driven knowledge extraction, validation, and entity management
for the TWS domain. Implements concepts from "Ontology-Driven GraphRAG" adapted
for IBM Tivoli Workload Scheduler.

Components:
- OntologyManager: Schema loading, prompt generation, validation
- EntityResolver: Hierarchical entity resolution (path + name)
- ProvenanceTracker: Source tracking and auditability

Usage:
    from resync.knowledge.ontology import (
        get_ontology_manager,
        create_job_resolver,
        track_entity,
    )

    # Get ontology manager
    ontology = get_ontology_manager()

    # Generate extraction prompt
    prompt = ontology.generate_extraction_prompt("Job", text)

    # Validate extracted entity
    result = ontology.validate_entity("Job", job_data)

    # Resolve entity (prevent duplicates)
    resolver = create_job_resolver()
    resolved = await resolver.resolve_job("BACKUP_DIARIO", folder="/root")

    # Track provenance
    record = track_entity(resolved.entity_id, "Job", chunk_metadata, "gpt-4o", 0.95)

Author: Resync Team
Version: 5.9.2
"""

from .entity_resolver import (
    EntityMergeLog,
    EntityResolver,
    ErrorCodeResolver,
    JobResolver,
    # Classes
    ResolvedEntity,
    # Factory functions
    create_entity_resolver,
    create_error_resolver,
    create_job_resolver,
)
from .ontology_manager import (
    EntityTypeDefinition,
    # Classes
    ExtractionStrategy,
    Ontology,
    OntologyManager,
    OntologyMetadata,
    PropertyDefinition,
    RelationshipTypeDefinition,
    ValidationError,
    ValidationResult,
    ValidationRule,
    generate_job_extraction_prompt,
    # Singleton
    get_ontology_manager,
    normalize_status,
    validate_error_code,
    # Convenience functions
    validate_job,
)
from .provenance import (
    ExtractionInfo,
    ExtractionMethod,
    ProvenanceRecord,
    ProvenanceTracker,
    # Classes
    SourceInfo,
    # Enums
    SourceType,
    VerificationInfo,
    VerificationStatus,
    get_entity_source,
    # Singleton
    get_provenance_tracker,
    # Convenience functions
    track_entity,
)

__all__ = [
    # Ontology Manager
    "ExtractionStrategy",
    "ValidationRule",
    "PropertyDefinition",
    "EntityTypeDefinition",
    "RelationshipTypeDefinition",
    "OntologyMetadata",
    "Ontology",
    "ValidationError",
    "ValidationResult",
    "OntologyManager",
    "get_ontology_manager",
    "validate_job",
    "validate_error_code",
    "generate_job_extraction_prompt",
    "normalize_status",
    # Entity Resolver
    "ResolvedEntity",
    "EntityMergeLog",
    "EntityResolver",
    "JobResolver",
    "ErrorCodeResolver",
    "create_entity_resolver",
    "create_job_resolver",
    "create_error_resolver",
    # Provenance
    "SourceType",
    "ExtractionMethod",
    "VerificationStatus",
    "SourceInfo",
    "ExtractionInfo",
    "VerificationInfo",
    "ProvenanceRecord",
    "ProvenanceTracker",
    "get_provenance_tracker",
    "track_entity",
    "get_entity_source",
]

__version__ = "5.9.2"
