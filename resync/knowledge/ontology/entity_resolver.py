"""
TWS Entity Resolver v5.9.2

Implements hierarchical entity resolution for TWS domain.
Prevents duplicate entities while handling same-name jobs in different folders.

Key Features:
1. Hierarchical Resolution - path + name = unique ID (e.g., /root/Job1 ≠ /finance/Job1)
2. Exact Match First - Performance optimization
3. Embedding Fallback - For aliases and variations
4. Merge Logging - Audit trail for entity merges

Author: Resync Team
Version: 5.9.2
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class ResolvedEntity:
    """Result of entity resolution."""

    entity_id: str
    canonical_id: str
    entity_type: str
    is_new: bool
    merged_from: str | None = None
    confidence: float = 1.0
    resolution_method: str = "exact"  # exact, embedding, new


@dataclass
class EntityMergeLog:
    """Log entry for entity merges (audit trail)."""

    timestamp: datetime
    entity_type: str
    source_representation: str
    target_id: str
    resolution_method: str
    confidence: float


# =============================================================================
# ENTITY RESOLVER
# =============================================================================


class EntityResolver:
    """
    Resolves entities to prevent duplicates in the knowledge graph.

    TWS-specific considerations:
    - Job names are only unique within their folder/job stream
    - Same job name in different folders = different jobs
    - Aliases like "BACKUP_DIARIO" vs "Job BACKUP_DIARIO" should resolve to same entity

    Resolution Strategy:
    1. Build canonical ID: {folder}/{entity_type}/{name}
    2. Try exact match on canonical ID
    3. If no match, try embedding similarity (for aliases)
    4. If still no match, create new entity

    Example:
        /root/Job/BACKUP_DIARIO ≠ /finance/Job/BACKUP_DIARIO
        "BACKUP_DIARIO" ≈ "Job BACKUP_DIARIO" (embedding similarity)
    """

    def __init__(
        self,
        embedding_service: Any | None = None,
        similarity_threshold: float = 0.92,
        enable_embedding_fallback: bool = True,
    ):
        """
        Initialize EntityResolver.

        Args:
            embedding_service: Service for generating embeddings (optional)
            similarity_threshold: Minimum similarity for embedding match
            enable_embedding_fallback: Whether to use embedding similarity
        """
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.enable_embedding_fallback = enable_embedding_fallback

        # In-memory cache for fast lookups
        self._entity_cache: dict[str, dict[str, Any]] = {}  # canonical_id -> entity
        self._embedding_cache: dict[str, list[float]] = {}  # canonical_id -> embedding
        self._merge_log: list[EntityMergeLog] = []

        logger.info(
            "entity_resolver_initialized",
            similarity_threshold=similarity_threshold,
            embedding_enabled=enable_embedding_fallback and embedding_service is not None,
        )

    # =========================================================================
    # CANONICAL ID GENERATION
    # =========================================================================

    def build_canonical_id(
        self,
        entity_type: str,
        name: str,
        folder: str | None = None,
        job_stream: str | None = None,
    ) -> str:
        """
        Build a canonical ID for an entity.

        The canonical ID is hierarchical:
        - {folder}/{job_stream}/{entity_type}/{name}
        - Falls back to /{entity_type}/{name} if no context

        Args:
            entity_type: Type of entity (Job, ErrorCode, etc.)
            name: Entity name
            folder: Folder path (optional)
            job_stream: Parent job stream (optional)

        Returns:
            Canonical ID string
        """
        parts = []

        # Add folder if available
        if folder:
            # Normalize folder path
            folder = folder.strip("/").upper()
            parts.append(folder)

        # Add job stream if available and entity is a Job
        if job_stream and entity_type == "Job":
            parts.append(job_stream.upper())

        # Add entity type and name
        parts.append(entity_type)
        parts.append(name.upper())

        canonical_id = "/" + "/".join(parts)

        logger.debug(
            "canonical_id_built",
            entity_type=entity_type,
            name=name,
            folder=folder,
            job_stream=job_stream,
            canonical_id=canonical_id,
        )

        return canonical_id

    def generate_entity_id(self, canonical_id: str) -> str:
        """
        Generate a short entity ID from canonical ID.

        Uses first 12 chars of SHA256 hash.

        Args:
            canonical_id: Full canonical ID

        Returns:
            Short entity ID (12 hex chars)
        """
        hash_obj = hashlib.sha256(canonical_id.encode("utf-8"))
        return hash_obj.hexdigest()[:12]

    # =========================================================================
    # RESOLUTION METHODS
    # =========================================================================

    async def resolve(
        self,
        entity_type: str,
        name: str,
        folder: str | None = None,
        job_stream: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> ResolvedEntity:
        """
        Resolve an entity, returning existing or creating new.

        Resolution order:
        1. Exact canonical ID match (fast, preferred)
        2. Embedding similarity match (if enabled, for aliases)
        3. Create new entity

        Args:
            entity_type: Type of entity
            name: Entity name
            folder: Folder path (optional)
            job_stream: Parent job stream (optional)
            properties: Additional entity properties

        Returns:
            ResolvedEntity with ID and resolution details
        """
        canonical_id = self.build_canonical_id(entity_type, name, folder, job_stream)

        # Step 1: Try exact match
        if canonical_id in self._entity_cache:
            logger.debug("entity_resolved_exact", canonical_id=canonical_id)
            return ResolvedEntity(
                entity_id=self._entity_cache[canonical_id]["id"],
                canonical_id=canonical_id,
                entity_type=entity_type,
                is_new=False,
                resolution_method="exact",
                confidence=1.0,
            )

        # Step 2: Try embedding similarity (if enabled)
        if self.enable_embedding_fallback and self.embedding_service:
            similar = await self._find_similar_entity(entity_type, name, properties)
            if similar:
                self._log_merge(
                    entity_type=entity_type,
                    source=f"{name} (folder={folder})",
                    target_id=similar["id"],
                    method="embedding",
                    confidence=similar["similarity"],
                )
                logger.info(
                    "entity_resolved_embedding",
                    source=name,
                    target=similar["canonical_id"],
                    similarity=similar["similarity"],
                )
                return ResolvedEntity(
                    entity_id=similar["id"],
                    canonical_id=similar["canonical_id"],
                    entity_type=entity_type,
                    is_new=False,
                    merged_from=canonical_id,
                    resolution_method="embedding",
                    confidence=similar["similarity"],
                )

        # Step 3: Create new entity
        entity_id = self.generate_entity_id(canonical_id)
        self._entity_cache[canonical_id] = {
            "id": entity_id,
            "type": entity_type,
            "name": name,
            "folder": folder,
            "job_stream": job_stream,
            "properties": properties or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "entity_created_new",
            entity_id=entity_id,
            canonical_id=canonical_id,
            entity_type=entity_type,
        )

        return ResolvedEntity(
            entity_id=entity_id,
            canonical_id=canonical_id,
            entity_type=entity_type,
            is_new=True,
            resolution_method="new",
            confidence=1.0,
        )

    async def _find_similar_entity(
        self,
        entity_type: str,
        name: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Find similar entity using embedding similarity.

        Args:
            entity_type: Type of entity to search
            name: Entity name to match
            properties: Additional properties for context

        Returns:
            Similar entity info or None
        """
        if not self.embedding_service:
            return None

        # Build text representation for embedding
        text = f"{entity_type}: {name}"
        if properties:
            for key, value in properties.items():
                if value and key not in ("id", "created_at"):
                    text += f", {key}: {value}"

        try:
            # Generate embedding for query
            query_embedding = await self.embedding_service.embed(text)

            # Search cached embeddings
            best_match = None
            best_similarity = 0.0

            for cached_id, cached_embedding in self._embedding_cache.items():
                # Only compare same entity types
                cached_entity = self._entity_cache.get(cached_id)
                if not cached_entity or cached_entity.get("type") != entity_type:
                    continue

                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = {
                        "id": cached_entity["id"],
                        "canonical_id": cached_id,
                        "similarity": similarity,
                    }

            return best_match

        except Exception as e:
            logger.warning("embedding_search_error", error=str(e))
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    # =========================================================================
    # BATCH RESOLUTION
    # =========================================================================

    async def resolve_batch(
        self,
        entities: list[dict[str, Any]],
        entity_type: str,
    ) -> list[ResolvedEntity]:
        """
        Resolve multiple entities efficiently.

        Args:
            entities: List of entity dicts with name, folder, etc.
            entity_type: Type of all entities

        Returns:
            List of ResolvedEntity results
        """
        results = []
        new_count = 0
        merged_count = 0

        for entity in entities:
            result = await self.resolve(
                entity_type=entity_type,
                name=entity.get("name", entity.get("job_name", "")),
                folder=entity.get("folder"),
                job_stream=entity.get("job_stream"),
                properties=entity,
            )
            results.append(result)

            if result.is_new:
                new_count += 1
            elif result.merged_from:
                merged_count += 1

        logger.info(
            "batch_resolution_complete",
            entity_type=entity_type,
            total=len(entities),
            new=new_count,
            merged=merged_count,
            existing=len(entities) - new_count - merged_count,
        )

        return results

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def register_entity(
        self,
        canonical_id: str,
        entity_id: str,
        entity_type: str,
        name: str,
        embedding: list[float] | None = None,
        **kwargs,
    ):
        """
        Register an existing entity in the cache.

        Used to warm up cache from database on startup.

        Args:
            canonical_id: Full canonical ID
            entity_id: Short entity ID
            entity_type: Type of entity
            name: Entity name
            embedding: Pre-computed embedding (optional)
            **kwargs: Additional properties
        """
        self._entity_cache[canonical_id] = {
            "id": entity_id,
            "type": entity_type,
            "name": name,
            **kwargs,
        }

        if embedding:
            self._embedding_cache[canonical_id] = embedding

        logger.debug("entity_registered", canonical_id=canonical_id, entity_id=entity_id)

    def clear_cache(self):
        """Clear all caches."""
        self._entity_cache.clear()
        self._embedding_cache.clear()
        logger.info("entity_cache_cleared")

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "entities_cached": len(self._entity_cache),
            "embeddings_cached": len(self._embedding_cache),
            "merges_logged": len(self._merge_log),
        }

    # =========================================================================
    # MERGE LOGGING (Audit Trail)
    # =========================================================================

    def _log_merge(
        self,
        entity_type: str,
        source: str,
        target_id: str,
        method: str,
        confidence: float,
    ):
        """Log an entity merge for audit trail."""
        log_entry = EntityMergeLog(
            timestamp=datetime.now(timezone.utc),
            entity_type=entity_type,
            source_representation=source,
            target_id=target_id,
            resolution_method=method,
            confidence=confidence,
        )
        self._merge_log.append(log_entry)

    def get_merge_log(
        self,
        entity_type: str | None = None,
        since: datetime | None = None,
    ) -> list[EntityMergeLog]:
        """
        Get merge log entries.

        Args:
            entity_type: Filter by entity type (optional)
            since: Filter by timestamp (optional)

        Returns:
            List of merge log entries
        """
        log = self._merge_log

        if entity_type:
            log = [e for e in log if e.entity_type == entity_type]

        if since:
            log = [e for e in log if e.timestamp >= since]

        return log

    def export_merge_log(self) -> list[dict]:
        """Export merge log as list of dicts."""
        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "entity_type": entry.entity_type,
                "source": entry.source_representation,
                "target_id": entry.target_id,
                "method": entry.resolution_method,
                "confidence": entry.confidence,
            }
            for entry in self._merge_log
        ]


# =============================================================================
# SPECIALIZED RESOLVERS
# =============================================================================


class JobResolver(EntityResolver):
    """
    Specialized resolver for TWS Jobs.

    Handles:
    - Job names within job streams
    - Folder hierarchy
    - Common job name variations
    """

    async def resolve_job(
        self,
        job_name: str,
        folder: str | None = None,
        job_stream: str | None = None,
        workstation: str | None = None,
        **properties,
    ) -> ResolvedEntity:
        """
        Resolve a Job entity.

        Args:
            job_name: Job name
            folder: Folder path
            job_stream: Parent job stream
            workstation: Target workstation
            **properties: Additional job properties

        Returns:
            ResolvedEntity for the job
        """
        # Normalize job name (uppercase, remove common prefixes)
        job_name = self._normalize_job_name(job_name)

        props = {
            "workstation": workstation,
            **properties,
        }

        return await self.resolve(
            entity_type="Job",
            name=job_name,
            folder=folder,
            job_stream=job_stream,
            properties=props,
        )

    @staticmethod
    def _normalize_job_name(job_name: str) -> str:
        """Normalize job name for consistent resolution."""
        # Remove common prefixes
        prefixes = ["job ", "JOB ", "Job "]
        for prefix in prefixes:
            if job_name.startswith(prefix):
                job_name = job_name[len(prefix):]

        # Uppercase and strip
        return job_name.upper().strip()


class ErrorCodeResolver(EntityResolver):
    """
    Specialized resolver for TWS Error Codes.

    Handles:
    - AWSB error codes
    - Return codes
    - Error message variations
    """

    async def resolve_error(
        self,
        code: str,
        message: str | None = None,
        severity: str | None = None,
        **properties,
    ) -> ResolvedEntity:
        """
        Resolve an ErrorCode entity.

        Args:
            code: Error code (AWSB####X or RC)
            message: Error message
            severity: Severity level
            **properties: Additional properties

        Returns:
            ResolvedEntity for the error code
        """
        # Normalize error code
        code = self._normalize_error_code(code)

        props = {
            "message": message,
            "severity": severity,
            **properties,
        }

        return await self.resolve(
            entity_type="ErrorCode",
            name=code,
            properties=props,
        )

    @staticmethod
    def _normalize_error_code(code: str) -> str:
        """Normalize error code format."""
        code = code.upper().strip()

        # Handle "RC=12" format
        if code.startswith(("RC=", "RC ")):
            code = f"RC{code[2:].strip()}"

        # Handle "rc 12" format
        if code.lower().startswith("rc"):
            code = f"RC{code[2:].strip()}"

        return code


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_entity_resolver(
    embedding_service: Any | None = None,
    similarity_threshold: float = 0.92,
) -> EntityResolver:
    """
    Create an EntityResolver with optional embedding service.

    Args:
        embedding_service: Embedding service for similarity search
        similarity_threshold: Minimum similarity for matches

    Returns:
        Configured EntityResolver
    """
    return EntityResolver(
        embedding_service=embedding_service,
        similarity_threshold=similarity_threshold,
        enable_embedding_fallback=embedding_service is not None,
    )


def create_job_resolver(
    embedding_service: Any | None = None,
    similarity_threshold: float = 0.92,
) -> JobResolver:
    """Create a specialized Job resolver."""
    return JobResolver(
        embedding_service=embedding_service,
        similarity_threshold=similarity_threshold,
        enable_embedding_fallback=embedding_service is not None,
    )


def create_error_resolver(
    embedding_service: Any | None = None,
    similarity_threshold: float = 0.95,  # Higher threshold for error codes
) -> ErrorCodeResolver:
    """Create a specialized ErrorCode resolver."""
    return ErrorCodeResolver(
        embedding_service=embedding_service,
        similarity_threshold=similarity_threshold,
        enable_embedding_fallback=embedding_service is not None,
    )
