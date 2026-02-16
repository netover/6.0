"""
Idempotent document ingestion service for RAG systems.

v6.0: Enhanced with all 9 RAG Chunking Decisions
- Structure-aware overlap (Decision #3)
- Citation-friendly IDs (Decision #7)
- Multi-view chunk indexing (Decision #8)
- Eval-driven tuning support (Decision #9)

v5.4.2: Enhanced with advanced chunking support
- Structure-aware parsing (markdown headers, code blocks, tables)
- Semantic chunking using sentence transformers
- TWS-specific entity extraction (error codes, job names)
- Contextual enrichment for improved retrieval

Handles chunking, deduplication by SHA-256, batch embedding, and upsert to pgvector.
Integrates Prometheus metrics for observability.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from resync.knowledge.config import CFG
from resync.knowledge.interfaces import Embedder, VectorStore
from resync.knowledge.monitoring import embed_seconds, jobs_total, upsert_seconds

from .chunking import chunk_text

logger = logging.getLogger(__name__)


class IngestService:
    """
    Idempotent ingestion service:
    - Token-aware chunking (basic or advanced)
    - Deduplication by normalized chunk SHA-256
    - Batch embedding with fixed batch size
    - Upsert to pgvector with complete payload

    v5.4.2: Added advanced chunking with structure awareness
    """

    def __init__(self, embedder: Embedder, store: VectorStore, batch_size: int = 128):
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size

    async def ingest_document(
        self,
        *,
        tenant: str,
        doc_id: str,
        source: str,
        text: str,
        ts_iso: str,
        tags: list[str] | None = None,
        graph_version: int = 1,
    ) -> int:
        """
        Ingest document using basic chunking.

        For improved accuracy, use ingest_document_advanced().
        """
        chunks = list(chunk_text(text, max_tokens=512, overlap_tokens=64))
        if not chunks:
            return 0

        ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        texts_for_embed: list[str] = []

        for i, ck in enumerate(chunks):
            ck_norm = ck.strip()
            sha = hashlib.sha256(ck_norm.encode("utf-8")).hexdigest()
            # dedup duro por sha256 (consulta por payload)
            exists = await self.store.exists_by_sha256(sha, collection=CFG.collection_read)
            if exists:
                continue
            chunk_id = "{doc_id}#c{i:06d}"
            ids.append(chunk_id)
            payloads.append(
                {
                    "tenant": tenant,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source": source,
                    "section": None,
                    "ts": ts_iso,
                    "tags": tags or [],
                    "neighbors": [],
                    "graph_version": graph_version,
                    "sha256": sha,
                }
            )
            texts_for_embed.append(ck_norm)

        if not ids:
            logger.info("No new chunks to ingest (dedup hit) doc_id=%s", doc_id)
            return 0

        # embed em lotes
        total_upsert = 0
        t0 = time.perf_counter()
        for start in range(0, len(texts_for_embed), self.batch_size):
            batch_texts = texts_for_embed[start : start + self.batch_size]
            with embed_seconds.time():
                vecs = await self.embedder.embed_batch(batch_texts)
            with upsert_seconds.time():
                await self.store.upsert_batch(
                    ids=ids[start : start + self.batch_size],
                    vectors=vecs,
                    payloads=payloads[start : start + self.batch_size],
                    collection=CFG.collection_write,
                )
            total_upsert += len(batch_texts)

        jobs_total.labels(status="ingested").inc()
        logger.info(
            "Ingested %s chunks for doc_id=%s in %.2fs",
            total_upsert,
            doc_id,
            time.perf_counter() - t0,
        )
        return total_upsert

    async def ingest_document_advanced(
        self,
        *,
        tenant: str,
        doc_id: str,
        source: str,
        text: str,
        ts_iso: str,
        document_title: str = "",
        tags: list[str] | None = None,
        graph_version: int = 1,
        chunking_strategy: str = "structure_aware",  # v5.7.0: Changed default to structure_aware
        max_tokens: int = 500,
        overlap_tokens: int = 75,
        use_contextual_content: bool = True,
        # v5.7.0 PR1: Authority and Freshness signals
        doc_type: str | None = None,
        source_tier: str = "unknown",
        authority_tier: int = 3,
        is_deprecated: bool = False,
        platform: str = "all",
        environment: str = "all",
        embedding_model: str = "",
        embedding_version: str = "",
        # v6.0: New chunking features
        overlap_strategy: str = "structure",  # "constant", "structure", "none"
        enable_multi_view: bool = False,  # Enable multi-view indexing
    ) -> int:
        """
        Ingest document using advanced chunking with rich metadata.

        v6.0 Features:
        - Structure-aware overlap (Decision #3)
        - Citation-friendly IDs (Decision #7)
        - Multi-view chunk indexing (Decision #8)

        v5.7.0 Features (PR1):
        - Structure-aware parsing as DEFAULT (preserves headers, code blocks, tables)
        - Authority signals (doc_type, source_tier, authority_tier)
        - Freshness signals (last_updated, is_deprecated, doc_version)
        - Platform/Environment for two-phase filtering
        - Embedding tracking for migration safety

        Args:
            tenant: Tenant identifier
            doc_id: Document identifier
            source: Source filename
            text: Document text
            ts_iso: Timestamp in ISO format
            document_title: Document title for context
            tags: Optional tags
            graph_version: Graph version
            chunking_strategy: One of 'fixed_size', 'recursive', 'semantic',
                             'structure_aware', 'hierarchical', 'tws_optimized'
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Token overlap
            use_contextual_content: Whether to use contextualized content for embedding
            doc_type: Document type for authority scoring (policy, manual, kb, blog, forum)
            source_tier: Source credibility tier (verified, official, curated, community, generated)
            authority_tier: Authority level 1-5 (lower = more authoritative)
            is_deprecated: Whether this document is deprecated
            platform: Target platform (ios, android, mobile, web, desktop, all)
            environment: Target environment (prod, staging, dev, all)
            embedding_model: Name of embedding model used
            embedding_version: Version of embedding model
            overlap_strategy: Overlap strategy - "constant", "structure", or "none"
            enable_multi_view: Enable multi-view indexing for different query types

        Returns:
            Number of chunks ingested
        """
        from .advanced_chunking import (
            AdvancedChunker,
            ChunkingConfig,
            ChunkingStrategy,
            OverlapStrategy,
        )
        from .authority import infer_doc_type
        from .filter_strategy import normalize_metadata_value

        # Map strategy string to enum
        strategy_map = {
            "fixed_size": ChunkingStrategy.FIXED_SIZE,
            "recursive": ChunkingStrategy.RECURSIVE,
            "semantic": ChunkingStrategy.SEMANTIC,
            "structure_aware": ChunkingStrategy.STRUCTURE_AWARE,
            "hierarchical": ChunkingStrategy.HIERARCHICAL,
            "tws_optimized": ChunkingStrategy.TWS_OPTIMIZED,
        }

        # Map overlap strategy
        overlap_map = {
            "constant": OverlapStrategy.CONSTANT,
            "structure": OverlapStrategy.STRUCTURE,
            "none": OverlapStrategy.NONE,
        }

        config = ChunkingConfig(
            strategy=strategy_map.get(chunking_strategy, ChunkingStrategy.STRUCTURE_AWARE),
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            overlap_strategy=overlap_map.get(overlap_strategy, OverlapStrategy.STRUCTURE),
            enable_multi_view=enable_multi_view,
        )

        chunker = AdvancedChunker(config)
        # v6.0: Pass doc_id for citation-friendly IDs
        enriched_chunks = chunker.chunk_document(
            text,
            source=source,
            document_title=document_title,
            doc_id=doc_id,
        )

        if not enriched_chunks:
            return 0

        # Auto-infer doc_type if not provided
        inferred_doc_type = doc_type or infer_doc_type(source)

        # Normalize platform and environment values
        normalized_platform = normalize_metadata_value("platform", platform)
        normalized_environment = normalize_metadata_value("environment", environment)

        ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        texts_for_embed: list[str] = []

        for i, chunk in enumerate(enriched_chunks):
            sha = chunk.sha256

            # Dedup by SHA256
            exists = await self.store.exists_by_sha256(sha, collection=CFG.collection_read)
            if exists:
                continue

            chunk_id = f"{doc_id}#c{i:06d}"
            ids.append(chunk_id)

            # Build rich payload with metadata including PR1 authority/freshness
            payload = {
                "tenant": tenant,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "source": source,
                "section": chunk.metadata.section_path,
                "ts": ts_iso,
                "tags": tags or [],
                "neighbors": [],
                "graph_version": graph_version,
                "sha256": sha,
                # v5.4.2: Rich metadata
                "document_title": document_title,
                "chunk_type": chunk.metadata.chunk_type.value,
                "parent_headers": chunk.metadata.parent_headers,
                "section_path": chunk.metadata.section_path,
                "error_codes": chunk.metadata.error_codes,
                "job_names": chunk.metadata.job_names,
                "commands": chunk.metadata.commands,
                "token_count": chunk.metadata.token_count,
                # v5.7.0 PR1: Authority signals
                "doc_type": inferred_doc_type,
                "source_tier": source_tier,
                "authority_tier": authority_tier,
                # v5.7.0 PR1: Freshness signals
                "doc_version": graph_version,
                "last_updated": ts_iso,
                "is_deprecated": is_deprecated,
                # v5.7.0 PR1: Filtering metadata (normalized)
                "platform": normalized_platform,
                "environment": normalized_environment,
                # v5.7.0 PR1: Embedding tracking
                "embedding_model": embedding_model or CFG.embed_model,
                "embedding_version": embedding_version,
                # v6.0: Citation-friendly fields (Decision #7)
                "stable_id": chunk.metadata.stable_id,
                "snippet_preview": chunk.metadata.snippet_preview,
            }
            payloads.append(payload)

            # Use contextualized content for better retrieval
            if use_contextual_content:
                texts_for_embed.append(chunk.contextualized_content)
            else:
                texts_for_embed.append(chunk.content)

        if not ids:
            logger.info("No new chunks to ingest (dedup hit) doc_id=%s", doc_id)
            return 0

        # Embed and upsert in batches
        total_upsert = 0
        t0 = time.perf_counter()

        for start in range(0, len(texts_for_embed), self.batch_size):
            batch_texts = texts_for_embed[start : start + self.batch_size]

            with embed_seconds.time():
                vecs = await self.embedder.embed_batch(batch_texts)

            with upsert_seconds.time():
                await self.store.upsert_batch(
                    ids=ids[start : start + self.batch_size],
                    vectors=vecs,
                    payloads=payloads[start : start + self.batch_size],
                    collection=CFG.collection_write,
                )

            total_upsert += len(batch_texts)

        # -----------------------------------------------------------------
        # v6.0: Multi-view indexing (Decision #8)
        # Index additional views for different query types
        # -----------------------------------------------------------------
        if enable_multi_view:
            try:
                multi_view_chunks = chunker.chunk_document_multi_view(
                    text,
                    source=source,
                    document_title=document_title,
                    doc_id=doc_id,
                )
                # Log multi-view generation
                logger.info(
                    "multi_view_generated",
                    doc_id=doc_id,
                    views_generated=len(multi_view_chunks),
                )
            except Exception as mv_e:
                logger.warning("multi_view_indexing_failed", error=str(mv_e))

        # -----------------------------------------------------------------
        # v6.1: Document KG extraction (optional, gated by settings)
        # Never blocks ingestion on failure.
        # -----------------------------------------------------------------
        try:
            from resync.settings import get_settings
            _s = get_settings()

            if _s.KG_EXTRACTION_ENABLED:
                from resync.knowledge.kg_extraction import KGExtractor
                from resync.knowledge.kg_store.store import PostgresGraphStore

                extractor = KGExtractor()
                chunk_payloads = [
                    {
                        "chunk_id": ids[i] if i < len(ids) else f"{doc_id}#c{i:06d}",
                        "content": texts_for_embed[i] if i < len(texts_for_embed) else "",
                    }
                    for i in range(len(texts_for_embed))
                ]

                extraction = await extractor.extract(
                    doc_id=doc_id,
                    chunks=chunk_payloads,
                    tenant=tenant,
                    graph_version=graph_version,
                )

                if extraction and (extraction.concepts or extraction.edges):
                    store = PostgresGraphStore()
                    await store.upsert_from_extraction(
                        tenant=tenant,
                        graph_version=graph_version,
                        doc_id=doc_id,
                        extraction=extraction,
                    )
                    logger.info(
                        "document_kg_extracted",
                        doc_id=doc_id,
                        concepts=len(extraction.concepts),
                        edges=len(extraction.edges),
                    )
        except Exception as _kg_e:
            logger.warning("document_kg_extraction_failed", error=str(_kg_e))

        # Log stats
        chunk_types = {}
        error_code_count = 0
        for chunk in enriched_chunks:
            ct = chunk.metadata.chunk_type.value
            chunk_types[ct] = chunk_types.get(ct, 0) + 1
            error_code_count += len(chunk.metadata.error_codes)

        jobs_total.labels(status="ingested").inc()
        logger.info(
            "Advanced ingest: %s chunks for doc_id=%s in %.2fs "
            "(strategy=%s, overlap=%s, types=%s, error_codes=%s)",
            total_upsert,
            doc_id,
            time.perf_counter() - t0,
            chunking_strategy,
            overlap_strategy,
            chunk_types,
            error_code_count,
        )

        return total_upsert

    async def reindex_document(
        self,
        *,
        tenant: str,
        doc_id: str,
        source: str,
        text: str,
        ts_iso: str,
        document_title: str = "",
        tags: list[str] | None = None,
        use_advanced: bool = True,
    ) -> int:
        """
        Reindex a document, removing old chunks first.

        Args:
            tenant: Tenant identifier
            doc_id: Document identifier
            source: Source filename
            text: Document text
            ts_iso: Timestamp
            document_title: Document title
            tags: Optional tags
            use_advanced: Use advanced chunking (recommended)

        Returns:
            Number of new chunks indexed
        """
        # Delete existing chunks for this document
        try:
            await self.store.delete_by_doc_id(doc_id, collection=CFG.collection_write)
            logger.info("Deleted existing chunks for doc_id=%s", doc_id)
        except Exception as e:
            logger.warning("Could not delete existing chunks: %s", e)

        # Reindex with chosen method
        if use_advanced:
            return await self.ingest_document_advanced(
                tenant=tenant,
                doc_id=doc_id,
                source=source,
                text=text,
                ts_iso=ts_iso,
                document_title=document_title,
                tags=tags,
            )
        return await self.ingest_document(
            tenant=tenant,
            doc_id=doc_id,
            source=source,
            text=text,
            ts_iso=ts_iso,
            tags=tags,
        )
