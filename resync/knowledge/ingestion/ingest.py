# pylint
# mypy
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
import structlog
import time
from typing import Any

from resync.knowledge.config import get_config

CFG = get_config()
from resync.knowledge.interfaces import Embedder, VectorStore
from resync.knowledge.monitoring import embed_seconds, jobs_total, upsert_seconds
from resync.settings import get_settings

from .chunking import chunk_text

logger = structlog.get_logger(__name__)


class IngestService:
    """
    Idempotent ingestion service:
    - Token-aware chunking (basic or advanced)
    - Deduplication by normalized chunk SHA-256
    - Batch embedding with fixed batch size
    - Upsert to pgvector with complete payload

    v5.4.2: Added advanced chunking with structure awareness
    """

    def __init__(
        self, 
        embedder: Embedder, 
        store: VectorStore, 
        kg_extractor: Any | None = None,
        kg_store: Any | None = None,
        batch_size: int = 128
    ):
        self.embedder = embedder
        self.store = store
        self.kg_extractor = kg_extractor
        self.kg_store = kg_store
        self.batch_size = batch_size
        self._settings = get_settings()

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
        import asyncio
        
        def _get_chunks() -> list[str]:
            return list(chunk_text(text, max_tokens=512, overlap_tokens=64))
            
        chunks = await asyncio.to_thread(_get_chunks)
        if not chunks:
            return 0
        ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        texts_for_embed: list[str] = []
        # FIX N+1: batch SHA-256 dedup — one DB call instead of N
        normalized = [ck.strip() for ck in chunks]
        shas = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in normalized]
        existing_shas = await self.store.exists_batch_by_sha256(
            shas, collection=CFG.collection_read
        )
        for i, (ck_norm, sha) in enumerate(zip(normalized, shas)):
            if sha in existing_shas:
                continue
            chunk_id = f"{doc_id}#c{i:06d}"
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
            logger.info("no_new_chunks_dedup_hit", doc_id=doc_id)
            return 0
        total_upsert = 0
        t0 = time.perf_counter()
        for start in range(0, len(texts_for_embed), self.batch_size):
            batch_texts = texts_for_embed[start : start + self.batch_size]
            # FIX P1-005: Use perf_counter for accurate async timing
            t_embed = time.perf_counter()
            vecs = await self.embedder.embed_batch(batch_texts)
            embed_seconds.observe(time.perf_counter() - t_embed)
            t_upsert = time.perf_counter()
            await self.store.upsert_batch(
                ids=ids[start : start + self.batch_size],
                vectors=vecs,
                payloads=payloads[start : start + self.batch_size],
                collection=CFG.collection_write,
            )
            upsert_seconds.observe(time.perf_counter() - t_upsert)
            total_upsert += len(batch_texts)
        jobs_total.labels(status="ingested").inc()
        logger.info(
            "document_ingested_basic",
            chunks=total_upsert,
            doc_id=doc_id,
            elapsed_seconds=time.perf_counter() - t0,
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
        chunking_strategy: str = "structure_aware",
        max_tokens: int = 500,
        overlap_tokens: int = 75,
        use_contextual_content: bool = True,
        doc_type: str | None = None,
        source_tier: str = "unknown",
        authority_tier: int = 3,
        is_deprecated: bool = False,
        platform: str = "all",
        environment: str = "all",
        embedding_model: str = "",
        embedding_version: str = "",
        overlap_strategy: str = "structure",
        enable_multi_view: bool = False,
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
            doc_type: Document type for authority scoring
                (policy, manual, kb, blog, forum)
            source_tier: Source credibility tier
                (verified, official, curated, community, generated)
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

        strategy_map = {
            "fixed_size": ChunkingStrategy.FIXED_SIZE,
            "recursive": ChunkingStrategy.RECURSIVE,
            "semantic": ChunkingStrategy.SEMANTIC,
            "structure_aware": ChunkingStrategy.STRUCTURE_AWARE,
            "hierarchical": ChunkingStrategy.HIERARCHICAL,
            "tws_optimized": ChunkingStrategy.TWS_OPTIMIZED,
        }
        overlap_map = {
            "constant": OverlapStrategy.CONSTANT,
            "structure": OverlapStrategy.STRUCTURE,
            "none": OverlapStrategy.NONE,
        }
        config = ChunkingConfig(
            strategy=strategy_map.get(
                chunking_strategy, ChunkingStrategy.STRUCTURE_AWARE
            ),
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            overlap_strategy=overlap_map.get(
                overlap_strategy, OverlapStrategy.STRUCTURE
            ),
            enable_multi_view=enable_multi_view,
        )
        chunker = AdvancedChunker(config)
        import asyncio
        enriched_chunks = await asyncio.to_thread(
            chunker.chunk_document, text, source=source, document_title=document_title, doc_id=doc_id
        )
        if not enriched_chunks:
            return 0
        inferred_doc_type = doc_type or infer_doc_type(source)
        normalized_platform = normalize_metadata_value("platform", platform)
        normalized_environment = normalize_metadata_value("environment", environment)
        ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        texts_for_embed: list[str] = []
        # FIX N+1: batch SHA-256 dedup — one DB call instead of N
        all_shas = [hashlib.sha256(chunk.content.encode("utf-8")).hexdigest() for chunk in enriched_chunks]
        existing_shas = await self.store.exists_batch_by_sha256(
            all_shas, collection=CFG.collection_read
        )
        for i, chunk in enumerate(enriched_chunks):
            sha = all_shas[i]
            if sha in existing_shas:
                continue
            chunk_id = f"{doc_id}#c{i:06d}"
            ids.append(chunk_id)
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
                "document_title": document_title,
                "chunk_type": chunk.metadata.chunk_type.value,
                "parent_headers": chunk.metadata.parent_headers,
                "section_path": chunk.metadata.section_path,
                "error_codes": chunk.metadata.error_codes,
                "job_names": chunk.metadata.job_names,
                "commands": chunk.metadata.commands,
                "token_count": chunk.metadata.token_count,
                "doc_type": inferred_doc_type,
                "source_tier": source_tier,
                "authority_tier": authority_tier,
                "doc_version": graph_version,
                "last_updated": ts_iso,
                "is_deprecated": is_deprecated,
                "platform": normalized_platform,
                "environment": normalized_environment,
                "embedding_model": embedding_model or CFG.embed_model,
                "embedding_version": embedding_version,
                "stable_id": chunk.metadata.stable_id,
                "snippet_preview": chunk.metadata.snippet_preview,
            }
            payloads.append(payload)
            if use_contextual_content:
                texts_for_embed.append(chunk.contextualized_content)
            else:
                texts_for_embed.append(chunk.content)
        if not ids:
            logger.info("no_new_chunks_dedup_hit", doc_id=doc_id)
            return 0
        total_upsert = 0
        t0 = time.perf_counter()
        for start in range(0, len(texts_for_embed), self.batch_size):
            batch_texts = texts_for_embed[start : start + self.batch_size]
            # FIX P1-005: Use perf_counter for accurate async timing
            t_embed = time.perf_counter()
            vecs = await self.embedder.embed_batch(batch_texts)
            embed_seconds.observe(time.perf_counter() - t_embed)
            t_upsert = time.perf_counter()
            await self.store.upsert_batch(
                ids=ids[start : start + self.batch_size],
                vectors=vecs,
                payloads=payloads[start : start + self.batch_size],
                collection=CFG.collection_write,
            )
            upsert_seconds.observe(time.perf_counter() - t_upsert)
            total_upsert += len(batch_texts)
        # FIX P0-004: Actually upsert multi-view chunks to vector store
        if enable_multi_view:
            try:
                import asyncio
                multi_view_chunks = await asyncio.to_thread(
                    chunker.chunk_document_multi_view,
                    text, source=source, document_title=document_title, doc_id=doc_id
                )
                if multi_view_chunks:
                    # FIX N+1: batch SHA-256 dedup for multi-view chunks — one DB call
                    mv_texts: list[str] = []
                    mv_ids: list[str] = []
                    mv_payloads: list[dict[str, Any]] = []
                    mv_all_shas = [
                        hashlib.sha256(mvc.raw_content.encode("utf-8")).hexdigest()
                        for mvc in multi_view_chunks
                    ]
                    mv_existing_shas = await self.store.exists_batch_by_sha256(
                        mv_all_shas, collection=CFG.collection_read
                    )
                    for mvc, sha_mv in zip(multi_view_chunks, mv_all_shas):
                        if sha_mv in mv_existing_shas:
                            continue
                            
                        # Generate a chunk for each enabled view
                        for view_type in config.enabled_views:
                            view_content = mvc.get_view(view_type)
                            if not view_content:
                                continue
                                
                            mv_id = f"{mvc.chunk_id}_view_{view_type.value}"
                            mv_ids.append(mv_id)
                            mv_payloads.append(
                                {
                                    "tenant": tenant,
                                    "doc_id": doc_id,
                                    "chunk_id": mv_id,
                                    "source": source,
                                    "section": mvc.metadata.get("section_path"),
                                    "ts": ts_iso,
                                    "tags": tags or [],
                                    "neighbors": [],
                                    "graph_version": graph_version,
                                    "sha256": sha_mv,
                                    "document_title": document_title,
                                    "chunk_type": mvc.metadata.get("chunk_type"),
                                    "parent_headers": mvc.metadata.get("parent_headers"),
                                    "section_path": mvc.metadata.get("section_path"),
                                    "view_type": view_type.value,
                                    "view_specific_content": view_content,
                                    "base_chunk_id": mvc.chunk_id,
                                    "embedding_model": embedding_model or CFG.embed_model,
                                    "embedding_version": embedding_version,
                                }
                            )
                            mv_texts.append(view_content)
                    # Embed and upsert multi-view chunks
                    if mv_texts:
                        for mv_start in range(0, len(mv_texts), self.batch_size):
                            mv_batch_texts = mv_texts[mv_start : mv_start + self.batch_size]
                            t_mv_embed = time.perf_counter()
                            mv_vecs = await self.embedder.embed_batch(mv_batch_texts)
                            embed_seconds.observe(time.perf_counter() - t_mv_embed)
                            t_mv_upsert = time.perf_counter()
                            await self.store.upsert_batch(
                                ids=mv_ids[mv_start : mv_start + self.batch_size],
                                vectors=mv_vecs,
                                payloads=mv_payloads[mv_start : mv_start + self.batch_size],
                                collection=CFG.collection_write,
                            )
                            upsert_seconds.observe(time.perf_counter() - t_mv_upsert)
                        logger.info(
                            "multi_view_upserted",
                            doc_id=doc_id,
                            views_generated=len(multi_view_chunks),
                            views_upserted=len(mv_texts),
                        )
                    else:
                        logger.info(
                            "multi_view_skipped_all_duplicates",
                            doc_id=doc_id,
                            views_total=len(multi_view_chunks)
                        )
                else:
                    logger.info(
                        "multi_view_no_chunks",
                        doc_id=doc_id
                    )
            except Exception as mv_e:
                logger.warning("multi_view_indexing_failed", error=str(mv_e))
        try:
            if self._settings.KG_EXTRACTION_ENABLED and self.kg_extractor and self.kg_store:
                chunk_payloads = [
                    {
                        "chunk_id": ids[i] if i < len(ids) else f"{doc_id}#c{i:06d}",
                        "content": texts_for_embed[i]
                        if i < len(texts_for_embed)
                        else "",
                    }
                    for i in range(len(texts_for_embed))
                ]
                extraction = await self.kg_extractor.extract(
                    doc_id=doc_id,
                    chunks=chunk_payloads,
                    tenant=tenant,
                    graph_version=graph_version,
                )
                if extraction and (extraction.concepts or extraction.edges):
                    await self.kg_store.upsert_from_extraction(
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
        chunk_types = {}
        error_code_count = 0
        for chunk in enriched_chunks:
            ct = chunk.metadata.chunk_type.value
            chunk_types[ct] = chunk_types.get(ct, 0) + 1
            error_code_count += len(chunk.metadata.error_codes)
        jobs_total.labels(status="ingested").inc()
        logger.info(
            "document_ingested_advanced",
            chunks=total_upsert,
            doc_id=doc_id,
            elapsed_seconds=time.perf_counter() - t0,
            strategy=chunking_strategy,
            overlap_strategy=overlap_strategy,
            chunk_types=chunk_types,
            error_code_count=error_code_count,
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
        try:
            await self.store.delete_by_doc_id(doc_id, collection=CFG.collection_write)
            logger.info("deleted_existing_chunks", doc_id=doc_id)
        except Exception as e:
            logger.warning("could_not_delete_existing_chunks", error=str(e))
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
