# pylint: skip-file
# mypy: ignore-errors
"""
File Ingestor implementation for RAG.

This module provides the concrete implementation of IFileIngestor,
handling file saving and ingestion into the knowledge graph.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from resync.core.interfaces import IFileIngestor
from resync.knowledge.ingestion.document_converter import DoclingConverter
from resync.knowledge.ingestion.ingest import IngestService
from resync.settings import get_settings

logger = logging.getLogger(__name__)


class FileIngestor(IFileIngestor):
    """
    Concrete implementation of IFileIngestor.
    """

    def __init__(self, ingest_service: IngestService):
        self.ingest_service = ingest_service
        self.converter = DoclingConverter()
        self.settings = get_settings()

    async def save_uploaded_file(self, file_name: str, file_content: Any) -> Path:
        """
        Saves an uploaded file to the RAG directory with security hardening.
        """
        # Security: Prevent path traversal by taking only the filename
        safe_name = os.path.basename(file_name)
        if not safe_name:
            raise ValueError("Invalid filename")

        upload_dir = Path(self.settings.upload_dir).resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)

        destination = upload_dir / safe_name

        # Ensure destination is within upload_dir
        if not str(destination.resolve()).startswith(str(upload_dir)):
            raise ValueError("Potential path traversal attack detected")

        try:
            # Use asyncio.to_thread to avoid blocking the event loop with synchronous I/O
            def _save():
                with open(destination, "wb") as buffer:
                    shutil.copyfileobj(file_content, buffer)

            await asyncio.to_thread(_save)

            logger.info("file_saved", extra={"path": str(destination)})
            return destination
        except Exception as e:
            logger.error(
                "failed_to_save_file", extra={"error": str(e), "path": str(destination)}
            )
            raise

    async def ingest_file(self, file_path: Path) -> bool:
        """
        Ingests a single file into the knowledge graph.
        """
        try:
            logger.info("starting_file_ingestion", extra={"path": str(file_path)})

            # 1. Convert document to markdown
            result = await self.converter.convert(file_path)
            if result.status != "success":
                logger.error(
                    "conversion_failed",
                    extra={"error": result.error, "path": str(file_path)},
                )
                return False

            # 2. Ingest into knowledge graph
            # We use a default tenant and graph version for now
            mtime = os.path.getmtime(file_path)
            ts_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

            chunks_count = await self.ingest_service.ingest_document_advanced(
                tenant="default",
                doc_id=file_path.name,
                source=file_path.name,
                text=result.markdown,
                ts_iso=ts_iso,
                document_title=result.metadata.get("title", file_path.stem),
            )

            logger.info(
                "file_ingested_successfully",
                extra={"path": str(file_path), "chunks": chunks_count},
            )
            return True
        except Exception as e:
            logger.error(
                "ingestion_failed", extra={"error": str(e), "path": str(file_path)}
            )
            return False

    async def shutdown(self) -> None:
        """
        Shutdown the file ingestor and its dependencies.
        """
        try:
            if hasattr(self.ingest_service.store, "close"):
                await self.ingest_service.store.close()
            logger.info("file_ingestor_shutdown_successfully")
        except Exception as e:
            logger.error("file_ingestor_shutdown_failed", extra={"error": str(e)})
