# pylint
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
        Save an uploaded file to the RAG upload directory with security hardening.

        Security/Performance:
            - Prevent path traversal by normalizing to a basename
            - Enforce a maximum upload size (default: 10 MiB)
            - Use `asyncio.to_thread()` for blocking disk I/O
            - Ensure the resolved destination stays within the configured upload_dir
        """
        # Security: Prevent path traversal by taking only the filename
        safe_name = os.path.basename(file_name)
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("Invalid filename")

        upload_dir = Path(self.settings.upload_dir).resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)

        destination = (upload_dir / safe_name)

        # Security: Ensure destination is within upload_dir (robust against prefix tricks)
        destination_resolved = destination.resolve()
        if not destination_resolved.is_relative_to(upload_dir):
            raise ValueError("Potential path traversal attack detected")

        # Security: Limit upload size to mitigate DoS via oversized files
        default_max_bytes = 10 * 1024 * 1024  # 10 MiB
        max_bytes_env = os.getenv("RAG_MAX_UPLOAD_BYTES") or os.getenv("APP_RAG_MAX_UPLOAD_BYTES")
        try:
            max_upload_bytes = int(max_bytes_env) if max_bytes_env else int(
                getattr(self.settings, "rag_max_upload_bytes", default_max_bytes)
            )
        except (TypeError, ValueError):
            max_upload_bytes = default_max_bytes

        if max_upload_bytes <= 0:
            # Fail-safe: do not allow "unlimited" via misconfiguration
            max_upload_bytes = default_max_bytes

        try:
            def _save() -> None:
                bytes_written = 0
                with open(destination_resolved, "wb") as buffer:
                    while True:
                        chunk = file_content.read(1024 * 1024)  # 1 MiB
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > max_upload_bytes:
                            raise ValueError("File too large")
                        buffer.write(chunk)

            await asyncio.to_thread(_save)

            logger.info("file_saved", extra={"path": str(destination_resolved), "bytes_written_max": max_upload_bytes})
            return destination_resolved
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error(
                "failed_to_save_file",
                extra={"error": str(e), "path": str(destination_resolved)},
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
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
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
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("file_ingestor_shutdown_failed", extra={"error": str(e)})
