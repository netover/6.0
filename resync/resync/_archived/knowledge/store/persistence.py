"""
PostgreSQL table snapshot management for versioning and rollback.

Provides export/import operations for document_embeddings table to enable safe migrations and rollbacks.
Uses PostgreSQL COPY command for efficient bulk operations.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from resync.knowledge.config import CFG
import aiofiles

logger = logging.getLogger(__name__)

# Optional import
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None
    ASYNCPG_AVAILABLE = False


class PgVectorPersistence:
    """
    Handles PostgreSQL table snapshots for versioning and rollback.

    Exports document_embeddings table to JSON files for backup,
    and can restore from those backups.
    """

    def __init__(self, database_url: str | None = None):
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is required for persistence")

        self._database_url = database_url or CFG.database_url
        # Clean URL for asyncpg
        if self._database_url.startswith("postgresql+asyncpg://"):
            self._database_url = self._database_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            )

        self._snapshot_dir = Path(os.getenv("RAG_SNAPSHOT_DIR", "/tmp/rag_snapshots"))
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    async def create_collection_snapshot(self, collection: str | None = None) -> str:
        """
        Create a snapshot of the specified collection.

        Exports all documents in the collection to a JSON file.

        Returns:
            Snapshot filename
        """
        col = collection or CFG.collection_write
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{col}_{timestamp}.json"
        filepath = self._snapshot_dir / filename

        conn = await asyncpg.connect(self._database_url)
        try:
            # Export documents (without embeddings for space efficiency)
            rows = await conn.fetch(
                """
                SELECT document_id, chunk_id, content, metadata, sha256
                FROM document_embeddings
                WHERE collection_name = $1
                ORDER BY document_id, chunk_id
            """,
                col,
            )

            documents = []
            for row in rows:
                documents.append(
                    {
                        "document_id": row["document_id"],
                        "chunk_id": row["chunk_id"],
                        "content": row["content"],
                        "metadata": dict(row["metadata"]) if row["metadata"] else {},
                        "sha256": row["sha256"],
                    }
                )

            # Write to file
            async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "collection": col,
                        "timestamp": timestamp,
                        "document_count": len(documents),
                        "documents": documents,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            logger.info("Snapshot created for %s: %s (%d docs)", col, filename, len(documents))
            return filename

        finally:
            await conn.close()

    async def list_collection_snapshots(self, collection: str | None = None) -> list[str]:
        """
        List all snapshots for the specified collection.

        Returns:
            List of snapshot filenames
        """
        col = collection or CFG.collection_write
        prefix = f"{col}_"

        snapshots = []
        for f in self._snapshot_dir.glob(f"{prefix}*.json"):
            snapshots.append(f.name)

        return sorted(snapshots, reverse=True)  # Most recent first

    async def delete_collection_snapshot(
        self, snapshot_name: str, collection: str | None = None
    ) -> None:
        """
        Delete a specific snapshot file.
        """
        filepath = self._snapshot_dir / snapshot_name
        if filepath.exists():
            filepath.unlink()
            logger.info("Snapshot deleted: %s", snapshot_name)
        else:
            logger.warning("Snapshot not found: %s", snapshot_name)

    async def restore_collection_snapshot(
        self,
        snapshot_name: str,
        collection: str | None = None,
        embedder=None,
    ) -> int:
        """
        Restore a collection from a snapshot.

        Note: Requires re-embedding documents since embeddings are not stored in snapshots.

        Args:
            snapshot_name: Snapshot filename
            collection: Target collection name (defaults to original)
            embedder: Embedder instance for regenerating embeddings

        Returns:
            Number of documents restored
        """
        filepath = self._snapshot_dir / snapshot_name
        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_name}")

        async with aiofiles.open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        target_col = collection or data["collection"]
        documents = data["documents"]

        if not documents:
            return 0

        conn = await asyncpg.connect(self._database_url)
        try:
            # Clear existing documents in target collection
            await conn.execute(
                """
                DELETE FROM document_embeddings
                WHERE collection_name = $1
            """,
                target_col,
            )

            # Re-insert documents
            # Note: If embedder is provided, regenerate embeddings
            # Otherwise, insert without embeddings (they'll need to be regenerated later)

            for doc in documents:
                embedding_str = None

                if embedder:
                    # Generate embedding
                    embedding = await embedder.embed(doc["content"])
                    embedding_str = f"[{','.join(str(x) for x in embedding)}]"

                await conn.execute(
                    """
                    INSERT INTO document_embeddings
                    (collection_name, document_id, chunk_id, content, embedding, metadata, sha256)
                    VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb, $7)
                    ON CONFLICT (collection_name, document_id, chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        sha256 = EXCLUDED.sha256,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    target_col,
                    doc["document_id"],
                    doc["chunk_id"],
                    doc["content"],
                    embedding_str,
                    json.dumps(doc["metadata"]),
                    doc["sha256"],
                )

            logger.info(
                "Restored %d documents to %s from %s", len(documents), target_col, snapshot_name
            )
            return len(documents)

        finally:
            await conn.close()
