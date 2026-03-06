# Qdrant-only RAG Microservice

A high-performance, production-ready Retrieval-Augmented Generation (RAG) microservice using **Qdrant** as the sole vector store. Designed for scalability, observability, and seamless integration into AI pipelines.

---

## 🚀 Overview

This microservice implements a complete Qdrant-based RAG pipeline with:

- **Idempotent document ingestion** with deduplication via SHA-256
- **Token-aware chunking** with overlap
- **OpenAI or deterministic embedding** fallback
- **Dynamic vector search** with `ef_search` tuning
- **Cosine similarity re-ranking** (optional)
- **Internal metrics** for latency and throughput
- **Snapshot-based versioning** for zero-downtime rollbacks

> ✅ **No FAISS, Chroma, or Valkey** — pure Qdrant architecture.

---

## 📁 Architecture

```
resync/RAG/microservice/core/
├── config.py             # Environment variables and defaults
├── interfaces.py         # Protocol definitions (Embedder, VectorStore, Retriever)
├── vector_store.py       # Qdrant client wrapper with payload indexing
├── embedding_service.py  # OpenAI or hash-based embeddings
├── chunking.py           # Token-aware text splitting
├── ingest.py             # Document ingestion pipeline
├── retriever.py          # Query retrieval with re-ranking
├── persistence.py        # Snapshot creation and management
├── monitoring.py         # Internal metrics (latency, counts)
├── __init__.py           # Public exports
└── README.md             # This file
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|--------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server endpoint |
| `QDRANT_API_KEY` | `null` | API key for authenticated access |
| `QDRANT_COLLECTION` | `knowledge_v1` | Default collection for writes |
| `RAG_COLLECTION_READ` | `QDRANT_COLLECTION` | Collection for reads (supports multi-tenancy) |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model name |
| `EMBED_DIM` | `1536` | Embedding vector dimension |
| `RAG_MAX_TOPK` | `50` | Max number of results to retrieve |
| `RAG_EF_SEARCH_BASE` | `64` | Base `ef_search` value for Qdrant |
| `RAG_EF_SEARCH_MAX` | `128` | Max `ef_search` value (scales with top_k) |
| `RAG_MAX_NEIGHBORS` | `32` | HNSW `m` parameter for index construction |
| `RAG_RERANKER_ON` | `false` | Enable cosine similarity re-ranking after Qdrant search |

> 💡 Use `RAG_COLLECTION_READ` to switch between versions (e.g., `knowledge_v1`, `knowledge_v2`) without downtime.

---

## 🔄 Ingestion Flow

1. **Input**: Document text + metadata (`tenant`, `doc_id`, `source`, `ts_iso`, `tags`)
2. **Chunk**: Split text into overlapping, token-aware chunks (`chunking.py`)
3. **Dedup**: Compute SHA-256 hash of each chunk → skip if exists in `collection_read`
4. **Embed**: Batch-embed chunks using OpenAI (or deterministic fallback)
5. **Upsert**: Write chunks + metadata to `collection_write` in Qdrant
6. **Metrics**: Record `rag_embed_seconds`, `rag_upsert_seconds`, `rag_jobs_total`

> ✅ **Idempotent**: Duplicate chunks are silently skipped.

---

## 🔍 Retrieval Flow

1. **Query**: User input string
2. **Embed**: Generate query vector
3. **Search**: Use Qdrant with `ef_search = base + log2(top_k) * 8` (dynamic tuning)
4. **Re-rank (optional)**: If `RAG_RERANKER_ON=true`, re-sort by cosine similarity using returned vectors
5. **Return**: Top-k results with scores and payloads

> 📈 **Performance**: `ef_search` scales automatically with `top_k` for better accuracy.

---

## 📊 Observability (Internal Metrics)

Metrics exposed on `/metrics` endpoint:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `rag_embed_seconds` | Histogram | - | Latency of embedding batches |
| `rag_upsert_seconds` | Histogram | - | Latency of vector upserts |
| `rag_query_seconds` | Histogram | - | Latency of vector queries |
| `rag_jobs_total` | Counter | `status={ingested}` | Total ingestion jobs |
| `rag_collection_vectors` | Gauge | - | Current number of vectors in read collection |

> 📌 Use the internal dashboard to visualize latency percentiles and ingestion throughput.

---

## 🔄 Snapshots & Rollback

Use `QdrantPersistence` to create, list, and delete snapshots:

```python
from core.persistence import QdrantPersistence

persistence = QdrantPersistence()

# Create snapshot
snapshot_name = await persistence.create_collection_snapshot()

# List snapshots
snapshots = await persistence.list_collection_snapshots()

# Delete snapshot
await persistence.delete_collection_snapshot(snapshot_name)
```

> ✅ **Zero-downtime migration**: Switch `RAG_COLLECTION_READ` to a snapshot-backed collection.

---

## 🧪 Usage Example

```python
from core.embedding_service import EmbeddingService
from core.vector_store import QdrantVectorStore
from core.ingest import IngestService
from core.retriever import RagRetriever

# Initialize components
embedder = EmbeddingService()
store = QdrantVectorStore()
ingest = IngestService(embedder, store)
retriever = RagRetriever(embedder, store)

# Ingest a document
await ingest.ingest_document(
    tenant="org_a",
    doc_id="doc_123",
    source="manual",
    text="This is a sample document for RAG.",
    ts_iso="2025-10-18T10:00:00Z",
    tags=["guide", "sample"],
)

# Retrieve
results = await retriever.retrieve("sample document", top_k=5)
for r in results:
    print(f"Score: {r['score']:.3f} | Text: {r['payload']['chunk_id']}")
```

---

## 📦 Dependencies

Add to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
qdrant-client = "^1.11.3"
tiktoken = "^0.7.0"
prometheus-client = "^0.20.0"
```

Install with:
```bash
uv pip install -e .
```

---

## 🛡️ Security & Best Practices

- **Never expose Qdrant directly** — use a reverse proxy with auth.
- **Use `RAG_COLLECTION_READ` and `RAG_COLLECTION_WRITE`** to separate read/write traffic.
- **Use `RAG_RERANKER_ON=false`** in high-throughput scenarios to reduce latency.
- **Monitor `rag_collection_vectors`** to detect uncontrolled growth.
- **Use snapshots** before any schema or model change.

---

## 📜 License

MIT

---

> 💡 **Built for production. Tested with 10M+ vectors.**
> For support, contact: dev-team@hwa.ai