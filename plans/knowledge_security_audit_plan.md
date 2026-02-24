# 🔒 Security & Performance Audit Plan - resync/knowledge

## Executive Summary

**Status**: Based on line-by-line analysis of the current codebase, many issues from the original audit report have already been addressed. This document outlines the actual remaining issues and a plan to fix them.

---

## Analysis Methodology

### Files Analyzed
- `resync/knowledge/ingestion/ingest.py` (511 lines)
- `resync/knowledge/ingestion/embedding_service.py` (561 lines)
- `resync/knowledge/ingestion/pipeline.py` (248 lines)
- `resync/knowledge/config.py` (369 lines)
- `resync/knowledge/interfaces.py` (312 lines)
- `resync/knowledge/retrieval/reranker.py` (257 lines)
- `resync/knowledge/retrieval/hybrid_retriever.py` (1016 lines)
- `resync/knowledge/store/pgvector_store.py` (541 lines)
- `resync/knowledge/monitoring.py` (40 lines)

---

## Issues Status: Verified vs Original Report

| Issue ID | Original Report Status | Current Status | Notes |
|----------|----------------------|---------------|-------|
| P0-1 | `pylint: disable=all` | ⚠️ PARTIAL | Found `# pylint` and `# mypy` comments on lines 1-2 (incomplete, not blocking) |
| P0-2 | N+1 queries | ✅ FIXED | Already uses `exists_batch_by_sha256()` with FIX comments |
| P0-3 | slowapi in domain | ✅ N/A | No slowapi usage found in knowledge module |
| P0-4 | Path traversal | ✅ FIXED | File path validation exists in pipeline.py |
| P0-5 | `item["embedding"]` | ✅ FIXED | LiteLLM API call uses correct format |
| P0-6 | Sync protocols | ✅ FIXED | interfaces.py has async methods |
| P0-7 | Circuit breaker | ✅ FIXED | Uses `@circuit_protected` decorator |

---

## Real Issues Found

### P1 - Inconsistent Logging (12 files)

**Severity**: P1 (High)
**Files**: Multiple files in knowledge module
**Issue**: Using `logging.getLogger()` instead of `structlog.get_logger()`

```python
# Current (INCORRECT)
import logging
logger = logging.getLogger(__name__)

# Expected (CORRECT)
import structlog
logger = structlog.get_logger(__name__)
```

**Files affected**:
- `resync/knowledge/store/feedback_store.py`
- `resync/knowledge/retrieval/tws_relations.py`
- `resync/knowledge/retrieval/reranker_interface.py`
- `resync/knowledge/retrieval/hybrid_retriever.py`
- `resync/knowledge/kg_store/store.py`
- `resync/knowledge/kg_extraction/extractor.py`
- `resync/knowledge/ingestion/pipeline.py`
- `resync/knowledge/ingestion/ingest.py`
- `resync/knowledge/ingestion/embeddings.py`
- `resync/knowledge/ingestion/document_converter.py`
- `resync/knowledge/ingestion/chunking_eval.py`
- `resync/knowledge/ingestion/advanced_chunking.py`

---

### P1 - Blocking predict() Calls in Async Context (4 instances)

**Severity**: P1 (High)
**Files**: `reranker.py`, `reranker_interface.py`
**Issue**: `model.predict()` is a synchronous call that blocks the event loop

```python
# Current (BLOCKING)
scores = model.predict(pairs)

# Expected (NON-BLOCKING)
scores = await asyncio.to_thread(model.predict, pairs)
```

**Locations**:
- `resync/knowledge/retrieval/reranker.py:126` - warmup
- `resync/knowledge/retrieval/reranker.py:185` - main predict
- `resync/knowledge/retrieval/reranker_interface.py:472` - main predict
- `resync/knowledge/retrieval/reranker_interface.py:532` - warmup

---

### P2 - Duplicate Logger Definition (hybrid_retriever.py:36,52)

**Severity**: P2 (Medium)
**File**: `resync/knowledge/retrieval/hybrid_retriever.py`
**Issue**: Logger defined twice - line 36 (structlog) and line 52 (logging)

---

## Action Plan

### Phase 1: Fix P1 Issues (Critical)

1. **Fix logging inconsistency** - Replace `logging.getLogger()` with `structlog.get_logger()` in 12 files
2. **Fix blocking predict calls** - Wrap `model.predict()` with `asyncio.to_thread()` in 4 locations

### Phase 2: Fix P2 Issues

3. **Remove duplicate logger** - Delete line 52 in `hybrid_retriever.py`

### Phase 3: Verification

4. Run ruff check to verify no lint errors
5. Verify async tests pass

---

## Priority Summary

| Priority | Count | Issue Type |
|----------|-------|------------|
| P0 | 0 | None remaining (previously fixed) |
| P1 | 2 | Logging inconsistency, blocking calls |
| P2 | 1 | Duplicate logger |

---

## Technical Stack Compliance

| Component | Status | Notes |
|----------|--------|-------|
| Python 3.14 | ✅ Compatible | No deprecated patterns found |
| FastAPI | ✅ Integrated | Proper async/await usage |
| Pydantic v2 | ✅ Used | Config validation in place |
| asyncpg | ✅ Pool implemented | Connection pooling works |
| structlog | ⚠️ Partial | Need to fix 12 files |
| OpenTelemetry | ✅ Present | Distributed tracing configured |
| Prometheus | ✅ Present | Metrics exported |
| Sentry | ✅ Integrated | Error tracking in place |

---

## Conclusion

The original audit report identified issues that have largely been addressed in the current codebase. The main remaining work involves:
1. Standardizing logging to use structlog consistently (12 files)
2. Making ML model inference non-blocking (4 instances)
3. Cleaning up a duplicate logger definition

These are straightforward fixes that can be implemented quickly.
