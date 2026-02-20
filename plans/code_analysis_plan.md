# Code Analysis and Fix Plan

## Overview
This project has **1513+ mypy errors** plus numerous Ruff and Pylint issues across the codebase. The errors are organized into 10 parts.

## Error Categories Identified

### 1. MYPY Errors (Type Checking)
- **Missing attributes**: `UserRepository` missing methods like `get_by_username`, `create`, `update`, `delete`, `list_all`
- **Incompatible types**: Assignment type mismatches (e.g., `bytes` to `str`)
- **Missing library stubs**: `passlib`, `jose`, `psutil`, `toml`, `pandas`, `scipy`, etc.
- **Missing `await`**: Async functions not awaited
- **Return type mismatches**: Functions returning wrong types
- **Undefined names**: Variables referenced before definition

### 2. RUFF Errors (Linting)
- **E402**: Module level imports not at top
- **F821**: Undefined names (`json` not imported)
- **F841**: Unused variables

### 3. PYLINT Issues (Code Smells)
- Duplicate code blocks
- Missing docstrings
- Convention issues

## Files with Most Critical Errors

### Priority 1 - Core API Files
- `resync/api/auth/service.py` - UserRepository missing methods
- `resync/api/core/security.py` - Type issues with password handling
- `resync/api/websocket/handlers.py` - Multiple type issues

### Priority 2 - Core Module Files  
- `resync/core/cache/*.py` - Cache implementations
- `resync/core/metrics/*.py` - Metrics collection
- `resync/core/langgraph/*.py` - LangGraph integration

### Priority 3 - Knowledge Module
- `resync/knowledge/retrieval/*.py` - RAG retrieval
- `resync/knowledge/ingestion/*.py` - Document ingestion

## Fix Strategy

### Phase 1: Quick Wins
1. Install missing type stubs: `pip install types-passlib types-python-jose types-psutil types-toml pandas-stubs`
2. Fix undefined name `json` in `hybrid_retriever.py` - add import
3. Fix unused variables (remove or use them)
4. Fix imports not at top of files

### Phase 2: Type Fixes
1. Fix type annotations in critical files
2. Add proper return types to async functions
3. Fix attribute access on Optional types

### Phase 3: Interface Alignment
1. Add missing methods to repository interfaces
2. Align method signatures across implementations
3. Fix protocol/implementation mismatches

## Implementation Notes

The errors are too numerous to fix manually in one pass. The recommended approach is:
1. Start with Phase 1 quick fixes
2. Run type checker again
3. Prioritize remaining errors by impact
4. Use `# type: ignore` sparingly for edge cases

---

## Task Execution Plan

This is a massive codebase with 1513+ errors. We will analyze and fix errors file by file.

### Task List:
- [ ] Task 1: Analyze and fix erros_parte_01.txt (RUFF + first part of MYPY)
- [ ] Task 2: Analyze and fix erros_parte_02.txt (MYPY continuation)
- [ ] Task 3: Analyze and fix erros_parte_03.txt (PYLINT)
- [ ] Task 4: Analyze and fix erros_parte_04.txt
- [ ] Task 5: Analyze and fix erros_parte_05.txt
- [ ] Task 6: Analyze and fix erros_parte_06.txt
- [ ] Task 7: Analyze and fix erros_parte_07.txt
- [ ] Task 8: Analyze and fix erros_parte_08.txt
- [ ] Task 9: Analyze and fix erros_parte_09.txt
- [ ] Task 10: Analyze and fix erros_parte_10.txt
