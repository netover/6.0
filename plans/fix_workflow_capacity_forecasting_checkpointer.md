# Fix Plan: workflow_capacity_forecasting.py PostgresSaver Issue

## Issue Summary

In [`resync/workflows/workflow_capacity_forecasting.py`](resync/workflows/workflow_capacity_forecasting.py:647-656), the current code attempts to instantiate `PostgresSaver` with `db.connection()` (a SQLAlchemy connection), but `PostgresSaver` expects a psycopg3 connection/pool. This will fail and silently fall back to `MemorySaver`.

## Current Code (lines 647-656)

```python
async with get_async_session() as db:
    if POSTGRES_SAVER_AVAILABLE and PostgresSaver is not None:
        try:
            checkpointer = PostgresSaver(db.connection())  # WRONG - SQLAlchemy connection
        except Exception as e:
            logger.warning("postgres_checkpointer_failed_using_memory", error=str(e))
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
    else:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
```

## Required Changes

### 1. Update Imports (lines 46-51)

**Current:**
```python
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_SAVER_AVAILABLE = True
except ImportError:
    PostgresSaver = None  # type: ignore
    POSTGRES_SAVER_AVAILABLE = False
```

**Should be:**
```python
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    POSTGRES_SAVER_AVAILABLE = True
except ImportError:
    # Fallback to older import path
    try:
        from langgraph_checkpoint_postgres import AsyncPostgresSaver
        POSTGRES_SAVER_AVAILABLE = True
    except ImportError:
        AsyncPostgresSaver = None  # type: ignore
        POSTGRES_SAVER_AVAILABLE = False
```

### 2. Add Helper Function for Database URL

Add a helper function to get the database URL (similar to checkpointer.py):

```python
def _get_checkpointer_db_url() -> str:
    """Get PostgreSQL connection URL for checkpointer."""
    from resync.settings import settings
    import os
    
    # Try settings first
    if hasattr(settings, "database_url") and settings.database_url:
        return settings.database_url
    
    # Try environment variable
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    # Build from components
    host = getattr(settings, "db_host", None) or os.getenv("DB_HOST", "localhost")
    port = getattr(settings, "db_port", None) or os.getenv("DB_PORT", "5432")
    user = getattr(settings, "db_user", None) or os.getenv("DB_USER", "postgres")
    password = getattr(settings, "db_password", None) or os.getenv("DB_PASSWORD", "")
    database = getattr(settings, "db_name", None) or os.getenv("DB_NAME", "resync")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"
```

### 3. Fix Checkpointer Instantiation (lines 647-656)

**Current:**
```python
async with get_async_session() as db:
    if POSTGRES_SAVER_AVAILABLE and PostgresSaver is not None:
        try:
            checkpointer = PostgresSaver(db.connection())
        except Exception as e:
            logger.warning("postgres_checkpointer_failed_using_memory", error=str(e))
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
    else:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
```

**Should be:**
```python
if POSTGRES_SAVER_AVAILABLE and AsyncPostgresSaver is not None:
    try:
        db_url = _get_checkpointer_db_url()
        checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
        await checkpointer.setup()
    except Exception as e:
        logger.warning("postgres_checkpointer_failed_using_memory", error=str(e))
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
else:
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
```

Note: The `db` session is no longer needed for checkpointer creation, so we don't need to wrap this in `async with get_async_session() as db:` for the checkpointer part. However, we still need the `db` for the `fetch_metrics_node`, so we need to restructure to get the checkpointer first, then use the db.

## Updated Function Structure

```python
async def run_capacity_forecast(
    workstation: str | None = None,
    lookback_days: int = 30,
    forecast_days: int = 90
) -> dict[str, Any]:
    """Run Capacity Forecasting workflow."""

    from resync.settings import settings
    
    model_name = getattr(settings, "agent_model_name", None) or getattr(settings, "llm_model", "gpt-4o")
    llm = LLMFactory.get_langchain_llm(model=model_name)

    # Create checkpointer first (outside the db session)
    if POSTGRES_SAVER_AVAILABLE and AsyncPostgresSaver is not None:
        try:
            db_url = _get_checkpointer_db_url()
            checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
            await checkpointer.setup()
        except Exception as e:
            logger.warning("postgres_checkpointer_failed_using_memory", error=str(e))
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
    else:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    # Then use db for the workflow
    async with get_async_session() as db:
        workflow = create_capacity_forecast_workflow(
            llm=llm,
            db=db,
            checkpointer=checkpointer
        )
        # ... rest of the function
```

## Verification Checklist

- [x] AsyncPostgresSaver can be imported from langgraph.checkpoint.postgres.aio
- [x] AsyncPostgresSaver.from_conn_string() is the correct method
- [x] await checkpointer.setup() is required to initialize the schema
- [x] Database URL can be obtained from settings or environment variables
- [x] Fallback to MemorySaver only when AsyncPostgresSaver creation fails

## Reference Files

- [`resync/core/langgraph/checkpointer.py`](resync/core/langgraph/checkpointer.py:61-79) - Contains the `get_database_url()` function pattern
- [`resync/workflows/workflow_predictive_maintenance.py`](resync/workflows/workflow_predictive_maintenance.py:75-121) - Uses PostgresSaver with connection pool (sync version)
