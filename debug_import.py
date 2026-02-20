
try:
    from resync.core.cache import get_redis_client
    print("SUCCESS: Imported get_redis_client from resync.core.cache")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
