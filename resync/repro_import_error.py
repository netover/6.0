
import sys
import os

# Adapt path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Attempting to import LLMCacheWrapper form resync.core.cache.llm_cache_wrapper...")
    from resync.core.cache.llm_cache_wrapper import LLMCacheWrapper
    print("Success wrapper direct import")
except ImportError as e:
    print(f"Failed direct wrapper import: {e}")

try:
    print("Attempting to import LLMCacheWrapper from resync.core.cache...")
    from resync.core.cache import LLMCacheWrapper
    print("Success package import")
except ImportError as e:
    print(f"Failed package import: {e}")

try:

    print("Attempting imports that triggered the error in log...")
    from resync.app_factory import ApplicationFactory
    from fastapi import FastAPI
    
    app = FastAPI()
    factory = ApplicationFactory()
    
    print("Entering lifespan...")
    import asyncio
    
    async def run_lifespan():
        async with factory.lifespan(app):
            print("Inside lifespan!")

    asyncio.run(run_lifespan())
    print("Success app_factory import and lifespan execution")

except ImportError as e:
    print(f"\n!!! CAUGHT IMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"\n!!! CAUGHT OTHER ERROR: {e}")
    import traceback
    traceback.print_exc()
