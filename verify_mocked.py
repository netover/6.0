"""
Mocked Verification script for RedisVL Semantic Cache.
Tests Python logic when Redis is unavailable.
"""

import asyncio
import logging
import json
from unittest.mock import MagicMock, AsyncMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_mocked")

async def verify_mocked():
    try:
        from resync.core.cache.semantic_cache import SemanticCache
        
        logger.info("Initializing SemanticCache with Mocked Redis...")
        
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.info = AsyncMock(return_value={"used_memory_human": "10MB"})
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.scan_iter = MagicMock()
        mock_redis.scan_iter.return_value = AsyncMock() # This is tricky for async iterators
        
        # Helper for async iteration
        async def mock_async_iter(items):
            for item in items:
                yield item
        
        mock_redis.scan_iter.side_return_value = mock_async_iter([])
        
        with patch("resync.core.cache.semantic_cache.get_redis_client", return_value=mock_redis), \
             patch("resync.core.cache.semantic_cache.check_redis_stack_available", return_value={"search": True}):
            
            cache = SemanticCache()
            cache._initialized = True
            cache._redis_stack_available = True
            
            # 1. Test update_threshold
            logger.info("Testing update_threshold()...")
            cache.update_threshold(0.2)
            if cache.threshold == 0.2:
                logger.info("SUCCESS: Threshold updated.")
            else:
                logger.error("FAILURE: Threshold not updated.")
                
            # 2. Test set_reranking_enabled
            logger.info("Testing set_reranking_enabled()...")
            with patch("resync.core.cache.semantic_cache.is_reranker_available", return_value=True):
                cache.set_reranking_enabled(True)
                if cache.enable_reranking:
                    logger.info("SUCCESS: Reranking enabled.")
            
            # 3. Test Invalidate
            logger.info("Testing invalidate()...")
            mock_redis.delete.return_value = 1
            res = await cache.invalidate("test query")
            if res:
                logger.info("SUCCESS: Invalidate called delete.")
            
            # 4. Test Stats logic
            logger.info("Testing get_stats() logic...")
            cache._stats["hits"] = 5
            cache._stats["misses"] = 5
            
            # Setup scan_iter for stats count
            mock_redis.scan_iter.return_value = mock_async_iter(["key1", "key2"])
            
            stats = await cache.get_stats()
            if stats["entries"] == 2 and stats["hit_rate_percent"] == 50.0:
                logger.info("SUCCESS: Stats calculation correct.")
                logger.info(f"Stats: {json.dumps(stats, indent=2)}")
            else:
                logger.error(f"FAILURE: Stats mismatch: {stats}")

        logger.info("=== MOCKED VERIFICATION COMPLETE ===")

    except Exception as e:
        logger.exception(f"Verification failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_mocked())
