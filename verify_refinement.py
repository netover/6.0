"""
Refined Verification script for RedisVL Semantic Cache.
Tests API parity, stats, and logical correctness.
"""

import asyncio
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_refinement")

async def verify_refinement():
    try:
        from resync.core.cache.semantic_cache import get_semantic_cache
        
        logger.info("Initializing Refined SemanticCache...")
        cache = await get_semantic_cache()
        
        # 1. Test SET with metadata
        query = "How to upgrade the system firmware?"
        response = "To upgrade firmware, navigate to Settings > Maintenance > Firmware Update."
        metadata = {"category": "hardware", "priority": "high"}
        
        logger.info(f"Setting cache for query: '{query}'")
        await cache.set(query, response, metadata=metadata)
            
        # 2. Test GET and verify metadata/hit_count
        logger.info("Testing retrieval and metadata...")
        result = await cache.get(query)
        if result.hit:
            logger.info("SUCCESS: Cache hit.")
            if result.entry.metadata.get("category") == "hardware":
                logger.info("SUCCESS: Metadata preserved.")
            else:
                logger.error(f"FAILURE: Metadata mismatch: {result.entry.metadata}")
        else:
            logger.error("FAILURE: Cache miss for exact match.")

        # 3. Test API Parity: invalidate
        logger.info("Testing invalidate()...")
        success = await cache.invalidate(query)
        if success:
            logger.info("SUCCESS: Entry invalidated.")
            result_after = await cache.get(query)
            if not result_after.hit:
                logger.info("SUCCESS: Confirmed miss after invalidation.")
            else:
                logger.error("FAILURE: Still hit after invalidation!")
        else:
            logger.error("FAILURE: Invalidation returned False.")

        # 4. Test API Parity: get_stats
        logger.info("Testing get_stats()...")
        # Re-populate for stats
        await cache.set(query, response)
        await cache.get(query) # Increment hits
        
        stats = await cache.get_stats()
        required_keys = ["entries", "hits", "misses", "sets", "avg_lookup_time_ms", "threshold", "redis_stack_available"]
        missing_keys = [k for k in required_keys if k not in stats]
        
        if not missing_keys:
            logger.info(f"SUCCESS: Stats keys match expectations. Entries: {stats['entries']}")
            logger.info(f"Full stats: {json.dumps(stats, indent=2)}")
        else:
            logger.error(f"FAILURE: Missing stats keys: {missing_keys}")

        # 5. Test API Parity: update_threshold
        logger.info("Testing update_threshold()...")
        old_t = cache.threshold
        cache.update_threshold(0.99)
        if cache.threshold == 0.99:
            logger.info("SUCCESS: Threshold updated.")
        else:
            logger.error(f"FAILURE: Threshold update failed. Value: {cache.threshold}")
        cache.update_threshold(old_t)

        # 6. Test invalidate_pattern
        logger.info("Testing invalidate_pattern()...")
        await cache.set("system update log", "log data")
        await cache.set("kernel update info", "info data")
        count = await cache.invalidate_pattern("update")
        logger.info(f"Invalidated {count} entries by pattern 'update'.")
        if count >= 2:
            logger.info("SUCCESS: Pattern invalidation worked.")
        else:
            logger.error(f"FAILURE: Pattern invalidation count too low: {count}")

        logger.info("=== REFINEMENT VERIFICATION COMPLETE ===")

    except Exception as e:
        logger.exception(f"Verification failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_refinement())
