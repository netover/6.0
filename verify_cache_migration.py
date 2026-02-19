"""
Verification script for RedisVL Semantic Cache.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_redisvl")

async def verify_cache():
    try:
        # Import after setting up logging
        from resync.core.cache.semantic_cache import get_semantic_cache
        
        logger.info("Initializing SemanticCache (RedisVL)...")
        cache = await get_semantic_cache()
        
        # 1. Test SET
        query = "How do I restart a failed background job in the system?"
        response = "To restart a failed job, you should use the Admin Dashboard or the /jobs/restart API endpoint."
        
        logger.info(f"Setting cache for query: '{query}'")
        success = await cache.set(query, response)
        if not success:
            logger.error("Failed to set cache entry")
            return
            
        # 2. Test EXACT GET
        logger.info("Testing exact match...")
        result = await cache.get(query)
        if result.hit and result.response == response:
            logger.info(f"SUCCESS: Exact match hit. Distance: {result.distance:.4f}")
        else:
            logger.error(f"FAILURE: Exact match failed. Hit: {result.hit}")

        # 3. Test SEMANTIC GET (Similar query)
        similar_query = "What is the procedure to start a job that has failed?"
        logger.info(f"Testing semantic match with: '{similar_query}'")
        result = await cache.get(similar_query)
        if result.hit:
            logger.info(f"SUCCESS: Semantic match hit! Distance: {result.distance:.4f}")
            logger.info(f"Cached Response: {result.response[:50]}...")
            if result.reranked:
                logger.info(f"Reranking was applied. Score: {result.rerank_score}")
        else:
            logger.info(f"MISS: Semantic match missed. Distance: {result.distance:.4f}")

        # 4. Test MISS (Unrelated query)
        unrelated_query = "What is the weather in Tokyo?"
        logger.info(f"Testing unrelated query: '{unrelated_query}'")
        result = await cache.get(unrelated_query)
        if not result.hit:
            logger.info("SUCCESS: Unrelated query resulted in a MISS.")
        else:
            logger.warning(f"WARNING: Unrelated query hit! Distance: {result.distance:.4f}")

        # 5. Get Stats
        stats = await cache.get_stats()
        logger.info(f"Cache Stats: {stats}")

    except Exception as e:
        logger.error(f"Verification failed with error: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(verify_cache())
