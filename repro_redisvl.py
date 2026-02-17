import asyncio
from redisvl.index import SearchIndex
import redis

SCHEMA = {
    "index": {
        "name": "test_index",
        "prefix": "test_prefix:",
    },
    "fields": [
        {"name": "text", "type": "text"},
        {
            "name": "vec",
            "type": "vector",
            "attrs": {"dims": 3, "algorithm": "flat", "distance_metric": "cosine"}
        }
    ],
}

async def test():
    try:
        # Some versions of RedisVL expect clients from redis.Redis (which can be async)
        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        index = SearchIndex.from_dict(SCHEMA)
        index.set_client(client)
        
        print(f"Client type: {type(client)}")
        
        data = [{"text": "hello", "vec": [0.1, 0.2, 0.3]}]
        print("Attempting to load data...")
        # Since this client might be sync, let's see if it works or if we need .asyncio
        # Actually SearchIndex in 0.3.1 might support both
        res = index.load(data)
        print(f"Success! Result: {res}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        
    try:
        from redis.asyncio import Redis as AsyncRedis
        aclient = AsyncRedis(host="localhost", port=6379, decode_responses=True)
        print(f"Async client type: {type(aclient)}")
        # If set_client failed for AsyncRedis, let's try to see what it wants
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(test())
