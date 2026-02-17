import asyncio
from resync.core.langgraph.roma_graph import execute_roma_query

async def main():
    res = await execute_roma_query("test atomic")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
