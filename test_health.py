import asyncio
import os
from resync.settings import settings

os.environ['APP_DATABASE_URL'] = settings.database_url.get_secret_value()

from resync.core.pools.pool_manager import initialize_pool_manager
from resync.core.health.health_models import ComponentHealth
from resync.core.health.health_checkers.database_health_checker import DatabaseHealthChecker
from resync.core.health.health_checkers.connection_pools_health_checker import ConnectionPoolsHealthChecker
from resync.core.health.health_checkers.redis_health_checker import RedisHealthChecker

async def test():
    await initialize_pool_manager()
    db = DatabaseHealthChecker(settings)
    cp = ConnectionPoolsHealthChecker(settings)
    rd = RedisHealthChecker(settings)
    print('DB:', await db.check_health())
    print('CP:', await cp.check_health())
    print('RD:', await rd.check_health())

asyncio.run(test())
