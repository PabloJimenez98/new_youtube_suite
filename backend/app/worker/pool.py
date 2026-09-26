"""FastAPI dependency for the arq Redis pool used to enqueue jobs.

Lazily-initialized module-level singleton, mirroring the `storage`
singleton in `app/routers/uploads.py` -- except this one has to be
created asynchronously (Redis connection), so it can't be built at
import time. Exposed as a dependency (rather than a bare module-level
object) so tests can override it with a pool pointed at
`test_redis_url`, the same way `get_db` is overridden.
"""

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config import get_settings

_pool: ArqRedis | None = None


async def get_arq_pool() -> ArqRedis:
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _pool
