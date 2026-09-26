"""arq CLI entrypoint. Run with:

    arq app.worker.settings.WorkerSettings

as a separate process alongside `uvicorn app.main:app` -- see
`docs/decisions/0006-async-job-pipeline.md`.
"""

from typing import Any

from arq.connections import RedisSettings

from app.config import get_settings
from app.database import async_session_factory
from app.worker.tasks import process_job

settings = get_settings()


async def startup(ctx: dict[str, Any]) -> None:
    ctx["session_factory"] = async_session_factory


class WorkerSettings:
    functions = [process_job]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    # M2 ships no job-orchestration retry policy (that's M9's
    # "hardening" territory) -- a handler exception is a terminal
    # `failed`, not an automatic re-run.
    max_tries = 1
