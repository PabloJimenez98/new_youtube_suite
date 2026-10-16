"""arq CLI entrypoint. Run with:

    arq app.worker.settings.WorkerSettings

as a separate process alongside `uvicorn app.main:app` -- see
`docs/decisions/0006-async-job-pipeline.md`.
"""

import logging
from typing import Any

from arq.connections import RedisSettings

from app.config import get_settings
from app.database import async_session_factory
from app.diarization.pipeline import startup_healthcheck
from app.worker.tasks import MAX_TRIES, fail_interrupted_jobs, process_job

settings = get_settings()


def configure_app_logging() -> None:
    """arq only configures its own logger: send `app.*` INFO lines (per-phase
    timings, M9) to stderr too, in arq's format."""
    app_log = logging.getLogger("app")
    if not app_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s: %(name)s %(message)s", "%H:%M:%S"))
        app_log.addHandler(handler)
        app_log.setLevel(logging.INFO)


async def startup(ctx: dict[str, Any]) -> None:
    configure_app_logging()
    ctx["session_factory"] = async_session_factory
    interrupted = await fail_interrupted_jobs(async_session_factory)
    if interrupted:
        logging.getLogger("app.worker").warning("marked %d interrupted job(s) as failed", interrupted)
    if settings.diarization_healthcheck:
        # Never raises: a failure only disables `diarize` jobs (M4, 0008 §5).
        await startup_healthcheck()


class WorkerSettings:
    functions = [process_job]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    # M9: `process_job` asks for a retry (arq `Retry`) only on transient
    # failures; anything else is a terminal `failed` (docs/decisions/0014).
    max_tries = MAX_TRIES
    # arq's 300s default would kill any real transcription (M3).
    job_timeout = settings.worker_job_timeout_seconds
