"""arq task functions.

Deliberately takes its DB session factory from `ctx["session_factory"]`
rather than importing `app.database.async_session_factory` directly --
that's the seam that lets tests point `process_job` at an isolated test
database (via their own `on_startup`) instead of the real one, without
any mocking. See `app/worker/settings.py` for the production wiring.

Retry policy (M9, docs/decisions/0014): a *transient* failure -- a lost
DB/Redis connection, CUDA running out of memory -- puts the job back to
`queued` and asks arq to run it again after a delay, up to `MAX_TRIES`
attempts. Everything else (bad input, "Run X first", a model missing,
a bug) fails at once: retrying would only fail again.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from arq import Retry
from sqlalchemy import update
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models.job import Job
from app.worker.handlers import HANDLERS

log = logging.getLogger(__name__)

MAX_TRIES = 3
RETRY_DELAY_S = 30  # x the attempt number


def is_transient(exc: BaseException) -> bool:
    """Worth another try: infrastructure hiccups, not the job's own fault."""
    if isinstance(exc, OperationalError) or (isinstance(exc, DBAPIError) and exc.connection_invalidated):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):  # includes redis' ConnectionError subclasses
        return True
    # torch says "CUDA out of memory", ctranslate2 "CUDA failed with error out
    # of memory"; llama.cpp can't build its context when VRAM is short.
    text = str(exc).lower()
    return any(s in text for s in ("out of memory", "cublas_status_alloc_failed", "failed to create llama_context"))


async def process_job(ctx: dict[str, Any], job_id: int) -> None:
    session_factory: async_sessionmaker = ctx["session_factory"]
    attempt = ctx.get("job_try", 1)

    async with session_factory() as db:
        job = await db.get(Job, job_id)
        if job is None:
            # Job row vanished (e.g. its upload was deleted) between
            # enqueue and pickup -- nothing to do.
            return

        job.status = "running"
        job.started_at = datetime.now(UTC)
        job.error_message = None
        await db.commit()

        try:
            handler = HANDLERS[job.tool_type]
            await handler(job, db)
        except Exception as exc:
            await db.rollback()
            if is_transient(exc) and attempt < MAX_TRIES:
                log.warning("job %d attempt %d/%d failed transiently, retrying: %s", job_id, attempt, MAX_TRIES, exc)
                job.status = "queued"
                job.error_message = f"Retrying (attempt {attempt}/{MAX_TRIES} failed): {exc}"
                await db.commit()
                raise Retry(defer=RETRY_DELAY_S * attempt) from exc
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.error_message = str(exc)
            await db.commit()
            raise
        else:
            job.status = "succeeded"
            job.completed_at = datetime.now(UTC)
            await db.commit()


async def fail_interrupted_jobs(session_factory: async_sessionmaker) -> int:
    """At worker startup: jobs still `running` were cut off by a stop/crash of
    the (single) worker. Mark them failed so the UI can offer Run again."""
    async with session_factory() as db:
        result = await db.execute(
            update(Job)
            .where(Job.status == "running")
            .values(status="failed", completed_at=datetime.now(UTC),
                    error_message="Interrupted: the worker stopped while this job was running")
        )
        await db.commit()
        return result.rowcount or 0
