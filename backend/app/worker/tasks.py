"""arq task functions.

Deliberately takes its DB session factory from `ctx["session_factory"]`
rather than importing `app.database.async_session_factory` directly --
that's the seam that lets tests point `process_job` at an isolated test
database (via their own `on_startup`) instead of the real one, without
any mocking. See `app/worker/settings.py` for the production wiring.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models.job import Job
from app.worker.handlers import HANDLERS


async def process_job(ctx: dict[str, Any], job_id: int) -> None:
    session_factory: async_sessionmaker = ctx["session_factory"]

    async with session_factory() as db:
        job = await db.get(Job, job_id)
        if job is None:
            # Job row vanished (e.g. its upload was deleted) between
            # enqueue and pickup -- nothing to do.
            return

        job.status = "running"
        job.started_at = datetime.now(UTC)
        await db.commit()

        try:
            handler = HANDLERS[job.tool_type]
            await handler(job, db)
        except Exception as exc:
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.error_message = str(exc)
            await db.commit()
            raise
        else:
            job.status = "succeeded"
            job.completed_at = datetime.now(UTC)
            await db.commit()
