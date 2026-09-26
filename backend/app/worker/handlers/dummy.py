"""M2's stand-in handler, used for every `tool_type` that doesn't have a
real one yet (everything but `transcribe` as of M3) -- see
`handlers/__init__.py`.

"Write a placeholder result" (per the M2 plan) means the job reaching
`succeeded`, not a persisted payload -- so `db` is unused here.
"""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.job import Job


async def run(job: Job, db: AsyncSession) -> None:
    await asyncio.sleep(get_settings().dummy_job_duration_seconds)
