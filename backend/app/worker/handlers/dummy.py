"""M2's stand-in handler, used for every `tool_type` until each tool's
own milestone (M3+) registers a real one in `handlers/__init__.py`.

There's no results table yet for any tool -- that's introduced
per-tool, starting with M3's `transcript_segments`/`transcript_words`
-- so "write a placeholder result" (per the M2 plan) means the job
reaching `succeeded`, not a persisted payload.
"""

import asyncio

from app.config import get_settings
from app.models.job import Job


async def run(job: Job) -> None:
    await asyncio.sleep(get_settings().dummy_job_duration_seconds)
