"""Per-`tool_type` job handler registry.

`transcribe` (M3), `diarize` (M4) and `chapters` (M5) have real handlers. Every other
`tool_type` still maps to `dummy.run` until its own milestone replaces the entry (see
`docs/decisions/0006-async-job-pipeline.md`). `app/worker/tasks.py` is
the only consumer of this registry.

Handlers get `process_job`'s DB session so they can persist results
through the same `ctx["session_factory"]` seam tests rely on
(`docs/decisions/0007-transcription-pipeline.md` §4).
"""

from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import JOB_TOOL_TYPES, Job
from app.worker.handlers import chapters, diarize, dummy, transcribe

Handler = Callable[[Job, AsyncSession], Awaitable[None]]

HANDLERS: dict[str, Handler] = {tool_type: dummy.run for tool_type in JOB_TOOL_TYPES}
HANDLERS["transcribe"] = transcribe.run
HANDLERS["diarize"] = diarize.run
HANDLERS["chapters"] = chapters.run
