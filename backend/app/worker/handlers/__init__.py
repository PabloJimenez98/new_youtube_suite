"""Per-`tool_type` job handler registry.

Every `tool_type` maps to a handler today -- `dummy.run` -- because no
real tool exists yet (see `docs/decisions/0006-async-job-pipeline.md`).
Each later milestone (M3 transcribe, M4 diarize, ...) replaces its own
entry with a real handler; the other entries stay on `dummy.run` until
their own milestone lands. `app/worker/tasks.py` is the only consumer
of this registry and never changes when an entry is swapped.
"""

from collections.abc import Awaitable, Callable

from app.models.job import JOB_TOOL_TYPES, Job
from app.worker.handlers import dummy

Handler = Callable[[Job], Awaitable[None]]

HANDLERS: dict[str, Handler] = {tool_type: dummy.run for tool_type in JOB_TOOL_TYPES}
