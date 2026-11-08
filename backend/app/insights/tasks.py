"""arq task: precompute the Analytics results (docs/decisions/0023).

Registered twice on the worker: as a cron hourly at :45 (after the :05 RSS
poll and the 09:30 daily pull have had time to land), and as a function the
API enqueues when a result is missing. Unchanged sources make a run a no-op.
"""

from typing import Any

from app.database import async_session_factory
from app.insights.store import RefreshReport, refresh_insights

#: arq job id prefix: one queued refresh per source state, never a pile-up.
JOB_ID_PREFIX = "insights_refresh"


async def insights_refresh(ctx: dict[str, Any], force: bool = False) -> dict:
    factory = ctx.get("session_factory") or async_session_factory
    report: RefreshReport = await refresh_insights(factory, force=force)
    return {"computed": report.computed, "skipped": len(report.skipped), "seconds": round(report.seconds, 2)}
