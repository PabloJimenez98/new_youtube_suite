"""`GET /uploads/{id}/relevance` (M7).

Status rules mirror chapters (0009 §7): 404 when no `relevance` job has
ever succeeded, 409 when none has yet but one is queued or running.
`top_moments` (the best windows at least 60 s apart) is for display
only -- real peak-finding is M8's. See docs/decisions/0012-relevance-v0.md §6.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.database import get_db
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.relevance import RelevanceScore
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.models.user import User
from app.schemas.relevance import RelevanceOut, RelevanceWindowOut, TopMomentOut

router = APIRouter(prefix="/uploads", tags=["relevance"])

TOP_MOMENTS = 5
MIN_GAP_SECONDS = 60.0


def pick_top_moments(rows: list[RelevanceScore], k: int = TOP_MOMENTS, gap: float = MIN_GAP_SECONDS) -> list[RelevanceScore]:
    """Greedy: best score first, skipping windows starting within `gap` s of one already picked."""
    picked: list[RelevanceScore] = []
    for r in sorted(rows, key=lambda r: (-r.score, r.window_start_seconds)):
        if all(abs(r.window_start_seconds - p.window_start_seconds) >= gap for p in picked):
            picked.append(r)
            if len(picked) == k:
                break
    return picked


async def _window_text(db: AsyncSession, upload_id: int, start: float, end: float) -> str:
    words = (
        await db.execute(
            select(TranscriptWord.word)
            .join(TranscriptSegment, TranscriptWord.segment_id == TranscriptSegment.id)
            .where(
                TranscriptSegment.upload_id == upload_id,
                TranscriptWord.start_seconds >= start,
                TranscriptWord.start_seconds < end,
            )
            .order_by(TranscriptWord.start_seconds)
        )
    ).scalars()
    return " ".join(w.strip() for w in words if w.strip())


@router.get("/{upload_id}/relevance", response_model=RelevanceOut)
async def get_relevance(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> RelevanceOut:
    if await db.get(Upload, upload_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")

    statuses = set(
        (
            await db.execute(
                select(Job.status).where(Job.upload_id == upload_id, Job.tool_type == "relevance")
            )
        ).scalars()
    )
    rows = list(
        (
            await db.execute(
                select(RelevanceScore).where(RelevanceScore.upload_id == upload_id).order_by(RelevanceScore.window_index)
            )
        ).scalars()
    )
    if "succeeded" not in statuses or not rows:
        if statuses & {"queued", "running"}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Relevance not ready")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Relevance not found")

    version = await db.get(ModelVersion, rows[0].model_version_id)
    return RelevanceOut(
        version=version.version_label,
        windows=[RelevanceWindowOut(start=r.window_start_seconds, end=r.window_end_seconds, score=r.score) for r in rows],
        top_moments=[
            TopMomentOut(
                start=r.window_start_seconds,
                end=r.window_end_seconds,
                score=r.score,
                text=await _window_text(db, upload_id, r.window_start_seconds, r.window_end_seconds),
            )
            for r in pick_top_moments(rows)
        ],
    )
