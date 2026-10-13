"""`GET /uploads/{id}/chapters` (M5).

Status rules mirror the subtitle endpoints (0007 §5), for the
`chapters` job: 404 when no chapters job has ever succeeded, 409 when
none has succeeded yet but one is queued/running. A previous successful
result keeps being served during a re-run (the handler swaps rows in one
transaction). See docs/decisions/0009-chapter-generator-local-models.md §7.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.database import get_db
from app.models.chapter import Chapter
from app.models.job import Job
from app.models.upload import Upload
from app.models.user import User
from app.schemas.chapter import ChapterOut

router = APIRouter(prefix="/uploads", tags=["chapters"])


@router.get("/{upload_id}/chapters", response_model=list[ChapterOut])
async def get_chapters(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> list[Chapter]:
    if await db.get(Upload, upload_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")

    statuses = set(
        (
            await db.execute(
                select(Job.status).where(Job.upload_id == upload_id, Job.tool_type == "chapters")
            )
        ).scalars()
    )
    if "succeeded" not in statuses:
        if statuses & {"queued", "running"}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Chapters not ready")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chapters not found")

    return list(
        (
            await db.execute(
                select(Chapter).where(Chapter.upload_id == upload_id).order_by(Chapter.order_index)
            )
        ).scalars()
    )
