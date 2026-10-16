"""`GET`/`PATCH /uploads/{id}/shownotes` (M6).

`GET` status rules mirror chapters (0009 §7): 404 when no `shownotes`
job has ever succeeded, 409 when none has yet but one is queued or
running. `PATCH` edits the stored draft within YouTube's limits. See
docs/decisions/0011-show-notes-local-llm.md §5.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.database import get_db
from app.models.job import Job
from app.models.show_notes import ShowNotesDraft
from app.models.upload import Upload
from app.models.user import User
from app.schemas.show_notes import MAX_DESCRIPTION_CHARS, ShowNotesOut, ShowNotesUpdate

router = APIRouter(prefix="/uploads", tags=["show notes"])


async def _draft(upload_id: int, db: AsyncSession) -> ShowNotesDraft:
    if await db.get(Upload, upload_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")

    statuses = set(
        (
            await db.execute(
                select(Job.status).where(Job.upload_id == upload_id, Job.tool_type == "shownotes")
            )
        ).scalars()
    )
    draft = await db.scalar(select(ShowNotesDraft).where(ShowNotesDraft.upload_id == upload_id))
    if "succeeded" not in statuses or draft is None:
        if statuses & {"queued", "running"}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Show notes not ready")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Show notes not found")
    return draft


@router.get("/{upload_id}/shownotes", response_model=ShowNotesOut)
async def get_show_notes(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> ShowNotesDraft:
    return await _draft(upload_id, db)


@router.patch("/{upload_id}/shownotes", response_model=ShowNotesOut)
async def update_show_notes(
    upload_id: int,
    body: ShowNotesUpdate,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> ShowNotesDraft:
    draft = await _draft(upload_id, db)
    for field, value in body.model_dump(exclude_unset=True).items():
        if value is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"{field} can't be null")
        setattr(draft, field, value)
    if len(draft.description) > MAX_DESCRIPTION_CHARS:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="description too long")
    await db.commit()
    await db.refresh(draft)
    return draft
