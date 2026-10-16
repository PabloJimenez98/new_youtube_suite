"""`GET /uploads/{id}/clips`, `GET /clips/{id}/file` and `GET /clips/count` (M8).

Status rules mirror chapters/relevance: 404 when no `shorts` job has
ever succeeded, 409 when none has yet but one is queued or running. A
previous run's clips keep being served during a re-run (the handler
swaps rows in one transaction). See docs/decisions/0013-shorts-generator.md §7.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user, get_current_user_for_media
from app.config import get_settings
from app.database import get_db
from app.models.clip import Clip
from app.models.job import Job
from app.models.upload import Upload
from app.models.user import User
from app.routers.media import stream_file
from app.schemas.clip import ClipOut
from app.storage import LocalStorageBackend

router = APIRouter(tags=["clips"])
storage = LocalStorageBackend(get_settings().storage_base_path)


@router.get("/clips/count")
async def count_clips(
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> dict[str, int]:
    """All rendered shorts, for the dashboard tile."""
    return {"count": await db.scalar(select(func.count()).select_from(Clip)) or 0}


@router.get("/uploads/{upload_id}/clips", response_model=list[ClipOut])
async def list_clips(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> list[ClipOut]:
    if await db.get(Upload, upload_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")

    statuses = set(
        (await db.execute(select(Job.status).where(Job.upload_id == upload_id, Job.tool_type == "shorts"))).scalars()
    )
    rows = list((await db.execute(select(Clip).where(Clip.upload_id == upload_id).order_by(Clip.rank))).scalars())
    if "succeeded" not in statuses or not rows:
        if statuses & {"queued", "running"}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Shorts not ready")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shorts not found")

    return [
        ClipOut(
            id=c.id, rank=c.rank, start=c.start_seconds, end=c.end_seconds,
            duration=c.end_seconds - c.start_seconds, score=c.score, aspect_ratio=c.aspect_ratio,
            caption_burned_in=c.caption_burned_in, framing=c.framing,
        )
        for c in rows
    ]


@router.get("/clips/{clip_id}/file")
async def stream_clip_file(
    clip_id: int,
    request: Request,
    download: bool = False,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user_for_media),
) -> StreamingResponse:
    clip = await db.get(Clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clip not found")
    headers = {"Content-Disposition": f'attachment; filename="short-{clip.upload_id}-{clip.rank}.mp4"'} if download else None
    return stream_file(storage.path_for(clip.storage_path), "video/mp4", request, headers)
