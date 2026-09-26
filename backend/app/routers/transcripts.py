"""`.srt` / `.vtt` downloads for an upload's transcript (M3).

Rendered on demand from `transcript_segments` -- see
docs/decisions/0007-transcription-pipeline.md §5 for the status-code
rules and why there's no blob cache.
"""

import re
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.database import get_db
from app.models.job import Job
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.models.user import User
from app.transcription.subtitles import to_srt, to_vtt

router = APIRouter(prefix="/uploads", tags=["transcripts"])

_MEDIA_TYPES = {
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
}


def _content_disposition(filename: str) -> str:
    """`attachment` header that survives non-ASCII filenames.

    Header values must be latin-1 encodable, and YouTube-style titles
    often aren't (e.g. a fullwidth `：`). Per RFC 6266, send an ASCII
    fallback in `filename` plus the real name, percent-encoded, in
    `filename*` -- browsers prefer the latter.
    """
    ascii_name = re.sub(r'[^\x20-\x7e]|["\\]', "_", filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"


async def _subtitle_response(
    upload_id: int, fmt: Literal["srt", "vtt"], db: AsyncSession
) -> Response:
    upload = await db.get(Upload, upload_id)
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")

    statuses = set(
        (
            await db.execute(
                select(Job.status).where(
                    Job.upload_id == upload_id, Job.tool_type == "transcribe"
                )
            )
        ).scalars()
    )
    # A previous successful transcript keeps being served while a re-run
    # is in flight (the handler swaps rows atomically).
    if "succeeded" not in statuses:
        if statuses & {"queued", "running"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Transcript not ready"
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        )

    segments = (
        await db.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.upload_id == upload_id)
            .order_by(TranscriptSegment.segment_index)
        )
    ).scalars()
    body = to_srt(segments) if fmt == "srt" else to_vtt(segments)

    stem = Path(upload.filename).stem or "transcript"
    return Response(
        content=body,
        media_type=_MEDIA_TYPES[fmt],
        headers={"Content-Disposition": _content_disposition(f"{stem}.{fmt}")},
    )


@router.get("/{upload_id}/transcript.srt")
async def get_transcript_srt(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> Response:
    return await _subtitle_response(upload_id, "srt", db)


@router.get("/{upload_id}/transcript.vtt")
async def get_transcript_vtt(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> Response:
    return await _subtitle_response(upload_id, "vtt", db)
