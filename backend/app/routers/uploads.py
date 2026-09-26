import mimetypes
import re
from collections.abc import Iterator
from pathlib import Path

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user, get_current_user_for_media
from app.config import get_settings
from app.database import get_db
from app.models.job import Job
from app.models.upload import Upload
from app.models.user import User
from app.schemas.job import JobCreate, JobOut
from app.schemas.upload import UploadOut
from app.storage import LocalStorageBackend, StorageKeyNotFoundError
from app.worker.pool import get_arq_pool

router = APIRouter(prefix="/uploads", tags=["uploads"])
settings = get_settings()
storage = LocalStorageBackend(settings.storage_base_path)

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK_SIZE = 64 * 1024


def _iter_file_range(path: Path, start: int, end: int) -> Iterator[bytes]:
    """Yield bytes [start, end] (inclusive) from `path`, in chunks."""
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


async def _get_upload_or_404(upload_id: int, db: AsyncSession) -> Upload:
    upload = await db.get(Upload, upload_id)
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found"
        )
    return upload


@router.post("", response_model=UploadOut, status_code=201)
async def create_upload(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> Upload:
    key = LocalStorageBackend.make_key(file.filename or "upload.bin")
    storage.save(key, file.file)

    upload = Upload(filename=file.filename or key, storage_path=key)
    db.add(upload)
    await db.commit()
    await db.refresh(upload)
    return upload


@router.get("", response_model=list[UploadOut])
async def list_uploads(
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> list[Upload]:
    result = await db.execute(select(Upload).order_by(Upload.uploaded_at.desc()))
    return list(result.scalars().all())


@router.get("/{upload_id}", response_model=UploadOut)
async def get_upload(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> Upload:
    return await _get_upload_or_404(upload_id, db)


@router.delete("/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_upload(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> None:
    upload = await _get_upload_or_404(upload_id, db)

    # The extracted audio (M3) goes with the video. Files already gone
    # don't block cleaning up the row (and its jobs/transcript rows,
    # via ON DELETE CASCADE).
    for key in (upload.storage_path, upload.audio_storage_path):
        if key is None:
            continue
        try:
            storage.delete(key)
        except StorageKeyNotFoundError:
            pass

    await db.delete(upload)
    await db.commit()


@router.get("/{upload_id}/file")
async def stream_upload_file(
    upload_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user_for_media),
) -> StreamingResponse:
    upload = await _get_upload_or_404(upload_id, db)

    path = storage.path_for(upload.storage_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    file_size = path.stat().st_size
    media_type, _ = mimetypes.guess_type(upload.filename)
    media_type = media_type or "application/octet-stream"

    # Starlette's FileResponse doesn't implement Range requests in the
    # version pinned here, and Range support is what lets a <video>
    # element seek without downloading the whole file first -- so this
    # is handled by hand rather than relying on FileResponse.
    range_header = request.headers.get("range")
    if range_header:
        match = _RANGE_RE.match(range_header)
        if match:
            start_str, end_str = match.groups()
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            if start <= end:
                return StreamingResponse(
                    _iter_file_range(path, start, end),
                    status_code=status.HTTP_206_PARTIAL_CONTENT,
                    media_type=media_type,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(end - start + 1),
                    },
                )

    return StreamingResponse(
        _iter_file_range(path, 0, file_size - 1),
        media_type=media_type,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


@router.post(
    "/{upload_id}/jobs", response_model=JobOut, status_code=status.HTTP_201_CREATED
)
async def create_job(
    upload_id: int,
    job_in: JobCreate,
    db: AsyncSession = Depends(get_db),
    arq_pool: ArqRedis = Depends(get_arq_pool),
    _current_user: User = Depends(get_current_user),
) -> Job:
    await _get_upload_or_404(upload_id, db)

    job = Job(upload_id=upload_id, tool_type=job_in.tool_type)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    await arq_pool.enqueue_job("process_job", job.id)
    return job


@router.get("/{upload_id}/jobs", response_model=list[JobOut])
async def list_jobs_for_upload(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> list[Job]:
    await _get_upload_or_404(upload_id, db)

    result = await db.execute(
        select(Job).where(Job.upload_id == upload_id).order_by(Job.created_at.desc())
    )
    return list(result.scalars().all())
