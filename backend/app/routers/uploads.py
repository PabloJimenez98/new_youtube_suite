import mimetypes

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user, get_current_user_for_media
from app.config import get_settings
from app.database import get_db
from app.models.clip import Clip
from app.models.job import Job
from app.models.upload import Upload
from app.models.user import User
from app.schemas.job import JobCreate, JobOut
from app.routers.media import stream_file
from app.schemas.upload import UploadOut
from app.storage import LocalStorageBackend, StorageKeyNotFoundError
from app.worker.pool import get_arq_pool

router = APIRouter(prefix="/uploads", tags=["uploads"])
settings = get_settings()
storage = LocalStorageBackend(settings.storage_base_path)

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

    # The extracted audio (M3) and rendered shorts (M8) go with the
    # video. Files already gone don't block cleaning up the row (and its
    # jobs/transcript/clip rows, via ON DELETE CASCADE).
    clip_keys = (await db.execute(select(Clip.storage_path).where(Clip.upload_id == upload.id))).scalars()
    for key in (upload.storage_path, upload.audio_storage_path, *clip_keys):
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

    media_type, _ = mimetypes.guess_type(upload.filename)
    return stream_file(storage.path_for(upload.storage_path), media_type or "application/octet-stream", request)


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

    job = Job(upload_id=upload_id, tool_type=job_in.tool_type, params=job_in.params)
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
