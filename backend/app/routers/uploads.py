from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models.job import Job
from app.models.upload import Upload
from app.models.user import User
from app.schemas.job import JobCreate, JobOut
from app.schemas.upload import UploadOut
from app.storage import LocalStorageBackend, StorageKeyNotFoundError

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

    try:
        storage.delete(upload.storage_path)
    except StorageKeyNotFoundError:
        # File already gone -- still let the row (and its jobs, via
        # ON DELETE CASCADE) be cleaned up rather than blocking on it.
        pass

    await db.delete(upload)
    await db.commit()


@router.post(
    "/{upload_id}/jobs", response_model=JobOut, status_code=status.HTTP_201_CREATED
)
async def create_job(
    upload_id: int,
    job_in: JobCreate,
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> Job:
    await _get_upload_or_404(upload_id, db)

    job = Job(upload_id=upload_id, tool_type=job_in.tool_type)
    db.add(job)
    await db.commit()
    await db.refresh(job)
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
