from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models.upload import Upload
from app.models.user import User
from app.schemas.upload import UploadOut
from app.storage import LocalStorageBackend

router = APIRouter(prefix="/uploads", tags=["uploads"])
settings = get_settings()
storage = LocalStorageBackend(settings.storage_base_path)


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
