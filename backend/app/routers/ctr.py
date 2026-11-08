"""Title/thumbnail candidates for the CTR predictor (M13, docs/decisions/0019 §4).

- `GET /uploads/{id}/ctr-candidates`: the candidates, ranked. Scored
  full candidates come first, then partial (title-only) ones, then
  unscored or errored ones.
- `POST /uploads/{id}/ctr-candidates`: multipart; `title` is required,
  `thumbnail` optional.
- `POST /uploads/{id}/ctr-candidates/from-show-notes`: title-only
  candidates from the latest show-notes draft, skipping titles already
  present.
- `PATCH /ctr-candidates/{cid}`: a new title and/or thumbnail. Clears the
  score.
- `DELETE /ctr-candidates/{cid}`: also deletes the stored thumbnail.
- `GET /ctr-candidates/{cid}/thumbnail`: the stored image.

Scoring is the `ctr` job (`POST /uploads/{id}/jobs`).
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user, get_current_user_for_media
from app.config import get_settings
from app.ctr.features import MAX_TITLE_CHARS
from app.ctr.pipeline import MAX_CANDIDATES_PER_UPLOAD, MAX_THUMBNAIL_BYTES, InvalidThumbnailError, sniff_type, validate_thumbnail
from app.database import get_db
from app.models.ctr import CtrPrediction
from app.models.show_notes import ShowNotesDraft
from app.models.upload import Upload
from app.models.user import User
from app.storage import LocalStorageBackend, StorageKeyNotFoundError

router = APIRouter(tags=["ctr"])
storage = LocalStorageBackend(get_settings().storage_base_path)

Group = Literal["full", "partial", "unscored"]


class CandidateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    upload_id: int | None
    candidate_title: str
    has_thumbnail: bool
    predicted_score: float | None
    partial: bool
    features: dict | None
    source: str
    scored_at: datetime | None
    group: Group
    rank: int | None  # 1-based within its group; None when unscored


def _group(c: CtrPrediction) -> Group:
    if c.predicted_score is None:
        return "unscored"
    return "partial" if c.partial else "full"


def ranked(candidates: list[CtrPrediction]) -> list[CandidateOut]:
    """Full first, then partial, each by score (desc); unscored last by id.

    Ties share a rank: a heuristic tie is shown as a tie, with no hidden
    tiebreak (0019).
    """
    order = {"full": 0, "partial": 1, "unscored": 2}
    rows = sorted(candidates, key=lambda c: (order[_group(c)], -(c.predicted_score or 0), c.id))
    out: list[CandidateOut] = []
    prev: tuple[Group, float | None] | None = None
    rank = 0
    position = {"full": 0, "partial": 0}
    for c in rows:
        group = _group(c)
        r = None
        if group != "unscored":
            position[group] += 1
            if prev != (group, c.predicted_score):
                rank = position[group]
            r = rank
            prev = (group, c.predicted_score)
        out.append(
            CandidateOut(
                id=c.id,
                upload_id=c.upload_id,
                candidate_title=c.candidate_title,
                has_thumbnail=c.candidate_thumbnail_path is not None,
                predicted_score=c.predicted_score,
                partial=c.partial,
                features=c.features,
                source=c.source,
                scored_at=c.scored_at,
                group=group,
                rank=r,
            )
        )
    return out


def _clean_title(title: str) -> str:
    title = " ".join(title.split())
    if not title:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "the title is empty")
    if len(title) > MAX_TITLE_CHARS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"the title is over {MAX_TITLE_CHARS} characters")
    return title


async def _read_thumbnail(file: UploadFile) -> bytes:
    data = await file.read(MAX_THUMBNAIL_BYTES + 1)
    try:
        validate_thumbnail(data)
    except InvalidThumbnailError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"thumbnail: {exc}") from exc
    return data


def _store_thumbnail(data: bytes) -> str:
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[sniff_type(data)]
    key = storage.make_key(f"ctr-thumbnail.{ext}")
    storage.save(key, data)
    return key


def _drop_file(key: str | None) -> None:
    if key:
        try:
            storage.delete(key)
        except StorageKeyNotFoundError:
            pass


def _clear_score(c: CtrPrediction) -> None:
    c.predicted_score = None
    c.features = None
    c.partial = False
    c.scored_at = None
    c.model_version_id = None


async def _upload_or_404(db: AsyncSession, upload_id: int) -> Upload:
    upload = await db.get(Upload, upload_id)
    if upload is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Upload not found")
    return upload


async def _candidate_or_404(db: AsyncSession, cid: int) -> CtrPrediction:
    c = await db.get(CtrPrediction, cid)
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Candidate not found")
    return c


async def _candidates(db: AsyncSession, upload_id: int) -> list[CtrPrediction]:
    return list((await db.execute(select(CtrPrediction).where(CtrPrediction.upload_id == upload_id))).scalars())


async def _room(db: AsyncSession, upload_id: int) -> int:
    n = await db.scalar(select(func.count()).select_from(CtrPrediction).where(CtrPrediction.upload_id == upload_id))
    return MAX_CANDIDATES_PER_UPLOAD - (n or 0)


@router.get("/uploads/{upload_id}/ctr-candidates", response_model=list[CandidateOut])
async def list_candidates(
    upload_id: int, db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)
) -> list[CandidateOut]:
    await _upload_or_404(db, upload_id)
    return ranked(await _candidates(db, upload_id))


@router.post("/uploads/{upload_id}/ctr-candidates", response_model=CandidateOut, status_code=201)
async def create_candidate(
    upload_id: int,
    title: str = Form(...),
    thumbnail: UploadFile | None = File(None),
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> CandidateOut:
    await _upload_or_404(db, upload_id)
    clean = _clean_title(title)
    if await _room(db, upload_id) <= 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"at most {MAX_CANDIDATES_PER_UPLOAD} candidates per upload")
    data = await _read_thumbnail(thumbnail) if thumbnail is not None else None
    key = _store_thumbnail(data) if data is not None else None
    c = CtrPrediction(upload_id=upload_id, candidate_title=clean, candidate_thumbnail_path=key, source="manual")
    db.add(c)
    await db.commit()
    await db.refresh(c)
    return ranked([c])[0]


@router.post("/uploads/{upload_id}/ctr-candidates/from-show-notes", response_model=list[CandidateOut])
async def candidates_from_show_notes(
    upload_id: int, db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)
) -> list[CandidateOut]:
    await _upload_or_404(db, upload_id)
    draft = await db.scalar(
        select(ShowNotesDraft)
        .where(ShowNotesDraft.upload_id == upload_id)
        .order_by(ShowNotesDraft.created_at.desc(), ShowNotesDraft.id.desc())
        .limit(1)
    )
    if draft is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No show notes for this upload -- run Show notes first")
    existing = await _candidates(db, upload_id)
    seen = {c.candidate_title.casefold() for c in existing}
    room = MAX_CANDIDATES_PER_UPLOAD - len(existing)
    for raw in draft.title_suggestions:
        title = " ".join(str(raw).split())
        if room <= 0 or not title or len(title) > MAX_TITLE_CHARS or title.casefold() in seen:
            continue
        db.add(CtrPrediction(upload_id=upload_id, candidate_title=title, source="show_notes"))
        seen.add(title.casefold())
        room -= 1
    await db.commit()
    return ranked(await _candidates(db, upload_id))


@router.patch("/ctr-candidates/{cid}", response_model=CandidateOut)
async def update_candidate(
    cid: int,
    title: str | None = Form(None),
    thumbnail: UploadFile | None = File(None),
    remove_thumbnail: bool = Form(False),
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> CandidateOut:
    c = await _candidate_or_404(db, cid)
    changed = False
    if title is not None:
        clean = _clean_title(title)
        changed |= clean != c.candidate_title
        c.candidate_title = clean
    if thumbnail is not None:
        data = await _read_thumbnail(thumbnail)
        old, c.candidate_thumbnail_path = c.candidate_thumbnail_path, _store_thumbnail(data)
        _drop_file(old)
        changed = True
    elif remove_thumbnail and c.candidate_thumbnail_path:
        _drop_file(c.candidate_thumbnail_path)
        c.candidate_thumbnail_path = None
        changed = True
    if changed:
        _clear_score(c)
    await db.commit()
    await db.refresh(c)
    return ranked([c])[0]


@router.delete("/ctr-candidates/{cid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_candidate(
    cid: int, db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)
) -> Response:
    c = await _candidate_or_404(db, cid)
    _drop_file(c.candidate_thumbnail_path)
    await db.delete(c)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/ctr-candidates/{cid}/thumbnail")
async def get_thumbnail(
    cid: int, db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user_for_media)
) -> Response:
    # `<img src>` can't send a header: the token may come as `?token=` (0005).
    c = await _candidate_or_404(db, cid)
    if not c.candidate_thumbnail_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This candidate has no thumbnail")
    try:
        with storage.get(c.candidate_thumbnail_path) as f:
            data = f.read()
    except StorageKeyNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thumbnail file missing") from exc
    return Response(content=data, media_type=sniff_type(data) or "application/octet-stream")
