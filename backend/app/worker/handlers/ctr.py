"""Real handler for `tool_type='ctr'` (M13, heuristic v0).

It (re)scores every title/thumbnail candidate of the upload and stores the
score, the features and the `model_versions` row on each candidate.

- CPU only; no GPU lock.
- No candidates → the job fails with a message.
- A candidate whose thumbnail can't be read gets `features.error`, and
  the others still score.
- A missing detector model fails the job: every thumbnail would fail the
  same way.

See docs/decisions/0019-ctr-heuristic-v0.md §4.
"""

import asyncio
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.ctr import detectors, scorer
from app.ctr.pipeline import InvalidThumbnailError, score_candidate
from app.models.ctr import CtrPrediction
from app.models.job import Job
from app.models.upload import Upload
from app.storage import LocalStorageBackend
from app.storage.base import StorageKeyNotFoundError
from app.worker.handlers.relevance import get_or_create_version


class NoCandidatesError(Exception):
    """The CTR tool scores candidates the owner entered -- there are none."""


def _read(storage: LocalStorageBackend, key: str) -> bytes:
    with storage.get(key) as f:
        return f.read()


async def run(job: Job, db: AsyncSession) -> None:
    upload = await db.get(Upload, job.upload_id)
    if upload is None:
        raise LookupError(f"Upload {job.upload_id} not found")
    candidates = list(
        (
            await db.execute(
                select(CtrPrediction).where(CtrPrediction.upload_id == upload.id).order_by(CtrPrediction.id)
            )
        ).scalars()
    )
    if not candidates:
        raise NoCandidatesError("No title/thumbnail candidates -- add candidates in the CTR panel first")

    settings = get_settings()
    face_model, text_model = settings.shorts_face_model_path, settings.ctr_text_model_path
    if any(c.candidate_thumbnail_path for c in candidates):
        # Fail once, clearly, instead of erroring every candidate.
        detectors.require_model(face_model, "SHORTS_FACE_MODEL_PATH")
        detectors.require_model(text_model, "CTR_TEXT_MODEL_PATH")

    storage = LocalStorageBackend(settings.storage_base_path)
    try:
        version = await get_or_create_version(
            db,
            scorer.VERSION_LABEL,
            {"architecture_notes": scorer.architecture_notes()},
            model_name="ctr",
        )
        now = datetime.now(timezone.utc)
        for c in candidates:
            try:
                data = await asyncio.to_thread(_read, storage, c.candidate_thumbnail_path) if c.candidate_thumbnail_path else None
                result = await asyncio.to_thread(score_candidate, c.candidate_title, data, face_model, text_model)
            except (InvalidThumbnailError, StorageKeyNotFoundError) as exc:
                c.predicted_score, c.partial = None, False
                c.features = {"version": scorer.VERSION_LABEL, "error": f"thumbnail unreadable: {exc}"}
            else:
                c.predicted_score = result.score.score
                c.partial = result.score.partial
                c.features = result.features
            c.model_version_id = version.id
            c.scored_at = now
        await db.commit()
    except Exception:
        await db.rollback()
        raise
