"""Real handler for `tool_type='hook'` (M14, heuristic v0).

It (re)scores the opening 15 s of every clip of the upload and upserts one
`hook_scores` row per clip.

- CPU only; no GPU lock.
- No clips → the job fails with a message.
- A missing spaCy or emotion model fails the job once, before any clip.

See docs/decisions/0020-hook-heuristic-v0.md §5.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.hook import nlp, scorer
from app.hook.pipeline import score_opening
from app.hook.window import detect_language, opening
from app.models.clip import Clip
from app.models.hook import HookScore
from app.models.job import Job
from app.models.show_notes import ShowNotesDraft
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.worker.handlers.relevance import get_or_create_version
from app.worker.handlers.shorts import words_from_segments


class NoClipsError(Exception):
    """The hook scorer scores shorts -- there are none yet."""


async def episode_title(db: AsyncSession, upload: Upload) -> tuple[str, str]:
    """(title, source): the show-notes draft's first suggestion, else the filename stem."""
    draft = await db.scalar(select(ShowNotesDraft).where(ShowNotesDraft.upload_id == upload.id))
    if draft and draft.title_suggestions:
        return str(draft.title_suggestions[0]), "show_notes"
    return " ".join(Path(upload.filename).stem.replace("_", " ").replace("-", " ").split()), "filename"


async def run(job: Job, db: AsyncSession) -> None:
    upload = await db.get(Upload, job.upload_id)
    if upload is None:
        raise LookupError(f"Upload {job.upload_id} not found")
    clips = list((await db.execute(select(Clip).where(Clip.upload_id == upload.id).order_by(Clip.rank))).scalars())
    if not clips:
        raise NoClipsError("No shorts yet -- run Shorts first")

    segments = list(
        (
            await db.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.upload_id == upload.id)
                .options(selectinload(TranscriptSegment.words))
                .order_by(TranscriptSegment.segment_index)
            )
        ).scalars()
    )
    words = words_from_segments(segments)
    lang, fallback = detect_language(" ".join(w.text for w in words)[:20000])
    title, title_source = await episode_title(db, upload)
    await asyncio.to_thread(nlp.require, lang)

    existing = {
        h.clip_id: h
        for h in (await db.execute(select(HookScore).where(HookScore.clip_id.in_([c.id for c in clips])))).scalars()
    }
    try:
        version = await get_or_create_version(
            db,
            scorer.VERSION_LABEL,
            {"architecture_notes": scorer.architecture_notes(), "training_data_reference": {"kind": "untrained"}},
            model_name="hook",
        )
        now = datetime.now(timezone.utc)
        for clip in clips:
            window = opening(words, clip.start_seconds, clip.end_seconds)
            result = await asyncio.to_thread(score_opening, window, lang, fallback, title, title_source)
            row = existing.get(clip.id) or HookScore(clip_id=clip.id)
            row.score, row.feature_vector = result.score, result.feature_vector
            row.model_version_id, row.scored_at = version.id, now
            db.add(row)
        await db.commit()
    except Exception:
        await db.rollback()
        raise
