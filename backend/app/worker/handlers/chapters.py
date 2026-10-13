"""Real handler for `tool_type='chapters'` (M5).

Transcript words -> text units -> boundary scores (BiLSTM or the
TextTiling baseline, per the configured bundle) -> decoded chapter
starts -> mT5 titles -> `chapters` rows. All on CPU; no GPU lock. See
docs/decisions/0009-chapter-generator-local-models.md §7.
"""

import asyncio

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.chapters import pipeline
from app.chapters.units import Atom
from app.models.chapter import Chapter
from app.models.job import Job
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload


class TranscriptMissingError(Exception):
    """Chapters are computed from an existing transcript -- there isn't one yet."""


def atoms_from_segments(segments: list[TranscriptSegment]) -> list[Atom]:
    """Words when a segment has them (timed or not), else the segment itself."""
    atoms: list[Atom] = []
    for seg in segments:
        if seg.words:
            atoms.extend(Atom(w.start_seconds, w.end_seconds, w.word) for w in seg.words)
        else:
            atoms.append(Atom(seg.start_seconds, seg.end_seconds, seg.text))
    return atoms


async def run(job: Job, db: AsyncSession) -> None:
    upload = await db.get(Upload, job.upload_id)
    if upload is None:
        raise LookupError(f"Upload {job.upload_id} not found")

    has_transcript = await db.scalar(
        select(Job.id).where(
            Job.upload_id == upload.id,
            Job.tool_type == "transcribe",
            Job.status == "succeeded",
        ).limit(1)
    )
    segments = list(
        (
            await db.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.upload_id == upload.id)
                .order_by(TranscriptSegment.segment_index)
                .options(selectinload(TranscriptSegment.words))
            )
        ).scalars()
    )
    if has_transcript is None or not segments:
        raise TranscriptMissingError("No transcript for this upload -- run Transcript first")

    duration = upload.duration_seconds or max(s.end_seconds for s in segments)
    version, results = await asyncio.to_thread(
        pipeline.generate_chapters, atoms_from_segments(segments), duration
    )

    try:
        # Replace, not append: a re-run is idempotent.
        await db.execute(delete(Chapter).where(Chapter.upload_id == upload.id))
        db.add_all(
            Chapter(
                upload_id=upload.id,
                order_index=i,
                start_seconds=r.start,
                title=r.title,
                model_version=version,
            )
            for i, r in enumerate(results)
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
