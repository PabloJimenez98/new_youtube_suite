"""Real handler for `tool_type='shownotes'` (M6).

Transcript (+ chapters, when they exist) -> local LLM map-reduce ->
one `show_notes_drafts` row per upload. CPU only; no GPU lock. See
docs/decisions/0011-show-notes-local-llm.md.
"""

import asyncio

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chapter import Chapter
from app.models.job import Job
from app.models.show_notes import ShowNotesDraft
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.shownotes import pipeline
from app.shownotes.chunking import ChapterMark, Seg


class TranscriptMissingError(Exception):
    """Show notes are written from an existing transcript -- there isn't one yet."""


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
    segs = [
        Seg(s.start_seconds, s.end_seconds, s.text)
        for s in (
            await db.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.upload_id == upload.id)
                .order_by(TranscriptSegment.segment_index)
            )
        ).scalars()
    ]
    if has_transcript is None or not segs:
        raise TranscriptMissingError("No transcript for this upload -- run Transcript first")

    # Chapters are optional input (0011 §2): whatever exists right now.
    chapters = [
        ChapterMark(c.start_seconds, c.title)
        for c in (
            await db.execute(select(Chapter).where(Chapter.upload_id == upload.id).order_by(Chapter.order_index))
        ).scalars()
    ]
    model_used, notes = await asyncio.to_thread(pipeline.generate_show_notes, segs, chapters)

    try:
        # Replace, not append: one draft per upload, a re-run overwrites it.
        await db.execute(delete(ShowNotesDraft).where(ShowNotesDraft.upload_id == upload.id))
        db.add(
            ShowNotesDraft(
                upload_id=upload.id,
                title_suggestions=notes.draft.title_suggestions,
                description=notes.draft.description,
                tags=notes.draft.tags,
                summary=notes.draft.summary,
                language=notes.language,
                llm_model_used=model_used,
            )
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
