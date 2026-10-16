"""Real handler for `tool_type='relevance'` (M7, bootstrap v0).

Transcript words (+ speaker labels and chapters, when they exist) ->
heuristic + batched LLM scores per 15 s window -> `relevance_scores`
rows tagged with a `model_versions` row. CPU only; no GPU lock. See
docs/decisions/0012-relevance-v0.md.
"""

import asyncio

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.chapter import Chapter
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.relevance import RelevanceScore
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.relevance import pipeline
from app.relevance.windows import Word


class TranscriptMissingError(Exception):
    """Relevance is scored on an existing transcript -- there isn't one yet."""


def words_from_segments(segments: list[TranscriptSegment]) -> list[Word]:
    """Words when a segment has them (word speaker, else the segment's), else the segment as one word."""
    words: list[Word] = []
    for seg in segments:
        if seg.words:
            words.extend(Word(w.start_seconds, w.word, w.speaker_label or seg.speaker_label) for w in seg.words)
        else:
            words.append(Word(seg.start_seconds, seg.text, seg.speaker_label))
    return words


async def get_or_create_version(db: AsyncSession, label: str, description: dict) -> ModelVersion:
    """The `relevance` row for this label; created live when no relevance version is live yet."""
    version = await db.scalar(
        select(ModelVersion).where(ModelVersion.model_name == "relevance", ModelVersion.version_label == label)
    )
    if version is None:
        live = await db.scalar(
            select(ModelVersion.id).where(ModelVersion.model_name == "relevance", ModelVersion.is_live)
        )
        version = ModelVersion(model_name="relevance", version_label=label, is_live=live is None, **description)
        db.add(version)
        await db.flush()
    return version


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

    chapters = [
        (c.start_seconds, c.title)
        for c in (
            await db.execute(select(Chapter).where(Chapter.upload_id == upload.id).order_by(Chapter.order_index))
        ).scalars()
    ]
    duration = upload.duration_seconds or max(s.end_seconds for s in segments)
    label, description, result = await asyncio.to_thread(
        pipeline.score_relevance, words_from_segments(segments), duration, chapters
    )

    try:
        version = await get_or_create_version(db, label, description)
        # Replace, not append: a re-run is idempotent.
        await db.execute(delete(RelevanceScore).where(RelevanceScore.upload_id == upload.id))
        db.add_all(
            RelevanceScore(
                upload_id=upload.id,
                window_index=s.window.index,
                window_start_seconds=s.window.start,
                window_end_seconds=s.window.end,
                score=s.score,
                heuristic_score=s.heuristic,
                llm_score=s.llm,
                model_version_id=version.id,
            )
            for s in result.scores
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
