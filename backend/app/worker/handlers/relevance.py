"""Real handler for `tool_type='relevance'` (M7, dispatch since M15).

The live `relevance` version decides the path (docs/decisions/0024 §6):

- **A teacher** (no `artifact_path`; M7's bootstrap): transcript words
  (+ speaker labels and chapters, when they exist) -> heuristic + batched
  LLM scores per 15 s window -> `relevance_scores`, and the same components
  into `relevance_pseudo_labels`. The LLM runs inside the GPU lock (since
  M9; 0016). See docs/decisions/0012-relevance-v0.md.
- **A trained BiLSTM** (an `artifact_path`): encoder + BiLSTM over the same
  windows -> `relevance_scores` only, tagged with that version.
"""

import asyncio

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.chapter import Chapter
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.relevance import RelevancePseudoLabel, RelevanceScore
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.relevance import bilstm, pipeline
from app.relevance.pipeline import WindowScore
from app.relevance.windows import Word
from app.llm import runtime
from app.worker.gpu import gpu_lock


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


async def get_or_create_version(
    db: AsyncSession, label: str, description: dict, *, model_name: str = "relevance"
) -> ModelVersion:
    """The `model_name` row for this label; created live when no version of that model is live yet.

    Shared with the CTR handler (M13, `model_name="ctr"`)."""
    version = await db.scalar(
        select(ModelVersion).where(ModelVersion.model_name == model_name, ModelVersion.version_label == label)
    )
    if version is None:
        live = await db.scalar(
            select(ModelVersion.id).where(ModelVersion.model_name == model_name, ModelVersion.is_live)
        )
        version = ModelVersion(model_name=model_name, version_label=label, is_live=live is None, **description)
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
    words = words_from_segments(segments)

    live = await db.scalar(
        select(ModelVersion).where(ModelVersion.model_name == "relevance", ModelVersion.is_live)
    )
    if live is not None and live.artifact_path is not None:
        # A trained model is live (0024 §6): encoder + BiLSTM, no LLM, no pseudo-labels.
        async with gpu_lock:
            result = await asyncio.to_thread(bilstm.score_with_bundle, live.artifact_path, words, duration)
        await _replace_scores(db, upload.id, live, result.scores)
        return

    label, description, result = await score_with_teacher(words, duration, chapters)
    try:
        version = await get_or_create_version(db, label, description)
        await _write_scores(db, upload.id, version, result.scores)
        # The teacher's components are also M15's training targets (0024 §2).
        await write_pseudo_labels(db, upload.id, version, result.scores)
        await db.commit()
    except Exception:
        await db.rollback()
        raise


async def score_with_teacher(words: list[Word], duration: float, chapters: list[tuple[float, str]]):
    """(label, description, result) from heuristics + LLM, inside the GPU lock."""
    # The LLM's prompt prefill runs on the GPU with the CUDA build (0014 §2).
    async with gpu_lock:
        try:
            return await asyncio.to_thread(pipeline.score_relevance, words, duration, chapters)
        finally:
            # Free the LLM's ~1 GB of VRAM for Whisper/pyannote (0014 §2).
            await asyncio.to_thread(runtime.release_all)


async def _write_scores(db: AsyncSession, upload_id: int, version: ModelVersion, scores: list[WindowScore]) -> None:
    # Replace, not append: a re-run is idempotent.
    await db.execute(delete(RelevanceScore).where(RelevanceScore.upload_id == upload_id))
    db.add_all(
        RelevanceScore(
            upload_id=upload_id,
            window_index=s.window.index,
            window_start_seconds=s.window.start,
            window_end_seconds=s.window.end,
            score=s.score,
            heuristic_score=s.heuristic,
            llm_score=s.llm,
            model_version_id=version.id,
        )
        for s in scores
    )


async def _replace_scores(db: AsyncSession, upload_id: int, version: ModelVersion, scores: list[WindowScore]) -> None:
    try:
        await _write_scores(db, upload_id, version, scores)
        await db.commit()
    except Exception:
        await db.rollback()
        raise


async def write_pseudo_labels(db: AsyncSession, upload_id: int, teacher: ModelVersion,
                              scores: list[WindowScore]) -> None:
    """Replace this teacher's labels for the upload (other teachers' labels are kept). No commit."""
    if teacher.artifact_path is not None:
        raise ValueError(f"{teacher.version_label!r} is a trained model; only teachers write pseudo-labels")
    await db.execute(
        delete(RelevancePseudoLabel).where(
            RelevancePseudoLabel.upload_id == upload_id, RelevancePseudoLabel.teacher_version_id == teacher.id
        )
    )
    db.add_all(
        RelevancePseudoLabel(
            upload_id=upload_id,
            window_index=s.window.index,
            window_start_seconds=s.window.start,
            window_end_seconds=s.window.end,
            heuristic_score=s.heuristic,
            llm_score=s.llm,
            teacher_version_id=teacher.id,
        )
        for s in scores
    )
