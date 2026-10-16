"""Real handler for `tool_type='shorts'` (M8).

Relevance curve + transcript words -> up to five spans -> for each:
face analysis, framing plan, karaoke ASS, one ffmpeg render -> `clips`
rows and files. CPU only; no GPU lock. See
docs/decisions/0013-shorts-generator.md §7.
"""

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models.clip import Clip
from app.models.job import Job
from app.models.relevance import RelevanceScore
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.schemas.job import ShortsParams
from app.shorts import captions, framing, render
from app.shorts.framing import FaceModelMissingError
from app.shorts.selection import ScoreWindow, Span, TimedWord, select_spans
from app.storage import LocalStorageBackend
from app.storage.base import StorageKeyNotFoundError


class RelevanceMissingError(Exception):
    """Shorts are cut from the relevance curve -- there isn't one yet."""


class NoClipsError(Exception):
    """The curve has no peak that makes a clip."""


@dataclass
class Rendered:
    span: Span
    key: str
    plan: framing.Plan | None = None


def words_from_segments(segments: list[TranscriptSegment]) -> list[TimedWord]:
    """Timed words (the segment as one word when it has none), marking segment starts."""
    words: list[TimedWord] = []
    for seg in segments:
        if seg.words:
            words.extend(
                TimedWord(w.start_seconds, w.end_seconds, w.word, segment_start=k == 0)
                for k, w in enumerate(seg.words)
            )
        else:
            words.append(TimedWord(seg.start_seconds, seg.end_seconds, seg.text, segment_start=True))
    return words


def make_clip(src: Path, span: Span, words: list[TimedWord], params: ShortsParams, model: Path,
              out: Path) -> framing.Plan:
    """Analyse, plan and render one span to `out`. Synchronous -- run in a thread."""
    analysis = framing.analyse(src, span.start, span.end, model)
    plan = framing.plan(analysis, span.end - span.start, params.aspect_ratio)
    ass = captions.build_ass(words, span.start, span.end, plan.out_w, plan.out_h) if params.captions else None
    render.render(src, span.start, span.end, plan, analysis.fps, ass, out)
    return plan


def _delete_files(storage: LocalStorageBackend, keys: list[str]) -> None:
    for key in keys:
        try:
            storage.delete(key)
        except StorageKeyNotFoundError:
            pass


async def run(job: Job, db: AsyncSession) -> None:
    upload = await db.get(Upload, job.upload_id)
    if upload is None:
        raise LookupError(f"Upload {job.upload_id} not found")

    scores = list(
        (
            await db.execute(
                select(RelevanceScore).where(RelevanceScore.upload_id == upload.id)
                .order_by(RelevanceScore.window_index)
            )
        ).scalars()
    )
    if not scores:
        raise RelevanceMissingError("No relevance scores for this upload -- run Relevance first")

    settings = get_settings()
    model = settings.shorts_face_model_path
    if not model or not Path(model).is_file():
        raise FaceModelMissingError(
            "No face detection model -- set SHORTS_FACE_MODEL_PATH (see README, 'Shorts')"
        )
    params = ShortsParams.model_validate(job.params or {})

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
    words = words_from_segments(segments)
    windows = [ScoreWindow(s.window_index, s.window_start_seconds, s.window_end_seconds, s.score) for s in scores]
    spans = select_spans(windows, words)
    if not spans:
        raise NoClipsError("The relevance curve has no clear peaks to cut shorts from")

    storage = LocalStorageBackend(settings.storage_base_path)
    src = storage.path_for(upload.storage_path)
    version_id = scores[0].model_version_id
    rendered: list[Rendered] = []
    try:
        for rank, span in enumerate(spans, 1):
            key = f"{uuid.uuid4()}_short_u{upload.id}_{rank}.mp4"
            rendered.append(Rendered(span, key))  # registered first: cleaned up if the render fails
            rendered[-1].plan = await asyncio.to_thread(
                make_clip, src, span, words, params, Path(model), storage.path_for(key)
            )
            job.progress_pct = round(100 * rank / len(spans))
            await db.commit()

        old_keys = list((await db.execute(select(Clip.storage_path).where(Clip.upload_id == upload.id))).scalars())
        # Replace, not append: a re-run is idempotent.
        await db.execute(delete(Clip).where(Clip.upload_id == upload.id))
        db.add_all(
            Clip(
                upload_id=upload.id,
                rank=rank,
                start_seconds=r.span.start,
                end_seconds=r.span.end,
                score=r.span.score,
                peak_window_index=r.span.peak_window_index,
                aspect_ratio=params.aspect_ratio,
                caption_burned_in=params.captions,
                storage_path=r.key,
                crop_bounding_box_track=r.plan.track,
                framing=r.plan.as_json(),
                model_version_id=version_id,
            )
            for rank, r in enumerate(rendered, 1)
        )
        await db.commit()
    except BaseException:
        # BaseException, not Exception: stopping the worker cancels the job
        # (CancelledError), and its half-rendered files must go too (M9 season run).
        _delete_files(storage, [r.key for r in rendered])
        await db.rollback()
        raise
    # The new clips are committed: only now drop the old files.
    _delete_files(storage, old_keys)
