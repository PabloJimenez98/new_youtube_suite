"""Structural checks per episode after a harness run (the M9 checks, 0014 §5):
segments, speakers, chapters, relevance coverage + version, show notes language, shorts.

    cd backend && <venv>/python ../experiments/hw-upgrade/structural_check.py
"""
import asyncio

from sqlalchemy import func, select

from app.database import async_session_factory
from app.models.chapter import Chapter
from app.models.clip import Clip
from app.models.model_version import ModelVersion
from app.models.relevance import RelevanceScore
from app.models.show_notes import ShowNotesDraft
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload


async def main() -> None:
    async with async_session_factory() as db:
        uploads = (await db.scalars(select(Upload).order_by(Upload.id))).all()
        live = (await db.execute(select(ModelVersion.version_label, ModelVersion.is_live)
                                 .where(ModelVersion.model_name == "relevance"))).all()
        print("relevance versions:", live)
        print(f"{'episode':<12} {'min':>5} {'segs':>5} {'spk':>4} {'chap':>5} {'rel_win':>8} {'rel_llm%':>8} {'notes':>6} {'clips':>6}  first chapter titles")
        for u in uploads:
            q = lambda m, *w: db.scalar(select(func.count()).select_from(m).where(*w))
            segs = await q(TranscriptSegment, TranscriptSegment.upload_id == u.id)
            spk = await db.scalar(select(func.count(func.distinct(TranscriptSegment.speaker_label)))
                                  .where(TranscriptSegment.upload_id == u.id))
            chaps = (await db.scalars(select(Chapter.title).where(Chapter.upload_id == u.id).order_by(Chapter.order_index))).all()
            rel = await q(RelevanceScore, RelevanceScore.upload_id == u.id)
            rel_llm = await q(RelevanceScore, RelevanceScore.upload_id == u.id, RelevanceScore.llm_score.isnot(None)) \
                if hasattr(RelevanceScore, "llm_score") else None
            notes = await db.scalar(select(ShowNotesDraft).where(ShowNotesDraft.upload_id == u.id))
            clips = await q(Clip, Clip.upload_id == u.id)
            ep = u.filename.split("[")[-1].rstrip("].webmkv") if "[" in u.filename else u.filename[:11]
            pct = f"{100 * rel_llm / rel:.0f}" if rel and rel_llm is not None else "?"
            lang = getattr(notes, "language", None) if notes else None
            print(f"{ep:<12} {(u.duration_seconds or 0) / 60:5.1f} {segs:>5} {spk:>4} {len(chaps):>5} {rel:>8} {pct:>8} "
                  f"{(lang or ('yes' if notes else 'no')):>6} {clips:>6}  {' | '.join(chaps[:3])}")


asyncio.run(main())
