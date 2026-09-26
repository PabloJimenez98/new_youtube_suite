"""The real pipeline on the fixture clip: ffmpeg + faster-whisper (`base`
on CPU in tests, per conftest) → rows in the DB, checked against the
LibriSpeech reference transcript.

Accuracy bar: WER ≤ 0.15 after lowercasing and stripping punctuation.
`base` is typically ~5% WER on clean LibriSpeech audio; the margin
absorbs single-utterance variance. See
docs/decisions/0007-transcription-pipeline.md.
"""

import re

import jiwer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.models.job import Job
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.storage import LocalStorageBackend
from app.worker.tasks import process_job
from tests.fixtures import FIXTURE_CLIP, FIXTURE_CLIP_REFERENCE

MAX_WER = 0.15
CLIP_DURATION = 18.3


def _normalize(text: str) -> str:
    text = text.lower().replace("-", " ")
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text).split())


async def test_fixture_clip_transcribes_within_tolerance(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    storage = LocalStorageBackend(get_settings().storage_base_path)
    key = LocalStorageBackend.make_key("librispeech_clip.mp4")
    storage.save(key, FIXTURE_CLIP.read_bytes())
    upload = Upload(filename="librispeech_clip.mp4", storage_path=key)
    db_session.add(upload)
    await db_session.commit()
    job = Job(upload_id=upload.id, tool_type="transcribe")
    db_session.add(job)
    await db_session.commit()

    await process_job({"session_factory": session_factory}, job.id)

    await db_session.refresh(job)
    assert job.status == "succeeded", job.error_message

    segments = list(
        (
            await db_session.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.upload_id == upload.id)
                .order_by(TranscriptSegment.segment_index)
            )
        ).scalars()
    )
    assert segments

    hypothesis = " ".join(s.text for s in segments)
    wer = jiwer.wer(_normalize(FIXTURE_CLIP_REFERENCE), _normalize(hypothesis))
    assert wer <= MAX_WER, f"WER {wer:.3f} > {MAX_WER}: {hypothesis!r}"

    # Timestamps: ordered, inside the clip, words inside their segment.
    previous_end = 0.0
    for seg in segments:
        assert 0 <= seg.start_seconds <= seg.end_seconds <= CLIP_DURATION + 0.5
        assert seg.start_seconds >= previous_end - 0.01
        previous_end = seg.end_seconds

        words = list(
            (
                await db_session.execute(
                    select(TranscriptWord)
                    .where(TranscriptWord.segment_id == seg.id)
                    .order_by(TranscriptWord.word_index)
                )
            ).scalars()
        )
        assert words
        timed = [w for w in words if w.start_seconds is not None]
        for w in timed:
            assert seg.start_seconds - 0.01 <= w.start_seconds <= w.end_seconds
            assert w.end_seconds <= seg.end_seconds + 0.01
            assert 0 <= w.probability <= 1
        starts = [w.start_seconds for w in timed]
        assert starts == sorted(starts)
