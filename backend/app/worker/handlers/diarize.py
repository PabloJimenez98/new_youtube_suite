"""Real handler for `tool_type='diarize'` (M4).

pyannote `community-1` on the WAV M3 stored → speaker turns → speaker
per word and per segment (`app.diarization.assign`) → `speaker_label`
on the existing transcript rows. See
docs/decisions/0008-speaker-diarization.md.
"""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.diarization import pipeline
from app.diarization.assign import SpeakerAssigner, segment_speaker
from app.models.job import Job
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.storage import LocalStorageBackend
from app.worker.gpu import gpu_lock


class TranscriptMissingError(Exception):
    """Diarization labels an existing transcript -- there isn't one yet."""


async def run(job: Job, db: AsyncSession) -> None:
    pipeline.ensure_available()

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
    if has_transcript is None or not segments or upload.audio_storage_path is None:
        raise TranscriptMissingError("No transcript for this upload -- run Transcript first")

    storage = LocalStorageBackend(get_settings().storage_base_path)
    wav_path = storage.path_for(upload.audio_storage_path)
    if not wav_path.is_file():
        raise TranscriptMissingError(
            "Extracted audio is missing from storage -- re-run Transcript first"
        )

    async with gpu_lock:
        turns = await asyncio.to_thread(pipeline.diarize, wav_path)

    assigner = SpeakerAssigner(turns)
    try:
        # Every word and segment of the upload is (re)assigned -- including
        # to None -- so a re-run never leaves stale labels behind.
        for segment in segments:
            word_speakers = [
                assigner.speaker_for(w.start_seconds, w.end_seconds) for w in segment.words
            ]
            for word, speaker in zip(segment.words, word_speakers, strict=True):
                word.speaker_label = speaker
            segment.speaker_label = segment_speaker(word_speakers)
        await db.commit()
    except Exception:
        await db.rollback()
        raise
