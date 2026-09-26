"""Real handler for `tool_type='transcribe'` (M3).

ffmpeg → 16kHz mono WAV → faster-whisper → `transcript_segments` /
`transcript_words`. The WAV is kept in storage (key on
`uploads.audio_storage_path`) for diarization (M4) to reuse. See
docs/decisions/0007-transcription-pipeline.md.
"""

import asyncio
import tempfile
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.job import Job
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.storage import LocalStorageBackend
from app.transcription import whisper
from app.transcription.audio import extract_audio

# arq runs several jobs concurrently, but only one Whisper model fits in
# the GPU's 3 GB -- serialize transcriptions within this worker process.
_transcribe_lock = asyncio.Lock()


def audio_key_for(upload_id: int) -> str:
    # Flat key: `LocalStorageBackend` flattens any directory components.
    return f"{upload_id}_audio.wav"


async def run(job: Job, db: AsyncSession) -> None:
    upload = await db.get(Upload, job.upload_id)
    if upload is None:
        raise LookupError(f"Upload {job.upload_id} not found")

    storage = LocalStorageBackend(get_settings().storage_base_path)
    source = storage.path_for(upload.storage_path)

    with tempfile.TemporaryDirectory(prefix="ys_transcribe_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        await extract_audio(source, wav_path)

        async with _transcribe_lock:
            results = await asyncio.to_thread(whisper.transcribe, wav_path)

        audio_key = audio_key_for(upload.id)
        with wav_path.open("rb") as wav:
            storage.save(audio_key, wav)

    try:
        upload.audio_storage_path = audio_key
        # Replace, don't append: a re-run must not duplicate rows. Done in
        # one transaction so readers see the old or the new transcript.
        await db.execute(
            delete(TranscriptSegment).where(TranscriptSegment.upload_id == upload.id)
        )
        db.add_all(
            TranscriptSegment(
                upload_id=upload.id,
                segment_index=seg_index,
                start_seconds=seg.start,
                end_seconds=seg.end,
                text=seg.text,
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
                words=[
                    TranscriptWord(
                        word_index=word_index,
                        word=w.word,
                        start_seconds=w.start,
                        end_seconds=w.end,
                        probability=w.probability,
                    )
                    for word_index, w in enumerate(seg.words)
                ],
            )
            for seg_index, seg in enumerate(results)
        )
        await db.commit()
    except Exception:
        # Leave the session usable for `process_job` to record `failed`.
        await db.rollback()
        raise
