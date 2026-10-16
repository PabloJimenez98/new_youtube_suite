"""Real handler for `tool_type='transcribe'` (M3).

ffmpeg → 16kHz mono WAV → faster-whisper → `transcript_segments` /
`transcript_words`. The WAV is kept in storage (key on
`uploads.audio_storage_path`) for diarization (M4) to reuse. See
docs/decisions/0007-transcription-pipeline.md.
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.job import Job
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.storage import LocalStorageBackend
from app.transcription import whisper
from app.transcription.audio import extract_audio, wav_duration
from app.worker.gpu import gpu_lock

log = logging.getLogger(__name__)


def audio_key_for(upload_id: int) -> str:
    # Flat key: `LocalStorageBackend` flattens any directory components.
    return f"{upload_id}_audio.wav"


async def run(job: Job, db: AsyncSession) -> None:
    upload = await db.get(Upload, job.upload_id)
    if upload is None:
        raise LookupError(f"Upload {job.upload_id} not found")

    storage = LocalStorageBackend(get_settings().storage_base_path)
    source = storage.path_for(upload.storage_path)

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ys_transcribe_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        await extract_audio(source, wav_path)
        duration = wav_duration(wav_path)
        t_extract = time.perf_counter()

        async with gpu_lock:
            t_gpu = time.perf_counter()
            results = await asyncio.to_thread(whisper.transcribe, wav_path)
        t_whisper = time.perf_counter()

        audio_key = audio_key_for(upload.id)
        with wav_path.open("rb") as wav:
            storage.save(audio_key, wav)
    t_saved = time.perf_counter()

    try:
        upload.audio_storage_path = audio_key
        # Nothing else fills `uploads.duration_seconds` (found in the M9 season
        # run): the dashboard's runtime and the LLM tools' windows rely on it.
        upload.duration_seconds = duration
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
        log.info(
            "transcribe upload %d: extract %.1fs, gpu wait %.1fs, whisper %.1fs, save wav %.1fs, db %.1fs",
            upload.id, t_extract - t0, t_gpu - t_extract, t_whisper - t_gpu, t_saved - t_whisper,
            time.perf_counter() - t_saved,
        )
    except Exception:
        # Leave the session usable for `process_job` to record `failed`.
        await db.rollback()
        raise
