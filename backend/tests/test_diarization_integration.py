"""The real pipeline end to end on the 2-speaker fixture: ffmpeg +
Whisper `base` (CPU) → transcript rows → pyannote `community-1` (CPU) →
speaker labels, checked against the clip's known turn boundaries.

Accuracy bar: ≥ 90% of timed words carry the right speaker under the
best label permutation, and every segment is labelled -- see
docs/decisions/0008-speaker-diarization.md.
"""

import json
from itertools import permutations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.diarization.pipeline import HEALTHCHECK_CLIP
from app.models.job import Job
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.storage import LocalStorageBackend
from app.worker.tasks import process_job
from tests.fixtures import CONVERSATION_CLIP

MIN_WORD_ACCURACY = 0.90
TRUTH = json.loads(HEALTHCHECK_CLIP.with_suffix(".json").read_text())["turns"]


def _true_speaker(start: float, end: float) -> str:
    mid = (start + end) / 2

    def distance(turn: dict) -> float:
        if turn["start"] <= mid <= turn["end"]:
            return 0.0
        return min(abs(mid - turn["start"]), abs(mid - turn["end"]))

    return min(TRUTH, key=distance)["speaker"]


async def test_transcribe_then_diarize_labels_speakers_correctly(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    storage = LocalStorageBackend(get_settings().storage_base_path)
    key = LocalStorageBackend.make_key("conversation_clip.mp4")
    storage.save(key, CONVERSATION_CLIP.read_bytes())
    upload = Upload(filename="conversation_clip.mp4", storage_path=key)
    db_session.add(upload)
    await db_session.commit()
    upload_id = upload.id

    for tool_type in ("transcribe", "diarize"):
        job = Job(upload_id=upload_id, tool_type=tool_type)
        db_session.add(job)
        await db_session.commit()
        await process_job({"session_factory": session_factory}, job.id)
        await db_session.refresh(job)
        assert job.status == "succeeded", f"{tool_type}: {job.error_message}"

    db_session.expire_all()
    segments = list(
        (
            await db_session.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.upload_id == upload_id)
                .order_by(TranscriptSegment.segment_index)
                .options(selectinload(TranscriptSegment.words))
            )
        ).scalars()
    )
    assert segments
    assert all(s.speaker_label is not None for s in segments)

    timed = [w for s in segments for w in s.words if w.start_seconds is not None]
    assert len(timed) > 50  # ~37 s of read speech
    predicted = [w.speaker_label for w in timed]
    truth = [_true_speaker(w.start_seconds, w.end_seconds) for w in timed]
    labels = sorted({p for p in predicted if p is not None})
    assert labels == ["SPEAKER_00", "SPEAKER_01"]

    best = max(
        sum(p is not None and mapping[p] == t for p, t in zip(predicted, truth, strict=True))
        for mapping in (dict(zip(labels, perm, strict=True)) for perm in permutations("AB"))
    ) / len(timed)
    assert best >= MIN_WORD_ACCURACY, f"word speaker accuracy {best:.3f} < {MIN_WORD_ACCURACY}"
