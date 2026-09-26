"""`diarize` handler with pyannote mocked out, driven through
`process_job` so the job-state transitions are covered too."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.diarization import pipeline
from app.diarization.assign import Turn
from app.models.job import Job
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.storage import LocalStorageBackend
from app.worker.tasks import process_job

storage = LocalStorageBackend(get_settings().storage_base_path)

FAKE_TURNS = [Turn(0.0, 2.0, "SPEAKER_00"), Turn(2.0, 4.0, "SPEAKER_01")]


@pytest.fixture(autouse=True)
def _reset_health() -> Iterator[None]:
    yield
    pipeline.health.error = None
    pipeline.health.checked = False


@pytest.fixture
def fake_diarize(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    seen: list[Path] = []

    def _diarize(wav_path: Path) -> list[Turn]:
        assert wav_path.is_file()
        seen.append(wav_path)
        return FAKE_TURNS

    monkeypatch.setattr(pipeline, "diarize", _diarize)
    return seen


async def _transcribed_upload(
    db: AsyncSession, *, with_audio: bool = True, transcribe_status: str | None = "succeeded"
) -> Upload:
    upload = Upload(filename="ep.mp4", storage_path="fake-video-key")
    if with_audio:
        upload.audio_storage_path = storage.save(
            LocalStorageBackend.make_key("audio.wav"),
            pipeline.HEALTHCHECK_CLIP.read_bytes(),
        )
    db.add(upload)
    await db.commit()
    if transcribe_status is not None:
        db.add(Job(upload_id=upload.id, tool_type="transcribe", status=transcribe_status))
    # Segment 0: words in the SPEAKER_00 turn, one untimed word, and one
    # straddler that overlaps SPEAKER_01 more. Segment 1: all SPEAKER_01.
    db.add_all(
        [
            TranscriptSegment(
                upload_id=upload.id,
                segment_index=0,
                start_seconds=0.0,
                end_seconds=2.3,
                text="hello there € friend",
                words=[
                    TranscriptWord(word_index=0, word="hello", start_seconds=0.1, end_seconds=0.5),
                    TranscriptWord(word_index=1, word="there", start_seconds=0.6, end_seconds=1.0),
                    TranscriptWord(word_index=2, word="€"),
                    TranscriptWord(word_index=3, word="friend", start_seconds=1.9, end_seconds=2.3),
                ],
            ),
            TranscriptSegment(
                upload_id=upload.id,
                segment_index=1,
                start_seconds=2.5,
                end_seconds=3.5,
                text="hi",
                words=[
                    TranscriptWord(word_index=0, word="hi", start_seconds=2.5, end_seconds=3.5),
                ],
            ),
        ]
    )
    await db.commit()
    return upload


async def _diarize_job(db: AsyncSession, upload: Upload) -> Job:
    job = Job(upload_id=upload.id, tool_type="diarize")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _labels(db: AsyncSession, upload_id: int) -> list[tuple[str | None, list[str | None]]]:
    db.expire_all()
    segments = (
        await db.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.upload_id == upload_id)
            .order_by(TranscriptSegment.segment_index)
            .options(selectinload(TranscriptSegment.words))
        )
    ).scalars()
    return [(s.speaker_label, [w.speaker_label for w in s.words]) for s in segments]


async def test_labels_words_and_segments(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    fake_diarize: list[Path],
) -> None:
    upload = await _transcribed_upload(db_session)
    upload_id = upload.id
    job = await _diarize_job(db_session, upload)

    await process_job({"session_factory": session_factory}, job.id)

    await db_session.refresh(job)
    assert job.status == "succeeded", job.error_message
    assert len(fake_diarize) == 1
    assert await _labels(db_session, upload_id) == [
        # "friend" (1.9-2.3) overlaps SPEAKER_01 more; "€" has no timing.
        # Majority of the segment is still SPEAKER_00.
        ("SPEAKER_00", ["SPEAKER_00", "SPEAKER_00", None, "SPEAKER_01"]),
        ("SPEAKER_01", ["SPEAKER_01"]),
    ]


async def test_rerun_overwrites_previous_labels(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = await _transcribed_upload(db_session)
    upload_id = upload.id

    monkeypatch.setattr(pipeline, "diarize", lambda _p: FAKE_TURNS)
    await process_job({"session_factory": session_factory}, (await _diarize_job(db_session, upload)).id)

    # Second run: one speaker for the whole clip.
    monkeypatch.setattr(pipeline, "diarize", lambda _p: [Turn(0.0, 4.0, "SPEAKER_00")])
    await process_job({"session_factory": session_factory}, (await _diarize_job(db_session, upload)).id)

    assert await _labels(db_session, upload_id) == [
        ("SPEAKER_00", ["SPEAKER_00", "SPEAKER_00", None, "SPEAKER_00"]),
        ("SPEAKER_00", ["SPEAKER_00"]),
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"transcribe_status": None}, "run Transcript first"),
        ({"transcribe_status": "failed"}, "run Transcript first"),
        ({"with_audio": False}, "run Transcript first"),
    ],
)
async def test_fails_cleanly_without_a_transcript(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    fake_diarize: list[Path],
    kwargs: dict,
    message: str,
) -> None:
    upload = await _transcribed_upload(db_session, **kwargs)
    job = await _diarize_job(db_session, upload)

    with pytest.raises(Exception, match=message):
        await process_job({"session_factory": session_factory}, job.id)

    await db_session.refresh(job)
    assert job.status == "failed"
    assert message in job.error_message
    assert fake_diarize == []  # never reached the model


async def test_fails_when_no_segments_exist(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    fake_diarize: list[Path],
) -> None:
    upload = Upload(filename="ep.mp4", storage_path="k", audio_storage_path="a.wav")
    db_session.add(upload)
    await db_session.commit()
    db_session.add(Job(upload_id=upload.id, tool_type="transcribe", status="succeeded"))
    await db_session.commit()
    job = await _diarize_job(db_session, upload)

    with pytest.raises(Exception, match="run Transcript first"):
        await process_job({"session_factory": session_factory}, job.id)
    assert fake_diarize == []


async def test_audio_file_missing_from_storage(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    fake_diarize: list[Path],
) -> None:
    upload = await _transcribed_upload(db_session)
    storage.delete(upload.audio_storage_path)
    job = await _diarize_job(db_session, upload)

    with pytest.raises(Exception, match="Extracted audio is missing"):
        await process_job({"session_factory": session_factory}, job.id)
    assert fake_diarize == []


async def test_fails_fast_when_the_healthcheck_failed(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    fake_diarize: list[Path],
) -> None:
    pipeline.health.checked = True
    pipeline.health.error = "DiarizationUnavailableError: 0 speaker(s)"
    upload = await _transcribed_upload(db_session)
    upload_id = upload.id
    job = await _diarize_job(db_session, upload)

    with pytest.raises(Exception, match="Diarization unavailable"):
        await process_job({"session_factory": session_factory}, job.id)

    await db_session.refresh(job)
    assert job.status == "failed"
    assert job.error_message == "Diarization unavailable: DiarizationUnavailableError: 0 speaker(s)"
    assert fake_diarize == []
    assert await _labels(db_session, upload_id) == [
        (None, [None, None, None, None]),
        (None, [None]),
    ]
