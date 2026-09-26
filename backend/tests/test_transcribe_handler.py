"""`transcribe` handler with Whisper mocked out but real ffmpeg, driven
through `process_job` so the job-state transitions are covered too."""

import tempfile
import wave
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.models.job import Job
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.storage import LocalStorageBackend
from app.transcription import whisper
from app.transcription.whisper import SegmentResult, WordResult
from app.worker.handlers.transcribe import audio_key_for
from app.worker.tasks import process_job
from tests.fixtures import FIXTURE_CLIP

storage = LocalStorageBackend(get_settings().storage_base_path)


def _fake_results(label: str) -> list[SegmentResult]:
    return [
        SegmentResult(
            start=0.0,
            end=1.5,
            text=f"{label} one",
            avg_logprob=-0.2,
            no_speech_prob=0.01,
            words=[
                WordResult(label, 0.0, 0.7, 0.9),
                WordResult("one", 0.7, 1.5, 0.8),
            ],
        ),
        SegmentResult(
            start=1.5,
            end=3.0,
            text=f"{label} two",
            avg_logprob=None,
            no_speech_prob=None,
            words=[WordResult(label, 1.5, 2.0, None), WordResult("€", None, None, 0.1)],
        ),
    ]


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the model call; record the WAV paths it was given and
    check they are what ffmpeg should produce (16kHz mono 16-bit)."""
    seen: list[Path] = []

    def _transcribe(audio_path: Path) -> list[SegmentResult]:
        with wave.open(str(audio_path)) as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert 17 < wav.getnframes() / 16000 < 20  # fixture is 18.3s
        seen.append(audio_path)
        return _fake_results(f"run{len(seen)}")

    monkeypatch.setattr(whisper, "transcribe", _transcribe)
    return seen


@pytest.fixture
def isolated_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    return temp_root


async def _job_for(db: AsyncSession, video: bytes) -> Job:
    key = LocalStorageBackend.make_key("clip.mp4")
    storage.save(key, video)
    upload = Upload(filename="clip.mp4", storage_path=key)
    db.add(upload)
    await db.commit()
    await db.refresh(upload)
    job = Job(upload_id=upload.id, tool_type="transcribe")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _segments(db: AsyncSession, upload_id: int) -> list[TranscriptSegment]:
    db.expire_all()
    result = await db.execute(
        select(TranscriptSegment)
        .where(TranscriptSegment.upload_id == upload_id)
        .order_by(TranscriptSegment.segment_index)
    )
    return list(result.scalars().all())


async def _words(db: AsyncSession, segment_id: int) -> list[TranscriptWord]:
    result = await db.execute(
        select(TranscriptWord)
        .where(TranscriptWord.segment_id == segment_id)
        .order_by(TranscriptWord.word_index)
    )
    return list(result.scalars().all())


async def test_persists_segments_words_and_audio(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    fake_whisper: list[Path],
    isolated_tempdir: Path,
) -> None:
    job = await _job_for(db_session, FIXTURE_CLIP.read_bytes())

    await process_job({"session_factory": session_factory}, job.id)

    await db_session.refresh(job)
    assert job.status == "succeeded", job.error_message
    upload_id = job.upload_id  # read before `_segments` expires `job`

    segments = await _segments(db_session, upload_id)
    assert [s.segment_index for s in segments] == [0, 1]
    assert [s.text for s in segments] == ["run1 one", "run1 two"]
    assert (segments[0].start_seconds, segments[0].end_seconds) == (0.0, 1.5)
    assert segments[0].avg_logprob == -0.2
    assert segments[1].avg_logprob is None
    assert all(s.speaker_label is None for s in segments)  # M4's job

    words = await _words(db_session, segments[1].id)
    assert [(w.word_index, w.word) for w in words] == [(0, "run1"), (1, "€")]
    assert words[1].start_seconds is None and words[1].end_seconds is None

    upload = await db_session.get(Upload, upload_id)
    await db_session.refresh(upload)
    assert upload.audio_storage_path == audio_key_for(upload_id)
    with storage.get(upload.audio_storage_path) as stored:
        assert stored.read(4) == b"RIFF"

    # The WAV was transcribed from a temp dir that's gone now.
    assert not fake_whisper[0].exists()
    assert list(isolated_tempdir.iterdir()) == []


async def test_rerun_replaces_instead_of_duplicating(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    fake_whisper: list[Path],
) -> None:
    job = await _job_for(db_session, FIXTURE_CLIP.read_bytes())
    await process_job({"session_factory": session_factory}, job.id)

    rerun = Job(upload_id=job.upload_id, tool_type="transcribe")
    db_session.add(rerun)
    await db_session.commit()
    await process_job({"session_factory": session_factory}, rerun.id)

    segments = await _segments(db_session, job.upload_id)
    assert [s.text for s in segments] == ["run2 one", "run2 two"]
    assert [len(await _words(db_session, s.id)) for s in segments] == [2, 2]


async def test_corrupt_video_fails_with_ffmpeg_error(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    fake_whisper: list[Path],
    isolated_tempdir: Path,
) -> None:
    job = await _job_for(db_session, b"definitely not a video")

    with pytest.raises(Exception, match="ffmpeg exited"):
        await process_job({"session_factory": session_factory}, job.id)

    await db_session.refresh(job)
    assert job.status == "failed"
    assert job.error_message.startswith("ffmpeg exited with")
    assert fake_whisper == []  # never got as far as the model
    assert await _segments(db_session, job.upload_id) == []
    assert list(isolated_tempdir.iterdir()) == []


async def test_delete_upload_removes_extracted_audio(
    client: AsyncClient,
    auth_headers: dict[str, str],
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    fake_whisper: list[Path],
) -> None:
    job = await _job_for(db_session, FIXTURE_CLIP.read_bytes())
    await process_job({"session_factory": session_factory}, job.id)
    audio_path = storage.path_for(audio_key_for(job.upload_id))
    assert audio_path.is_file()

    response = await client.delete(f"/uploads/{job.upload_id}", headers=auth_headers)

    assert response.status_code == 204
    assert not audio_path.exists()
    # Transcript rows went with the upload (ON DELETE CASCADE).
    assert await _segments(db_session, job.upload_id) == []
