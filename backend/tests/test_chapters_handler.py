"""`chapters` handler with the model pipeline mocked out, driven through
`process_job` so the job-state transitions are covered too."""

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chapters import pipeline
from app.chapters.pipeline import ChapterResult
from app.chapters.units import Atom
from app.models.chapter import Chapter
from app.models.job import Job
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.worker.handlers.chapters import TranscriptMissingError, atoms_from_segments
from app.worker.tasks import process_job


@pytest.fixture
def fake_generate(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[Atom], float]]:
    calls: list[tuple[list[Atom], float]] = []

    def _generate(atoms: list[Atom], duration: float) -> tuple[str, list[ChapterResult]]:
        calls.append((atoms, duration))
        return "fake-v1", [ChapterResult(0.0, "Intro"), ChapterResult(600.0, "Main topic")]

    monkeypatch.setattr(pipeline, "generate_chapters", _generate)
    return calls


async def _transcribed_upload(db: AsyncSession, *, transcribe_status: str | None = "succeeded") -> Upload:
    upload = Upload(filename="ep.mp4", storage_path="fake-key", duration_seconds=1200.0)
    db.add(upload)
    await db.commit()
    if transcribe_status is not None:
        db.add(Job(upload_id=upload.id, tool_type="transcribe", status=transcribe_status))
        db.add_all(
            [
                TranscriptSegment(
                    upload_id=upload.id, segment_index=0, start_seconds=0.0, end_seconds=2.0, text="hello there",
                    words=[
                        TranscriptWord(word_index=0, word="hello", start_seconds=0.0, end_seconds=0.5),
                        TranscriptWord(word_index=1, word="€"),
                    ],
                ),
                TranscriptSegment(upload_id=upload.id, segment_index=1, start_seconds=2.0, end_seconds=4.0, text="no words"),
            ]
        )
    await db.commit()
    return upload


async def _run_job(
    db: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    upload: Upload,
    raises: type[Exception] | None = None,
) -> Job:
    job = Job(upload_id=upload.id, tool_type="chapters")
    db.add(job)
    await db.commit()
    if raises is None:
        await process_job({"session_factory": session_factory}, job.id)
    else:
        # process_job marks the job failed, then re-raises for arq to log.
        with pytest.raises(raises):
            await process_job({"session_factory": session_factory}, job.id)
    await db.refresh(job)
    return job


async def test_persists_chapters_with_model_version(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_generate
) -> None:
    upload = await _transcribed_upload(db_session)
    job = await _run_job(db_session, session_factory, upload)

    assert job.status == "succeeded", job.error_message
    rows = (await db_session.execute(select(Chapter).order_by(Chapter.order_index))).scalars().all()
    assert [(r.order_index, r.start_seconds, r.title, r.model_version) for r in rows] == [
        (0, 0.0, "Intro", "fake-v1"),
        (1, 600.0, "Main topic", "fake-v1"),
    ]
    (atoms, duration), = fake_generate
    assert duration == 1200.0
    assert [(a.start, a.text) for a in atoms] == [(0.0, "hello"), (None, "€"), (2.0, "no words")]


async def test_rerun_replaces_rows(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_generate
) -> None:
    upload = await _transcribed_upload(db_session)
    await _run_job(db_session, session_factory, upload)
    await _run_job(db_session, session_factory, upload)
    count = len((await db_session.execute(select(Chapter))).scalars().all())
    assert count == 2


@pytest.mark.parametrize("transcribe_status", [None, "failed"])
async def test_fails_without_transcript(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_generate, transcribe_status
) -> None:
    upload = await _transcribed_upload(db_session, transcribe_status=transcribe_status)
    job = await _run_job(db_session, session_factory, upload, raises=TranscriptMissingError)
    assert job.status == "failed"
    assert "run Transcript first" in job.error_message
    assert fake_generate == []


async def test_fails_fast_without_bundle(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pipeline, "_chapterer", None)
    monkeypatch.setattr(pipeline.get_settings(), "chapters_model_dir", tmp_path / "nope")
    upload = await _transcribed_upload(db_session)
    job = await _run_job(db_session, session_factory, upload, raises=pipeline.ChaptersUnavailableError)
    assert job.status == "failed"
    assert "CHAPTERS_MODEL_DIR" in job.error_message


def test_atoms_from_segments_uses_segment_when_no_words() -> None:
    seg = TranscriptSegment(segment_index=0, start_seconds=1.0, end_seconds=2.0, text="whole", words=[])
    assert atoms_from_segments([seg]) == [Atom(1.0, 2.0, "whole")]
