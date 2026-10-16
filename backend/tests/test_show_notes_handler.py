"""`shownotes` handler with the generator mocked out, driven through
`process_job` so the job-state transitions are covered too (0011 §5)."""

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.chapter import Chapter
from app.models.job import Job
from app.models.show_notes import ShowNotesDraft
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.shownotes import pipeline
from app.shownotes.chunking import ChapterMark, Seg
from app.shownotes.pipeline import ShowNotes
from app.shownotes.validate import Draft
from app.worker.handlers.shownotes import TranscriptMissingError
from app.worker.tasks import process_job


@pytest.fixture
def fake_generate(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[Seg], list[ChapterMark]]]:
    calls: list[tuple[list[Seg], list[ChapterMark]]] = []

    def _generate(segs: list[Seg], chapters: list[ChapterMark]) -> tuple[str, ShowNotes]:
        calls.append((segs, chapters))
        n = len(calls)
        draft = Draft([f"Title {n}", "Other", "Third"], f"Description {n}", ["a", "b", "c", "d", "e"], "Summary.")
        return "fake-model+shownotes-v1", ShowNotes(draft, "es")

    monkeypatch.setattr(pipeline, "generate_show_notes", _generate)
    return calls


async def _transcribed_upload(db: AsyncSession, *, transcribe_status: str | None = "succeeded") -> Upload:
    upload = Upload(filename="ep.mp4", storage_path="fake-key", duration_seconds=1200.0)
    db.add(upload)
    await db.commit()
    if transcribe_status is not None:
        db.add(Job(upload_id=upload.id, tool_type="transcribe", status=transcribe_status))
        db.add_all([
            TranscriptSegment(upload_id=upload.id, segment_index=1, start_seconds=2.0, end_seconds=4.0, text="second"),
            TranscriptSegment(upload_id=upload.id, segment_index=0, start_seconds=0.0, end_seconds=2.0, text="first"),
        ])
    await db.commit()
    return upload


async def _run_job(
    db: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    upload: Upload,
    raises: type[Exception] | None = None,
) -> Job:
    job = Job(upload_id=upload.id, tool_type="shownotes")
    db.add(job)
    await db.commit()
    if raises is None:
        await process_job({"session_factory": session_factory}, job.id)
    else:
        with pytest.raises(raises):
            await process_job({"session_factory": session_factory}, job.id)
    await db.refresh(job)
    return job


async def test_persists_the_draft(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_generate
) -> None:
    upload = await _transcribed_upload(db_session)
    job = await _run_job(db_session, session_factory, upload)

    assert job.status == "succeeded", job.error_message
    row = (await db_session.execute(select(ShowNotesDraft))).scalar_one()
    assert row.upload_id == upload.id
    assert row.title_suggestions == ["Title 1", "Other", "Third"]
    assert row.description == "Description 1"
    assert row.tags == ["a", "b", "c", "d", "e"]
    assert row.summary == "Summary." and row.language == "es"
    assert row.llm_model_used == "fake-model+shownotes-v1"
    (segs, chapters), = fake_generate
    assert [s.text for s in segs] == ["first", "second"] and chapters == []


async def test_rerun_replaces_the_draft(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_generate
) -> None:
    upload = await _transcribed_upload(db_session)
    await _run_job(db_session, session_factory, upload)
    await _run_job(db_session, session_factory, upload)
    rows = (await db_session.execute(select(ShowNotesDraft))).scalars().all()
    assert len(rows) == 1 and rows[0].description == "Description 2"


async def test_uses_chapters_when_present(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_generate
) -> None:
    upload = await _transcribed_upload(db_session)
    db_session.add_all([
        Chapter(upload_id=upload.id, order_index=1, start_seconds=600.0, title="Main", model_version="v"),
        Chapter(upload_id=upload.id, order_index=0, start_seconds=0.0, title="Intro", model_version="v"),
    ])
    await db_session.commit()
    await _run_job(db_session, session_factory, upload)
    (_, chapters), = fake_generate
    assert chapters == [ChapterMark(0.0, "Intro"), ChapterMark(600.0, "Main")]


@pytest.mark.parametrize("transcribe_status", [None, "failed"])
async def test_fails_without_transcript(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_generate, transcribe_status
) -> None:
    upload = await _transcribed_upload(db_session, transcribe_status=transcribe_status)
    job = await _run_job(db_session, session_factory, upload, raises=TranscriptMissingError)
    assert job.status == "failed"
    assert "run Transcript first" in job.error_message
    assert fake_generate == []


async def test_failed_generation_keeps_the_previous_draft(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_generate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = await _transcribed_upload(db_session)
    await _run_job(db_session, session_factory, upload)

    def _fail(segs, chapters):
        raise pipeline.ShowNotesGenerationError("Show notes: model returned no valid draft after 3 attempts")

    monkeypatch.setattr(pipeline, "generate_show_notes", _fail)
    job = await _run_job(db_session, session_factory, upload, raises=pipeline.ShowNotesGenerationError)
    assert job.status == "failed" and "after 3 attempts" in job.error_message
    row = (await db_session.execute(select(ShowNotesDraft))).scalar_one()
    assert row.description == "Description 1"


async def test_fails_fast_without_bundle(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pipeline, "_generator", None)
    monkeypatch.setattr(pipeline.get_settings(), "chapters_model_dir", tmp_path / "nope")
    upload = await _transcribed_upload(db_session)
    job = await _run_job(db_session, session_factory, upload, raises=pipeline.ShowNotesUnavailableError)
    assert job.status == "failed"
    assert "CHAPTERS_MODEL_DIR" in job.error_message
