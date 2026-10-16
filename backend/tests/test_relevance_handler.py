"""`relevance` handler with the scorer mocked out, driven through
`process_job`, plus `model_versions` rules (docs/decisions/0012 §5-6)."""

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.chapter import Chapter
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.relevance import RelevanceScore
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.relevance import pipeline
from app.relevance.pipeline import RelevanceResult, WindowScore
from app.relevance.windows import Window, Word
from app.worker.handlers.relevance import TranscriptMissingError, get_or_create_version, words_from_segments
from app.worker.tasks import process_job

DESCRIPTION = {"architecture_notes": "notes", "training_data_reference": {"kind": "untrained"}}


@pytest.fixture
def fake_score(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []

    def _score(words, duration, chapters):
        calls.append((words, duration, chapters))
        scores = [
            WindowScore(Window(0, 0.0, 15.0, ()), 0.5, 0.9, 0.74),
            WindowScore(Window(1, 7.5, 22.5, ()), 0.2, None, 0.2),
        ]
        return "fake-v0", DESCRIPTION, RelevanceResult("en", scores, 1, 0)

    monkeypatch.setattr(pipeline, "score_relevance", _score)
    return calls


async def _transcribed_upload(db: AsyncSession, *, transcribe_status: str | None = "succeeded") -> Upload:
    upload = Upload(filename="ep.mp4", storage_path="fake-key", duration_seconds=600.0)
    db.add(upload)
    await db.commit()
    if transcribe_status is not None:
        db.add(Job(upload_id=upload.id, tool_type="transcribe", status=transcribe_status))
        db.add_all([
            TranscriptSegment(
                upload_id=upload.id, segment_index=0, start_seconds=0.0, end_seconds=2.0, text="hello there",
                speaker_label="SPEAKER_00",
                words=[
                    TranscriptWord(word_index=0, word="hello", start_seconds=0.0, end_seconds=0.5, speaker_label="SPEAKER_01"),
                    TranscriptWord(word_index=1, word="there", start_seconds=0.6, end_seconds=1.0),
                ],
            ),
            TranscriptSegment(upload_id=upload.id, segment_index=1, start_seconds=2.0, end_seconds=4.0, text="no words"),
        ])
    await db.commit()
    return upload


async def _run_job(db, session_factory, upload, raises=None) -> Job:
    job = Job(upload_id=upload.id, tool_type="relevance")
    db.add(job)
    await db.commit()
    if raises is None:
        await process_job({"session_factory": session_factory}, job.id)
    else:
        with pytest.raises(raises):
            await process_job({"session_factory": session_factory}, job.id)
    await db.refresh(job)
    return job


async def test_persists_windows_with_a_live_model_version(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_score
) -> None:
    upload = await _transcribed_upload(db_session)
    job = await _run_job(db_session, session_factory, upload)

    assert job.status == "succeeded", job.error_message
    version = (await db_session.execute(select(ModelVersion))).scalar_one()
    assert (version.model_name, version.version_label, version.is_live) == ("relevance", "fake-v0", True)
    assert version.architecture_notes == "notes" and version.training_data_reference == {"kind": "untrained"}
    rows = (await db_session.execute(select(RelevanceScore).order_by(RelevanceScore.window_index))).scalars().all()
    assert [(r.window_index, r.window_start_seconds, r.window_end_seconds, r.score, r.heuristic_score, r.llm_score)
            for r in rows] == [(0, 0.0, 15.0, 0.74, 0.5, 0.9), (1, 7.5, 22.5, 0.2, 0.2, None)]
    assert {r.model_version_id for r in rows} == {version.id}


async def test_passes_words_speakers_and_chapters(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_score
) -> None:
    upload = await _transcribed_upload(db_session)
    db_session.add(Chapter(upload_id=upload.id, order_index=0, start_seconds=0.0, title="Intro", model_version="v"))
    await db_session.commit()
    await _run_job(db_session, session_factory, upload)
    (words, duration, chapters), = fake_score
    assert words == [Word(0.0, "hello", "SPEAKER_01"), Word(0.6, "there", "SPEAKER_00"), Word(2.0, "no words", None)]
    assert duration == 600.0 and chapters == [(0.0, "Intro")]


async def test_rerun_replaces_rows_and_reuses_the_version(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_score
) -> None:
    upload = await _transcribed_upload(db_session)
    await _run_job(db_session, session_factory, upload)
    await _run_job(db_session, session_factory, upload)
    assert len((await db_session.execute(select(RelevanceScore))).scalars().all()) == 2
    assert len((await db_session.execute(select(ModelVersion))).scalars().all()) == 1


@pytest.mark.parametrize("transcribe_status", [None, "failed"])
async def test_fails_without_transcript(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_score, transcribe_status
) -> None:
    upload = await _transcribed_upload(db_session, transcribe_status=transcribe_status)
    job = await _run_job(db_session, session_factory, upload, raises=TranscriptMissingError)
    assert job.status == "failed" and "run Transcript first" in job.error_message
    assert fake_score == []


async def test_fails_fast_without_bundle(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pipeline, "_scorer", None)
    monkeypatch.setattr(pipeline.get_settings(), "chapters_model_dir", tmp_path / "nope")
    upload = await _transcribed_upload(db_session)
    job = await _run_job(db_session, session_factory, upload, raises=pipeline.RelevanceUnavailableError)
    assert job.status == "failed" and "CHAPTERS_MODEL_DIR" in job.error_message


# --- model_versions ------------------------------------------------------------


async def test_a_new_label_is_not_live_when_another_version_is(db_session: AsyncSession) -> None:
    first = await get_or_create_version(db_session, "v0", DESCRIPTION)
    second = await get_or_create_version(db_session, "v0b", DESCRIPTION)
    again = await get_or_create_version(db_session, "v0", DESCRIPTION)
    await db_session.commit()
    assert first.is_live and not second.is_live and again.id == first.id


async def test_only_one_live_version_per_model(db_session: AsyncSession) -> None:
    db_session.add_all([
        ModelVersion(model_name="relevance", version_label="a", is_live=True),
        ModelVersion(model_name="ctr", version_label="a", is_live=True),  # another model: fine
    ])
    await db_session.commit()
    db_session.add(ModelVersion(model_name="relevance", version_label="b", is_live=True))
    with pytest.raises(IntegrityError):
        await db_session.commit()


def test_words_from_a_segment_without_words() -> None:
    seg = TranscriptSegment(segment_index=0, start_seconds=1.0, end_seconds=2.0, text="whole", words=[],
                            speaker_label="S")
    assert words_from_segments([seg]) == [Word(1.0, "whole", "S")]
