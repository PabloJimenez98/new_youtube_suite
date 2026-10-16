"""`shorts` handler driven through `process_job` (docs/decisions/0013 §7).

Most tests replace `make_clip` with a fake that writes a stub file; one
renders for real on a synthetic video (skips without the YuNet model).
"""

import subprocess
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.models.clip import Clip
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.relevance import RelevanceScore
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.shorts import render
from app.shorts.framing import FaceModelMissingError, Plan, Shot
from app.storage import LocalStorageBackend
from app.worker.handlers import shorts
from app.worker.handlers.shorts import NoClipsError, RelevanceMissingError
from app.worker.tasks import process_job

storage = LocalStorageBackend(get_settings().storage_base_path)
MODEL = get_settings().shorts_face_model_path
has_model = bool(MODEL) and Path(MODEL).is_file()
DURATION = 120.0


@pytest.fixture
def fake_make_clip(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _make(src, span, words, params, model, out):
        calls.append({"span": span, "params": params, "words": words, "src": src})
        Path(out).write_bytes(b"fake mp4")
        return Plan(1280, 720, 404, 720, 1280, [Shot(0, span.duration, "crop")],
                    [{"t": 0.0, "x": 10, "y": 0, "w": 404, "h": 720}])

    monkeypatch.setattr(shorts, "make_clip", _make)
    return calls


@pytest.fixture
def face_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A model file that exists, for the fail-fast check (the fake make_clip never loads it)."""
    if not has_model:
        stub = tmp_path / "yunet.onnx"
        stub.write_bytes(b"stub")
        monkeypatch.setattr(get_settings(), "shorts_face_model_path", stub)


async def _scored_upload(db: AsyncSession, *, peaks: tuple[int, ...] = (3, 12), video_key: str = "fake-key",
                         with_scores: bool = True) -> Upload:
    """120 s of words (a sentence every 5 s) and a relevance curve with bumps at the given windows."""
    upload = Upload(filename="ep.mp4", storage_path=video_key, duration_seconds=DURATION)
    db.add(upload)
    await db.commit()
    db.add(Job(upload_id=upload.id, tool_type="transcribe", status="succeeded"))
    for s in range(int(DURATION / 5)):
        db.add(TranscriptSegment(
            upload_id=upload.id, segment_index=s, start_seconds=s * 5.0, end_seconds=s * 5.0 + 5, text="x",
            words=[TranscriptWord(word_index=k, word=f"w{k}." if k == 9 else f"w{k}",
                                  start_seconds=s * 5.0 + k * 0.5, end_seconds=s * 5.0 + k * 0.5 + 0.4)
                   for k in range(10)],
        ))
    if with_scores:
        version = ModelVersion(model_name="relevance", version_label="fake-v0", is_live=True)
        db.add(version)
        await db.flush()
        n = int((DURATION - 15) / 7.5) + 1
        db.add_all(
            RelevanceScore(upload_id=upload.id, window_index=i, window_start_seconds=i * 7.5,
                           window_end_seconds=i * 7.5 + 15, score=0.9 if any(abs(i - p) <= 1 for p in peaks) else 0.3,
                           heuristic_score=0.3, model_version_id=version.id)
            for i in range(n)
        )
        db.add(Job(upload_id=upload.id, tool_type="relevance", status="succeeded"))
    await db.commit()
    return upload


async def _run(db, session_factory, upload, params=None, raises=None) -> Job:
    job = Job(upload_id=upload.id, tool_type="shorts", params=params)
    db.add(job)
    await db.commit()
    if raises is None:
        await process_job({"session_factory": session_factory}, job.id)
    else:
        with pytest.raises(raises):
            await process_job({"session_factory": session_factory}, job.id)
    await db.refresh(job)
    return job


async def _clips(db: AsyncSession, upload: Upload) -> list[Clip]:
    query = select(Clip).where(Clip.upload_id == upload.id).order_by(Clip.rank)
    return list((await db.execute(query.execution_options(populate_existing=True))).scalars())


async def test_persists_ranked_clips_with_files_and_version(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_make_clip, face_model
) -> None:
    upload = await _scored_upload(db_session)
    job = await _run(db_session, session_factory, upload, params={"aspect_ratio": "1:1", "captions": False})
    assert job.status == "succeeded" and job.progress_pct == 100

    clips = await _clips(db_session, upload)
    assert [c.rank for c in clips] == [1, 2]
    version = await db_session.scalar(select(ModelVersion.id))
    for c in clips:
        assert 20 <= c.end_seconds - c.start_seconds <= 60
        assert c.aspect_ratio == "1:1" and c.caption_burned_in is False
        assert c.model_version_id == version
        assert c.crop_bounding_box_track[0]["w"] == 404 and c.framing["shots"][0]["mode"] == "crop"
        assert storage.path_for(c.storage_path).read_bytes() == b"fake mp4"
    assert fake_make_clip[0]["params"].aspect_ratio == "1:1"
    assert fake_make_clip[0]["words"][0].segment_start is True


async def test_defaults_when_the_job_has_no_params(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_make_clip, face_model
) -> None:
    upload = await _scored_upload(db_session, peaks=(8,))
    await _run(db_session, session_factory, upload)
    (clip,) = await _clips(db_session, upload)
    assert clip.aspect_ratio == "9:16" and clip.caption_burned_in is True


async def test_rerun_replaces_rows_and_files(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_make_clip, face_model
) -> None:
    upload = await _scored_upload(db_session)
    await _run(db_session, session_factory, upload)
    old = [c.storage_path for c in await _clips(db_session, upload)]
    await _run(db_session, session_factory, upload)
    new = [c.storage_path for c in await _clips(db_session, upload)]
    assert len(new) == 2 and not set(old) & set(new)
    assert all(not storage.path_for(k).exists() for k in old)
    assert all(storage.path_for(k).exists() for k in new)


async def test_a_failed_render_keeps_the_previous_clips(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_make_clip, face_model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = await _scored_upload(db_session)
    await _run(db_session, session_factory, upload)
    before = [c.storage_path for c in await _clips(db_session, upload)]
    files_before = set(storage.base_path.iterdir())

    def _boom(src, span, words, params, model, out):
        Path(out).write_bytes(b"partial")
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(shorts, "make_clip", _boom)
    job = await _run(db_session, session_factory, upload, raises=RuntimeError)
    assert job.status == "failed" and "ffmpeg exploded" in job.error_message
    assert [c.storage_path for c in await _clips(db_session, upload)] == before
    assert all(storage.path_for(k).exists() for k in before)
    assert set(storage.base_path.iterdir()) == files_before  # the partial file was removed


async def test_needs_relevance_first(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_make_clip, face_model
) -> None:
    upload = await _scored_upload(db_session, with_scores=False)
    job = await _run(db_session, session_factory, upload, raises=RelevanceMissingError)
    assert "run Relevance first" in job.error_message
    assert fake_make_clip == []


async def test_a_flat_curve_fails_with_a_clear_message(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_make_clip, face_model
) -> None:
    upload = await _scored_upload(db_session, peaks=())
    job = await _run(db_session, session_factory, upload, raises=NoClipsError)
    assert "no clear peaks" in job.error_message


async def test_missing_face_model_fails_fast(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_make_clip,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(get_settings(), "shorts_face_model_path", tmp_path / "missing.onnx")
    upload = await _scored_upload(db_session)
    job = await _run(db_session, session_factory, upload, raises=FaceModelMissingError)
    assert "SHORTS_FACE_MODEL_PATH" in job.error_message and fake_make_clip == []


@pytest.mark.skipif(not has_model, reason="YuNet model not downloaded")
async def test_real_render_on_a_synthetic_episode(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    src = tmp_path / "ep.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=25:duration={DURATION}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={DURATION}", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)],
        check=True,
    )
    key = storage.save("synthetic_ep.mp4", src.read_bytes())
    upload = await _scored_upload(db_session, peaks=(8,), video_key=key)
    job = await _run(db_session, session_factory, upload)
    assert job.status == "succeeded"
    (clip,) = await _clips(db_session, upload)
    info = render.probe(storage.path_for(clip.storage_path))
    assert (info["width"], info["height"]) == (360, 640)  # testsrc has no face: fit, 9:16 at the source's short side
    assert info["duration"] == pytest.approx(clip.end_seconds - clip.start_seconds, abs=0.5)
    assert clip.framing["shots"] and all(s["mode"] == "fit" for s in clip.framing["shots"])
