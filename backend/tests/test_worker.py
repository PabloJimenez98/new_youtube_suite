import pytest
from arq.connections import RedisSettings
from arq.worker import Worker
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from tests.fixtures import CONVERSATION_CLIP, FIXTURE_CLIP
from app.models.job import Job
from app.models.upload import Upload
from app.worker import handlers
from app.worker.tasks import process_job

settings = get_settings()


# `hook` still runs the M2 dummy handler; `transcribe` is real as of M3
# (and covered by its own real-worker test at the bottom of this file).
async def _create_job(db_session: AsyncSession, tool_type: str = "hook") -> Job:
    upload = Upload(filename="clip.mp4", storage_path="fake-key")
    db_session.add(upload)
    await db_session.commit()
    await db_session.refresh(upload)

    job = Job(upload_id=upload.id, tool_type=tool_type)
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    return job


# ---- Unit: the state-machine logic in `process_job`, no Redis/worker involved ----


async def test_process_job_success_transitions_to_succeeded(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    job = await _create_job(db_session)

    await process_job({"session_factory": session_factory}, job.id)

    await db_session.refresh(job)
    assert job.status == "succeeded"
    assert job.started_at is not None
    assert job.completed_at is not None
    assert job.error_message is None


async def test_process_job_failure_transitions_to_failed(
    db_session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = await _create_job(db_session)

    async def _boom(_job: Job, _db: AsyncSession) -> None:
        raise RuntimeError("handler exploded")

    monkeypatch.setitem(handlers.HANDLERS, job.tool_type, _boom)

    with pytest.raises(RuntimeError):
        await process_job({"session_factory": session_factory}, job.id)

    await db_session.refresh(job)
    assert job.status == "failed"
    assert job.completed_at is not None
    assert job.error_message == "handler exploded"


async def test_process_job_missing_job_is_a_noop(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # Upload/job deleted between enqueue and pickup -- shouldn't raise.
    await process_job({"session_factory": session_factory}, 999999)


# ---- Integration: real enqueue (via the HTTP API) + a real arq worker ----


async def _run_burst_worker(session_factory: async_sessionmaker[AsyncSession]) -> None:
    """Run a real arq `Worker` in burst mode: drain the queue, then return."""

    async def _on_startup(ctx: dict) -> None:
        ctx["session_factory"] = session_factory

    worker = Worker(
        functions=[process_job],
        redis_settings=RedisSettings.from_dsn(settings.test_redis_url),
        on_startup=_on_startup,
        burst=True,
        max_tries=1,
        handle_signals=False,
    )
    try:
        await worker.async_run()
    finally:
        await worker.close()


async def test_enqueue_and_real_worker_pickup(
    client: AsyncClient,
    auth_headers: dict[str, str],
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The milestone's actual checkpoint: upload -> create a job through
    the real endpoint (which enqueues into arq/Redis via the overridden
    `get_arq_pool` dependency) -> run a real arq `Worker` in burst mode
    (drains whatever's queued, then returns instead of running forever)
    -> assert the status transition landed in the DB, not just that
    `process_job` works when called directly.
    """
    monkeypatch.setattr(get_settings(), "dummy_job_duration_seconds", 0.01)

    upload_response = await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("clip.mp4", b"fake bytes", "video/mp4")},
    )
    upload_id = upload_response.json()["id"]

    job_response = await client.post(
        f"/uploads/{upload_id}/jobs",
        headers=auth_headers,
        json={"tool_type": "hook"},
    )
    job_id = job_response.json()["id"]
    assert job_response.json()["status"] == "queued"

    await _run_burst_worker(session_factory)

    jobs_response = await client.get(f"/uploads/{upload_id}/jobs", headers=auth_headers)
    job = next(j for j in jobs_response.json() if j["id"] == job_id)
    assert job["status"] == "succeeded"
    assert job["started_at"] is not None
    assert job["completed_at"] is not None


async def test_real_transcribe_job_through_real_worker(
    client: AsyncClient,
    auth_headers: dict[str, str],
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """M3 regression of the M2 checkpoint with a real (not dummy) job:
    upload the fixture clip through the API, enqueue `transcribe`, let a
    real arq worker run the real handler (ffmpeg + Whisper `base` on CPU,
    see conftest), and check the status transitions and persisted rows.
    """
    upload_response = await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("clip.mp4", FIXTURE_CLIP.read_bytes(), "video/mp4")},
    )
    upload_id = upload_response.json()["id"]

    job_response = await client.post(
        f"/uploads/{upload_id}/jobs",
        headers=auth_headers,
        json={"tool_type": "transcribe"},
    )
    job_id = job_response.json()["id"]
    assert job_response.json()["status"] == "queued"

    await _run_burst_worker(session_factory)

    jobs_response = await client.get(f"/uploads/{upload_id}/jobs", headers=auth_headers)
    job = next(j for j in jobs_response.json() if j["id"] == job_id)
    assert job["status"] == "succeeded", job["error_message"]
    assert job["started_at"] is not None
    assert job["completed_at"] is not None
    assert job["started_at"] <= job["completed_at"]

    srt = await client.get(f"/uploads/{upload_id}/transcript.srt", headers=auth_headers)
    assert srt.status_code == 200
    assert srt.text.startswith("1\n00:00:")


async def test_transcribe_then_diarize_through_real_worker(
    client: AsyncClient,
    auth_headers: dict[str, str],
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """M4 regression of the M2 checkpoint: two real jobs through a real
    arq worker, chained the way the Manage page drives them -- diarize
    only after transcribe has succeeded -- then the speaker labels show
    up in the VTT download.
    """
    upload_response = await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("conversation.mp4", CONVERSATION_CLIP.read_bytes(), "video/mp4")},
    )
    upload_id = upload_response.json()["id"]

    for tool_type in ("transcribe", "diarize"):
        job_response = await client.post(
            f"/uploads/{upload_id}/jobs", headers=auth_headers, json={"tool_type": tool_type}
        )
        job_id = job_response.json()["id"]
        await _run_burst_worker(session_factory)

        jobs_response = await client.get(f"/uploads/{upload_id}/jobs", headers=auth_headers)
        job = next(j for j in jobs_response.json() if j["id"] == job_id)
        assert job["status"] == "succeeded", f"{tool_type}: {job['error_message']}"

    vtt = await client.get(f"/uploads/{upload_id}/transcript.vtt", headers=auth_headers)
    assert vtt.status_code == 200
    assert "<v SPEAKER_00>" in vtt.text
    assert "<v SPEAKER_01>" in vtt.text

