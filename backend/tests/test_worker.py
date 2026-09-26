import pytest
from arq.connections import RedisSettings
from arq.worker import Worker
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.models.job import Job
from app.models.upload import Upload
from app.worker import handlers
from app.worker.tasks import process_job

settings = get_settings()


async def _create_job(db_session: AsyncSession, tool_type: str = "transcribe") -> Job:
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

    async def _boom(_job: Job) -> None:
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
        json={"tool_type": "transcribe"},
    )
    job_id = job_response.json()["id"]
    assert job_response.json()["status"] == "queued"

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

    jobs_response = await client.get(f"/uploads/{upload_id}/jobs", headers=auth_headers)
    job = next(j for j in jobs_response.json() if j["id"] == job_id)
    assert job["status"] == "succeeded"
    assert job["started_at"] is not None
    assert job["completed_at"] is not None
