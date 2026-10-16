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
from arq import Retry
from sqlalchemy.exc import OperationalError

from app.worker.tasks import MAX_TRIES, fail_interrupted_jobs, is_transient, process_job

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


# ---- Retry policy (M9, docs/decisions/0014) ----


@pytest.mark.parametrize(
    "exc,transient",
    [
        (OperationalError("SELECT 1", {}, Exception("server closed the connection")), True),
        (ConnectionError("redis down"), True),
        (RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"), True),
        (RuntimeError("CUDA failed with error out of memory"), True),  # ctranslate2, season-1 run
        (ValueError("Failed to create llama_context"), True),
        (RuntimeError("No relevance scores for this upload -- run Relevance first"), False),
        (ValueError("bad input"), False),
    ],
)
def test_transient_failures_are_recognised(exc: Exception, transient: bool) -> None:
    assert is_transient(exc) is transient


async def test_transient_failure_requeues_and_asks_arq_to_retry(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    job = await _create_job(db_session)

    async def _oom(_job: Job, _db: AsyncSession) -> None:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setitem(handlers.HANDLERS, job.tool_type, _oom)
    with pytest.raises(Retry):
        await process_job({"session_factory": session_factory, "job_try": 1}, job.id)
    await db_session.refresh(job)
    assert job.status == "queued" and job.error_message.startswith("Retrying (attempt 1/")

    # The last attempt fails for good.
    with pytest.raises(RuntimeError):
        await process_job({"session_factory": session_factory, "job_try": MAX_TRIES}, job.id)
    await db_session.refresh(job)
    assert job.status == "failed" and job.error_message == "CUDA out of memory"


async def test_a_retry_that_succeeds_clears_the_error(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    job = await _create_job(db_session)
    calls = []

    async def _flaky(_job: Job, _db: AsyncSession) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("redis down")

    monkeypatch.setitem(handlers.HANDLERS, job.tool_type, _flaky)
    with pytest.raises(Retry):
        await process_job({"session_factory": session_factory, "job_try": 1}, job.id)
    await process_job({"session_factory": session_factory, "job_try": 2}, job.id)
    await db_session.refresh(job)
    assert job.status == "succeeded" and job.error_message is None


async def test_non_transient_failures_are_not_retried(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    job = await _create_job(db_session)

    async def _precondition(_job: Job, _db: AsyncSession) -> None:
        raise LookupError("run Transcript first")

    monkeypatch.setitem(handlers.HANDLERS, job.tool_type, _precondition)
    with pytest.raises(LookupError):
        await process_job({"session_factory": session_factory, "job_try": 1}, job.id)
    await db_session.refresh(job)
    assert job.status == "failed"


async def test_startup_fails_jobs_left_running(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    running = await _create_job(db_session)
    running.status = "running"
    queued = await _create_job(db_session)
    await db_session.commit()

    assert await fail_interrupted_jobs(session_factory) == 1
    await db_session.refresh(running)
    await db_session.refresh(queued)
    assert running.status == "failed" and "Interrupted" in running.error_message
    assert queued.status == "queued"


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

