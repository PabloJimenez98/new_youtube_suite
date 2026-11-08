"""M10 worker wiring: both cron tasks registered on the existing worker
with the 0015 §4 schedule, in UTC; no API key -> logged skip, no raise."""

import logging
from datetime import timezone

import pytest

from app.config import get_settings
from app.market import tasks
from app.worker.settings import WorkerSettings


def test_cron_jobs_registered_with_expected_schedule() -> None:
    jobs = {c.name: c for c in WorkerSettings.cron_jobs}
    # + M11's analytics pull (docs/decisions/0017 §5), tested in test_analytics_pull.
    assert set(jobs) == {
        "cron:market_rss_poll",
        "cron:market_daily_pull",
        "cron:analytics_daily_pull",
        "cron:reporting_daily_poll",
    }
    rss, daily = jobs["cron:market_rss_poll"], jobs["cron:market_daily_pull"]
    assert rss.minute == 5 and rss.hour is None
    assert daily.hour == 9 and daily.minute == 30
    assert rss.unique and daily.unique
    assert WorkerSettings.timezone is timezone.utc
    # The tool-job function is still there.
    assert [f.__name__ for f in WorkerSettings.functions] == ["process_job"]


@pytest.mark.parametrize("task", [tasks.market_daily_pull, tasks.market_rss_poll])
async def test_missing_api_key_logs_skip_and_returns(
    task, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    no_key = get_settings().model_copy(update={"youtube_api_key": None})
    monkeypatch.setattr(tasks, "get_settings", lambda: no_key)
    with caplog.at_level(logging.INFO, logger="app.market"):
        assert await task({}) is None
    assert "YOUTUBE_API_KEY not set" in caplog.text


async def test_worker_with_market_crons_still_processes_tool_jobs(
    client, auth_headers, session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a real arq worker built with the M10 cron jobs and UTC
    timezone still drains a queued tool job (the M2 checkpoint)."""
    from arq.connections import RedisSettings
    from arq.worker import Worker

    from app.worker.tasks import process_job

    monkeypatch.setattr(get_settings(), "dummy_job_duration_seconds", 0.01)
    upload = await client.post(
        "/uploads", headers=auth_headers, files={"file": ("clip.mp4", b"fake bytes", "video/mp4")}
    )
    job = await client.post(
        f"/uploads/{upload.json()['id']}/jobs", headers=auth_headers, json={"tool_type": "hook"}
    )

    async def _on_startup(ctx: dict) -> None:
        ctx["session_factory"] = session_factory

    worker = Worker(
        functions=[process_job],
        cron_jobs=WorkerSettings.cron_jobs,
        timezone=WorkerSettings.timezone,
        redis_settings=RedisSettings.from_dsn(get_settings().test_redis_url),
        on_startup=_on_startup,
        burst=True,
        max_tries=1,
        handle_signals=False,
    )
    try:
        await worker.async_run()
    finally:
        await worker.close()

    jobs = await client.get(f"/uploads/{upload.json()['id']}/jobs", headers=auth_headers)
    assert next(j for j in jobs.json() if j["id"] == job.json()["id"])["status"] == "succeeded"


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_api_key_counts_as_unset(raw: str) -> None:
    from app.config import Settings

    assert Settings(_env_file=None, youtube_api_key=raw).youtube_api_key is None
