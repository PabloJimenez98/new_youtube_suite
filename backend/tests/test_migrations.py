"""Migration up/down smoke test.

Runs the real Alembic migrations (not `Base.metadata.create_all`)
against the test database, in a subprocess with DATABASE_URL pointed
at `settings.test_database_url`, and checks the expected tables
appear/disappear.
"""

import os
import subprocess
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings

BACKEND_DIR = Path(__file__).resolve().parent.parent
EXPECTED_TABLES = {
    "users",
    "uploads",
    "jobs",
    "transcript_segments",
    "transcript_words",
    "chapters",
    "show_notes_drafts",
    "model_versions",
    "relevance_scores",
    "clips",
    "market_channels",
    "market_videos",
    "market_video_snapshots",
    "oauth_credentials",
    "analytics_videos",
    "analytics_snapshots",
    "analytics_retention_curve",
    "reporting_jobs",
    "reporting_reports",
    "analytics_reach_snapshots",
    "ctr_predictions",
    "hook_scores",
}
ANALYTICS_TABLES = {"oauth_credentials", "analytics_videos", "analytics_snapshots", "analytics_retention_curve"}
MARKET_TABLES = {"market_channels", "market_videos", "market_video_snapshots"}
REPORTING_TABLES = {"reporting_jobs", "reporting_reports", "analytics_reach_snapshots"}


@pytest_asyncio.fixture(autouse=True)
async def _override_get_db() -> AsyncGenerator[None, None]:
    """Shadow conftest's autouse fixture (same name): this test manages
    the schema itself via real Alembic migrations, not
    `Base.metadata.create_all`, so nothing should pre-create tables.
    """
    yield


def _run_alembic(*args: str) -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = get_settings().test_database_url
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


async def _job_columns() -> set[str]:
    engine = create_async_engine(get_settings().test_database_url)
    async with engine.connect() as conn:
        names = await conn.run_sync(lambda c: {col["name"] for col in inspect(c).get_columns("jobs")})
    await engine.dispose()
    return names


async def _table_names() -> set[str]:
    engine = create_async_engine(get_settings().test_database_url)
    async with engine.connect() as conn:
        names = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
    await engine.dispose()
    return names


async def test_migration_up_then_down_creates_and_drops_tables() -> None:
    # Start from a clean slate in case a previous test left tables behind.
    _run_alembic("downgrade", "base")
    assert not (EXPECTED_TABLES & await _table_names())

    _run_alembic("upgrade", "head")
    assert EXPECTED_TABLES <= await _table_names()

    _run_alembic("downgrade", "base")
    assert not (EXPECTED_TABLES & await _table_names())


async def test_0009_adds_job_params_and_clips_and_reverts() -> None:
    _run_alembic("downgrade", "base")
    _run_alembic("upgrade", "head")
    assert "params" in await _job_columns() and "clips" in await _table_names()

    _run_alembic("downgrade", "0008")
    assert "params" not in await _job_columns() and "clips" not in await _table_names()
    assert "relevance_scores" in await _table_names()

    _run_alembic("downgrade", "base")


async def test_0010_creates_market_tables_and_reverts() -> None:
    _run_alembic("downgrade", "base")
    _run_alembic("upgrade", "head")
    assert MARKET_TABLES <= await _table_names()

    _run_alembic("downgrade", "0009")
    tables = await _table_names()
    assert not (MARKET_TABLES & tables) and "clips" in tables

    _run_alembic("downgrade", "base")


async def test_0011_creates_oauth_and_analytics_tables_and_reverts() -> None:
    _run_alembic("downgrade", "base")
    _run_alembic("upgrade", "head")
    assert ANALYTICS_TABLES <= await _table_names()

    _run_alembic("downgrade", "0010")
    tables = await _table_names()
    assert not (ANALYTICS_TABLES & tables) and MARKET_TABLES <= tables

    _run_alembic("downgrade", "base")


async def test_0012_creates_reporting_tables_with_constraints_and_reverts() -> None:
    _run_alembic("downgrade", "base")
    _run_alembic("upgrade", "head")
    assert REPORTING_TABLES <= await _table_names()

    engine = create_async_engine(get_settings().test_database_url)
    try:
        async with engine.begin() as conn:
            insert_job = text(
                "INSERT INTO reporting_jobs (channel_id, report_type, youtube_job_id, status, job_created_at) "
                "VALUES ('UC1', 'channel_reach_basic_a1', :jid, :status, now())"
            )
            await conn.execute(insert_job, {"jid": "j1", "status": "active"})
            await conn.execute(insert_job, {"jid": "j0", "status": "expired"})  # non-active rows don't count
        # A second active job for the same (channel, type) breaks the partial unique index.
        with pytest.raises(IntegrityError):
            async with engine.begin() as conn:
                await conn.execute(insert_job, {"jid": "j2", "status": "active"})
        with pytest.raises(IntegrityError):  # status CHECK
            async with engine.begin() as conn:
                await conn.execute(insert_job, {"jid": "j3", "status": "ready"})
    finally:
        await engine.dispose()

    _run_alembic("downgrade", "0011")
    tables = await _table_names()
    assert not (REPORTING_TABLES & tables) and ANALYTICS_TABLES <= tables

    _run_alembic("downgrade", "base")


async def test_0013_creates_ctr_predictions_with_cascade_and_reverts() -> None:
    _run_alembic("downgrade", "base")
    _run_alembic("upgrade", "head")
    assert "ctr_predictions" in await _table_names()

    engine = create_async_engine(get_settings().test_database_url)
    try:
        async with engine.begin() as conn:
            upload_id = (
                await conn.execute(text("INSERT INTO uploads (filename, storage_path) VALUES ('a', 'k') RETURNING id"))
            ).scalar_one()
            await conn.execute(
                text("INSERT INTO ctr_predictions (upload_id, candidate_title) VALUES (:u, 't')"), {"u": upload_id}
            )
            await conn.execute(text("DELETE FROM uploads WHERE id = :u"), {"u": upload_id})
            assert (await conn.execute(text("SELECT count(*) FROM ctr_predictions"))).scalar_one() == 0
        with pytest.raises(IntegrityError):  # source CHECK
            async with engine.begin() as conn:
                await conn.execute(text("INSERT INTO ctr_predictions (candidate_title, source) VALUES ('t', 'llm')"))
    finally:
        await engine.dispose()

    _run_alembic("downgrade", "0012")
    tables = await _table_names()
    assert "ctr_predictions" not in tables and REPORTING_TABLES <= tables

    _run_alembic("downgrade", "base")


async def test_0014_creates_hook_scores_unique_per_clip_with_cascade_and_reverts() -> None:
    _run_alembic("downgrade", "base")
    _run_alembic("upgrade", "head")
    assert "hook_scores" in await _table_names()

    engine = create_async_engine(get_settings().test_database_url)
    try:
        async with engine.begin() as conn:
            upload_id = (
                await conn.execute(text("INSERT INTO uploads (filename, storage_path) VALUES ('a', 'k') RETURNING id"))
            ).scalar_one()
            clip_id = (
                await conn.execute(
                    text(
                        "INSERT INTO clips (upload_id, rank, start_seconds, end_seconds, score, peak_window_index, "
                        "aspect_ratio, caption_burned_in, storage_path, crop_bounding_box_track, framing) "
                        "VALUES (:u, 1, 0, 30, 0.5, 0, '9:16', true, 'c', '[]', '{}') RETURNING id"
                    ),
                    {"u": upload_id},
                )
            ).scalar_one()
            await conn.execute(
                text("INSERT INTO hook_scores (clip_id, score, feature_vector) VALUES (:c, 50, '{}')"), {"c": clip_id}
            )
        with pytest.raises(IntegrityError):  # one row per clip
            async with engine.begin() as conn:
                await conn.execute(
                    text("INSERT INTO hook_scores (clip_id, score, feature_vector) VALUES (:c, 60, '{}')"),
                    {"c": clip_id},
                )
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM clips WHERE id = :c"), {"c": clip_id})
            assert (await conn.execute(text("SELECT count(*) FROM hook_scores"))).scalar_one() == 0
    finally:
        await engine.dispose()

    _run_alembic("downgrade", "0013")
    tables = await _table_names()
    assert "hook_scores" not in tables and "ctr_predictions" in tables

    _run_alembic("downgrade", "base")
