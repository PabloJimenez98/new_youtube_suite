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

import pytest_asyncio
from sqlalchemy import inspect
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
}


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
