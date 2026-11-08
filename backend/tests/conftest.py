import os
import tempfile
from collections.abc import AsyncGenerator

# Must happen before `app.main` (and therefore `app.routers.uploads`) is
# imported anywhere -- the uploads router builds its `LocalStorageBackend`
# once at module-import time from `get_settings()`, so tests need their
# own isolated storage dir in place before that first import, not
# patched in per-test.
os.environ.setdefault("STORAGE_BASE_PATH", tempfile.mkdtemp(prefix="ys_test_uploads_"))

# Transcription tests (M3) run a small Whisper model on CPU: fast, and
# independent of the CUDA setup. Production uses large-v3 on the GPU
# -- see docs/decisions/0007-transcription-pipeline.md. Set before any
# `get_settings()` call, which caches.
# Same for diarization (M4): pyannote on CPU in tests. HF_TOKEN still
# comes from backend/.env -- the real-pipeline tests need it.
# And chapters + the shared LLM (M5-M7): the fast suite stays CPU-only
# even when backend/.env points production at the GPU (docs/decisions/0016).
# The GPU path has its own opt-in suite, `pytest -m gpu`, which undoes
# exactly the pins set here (FORCED_CPU_ENV) to get the production config.
CPU_PINS = {
    "WHISPER_MODEL": "base",
    "WHISPER_DEVICE": "cpu",
    "WHISPER_COMPUTE_TYPE": "int8",
    "DIARIZATION_DEVICE": "cpu",
    "CHAPTERS_DEVICE": "cpu",
    "LLM_N_GPU_LAYERS": "0",
}
FORCED_CPU_ENV = [k for k in CPU_PINS if k not in os.environ]
for _key, _value in CPU_PINS.items():
    os.environ.setdefault(_key, _value)

import pytest
import pytest_asyncio
from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.auth.security import hash_password
from app.config import get_settings
from app.database import Base, get_db, get_session_factory
from app.main import app
from app.models.user import User
from app.worker.pool import get_arq_pool

settings = get_settings()


@pytest_asyncio.fixture
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Fresh engine per test, bound to that test's event loop.

    Async SQLAlchemy/asyncpg connections are tied to the event loop
    they were created on; pytest-asyncio gives each test function its
    own loop, so the engine must be created (and disposed) per test
    rather than once at module scope.
    """
    engine = create_async_engine(settings.test_database_url, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(
    db_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def _override_get_db(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[None, None]:
    async def _get_db_override() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db_override
    # M11: endpoints that pass a session factory to worker-shared code.
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    yield
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_session_factory, None)


@pytest_asyncio.fixture
async def db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def arq_redis_pool() -> AsyncGenerator[ArqRedis, None]:
    """Pool bound to `test_redis_url` (a separate logical Redis DB from
    dev), flushed before and after each test so jobs from one test
    can't leak into the next -- same isolation goal as `db_engine`'s
    per-test create/drop, just for Redis instead of Postgres.
    """
    pool = await create_pool(RedisSettings.from_dsn(settings.test_redis_url))
    await pool.flushdb()
    yield pool
    await pool.flushdb()
    await pool.aclose()


@pytest_asyncio.fixture(autouse=True)
async def _override_get_arq_pool(
    arq_redis_pool: ArqRedis,
) -> AsyncGenerator[None, None]:
    async def _get_arq_pool_override() -> ArqRedis:
        return arq_redis_pool

    app.dependency_overrides[get_arq_pool] = _get_arq_pool_override
    yield
    app.dependency_overrides.pop(get_arq_pool, None)


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    user = User(email="creator@example.com", password_hash=hash_password("s3cret-pw"))
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient, test_user: User) -> dict[str, str]:
    response = await client.post(
        "/auth/login",
        data={"username": test_user.email, "password": "s3cret-pw"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def dummy_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run `hook` jobs through the M2 dummy handler.

    The M2 state-machine and real-worker tests used `hook` as "the dummy
    tool"; since M14 every `tool_type` has a real handler, so they patch
    it back (docs/decisions/0020 §5)."""
    from app.worker import handlers
    from app.worker.handlers import dummy

    monkeypatch.setitem(handlers.HANDLERS, "hook", dummy.run)
