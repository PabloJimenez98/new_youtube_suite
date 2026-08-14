from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.auth.security import hash_password
from app.config import get_settings
from app.database import Base, get_db
from app.main import app
from app.models.user import User

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
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session


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
