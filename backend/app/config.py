from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = (
        "postgresql+asyncpg://youtube_suite:youtube_suite@localhost:5432/youtube_suite"
    )
    test_database_url: str = (
        "postgresql+asyncpg://youtube_suite:youtube_suite@localhost:5432/youtube_suite_test"
    )

    jwt_secret: str = "change-me-dev-only"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720

    storage_base_path: Path = Path("../uploads")

    # arq job queue (M2). Separate logical Redis DB for tests, mirroring
    # the database_url/test_database_url split above.
    redis_url: str = "redis://localhost:6379/0"
    test_redis_url: str = "redis://localhost:6379/1"

    # How long the M2 dummy job handler sleeps before succeeding -- a
    # stand-in for real per-tool processing time until each tool's own
    # milestone (M3+) registers a real handler.
    dummy_job_duration_seconds: float = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
