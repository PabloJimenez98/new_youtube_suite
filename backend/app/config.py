from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
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

    # arq kills a job after this long. arq's own 300s default is far
    # shorter than a real episode's transcription.
    worker_job_timeout_seconds: int = 10800

    # Transcription (M3) -- see docs/decisions/0007-transcription-pipeline.md.
    # CC 6.1 GPUs (the GTX 1060 this runs on) only support int8/float32;
    # `WHISPER_DEVICE=cpu` is the no-GPU fallback, no code changes needed.
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True

    # Speaker diarization (M4) -- see docs/decisions/0008-speaker-diarization.md.
    # Read token for the gated pyannote/speaker-diarization-community-1 model.
    hf_token: SecretStr | None = None
    diarization_device: str = "cuda"
    # 8 fits alongside a resident Whisper on 3 GB at no speed cost vs the
    # default 32 (measured, docs/research/m4-diarization-spike.md).
    diarization_embedding_batch_size: int = 8
    # Run the pipeline on a known 2-speaker clip at worker startup; if it
    # fails, only `diarize` jobs degrade (they fail fast with the reason).
    diarization_healthcheck: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
