from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
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
    # RTX 5070 pass (docs/decisions/0016, research hw-upgrade §4-5): large-v3
    # int8_float16 instead of M9's turbo int8 -- 6.3 WER points better on the
    # Sastre y Maldonado excerpts, 48x real time at batch 16, peak +6.5 GB,
    # ~2.2 GB resident after a job. `WHISPER_DEVICE=cpu` (with
    # `WHISPER_COMPUTE_TYPE=int8`) is the no-GPU fallback.
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"
    # Batched greedy decoding (M9, 0014 §1); batch only changes speed, not WER
    # (0016). `WHISPER_BATCH_SIZE=0` restores sequential decoding.
    whisper_beam_size: int = 1
    whisper_batch_size: int = 16
    whisper_vad_filter: bool = True

    # Speaker diarization (M4) -- see docs/decisions/0008-speaker-diarization.md.
    # Read token for the gated pyannote/speaker-diarization-community-1 model.
    hf_token: SecretStr | None = None
    diarization_device: str = "cuda"
    # pyannote's default 32 (the GTX 1060 needed 8, 0008): on a 43-min episode
    # 55 s vs 71 s, identical output (DER 0.0000 between the two), peak +2.6 GB
    # handed back by empty_cache (0016).
    diarization_embedding_batch_size: int = 32
    # Run the pipeline on a known 2-speaker clip at worker startup; if it
    # fails, only `diarize` jobs degrade (they fail fast with the reason).
    diarization_healthcheck: bool = True

    # Chapter generation (M5) -- see docs/decisions/0009-chapter-generator-local-models.md.
    # Directory of a model bundle exported by experiments/m5-chapters/05_eval
    # (gitignored). Unset -> `chapters` jobs fail fast with a clear message.
    chapters_model_dir: Path | None = None
    # GPU since the RTX 5070 pass (0016): same boundaries as on CPU, and it
    # fits next to Whisper + pyannote + the LLM. `cpu` works without a GPU.
    chapters_device: str = "cuda"
    # CPU threads for the bundle's title LLM, if it has one (0010 §4); the
    # 7800X3D's physical cores. With full offload they barely matter.
    chapters_llm_threads: int = 8
    # Layers of the shared LLM (chapters titles, show notes, relevance) on
    # the GPU: -1 = all (RTX 5070, 0016: Qwen3.5-9B Q4_K_M ~5.5 GB, ~93 tok/s);
    # 0 = weights in RAM with the prefill on the GPU (M9, 0014 §2).
    llm_n_gpu_layers: int = -1

    # Shorts (M8) -- see docs/decisions/0013-shorts-generator.md. YuNet face
    # detector, downloaded by hand (gitignored, see README). A missing file
    # makes `shorts` jobs fail fast.
    shorts_face_model_path: Path | None = Path("models/shorts/face_detection_yunet_2023mar.onnx")

    # Market-analysis pipeline (M10) -- see docs/decisions/0015-market-pipeline-v0.md.
    # Read-only key with YouTube Data API v3 enabled. Unset -> the market
    # cron tasks log a skip and return (the tool jobs are unaffected).
    youtube_api_key: SecretStr | None = None
    # Videos published within this many days get a snapshot every day...
    market_recent_window_days: int = 90
    # ...older ones once their latest snapshot is this many days old.
    market_older_interval_days: int = 7
    # Soft cap on Data API units per quota day (Pacific), per process.
    market_daily_unit_budget: int = 5000

    @field_validator("youtube_api_key", mode="before")
    @classmethod
    def _empty_key_is_unset(cls, value: object) -> object:
        # `YOUTUBE_API_KEY=` (as in .env.example) means "no key".
        return None if isinstance(value, str) and not value.strip() else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
