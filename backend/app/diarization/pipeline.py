"""pyannote `community-1` wrapper + the worker's startup healthcheck.

Everything but `startup_healthcheck` is synchronous and GPU/CPU-bound --
callers run it in a worker thread under `app.worker.gpu.gpu_lock`. See
docs/decisions/0008-speaker-diarization.md.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline

from app.config import get_settings
from app.diarization.assign import Turn, relabel_by_first_appearance

logger = logging.getLogger(__name__)

PIPELINE_ID = "pyannote/speaker-diarization-community-1"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
HEALTHCHECK_CLIP = ASSETS_DIR / "diarization_healthcheck.wav"


class DiarizationUnavailableError(Exception):
    """The pipeline can't be used (no token, gated model, failed healthcheck)."""


_pipeline: Pipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> Pipeline:
    """Load once per process (the first load downloads the model)."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            settings = get_settings()
            if settings.hf_token is None:
                raise DiarizationUnavailableError(
                    "HF_TOKEN is not set -- needed for the gated "
                    f"{PIPELINE_ID} model (see README)"
                )
            pipeline = Pipeline.from_pretrained(
                PIPELINE_ID, token=settings.hf_token.get_secret_value()
            )
            if pipeline is None:
                # Older pyannote versions returned None instead of raising
                # on gated-model access problems -- the classic silent failure.
                raise DiarizationUnavailableError(
                    f"Could not load {PIPELINE_ID}: check HF_TOKEN and that its "
                    "conditions are accepted on huggingface.co"
                )
            pipeline.to(torch.device(settings.diarization_device))
            pipeline.embedding_batch_size = settings.diarization_embedding_batch_size
            _pipeline = pipeline
        return _pipeline


def diarize(wav_path: Path) -> list[Turn]:
    """Speaker turns for a WAV file, relabelled by first appearance.

    The audio is passed preloaded: pyannote's own decoder (torchcodec)
    can't load in this environment (docs/decisions/0008 §2).
    """
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())  # (channel, time)
    try:
        output = get_pipeline()({"waveform": waveform, "sample_rate": sample_rate})
    finally:
        # torch's caching allocator keeps the ~1 GB of inference memory
        # reserved after a run. On a 3 GB card shared with Whisper that
        # starves the next transcription (CUDA OOM) -- hand it back.
        # See docs/decisions/0008-speaker-diarization.md "Post-review notes".
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # The overlap-free variant: at most one speaker at any instant, which
    # is what word → speaker assignment needs.
    annotation = output.exclusive_speaker_diarization
    return relabel_by_first_appearance(
        [
            Turn(start=segment.start, end=segment.end, speaker=label)
            for segment, _track, label in annotation.itertracks(yield_label=True)
        ]
    )


# ---- Startup healthcheck -------------------------------------------------


@dataclass
class HealthState:
    checked: bool = False
    error: str | None = None


health = HealthState()


def check_health() -> None:
    """Raise unless the pipeline finds 2+ speakers on the bundled clip."""
    turns = diarize(HEALTHCHECK_CLIP)
    speakers = {t.speaker for t in turns}
    if len(speakers) < 2:
        raise DiarizationUnavailableError(
            f"healthcheck clip produced {len(turns)} turns / {len(speakers)} "
            "speaker(s), expected 2+ -- wrong model license or token?"
        )


async def startup_healthcheck() -> None:
    """Run at worker startup. Never raises: a failure only degrades
    `diarize` (its jobs fail fast with `health.error`)."""
    from app.worker.gpu import gpu_lock  # avoid an import cycle at module load

    try:
        async with gpu_lock:
            await asyncio.to_thread(check_health)
    except Exception as exc:  # noqa: BLE001 -- any failure means "unavailable"
        health.error = f"{type(exc).__name__}: {exc}"
        logger.error("Diarization healthcheck failed, diarize jobs disabled: %s", health.error)
    else:
        health.error = None
        logger.info("Diarization healthcheck passed")
    health.checked = True


def ensure_available() -> None:
    if health.error is not None:
        raise DiarizationUnavailableError(f"Diarization unavailable: {health.error}")
