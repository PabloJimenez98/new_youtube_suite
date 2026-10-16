"""faster-whisper wrapper.

Everything here is synchronous and CPU/GPU-bound -- callers run
`transcribe` in a worker thread (`asyncio.to_thread`), never on the
event loop.
"""

import threading
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

from app.config import get_settings


@dataclass(frozen=True)
class WordResult:
    word: str
    start: float | None
    end: float | None
    probability: float | None


@dataclass(frozen=True)
class SegmentResult:
    start: float
    end: float
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None
    words: list[WordResult]


_model: WhisperModel | None = None
_model_lock = threading.Lock()


def get_model() -> WhisperModel:
    """Load the model once per process (the first load may download it)."""
    global _model
    with _model_lock:
        if _model is None:
            settings = get_settings()
            _model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
        return _model


def transcribe(audio_path: Path) -> list[SegmentResult]:
    settings = get_settings()
    kwargs = dict(
        language=None,  # auto-detect per episode
        beam_size=settings.whisper_beam_size,
        vad_filter=settings.whisper_vad_filter,
        word_timestamps=True,
    )
    if settings.whisper_batch_size > 0:
        # Batched decoding always segments with VAD (docs/decisions/0014 §1).
        segments, _info = BatchedInferencePipeline(get_model()).transcribe(
            str(audio_path), batch_size=settings.whisper_batch_size, **kwargs
        )
    else:
        segments, _info = get_model().transcribe(str(audio_path), **kwargs)
    # `segments` is a lazy generator -- the actual decoding happens
    # while it's consumed, so it must be drained here, inside the thread.
    return [
        SegmentResult(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            avg_logprob=seg.avg_logprob,
            no_speech_prob=seg.no_speech_prob,
            words=[
                WordResult(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in (seg.words or [])
            ],
        )
        for seg in segments
    ]
