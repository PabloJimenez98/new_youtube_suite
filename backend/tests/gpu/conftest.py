"""Opt-in GPU suite (`pytest -m gpu`, docs/decisions/0016).

Loads the *production* configuration (backend/.env + defaults), not the
CPU pins the top-level conftest forces for the fast suite, and checks it
on the real GPU: the stack, each model's speed and VRAM, and that the
worst-case set of resident models fits on the 12 GB card.
"""

import json
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from tests.conftest import FORCED_CPU_ENV

# Whole-card budget for everything the worker keeps resident at once,
# Windows desktop included (12,227 MiB card; see the research note).
CARD_BUDGET_MIB = 11_264


def vram_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


class VramPeak:
    """Sample nvidia-smi in a thread (as experiments/m9-perf/bench_transcribe.py does)."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, vram_used_mib())
            time.sleep(self.interval)

    def __enter__(self) -> "VramPeak":
        self.peak = vram_used_mib()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def reset_models() -> None:
    from app.chapters import pipeline as chapters
    from app.diarization import pipeline as diarization
    from app.llm import runtime
    from app.transcription import whisper

    runtime.release_all()  # also drops the chapterer / scorer / generator
    whisper._model = None
    diarization._pipeline = None
    chapters._chapterer = None
    import gc

    gc.collect()
    import torch

    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def prod_settings() -> Iterator[Settings]:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    saved = {k: os.environ.pop(k) for k in FORCED_CPU_ENV if k in os.environ}
    get_settings.cache_clear()
    reset_models()
    try:
        yield get_settings()
    finally:
        reset_models()
        os.environ.update(saved)
        get_settings.cache_clear()


def bundle_llm(settings: Settings) -> tuple[Path, int] | None:
    """(GGUF path, n_ctx) of the configured bundle's LLM, or None."""
    bundle = settings.chapters_model_dir
    if bundle is None or not (Path(bundle) / "manifest.json").is_file():
        return None
    llm = json.loads((Path(bundle) / "manifest.json").read_text()).get("titles", {}).get("llm")
    if not llm:
        return None
    return Path(bundle) / llm["file"], int(llm.get("n_ctx", 4096))
