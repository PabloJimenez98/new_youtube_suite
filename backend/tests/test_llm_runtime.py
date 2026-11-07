"""Releasing the shared llama.cpp models frees their VRAM (M9, docs/decisions/0014 §2).

Regression for the season-1 run: with the CUDA build a resident context
kept ~1 GB of the 3 GB GPU and batched Whisper ran out of memory.
"""

from app.chapters import pipeline as chapters_pipeline
from app.llm import runtime
from app.relevance import pipeline as relevance_pipeline
from app.shownotes import pipeline as shownotes_pipeline


class FakeLlama:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_release_all_closes_models_and_drops_their_holders(monkeypatch) -> None:
    fake = FakeLlama()
    monkeypatch.setitem(runtime._cache, ("x.gguf", 4096), fake)
    monkeypatch.setattr(relevance_pipeline, "_scorer", object())
    monkeypatch.setattr(shownotes_pipeline, "_generator", object())
    monkeypatch.setattr(chapters_pipeline, "_chapterer", object())

    runtime.release_all()

    assert fake.closed and ("x.gguf", 4096) not in runtime._cache
    assert relevance_pipeline._scorer is None
    assert shownotes_pipeline._generator is None
    assert chapters_pipeline._chapterer is None


def test_release_all_with_nothing_loaded_is_a_noop() -> None:
    runtime.release_all()


def test_gpu_layers_come_from_settings_and_key_the_cache(monkeypatch, tmp_path) -> None:
    """LLM_N_GPU_LAYERS (docs/decisions/0016) reaches llama.cpp, and the same
    file with a different offload is a different instance."""
    from app.config import get_settings

    loads = []
    monkeypatch.setattr(runtime, "_cache", {})
    monkeypatch.setattr(runtime, "_load", lambda p, n, t, g: loads.append(g) or FakeLlama())
    monkeypatch.setattr(get_settings(), "llm_n_gpu_layers", -1)
    gguf = tmp_path / "m.gguf"

    a = runtime.get_llama(gguf, 4096)
    assert runtime.get_llama(gguf, 4096) is a  # cached
    b = runtime.get_llama(gguf, 4096, n_gpu_layers=0)
    assert b is not a and loads == [-1, 0]
