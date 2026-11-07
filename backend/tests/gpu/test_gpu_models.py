"""Each production model on the GPU: right output, GPU-class speed (catches a
silent CPU or sysmem fallback), VRAM within budget, memory handed back."""

import re
import time

import jiwer
import pytest

from tests.fixtures import FIXTURE_CLIP, FIXTURE_CLIP_REFERENCE
from tests.gpu.conftest import CARD_BUDGET_MIB, VramPeak, bundle_llm, reset_models, vram_used_mib

pytestmark = pytest.mark.gpu


def _words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z' ]", " ", text.lower()).split())


def test_whisper_production_config(prod_settings) -> None:
    from app.transcription import whisper

    reset_models()
    before = vram_used_mib()
    with VramPeak() as peak:
        whisper.transcribe(FIXTURE_CLIP)  # load + warm-up
        t0 = time.time()
        segments = whisper.transcribe(FIXTURE_CLIP)
        elapsed = time.time() - t0
    wer = jiwer.wer(_words(FIXTURE_CLIP_REFERENCE), _words(" ".join(s.text for s in segments)))
    assert wer <= 0.10
    assert elapsed < 10, f"18 s clip took {elapsed:.1f}s: not running on the GPU?"
    assert peak.peak - before < 5_000, f"Whisper used {peak.peak - before} MiB"


def test_diarization_production_config_and_memory_handback(prod_settings) -> None:
    import torch

    from app.diarization import pipeline

    assert prod_settings.diarization_device == "cuda"
    t0 = time.time()
    pipeline.check_health()  # >= 2 speakers on the A/B/A clip, or raises
    assert time.time() - t0 < 60
    assert torch.cuda.memory_reserved() < 512 * 2**20, "diarize() should hand its inference memory back"


def test_llm_production_config_speed_and_release(prod_settings) -> None:
    found = bundle_llm(prod_settings)
    if found is None:
        pytest.skip("no chapter bundle with an LLM configured")
    from app.llm import runtime

    assert prod_settings.llm_n_gpu_layers != 0, "production LLM should be offloaded to the GPU"
    reset_models()
    before = vram_used_mib()
    llm = runtime.get_llama(found[0], found[1], prod_settings.chapters_llm_threads)
    loaded = vram_used_mib()
    t0 = time.time()
    out = llm.create_completion("Escribe una lista de diez ciudades de España:\n1.", max_tokens=128, temperature=0.0)
    tok_s = out["usage"]["completion_tokens"] / (time.time() - t0)
    runtime.release_all()
    after = vram_used_mib()
    assert tok_s > 30, f"{tok_s:.1f} tok/s: layers not on the GPU?"
    assert loaded - before > 1_000, "the weights should be in VRAM"
    assert after - before < 300, f"release_all left {after - before} MiB behind"


def test_chapters_on_cuda_match_cpu(prod_settings) -> None:
    from pathlib import Path

    from app.chapters import pipeline
    from app.chapters.units import Atom

    bundle = prod_settings.chapters_model_dir
    if bundle is None or not (Path(bundle) / "manifest.json").is_file():
        pytest.skip("no chapter bundle configured")
    cooking = "Today we slowly cook the tomato sauce with garlic, basil and olive oil."
    football = "The striker scored twice in the second half and the stadium went wild."
    atoms = [Atom(5.0 * i, 5.0 * i + 4.5, cooking if i < 120 else football) for i in range(240)]
    starts = {}
    for device in ("cpu", "cuda"):
        c = pipeline.Chapterer(Path(bundle), device, prod_settings.chapters_llm_threads)
        results = c.run(atoms, duration=1200.0)
        starts[device] = [r.start for r in results]
        assert all(r.title for r in results)
    reset_models()
    assert starts["cuda"] == starts["cpu"]


def test_worst_case_resident_set_fits_the_card(prod_settings) -> None:
    """Whisper + pyannote + chapters + the LLM all resident (the worker never
    unloads the first three), each run in turn as the gpu_lock does."""
    found = bundle_llm(prod_settings)
    if found is None:
        pytest.skip("no chapter bundle with an LLM configured")
    from app.chapters import pipeline as chapters
    from app.diarization import pipeline as diarization
    from app.llm import runtime
    from app.transcription import whisper

    reset_models()
    with VramPeak() as peak:
        whisper.transcribe(FIXTURE_CLIP)
        diarization.check_health()
        chapters.get_chapterer()
        llm = runtime.get_llama(found[0], found[1], prod_settings.chapters_llm_threads)
        llm.create_completion("Resume en una frase: " + FIXTURE_CLIP_REFERENCE, max_tokens=64, temperature=0.0)
        whisper.transcribe(FIXTURE_CLIP)  # Whisper again with everything loaded
    reset_models()
    print(f"worst-case resident set: peak {peak.peak} MiB of {CARD_BUDGET_MIB} MiB budget")  # shown with -s
    assert peak.peak <= CARD_BUDGET_MIB, f"peak {peak.peak} MiB > budget {CARD_BUDGET_MIB} MiB"
