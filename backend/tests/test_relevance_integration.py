"""Relevance LLM scoring with the real local model, on CPU (0012 test plan).

Skipped when `CHAPTERS_MODEL_DIR` points at no bundle, as the chapters
integration test is (0009, "Test plan"): the GGUF is gitignored.
"""

from pathlib import Path

import pytest

from app.config import get_settings
from app.relevance import pipeline
from app.relevance.llm_scoring import PROMPT_VERSION, score_batch
from app.relevance.windows import Word
from tests.fixtures import FIXTURE_CLIP_REFERENCE

bundle = get_settings().chapters_model_dir
pytestmark = pytest.mark.skipif(
    bundle is None or not (Path(bundle) / "manifest.json").is_file(),
    reason="no exported chapter bundle (run experiments/m5-chapters/05_eval)",
)


def test_real_llm_scores_a_batch() -> None:
    texts = [
        "So the biggest mistake founders make is hiring too fast, and here is the story of how it nearly killed us.",
        "Okay, uh, let me just check the mic, can you hear me, yeah, okay.",
        FIXTURE_CLIP_REFERENCE[:400],
    ]
    scores = score_batch(pipeline.get_scorer().llm, texts, "en", [None, None, None])
    assert scores is not None and len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_real_scorer_on_the_m3_fixture_text() -> None:
    words = [Word(0.4 * i, w) for i, w in enumerate(FIXTURE_CLIP_REFERENCE.split())]
    label, description, result = pipeline.score_relevance(words, 0.4 * len(words), [])
    assert label.startswith("heuristic-v0+") and label.endswith(f"+{PROMPT_VERSION}")
    assert result.scores and result.failed_batches == 0 and result.batches >= 2
    assert all(0.0 <= s.score <= 1.0 and s.llm is not None for s in result.scores)


def test_real_encoder_and_bilstm_bundle_score_the_m3_fixture(tmp_path) -> None:
    """M15 (0024 test plan): the real e5 encoder + a (randomly initialized) BiLSTM bundle."""
    import json as _json

    import torch

    from app.relevance.bilstm import BiLSTMScorer, BundleConfig, RelevanceBiLSTM, save_bundle

    boundary = _json.loads((Path(bundle) / "manifest.json").read_text())["boundary"]
    torch.manual_seed(0)
    model = RelevanceBiLSTM(emb_dim=384)
    config = BundleConfig(label="it", encoder_name=boundary["encoder"], encoder_prefix=boundary["encoder_prefix"],
                          encoder_max_length=128, model_config=model.config)
    save_bundle(tmp_path / "it", config, model, {}, None)
    (tmp_path / "it" / "encoder").symlink_to(Path(bundle).resolve() / "encoder", target_is_directory=True)

    words = [Word(0.4 * i, w) for i, w in enumerate(FIXTURE_CLIP_REFERENCE.split())]
    duration = 0.4 * len(words)
    result = BiLSTMScorer(tmp_path / "it", cache_dir=tmp_path / "cache").run(words, duration)
    from app.relevance.windows import build_blocks, build_windows

    n_windows = len(build_windows(build_blocks(words, duration)))
    assert len(result.scores) == n_windows
    assert all(0.0 <= s.score <= 1.0 and 0.0 <= s.heuristic <= 1.0 and 0.0 <= s.llm <= 1.0 for s in result.scores)
    assert list((tmp_path / "cache").glob("*.npy"))
