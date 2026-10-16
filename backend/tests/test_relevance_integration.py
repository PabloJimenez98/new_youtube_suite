"""Relevance LLM scoring with the real local model, on CPU (0012 test plan).

Skipped when `CHAPTERS_MODEL_DIR` points at no bundle, as the chapters
integration test is (0009, "Test plan"): the GGUF is gitignored.
"""

from pathlib import Path

import pytest

from app.config import get_settings
from app.relevance import pipeline
from app.relevance.llm_scoring import score_batch
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
    assert label.startswith("heuristic-v0+") and label.endswith("+relevance-p2")
    assert result.scores and result.failed_batches == 0 and result.batches >= 2
    assert all(0.0 <= s.score <= 1.0 and s.llm is not None for s in result.scores)
