"""The real exported chapter bundle, on CPU (0009 test plan).

Skipped when `CHAPTERS_MODEL_DIR` points at no bundle: the weights are
gitignored and come from experiments/m5-chapters/05_eval, so a fresh
clone can't have them. This is the suite's only skip-if-unavailable
test, by design (0009, "Test plan").
"""

from pathlib import Path

import pytest

from app.chapters import pipeline
from app.chapters.units import Atom
from app.config import get_settings
from tests.fixtures import FIXTURE_CLIP_REFERENCE

bundle = get_settings().chapters_model_dir
pytestmark = pytest.mark.skipif(
    bundle is None or not (Path(bundle) / "manifest.json").is_file(),
    reason="no exported chapter bundle (run experiments/m5-chapters/05_eval)",
)


def _long_atoms() -> list[Atom]:
    """~20 min of two clearly different topics, one ~5 s sentence at a time."""
    cooking = "Today we slowly cook the tomato sauce with garlic, basil and olive oil."
    football = "The striker scored twice in the second half and the stadium went wild."
    return [
        Atom(5.0 * i, 5.0 * i + 4.5, cooking if i < 120 else football)
        for i in range(240)
    ]


def test_real_bundle_on_the_m3_fixture_text() -> None:
    words = FIXTURE_CLIP_REFERENCE.split()
    atoms = [Atom(0.4 * i, 0.4 * i + 0.35, w.lower() + ("." if i % 12 == 11 else "")) for i, w in enumerate(words)]
    c = pipeline.Chapterer(Path(bundle))
    results = c.run(atoms, duration=0.4 * len(words))
    assert results and results[0].start == 0.0
    assert all(r.title.strip() for r in results)


def test_current_bundle_is_v2_with_llm_titles() -> None:
    c = pipeline.Chapterer(Path(bundle))
    assert c.version.endswith("+qwen3.5-4b")  # 0010: titles from the local LLM
    assert c.llm is not None and c.lid is not None


def test_real_bundle_on_a_two_topic_episode() -> None:
    c = pipeline.Chapterer(Path(bundle))
    results = c.run(_long_atoms(), duration=1200.0)
    starts = [r.start for r in results]
    assert starts[0] == 0.0 and starts == sorted(starts)
    assert len(results) >= 3  # YouTube floor, enforced by decoding
    assert all(r.title.strip() for r in results)
