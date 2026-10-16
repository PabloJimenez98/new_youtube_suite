"""Show notes with the real local LLM, on CPU (0011 test plan).

Skipped when `CHAPTERS_MODEL_DIR` points at no bundle, as the chapters
integration test is (0009, "Test plan"): the GGUF is gitignored.
"""

from pathlib import Path

import pytest

from app.config import get_settings
from app.shownotes import pipeline
from app.shownotes.chunking import ChapterMark, Seg
from app.shownotes.validate import MIN_TAGS, MIN_TITLES
from tests.fixtures import FIXTURE_CLIP_REFERENCE

bundle = get_settings().chapters_model_dir
pytestmark = pytest.mark.skipif(
    bundle is None or not (Path(bundle) / "manifest.json").is_file(),
    reason="no exported chapter bundle (run experiments/m5-chapters/05_eval)",
)


def test_real_llm_on_the_m3_fixture_transcript() -> None:
    words = FIXTURE_CLIP_REFERENCE.split()
    segs = [Seg(4.0 * i, 4.0 * i + 4, " ".join(words[j : j + 10])) for i, j in enumerate(range(0, len(words), 10))]
    chapters = [ChapterMark(0.0, "Opening"), ChapterMark(8.0, "The story")]

    model, notes = pipeline.generate_show_notes(segs, chapters)

    assert model.endswith("+shownotes-v1")
    d = notes.draft
    assert MIN_TITLES <= len(d.title_suggestions) <= 5 and all(0 < len(t) <= 100 for t in d.title_suggestions)
    assert len(d.tags) >= MIN_TAGS
    assert d.summary.strip()
    assert d.description.endswith("00:00 Opening\n00:08 The story")
    assert notes.language == "en"
