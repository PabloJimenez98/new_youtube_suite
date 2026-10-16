"""Relevance v0: heuristics + batched zero-shot LLM per window (docs/decisions/0012).

Words -> 7.5 s blocks -> 15 s windows (50 % overlap) -> heuristic score
and LLM score per window -> `score = 0.4 * heuristic + 0.6 * llm`, or
the heuristic alone where the LLM gave nothing. Model and LID come from
the chapters bundle; the `Llama` is the process-wide one. Synchronous
and CPU-only: callers use `asyncio.to_thread`.
"""

import bisect
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from app.chapters.text import LanguageId
from app.config import get_settings
from app.llm import runtime
from app.llm.chat import ChatLLM, LlamaChat
from app.relevance import heuristics
from app.relevance.llm_scoring import PROMPT_VERSION, block_scores, window_llm_scores
from app.relevance.windows import BLOCK_S, BLOCKS_PER_WINDOW, Window, Word, build_blocks, build_windows

HEURISTIC_WEIGHT = 0.4
LLM_WEIGHT = 0.6


class RelevanceUnavailableError(Exception):
    """No usable LLM is configured."""


class RelevanceGenerationError(Exception):
    """Every LLM batch failed."""


@dataclass(frozen=True)
class WindowScore:
    window: Window
    heuristic: float
    llm: float | None
    score: float


@dataclass
class RelevanceResult:
    language: str
    scores: list[WindowScore]
    batches: int = 0
    failed_batches: int = 0


def combine(heuristic: float, llm: float | None) -> float:
    return heuristic if llm is None else HEURISTIC_WEIGHT * heuristic + LLM_WEIGHT * llm


class RelevanceScorer:
    def __init__(self, llm: ChatLLM, lid, version_label: str):
        self.llm, self.lid = llm, lid
        self.version_label = version_label

    @staticmethod
    def description() -> dict:
        """What this version is, for `model_versions.architecture_notes` / `training_data_reference`."""
        return {
            "architecture_notes": (
                f"Heuristics (weights {heuristics.WEIGHTS}) + zero-shot LLM over 15 s blocks, "
                f"{HEURISTIC_WEIGHT}/{LLM_WEIGHT}; windows {BLOCK_S * BLOCKS_PER_WINDOW:g} s, hop {BLOCK_S:g} s. "
                "docs/decisions/0012-relevance-v0.md"
            ),
            "training_data_reference": {"kind": "untrained", "prompt_version": PROMPT_VERSION},
        }

    def run(self, words: list[Word], duration: float, chapters: list[tuple[float, str]]) -> RelevanceResult:
        text = " ".join(w.text for w in words)
        detected, _ = self.lid(text[:5000])
        lang = "es" if detected == "es" else "en"
        blocks = build_blocks(words, duration)
        windows = build_windows(blocks)
        heur = heuristics.heuristic_scores(windows, lang)

        starts = [s for s, _ in chapters]

        def chapter_at(t: float) -> str | None:
            i = bisect.bisect_right(starts, t) - 1
            return chapters[i][1] if i >= 0 else None

        block_llm, batches, failed = block_scores(self.llm, blocks, lang, chapter_at)
        if batches and failed == batches:
            raise RelevanceGenerationError(f"Relevance: every LLM batch failed ({batches} batches)")
        llm = window_llm_scores(block_llm, len(windows))
        scores = [WindowScore(w, h, l, combine(h, l)) for w, h, l in zip(windows, heur, llm, strict=True)]
        return RelevanceResult(lang, scores, batches, failed)


_scorer: RelevanceScorer | None = None
_lock = threading.Lock()


def get_scorer() -> RelevanceScorer:
    """Build the scorer once per process from the chapters bundle's LLM and LID."""
    global _scorer
    with _lock:
        if _scorer is None:
            settings = get_settings()
            bundle = settings.chapters_model_dir
            manifest = Path(bundle) / "manifest.json" if bundle else None
            titles = json.loads(manifest.read_text()).get("titles", {}) if manifest and manifest.is_file() else {}
            if "llm" not in titles or "lid" not in titles:
                raise RelevanceUnavailableError(
                    "No LLM configured for relevance -- set CHAPTERS_MODEL_DIR to a chapter bundle "
                    "with an LLM (see README, 'Chapter models')"
                )
            path = Path(bundle) / titles["llm"]["file"]
            n_ctx = titles["llm"].get("n_ctx", 4096)
            llm = LlamaChat(runtime.get_llama(path, n_ctx, settings.chapters_llm_threads), n_ctx)
            _scorer = RelevanceScorer(llm, LanguageId(Path(bundle) / titles["lid"]["file"]),
                                      f"heuristic-v0+{path.stem}+{PROMPT_VERSION}")
        return _scorer


def score_relevance(words: list[Word], duration: float,
                    chapters: list[tuple[float, str]]) -> tuple[str, dict, RelevanceResult]:
    """(version label, version description, result). Synchronous -- run in a thread."""
    s = get_scorer()
    return s.version_label, s.description(), s.run(words, duration, chapters)
