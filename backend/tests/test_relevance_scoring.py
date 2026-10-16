"""Relevance v0 building blocks: windows, heuristics, batched LLM
scoring and the combination (docs/decisions/0012 §1-4)."""

import json

import pytest

from app.relevance import heuristics, llm_scoring
from app.relevance.heuristics import (
    episode_keywords,
    filler_score,
    heuristic_scores,
    question_answer,
    speaker_changes,
)
from app.relevance.llm_scoring import block_scores, parse_scores, window_llm_scores
from app.relevance.pipeline import RelevanceGenerationError, RelevanceScorer, combine
from app.relevance.windows import Block, Window, Word, build_blocks, build_windows


def win(text: str, start: float = 0.0, end: float = 15.0, speakers: list[str] | None = None) -> Window:
    toks = text.split()
    labels = speakers or [None] * len(toks)
    return Window(0, start, end, tuple(Word(start, t, s) for t, s in zip(toks, labels, strict=True)))


def timed_words(n: int, step: float = 0.5, speaker: str | None = None) -> list[Word]:
    return [Word(i * step, f"w{i}", speaker) for i in range(n)]


# --- windows -------------------------------------------------------------------


def test_blocks_and_windows_overlap_by_half() -> None:
    blocks = build_blocks(timed_words(60), duration=30.0)  # 30 s -> 4 blocks of 7.5 s
    assert [(b.start, b.end) for b in blocks] == [(0.0, 7.5), (7.5, 15.0), (15.0, 22.5), (22.5, 30.0)]
    assert [len(b.words) for b in blocks] == [15, 15, 15, 15]
    windows = build_windows(blocks)
    assert [(w.start, w.end) for w in windows] == [(0.0, 15.0), (7.5, 22.5), (15.0, 30.0)]
    assert windows[1].words == blocks[1].words + blocks[2].words


def test_untimed_words_join_the_previous_timed_word() -> None:
    words = [Word(None, "lead"), Word(1.0, "a"), Word(None, "€"), Word(8.0, "b"), Word(None, "%")]
    blocks = build_blocks(words, duration=15.0)
    assert [[w.text for w in b.words] for b in blocks] == [["lead", "a", "€"], ["b", "%"]]


def test_gaps_give_empty_blocks_and_the_last_block_ends_at_duration() -> None:
    blocks = build_blocks([Word(0.0, "a"), Word(20.0, "b")], duration=21.0)
    assert [len(b.words) for b in blocks] == [1, 0, 1]
    assert blocks[-1].end == 21.0


def test_an_episode_shorter_than_one_window_gives_one_window() -> None:
    windows = build_windows(build_blocks([Word(0.0, "hi")], duration=5.0))
    assert len(windows) == 1 and (windows[0].start, windows[0].end) == (0.0, 5.0)


def test_words_past_the_duration_land_in_the_last_block() -> None:
    blocks = build_blocks([Word(99.0, "late")], duration=10.0)
    assert blocks[-1].words[0].text == "late"


# --- heuristics ------------------------------------------------------------------


def test_keywords_recur_and_skip_stopwords() -> None:
    windows = [win("neural networks learn quickly"), win("neural networks overfit data"), win("lunch break pizza")]
    kw = episode_keywords(windows, "en")
    assert {"neural", "networks"} <= kw
    assert "lunch" not in kw  # appears once only


def test_spanish_stopwords_are_used_for_spanish() -> None:
    windows = [win("porque la inflación sube"), win("porque la inflación baja"), win("hola")]
    assert episode_keywords(windows, "es") == {"inflación"}


@pytest.mark.parametrize(
    "text,following,expected",
    [
        ("so what do you think about it? well I think it is great and also useful", None, 1.0),
        ("¿qué opinas tú? pues creo que es genial", "y además muy útil para todos", 1.0),
        ("what? ok", None, 1 / 8),
        ("no questions here at all", None, 0.0),
    ],
)
def test_question_followed_by_an_answer(text: str, following: str | None, expected: float) -> None:
    assert question_answer(win(text), win(following) if following else None) == pytest.approx(expected)


def test_speaker_changes_per_minute_saturate() -> None:
    assert speaker_changes(win("a b c d", speakers=["S0", "S1", "S0", "S1"])) == 1.0  # 3 in 15 s = 12/min
    assert speaker_changes(win("a b", start=0, end=60, speakers=["S0", "S1"])) == pytest.approx(1 / 6)
    assert speaker_changes(win("a b")) == 0.0


@pytest.mark.parametrize(
    "text,lang,expected",
    [
        ("this is clean speech with no hesitation at all here", "en", 1.0),
        ("um so you know it is um like this", "en", 0.0),  # 4/9 fillers > 15 %
        ("o sea es que eh no sé muy bien qué decir ahora mismo pues", "es", 0.0),  # 3/14 fillers > 15 %
        ("bueno pues hoy hablamos de economía y política con eh nuestros invitados especiales", "es", 1 - (1 / 13) / 0.15),
    ],
)
def test_filler_score(text: str, lang: str, expected: float) -> None:
    assert filler_score(win(text), lang) == pytest.approx(expected)


def test_short_windows_score_zero_and_weights_renormalize_without_speakers(monkeypatch) -> None:
    windows = [win("tiny"), win("a full sentence with enough words to count")]
    monkeypatch.setattr(heuristics, "topic_density", lambda w, k: 1.0)
    monkeypatch.setattr(heuristics, "question_answer", lambda w, f: 0.0)
    monkeypatch.setattr(heuristics, "filler_score", lambda w, lang: 1.0)
    scores = heuristic_scores(windows, "en")
    assert scores[0] == 0.0
    assert scores[1] == pytest.approx((0.35 + 0.20) / 0.80)  # no speakers -> weights sum 0.8


def test_speaker_feature_used_when_diarized(monkeypatch) -> None:
    w = win("a b c d e f", start=0, end=60, speakers=["S0", "S1"] * 3)
    monkeypatch.setattr(heuristics, "topic_density", lambda w, k: 0.0)
    monkeypatch.setattr(heuristics, "question_answer", lambda w, f: 0.0)
    monkeypatch.setattr(heuristics, "filler_score", lambda w, lang: 0.0)
    assert heuristic_scores([w], "en") == [pytest.approx(0.20 * (5 / 6))]


# --- LLM scoring -----------------------------------------------------------------


class FakeLLM:
    """Answers each batch with scripted outputs, else N copies of 50."""

    def __init__(self, outputs: list[str] | None = None):
        self.outputs = list(outputs or [])
        self.calls: list[dict] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete(self, messages, *, max_tokens, temperature=0.0, seed=0, json_schema=None) -> str:
        n = json_schema["properties"]["scores"]["minItems"]
        self.calls.append(dict(messages=messages, n=n, temperature=temperature, seed=seed))
        return self.outputs.pop(0) if self.outputs else json.dumps({"scores": [50] * n})


def blocks_with_text(n: int) -> list[Block]:
    return [Block(i, i * 7.5, (i + 1) * 7.5, (Word(i * 7.5, f"word{i}"),)) for i in range(n)]


def test_parse_scores_clamps_and_validates() -> None:
    assert parse_scores('{"scores": [0, 150, -5, 42]}', 4) == [0.0, 1.0, 0.0, 0.42]
    for raw in ('{"scores": [1, 2]}', "nope", '{"scores": ["a", 1, 2, 3]}', "[1,2,3,4]"):
        with pytest.raises(llm_scoring.ScoresInvalid):
            parse_scores(raw, 4)


def test_blocks_are_paired_and_batched_by_20() -> None:
    llm = FakeLLM()
    scores, batches, failed = block_scores(llm, blocks_with_text(90), "en")  # 45 LLM blocks
    assert [c["n"] for c in llm.calls] == [20, 20, 5]
    assert (batches, failed) == (3, 0) and scores == [0.5] * 45
    assert "[1] word0 word1" in llm.calls[0]["messages"][-1]["content"]


def test_empty_blocks_score_zero_without_a_call() -> None:
    blocks = [Block(0, 0, 7.5, ()), Block(1, 7.5, 15, ()), *blocks_with_text(2)]
    llm = FakeLLM()
    scores, _, _ = block_scores(llm, blocks, "en")
    assert scores == [0.0, 0.5] and llm.calls[0]["n"] == 1


def test_chapter_titles_and_spanish_prompt() -> None:
    llm = FakeLLM()
    block_scores(llm, blocks_with_text(2), "es", chapter_at=lambda t: "Intro")
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "(capítulo: Intro)" in prompt and "exactamente 1 enteros" in prompt


def test_invalid_batch_is_retried_then_succeeds() -> None:
    llm = FakeLLM(['{"scores": [1]}', json.dumps({"scores": [80, 20]})])
    scores, _, failed = block_scores(llm, blocks_with_text(4), "en")
    assert [(c["temperature"], c["seed"]) for c in llm.calls] == [(0.0, 0), (0.7, 1)]
    assert scores == [0.8, 0.2] and failed == 0


def test_exhausted_batch_leaves_its_blocks_unscored() -> None:
    llm = FakeLLM(["x", "y", "z"])  # first batch fails 3 times, second uses the default
    scores, batches, failed = block_scores(llm, blocks_with_text(42), "en")
    assert scores[:20] == [None] * 20 and scores[20] == 0.5
    assert (batches, failed) == (2, 1)


def test_window_scores_average_the_overlapping_blocks() -> None:
    assert window_llm_scores([0.2, 0.6, None], 5) == [0.2, pytest.approx(0.4), 0.6, 0.6, None]


def test_combine_weights_and_fallback() -> None:
    assert combine(0.5, 1.0) == pytest.approx(0.4 * 0.5 + 0.6 * 1.0)
    assert combine(0.5, None) == 0.5


class FakeLid:
    def __init__(self, lang: str = "en"):
        self.lang = lang

    def __call__(self, text: str) -> tuple[str, float]:
        return self.lang, 0.99


def test_scorer_end_to_end_with_fakes() -> None:
    words = [Word(i * 0.3, f"palabra{i % 7}") for i in range(200)]  # 60 s
    r = RelevanceScorer(FakeLLM(), FakeLid("es"), "v").run(words, 60.0, [(0.0, "Intro")])
    assert r.language == "es" and len(r.scores) == 7  # 8 blocks -> 7 windows
    assert all(s.llm == 0.5 and 0 <= s.score <= 1 for s in r.scores)
    assert (r.batches, r.failed_batches) == (1, 0)


def test_scorer_fails_when_every_batch_fails() -> None:
    words = [Word(i * 0.3, "word") for i in range(100)]
    with pytest.raises(RelevanceGenerationError, match="every LLM batch failed"):
        RelevanceScorer(FakeLLM(["x"] * 3), FakeLid(), "v").run(words, 30.0, [])
