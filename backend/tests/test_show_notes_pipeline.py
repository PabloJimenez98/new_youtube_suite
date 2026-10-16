"""Show notes generator with a fake LLM: map-reduce routing, the
retry/repair policy, language and the shared runtime (docs/decisions/0011)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.llm import runtime
from app.shownotes import pipeline
from app.shownotes.chunking import ChapterMark, Seg
from app.shownotes.pipeline import ShowNotesGenerationError, ShowNotesGenerator
from app.shownotes.prompts import DRAFT_JSON_SCHEMA

VALID = json.dumps({
    "title_suggestions": ["Title one", "Title two", "Title three", "Title four"],
    "description": "An episode about things.",
    "tags": ["one", "two", "three", "four", "five", "six"],
    "summary": "They talk about things.",
})


class FakeLLM:
    """One token per word; scripted reduce outputs, a fixed map summary."""

    def __init__(self, reduce_outputs: list[str] | None = None, map_outputs: list[str] | None = None):
        self.reduce_outputs = list(reduce_outputs or [VALID])
        self.map_outputs = list(map_outputs or [])
        self.calls: list[dict] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete(self, messages, *, max_tokens, temperature=0.0, seed=0, json_schema=None) -> str:
        self.calls.append(dict(messages=messages, max_tokens=max_tokens, temperature=temperature, seed=seed,
                               json_schema=json_schema))
        if json_schema is not None:
            return self.reduce_outputs.pop(0)
        return self.map_outputs.pop(0) if self.map_outputs else "- a point\n- another point"

    @property
    def reduce_calls(self) -> list[dict]:
        return [c for c in self.calls if c["json_schema"] is not None]

    @property
    def map_calls(self) -> list[dict]:
        return [c for c in self.calls if c["json_schema"] is None]


class FakeLid:
    """The episode language, except text containing "WRONG" reads as confidently the other one."""

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def __call__(self, text: str) -> tuple[str, float]:
        if "WRONG" in text:
            return ("en" if self.lang == "es" else "es"), 0.99
        return self.lang, 0.99


def segs(n_words: int, per_seg: int = 50) -> list[Seg]:
    return [Seg(5.0 * i, 5.0 * i + 5, " ".join(["word"] * per_seg)) for i in range(n_words // per_seg)]


def gen(llm: FakeLLM, lang: str = "en") -> ShowNotesGenerator:
    return ShowNotesGenerator(llm, FakeLid(lang), "fake-model+shownotes-v1")


def user_prompt(call: dict) -> str:
    return call["messages"][-1]["content"]


# --- routing -----------------------------------------------------------------


def test_short_transcript_skips_the_map_step() -> None:
    llm = FakeLLM()
    notes = gen(llm).run(segs(1000), [])
    assert llm.map_calls == [] and len(llm.reduce_calls) == 1
    assert "transcript" in user_prompt(llm.reduce_calls[0])
    assert llm.reduce_calls[0]["json_schema"] == DRAFT_JSON_SCHEMA
    assert notes.draft.title_suggestions[0] == "Title one" and notes.language == "en"


def test_long_transcript_is_mapped_per_chunk_then_reduced() -> None:
    llm = FakeLLM()
    gen(llm).run(segs(5000), [])
    assert len(llm.map_calls) == 4  # 5000 words / 1600-word chunks
    assert all(c["temperature"] == 0.0 for c in llm.map_calls)
    assert "notes" in user_prompt(llm.reduce_calls[0])


def test_chapters_drive_the_chunks_and_the_description() -> None:
    llm = FakeLLM()
    chapters = [ChapterMark(0.0, "Intro"), ChapterMark(250.0, "Main topic")]
    notes = gen(llm).run(segs(5000), chapters)
    assert len(llm.map_calls) == 4  # 2500 words per chapter -> 2 chunks each
    assert 'chapter "Intro"' in user_prompt(llm.map_calls[0])
    reduce_prompt = user_prompt(llm.reduce_calls[0])
    assert "## Intro" in reduce_prompt and "Chapters: Intro; Main topic" in reduce_prompt
    assert notes.draft.description == "An episode about things.\n\n00:00 Intro\n04:10 Main topic"


def test_notes_too_long_for_the_reduce_are_summarized_again() -> None:
    long_notes = " ".join(["note"] * 700)
    llm = FakeLLM(map_outputs=[long_notes] * 10 + ["- regrouped"] * 10)
    gen(llm).run(segs(16000), [])
    first_level = 10  # 16000 words / 1600
    assert len(llm.map_calls) > first_level
    assert "- regrouped" in user_prompt(llm.reduce_calls[0])


def test_spanish_episode_gets_the_spanish_prompts() -> None:
    llm = FakeLLM()
    notes = gen(llm, "es").run(segs(5000), [])
    assert "en español" in user_prompt(llm.map_calls[0])
    assert "en español" in user_prompt(llm.reduce_calls[0])
    assert notes.language == "es"


# --- retry / repair ----------------------------------------------------------


def test_malformed_then_valid_succeeds_on_attempt_two() -> None:
    llm = FakeLLM(reduce_outputs=['{"title_suggestions": ["cut off', VALID])
    notes = gen(llm).run(segs(500), [])
    assert [(c["temperature"], c["seed"]) for c in llm.reduce_calls] == [(0.0, 0), (0.7, 1)]
    assert notes.draft.summary == "They talk about things."


def test_wrong_language_description_is_retried() -> None:
    wrong = json.loads(VALID) | {"description": "WRONG language here."}
    llm = FakeLLM(reduce_outputs=[json.dumps(wrong), VALID])
    notes = gen(llm, "es").run(segs(500), [])
    assert len(llm.reduce_calls) == 2 and notes.draft.description == "An episode about things."


def test_three_failures_raise_with_the_reasons() -> None:
    llm = FakeLLM(reduce_outputs=["nope", "[]", json.dumps({"title_suggestions": ["a"]})])
    with pytest.raises(ShowNotesGenerationError, match="no valid draft after 3 attempts") as e:
        gen(llm).run(segs(500), [])
    assert "not valid JSON" in str(e.value) and "not a JSON object" in str(e.value)
    assert [(c["temperature"], c["seed"]) for c in llm.reduce_calls] == [(0.0, 0), (0.7, 1), (0.7, 2)]


def test_empty_map_summary_is_retried_then_degrades_to_the_chunk_head() -> None:
    llm = FakeLLM(map_outputs=["", "  ", "- fine"])
    gen(llm).run(segs(3200), [])
    assert [c["temperature"] for c in llm.map_calls[:2]] == [0.0, 0.7]
    reduce_prompt = user_prompt(llm.reduce_calls[0])
    assert "word word word" in reduce_prompt and "- fine" in reduce_prompt


# --- configuration and shared runtime -------------------------------------


@pytest.fixture
def no_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_generator", None)


@pytest.mark.parametrize("titles", [None, {}, {"lid": {"file": "lid.176.ftz"}}])
def test_no_llm_in_the_bundle_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_generator,
                                         titles) -> None:
    if titles is not None:
        (tmp_path / "manifest.json").write_text(json.dumps({"titles": titles}))
    monkeypatch.setattr(pipeline.get_settings(), "chapters_model_dir", tmp_path)
    with pytest.raises(pipeline.ShowNotesUnavailableError, match="CHAPTERS_MODEL_DIR"):
        pipeline.get_generator()


def test_generator_reuses_the_process_wide_llama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 no_generator) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"titles": {
        "llm": {"file": "llm/Model-Q4.gguf", "n_ctx": 2048}, "lid": {"file": "lid.176.ftz"}}}))
    monkeypatch.setattr(pipeline.get_settings(), "chapters_model_dir", tmp_path)
    loads = []
    monkeypatch.setattr(runtime, "_cache", {})
    monkeypatch.setattr(runtime, "_load", lambda p, n, t: loads.append((p, n)) or SimpleNamespace())
    monkeypatch.setattr(runtime, "chat_tokens", lambda llm: ("tpl", "", ""))
    monkeypatch.setattr(pipeline, "LanguageId", lambda path: FakeLid())

    g = pipeline.get_generator()

    assert g.model_name == "Model-Q4+shownotes-v1"
    assert g.llm.n_ctx == 2048
    assert g.llm.llm is runtime.get_llama(tmp_path / "llm" / "Model-Q4.gguf", 2048)
    assert loads == [(tmp_path / "llm" / "Model-Q4.gguf", 2048)]  # loaded once
    assert pipeline.get_generator() is g
