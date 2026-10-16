"""Show notes: output validation/normalization, chunking, description
composition and the PATCH schema (docs/decisions/0011 §2-3)."""

import json

import pytest
from pydantic import ValidationError

from app.schemas.show_notes import ShowNotesUpdate, tags_length
from app.shownotes.chunking import ChapterMark, Seg, make_chunks
from app.shownotes.validate import (
    DraftInvalid,
    chapters_block,
    compose_description,
    cut_at_sentence,
    normalize_tags,
    normalize_titles,
    parse_draft,
)


def raw(**over) -> str:
    data = {
        "title_suggestions": ["One", "Two", "Three"],
        "description": "A good episode.",
        "tags": ["a1", "b2", "c3", "d4", "e5"],
        "summary": "Short.",
    }
    data.update(over)
    return json.dumps(data)


# --- parse_draft ---------------------------------------------------------


def test_valid_output_parses() -> None:
    d = parse_draft(raw())
    assert d.title_suggestions == ["One", "Two", "Three"]
    assert d.tags == ["a1", "b2", "c3", "d4", "e5"]


def test_think_block_and_code_fence_are_stripped() -> None:
    d = parse_draft("<think>hmm</think>\n```json\n" + raw() + "\n```")
    assert d.description == "A good episode."


@pytest.mark.parametrize(
    "text,reason",
    [
        ("not json at all", "not valid JSON"),
        ('{"title_suggestions": ["a", "b"', "not valid JSON"),  # cut off by max_tokens
        ("[1, 2]", "not a JSON object"),
        (json.dumps({"description": "x", "tags": [], "summary": "y"}), "title_suggestions"),
        (raw(summary=3), "summary"),
    ],
)
def test_hard_failures(text: str, reason: str) -> None:
    with pytest.raises(DraftInvalid, match=reason):
        parse_draft(text)


def test_too_few_titles_after_dedup_is_a_hard_failure() -> None:
    with pytest.raises(DraftInvalid, match="usable titles"):
        parse_draft(raw(title_suggestions=["Same", "same", "SAME ", "x"]))


def test_too_few_tags_after_dedup_is_a_hard_failure() -> None:
    with pytest.raises(DraftInvalid, match="usable tags"):
        parse_draft(raw(tags=["#AI", "ai", "a,i", "b", "c"]))


@pytest.mark.parametrize("field", ["description", "summary"])
def test_empty_text_fields_fail(field: str) -> None:
    with pytest.raises(DraftInvalid, match=f"empty {field}"):
        parse_draft(raw(**{field: "   "}))


def test_titles_are_cleaned_deduped_capped() -> None:
    titles = ['1. "First title"', "first title", "- Second", "x" * 150, "Third", "Fourth", "Fifth", "Sixth"]
    out = normalize_titles(titles)
    assert out[:2] == ["First title", "Second"]
    assert len(out[2]) == 100
    assert len(out) == 5


def test_tags_are_normalized_and_pruned_to_500_chars() -> None:
    assert normalize_tags(["#Podcast", "podcast", " Deep  Learning ", "a,b", "<x>"]) == [
        "podcast", "deep learning", "a b", "x",
    ]
    long = [f"tag number {i:02d} " + "x" * 30 for i in range(15)]
    out = normalize_tags(long)
    assert tags_length(out) <= 500 and out == [t.lower().strip() for t in long[: len(out)]]


def test_long_description_is_cut_at_a_sentence() -> None:
    text = ("This is a sentence. " * 100).strip()
    d = parse_draft(raw(description=text, summary="One. " * 200))
    assert len(d.description) <= 1500 and d.description.endswith(".")
    assert len(d.summary) <= 600 and d.summary.endswith(".")


def test_angle_brackets_are_removed_from_description() -> None:
    assert parse_draft(raw(description="Use <b>bold</b> here.")).description == "Use bbold/b here."


def test_cut_without_sentence_end_cuts_on_a_word() -> None:
    out = cut_at_sentence("word " * 100, 50)
    assert len(out) <= 50 and out.endswith("…") and "wor…" not in out


def test_tags_length_counts_commas_and_quotes() -> None:
    assert tags_length([]) == 0
    assert tags_length(["ab", "c d"]) == 2 + 5 + 1


# --- description composition ---------------------------------------------


def test_chapters_block_without_and_with_hours() -> None:
    assert chapters_block([(0.0, "Intro"), (75.4, "Topic")]) == "00:00 Intro\n01:15 Topic"
    assert chapters_block([(0.0, "Intro"), (3725.0, "Late")]) == "0:00:00 Intro\n1:02:05 Late"


def test_compose_description_appends_the_chapter_block() -> None:
    assert compose_description("Text.", []) == "Text."
    assert compose_description("Text.", [(0.0, "Intro"), (90.0, "Main")]) == "Text.\n\n00:00 Intro\n01:30 Main"


def test_compose_description_stays_within_youtube_limit() -> None:
    chapters = [(60.0 * i, "t" * 90) for i in range(50)]
    out = compose_description("Sentence here. " * 100, chapters)
    assert len(out) <= 5000 and out.endswith(chapters_block(chapters))


# --- chunking ---------------------------------------------------------------


def segs(n: int, words: int = 10, step: float = 5.0) -> list[Seg]:
    return [Seg(step * i, step * i + step, " ".join(f"w{i}_{j}" for j in range(words))) for i in range(n)]


def test_fixed_blocks_cut_on_segment_boundaries() -> None:
    chunks = make_chunks(segs(10), max_words=35)
    assert [len(c.text.split()) for c in chunks] == [30, 30, 30, 10]
    assert [c.start for c in chunks] == [0.0, 15.0, 30.0, 45.0]
    assert all(c.title is None for c in chunks)


def test_a_huge_segment_is_split_on_words() -> None:
    chunks = make_chunks([Seg(0.0, 100.0, " ".join(["w"] * 25))], max_words=10)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 5]


def test_one_chunk_per_chapter_long_ones_split() -> None:
    s = segs(12)
    chunks = make_chunks(s, [ChapterMark(0.0, "A"), ChapterMark(20.0, "B")], max_words=50)
    assert [(c.title, len(c.text.split())) for c in chunks] == [("A", 40), ("B", 50), ("B", 30)]
    assert chunks[1].start == 20.0


def test_text_before_the_first_chapter_goes_to_it() -> None:
    chunks = make_chunks(segs(4), [ChapterMark(10.0, "Only")], max_words=1000)
    assert len(chunks) == 1 and chunks[0].start == 0.0


def test_empty_segments_are_skipped() -> None:
    assert make_chunks([Seg(0, 1, "  ")]) == []


# --- PATCH schema -----------------------------------------------------------


def test_update_accepts_a_partial_edit() -> None:
    u = ShowNotesUpdate(tags=[" a ", "", "b c"])
    assert u.model_dump(exclude_unset=True) == {"tags": ["a", "b c"]}


@pytest.mark.parametrize(
    "body",
    [
        {"title_suggestions": []},
        {"title_suggestions": ["x" * 101]},
        {"title_suggestions": ["a"] * 6},
        {"tags": ["a,b"]},
        {"tags": ["x" * 40] * 13},
        {"description": "x" * 5001},
        {"description": "a <b>"},
        {"summary": ""},
        {"unknown": 1},
    ],
)
def test_update_rejects_out_of_limit_edits(body: dict) -> None:
    with pytest.raises(ValidationError):
        ShowNotesUpdate(**body)
