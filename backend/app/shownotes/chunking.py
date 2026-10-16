"""Transcript -> chunks for the map step (docs/decisions/0011 §2).

One chunk per chapter when chapters exist, otherwise fixed-size blocks.
Every chunk is cut on segment boundaries and holds at most `max_words`
words. A single segment longer than that is split on words.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Seg:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ChapterMark:
    start: float
    title: str


@dataclass(frozen=True)
class Chunk:
    start: float
    text: str
    title: str | None = None


def word_count(text: str) -> int:
    return len(text.split())


def _split(segs: list[Seg], max_words: int, title: str | None) -> list[Chunk]:
    chunks: list[Chunk] = []
    words: list[str] = []
    start: float | None = None

    def flush() -> None:
        nonlocal words, start
        if words:
            chunks.append(Chunk(start if start is not None else 0.0, " ".join(words), title))
        words, start = [], None

    for seg in segs:
        seg_words = seg.text.split()
        if not seg_words:
            continue
        if words and len(words) + len(seg_words) > max_words:
            flush()
        if start is None:
            start = seg.start
        while len(seg_words) > max_words - len(words):
            take = max_words - len(words)
            words.extend(seg_words[:take])
            seg_words = seg_words[take:]
            flush()
            start = seg.start
        words.extend(seg_words)
    flush()
    return chunks


def make_chunks(segs: list[Seg], chapters: list[ChapterMark] | None = None, max_words: int = 1600) -> list[Chunk]:
    if not chapters:
        return _split(segs, max_words, None)
    marks = sorted(chapters, key=lambda c: c.start)
    chunks: list[Chunk] = []
    for i, mark in enumerate(marks):
        # The first chapter also takes anything before it (it should start at 0).
        lo = mark.start if i else float("-inf")
        hi = marks[i + 1].start if i + 1 < len(marks) else float("inf")
        chunks.extend(_split([s for s in segs if lo <= s.start < hi], max_words, mark.title))
    return chunks
