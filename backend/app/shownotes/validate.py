"""LLM output -> a normalized draft, or `DraftInvalid` (docs/decisions/0011 §3-4).

Normalization repairs what doesn't need another model call: lengths,
duplicates, quotes/enumerators, `#` on tags, excess tags. What it can't
repair (unparseable JSON, a missing field, too few titles or tags) is a
hard failure, and the caller retries the model.
"""

import json
import re
from dataclasses import dataclass

from app.chapters.text import clean_title
from app.schemas.show_notes import (
    MAX_DESCRIPTION_CHARS,
    MAX_SUMMARY_CHARS,
    MAX_TAGS,
    MAX_TAGS_CHARS,
    MAX_TITLES,
    tags_length,
)

MIN_TITLES = 3
MIN_TAGS = 5
# The LLM's own text; the chapter block is added on top (compose_description).
MAX_LLM_DESCRIPTION_CHARS = 1500

_THINK = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
_SENTENCE_END = re.compile(r"[.!?…](?=\s|$)")


class DraftInvalid(ValueError):
    """The model's output can't be repaired into a valid draft."""


@dataclass(frozen=True)
class Draft:
    title_suggestions: list[str]
    description: str
    tags: list[str]
    summary: str


def cut_at_sentence(text: str, limit: int) -> str:
    """`text` if it fits, else cut at the last sentence end within `limit` (or a word, with "…")."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [m.end() for m in _SENTENCE_END.finditer(head)]
    if ends and ends[-1] >= limit // 2:
        return head[: ends[-1]].strip()
    return head[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _strip_brackets(text: str) -> str:
    # YouTube rejects "<" and ">" in descriptions and tags.
    return text.replace("<", "").replace(">", "")


def normalize_titles(titles: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in titles:
        c = clean_title(str(t).strip().strip("\"'“”«»*").strip())
        if c and c.casefold() not in seen:
            seen.add(c.casefold())
            out.append(c)
    return out[:MAX_TITLES]


def normalize_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        c = " ".join(_strip_brackets(str(t)).replace(",", " ").lstrip("#").lower().split())
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    out = out[:MAX_TAGS]
    while out and tags_length(out) > MAX_TAGS_CHARS:
        out.pop()
    return out


def parse_draft(raw: str) -> Draft:
    """Raw model output -> normalized `Draft`. Raises `DraftInvalid` with the reason."""
    text = _FENCE.sub("", _THINK.sub("", raw)).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DraftInvalid(f"not valid JSON ({e.msg})") from None
    if not isinstance(data, dict):
        raise DraftInvalid("not a JSON object")
    fields = {"title_suggestions": list, "description": str, "tags": list, "summary": str}
    for name, kind in fields.items():
        if not isinstance(data.get(name), kind):
            raise DraftInvalid(f"missing or mistyped field {name!r}")

    titles = normalize_titles(data["title_suggestions"])
    if len(titles) < MIN_TITLES:
        raise DraftInvalid(f"{len(titles)} usable titles, need {MIN_TITLES}")
    tags = normalize_tags(data["tags"])
    if len(tags) < MIN_TAGS:
        raise DraftInvalid(f"{len(tags)} usable tags, need {MIN_TAGS}")
    description = cut_at_sentence(_strip_brackets(data["description"]), MAX_LLM_DESCRIPTION_CHARS)
    if not description:
        raise DraftInvalid("empty description")
    summary = cut_at_sentence(data["summary"], MAX_SUMMARY_CHARS)
    if not summary:
        raise DraftInvalid("empty summary")
    return Draft(titles, description, tags, summary)


def format_timestamp(seconds: float, with_hours: bool) -> str:
    """`MM:SS`, or `H:MM:SS` -- same as the frontend's `formatTimestamp`."""
    s = int(seconds)
    h, m = s // 3600, (s % 3600) // 60
    return f"{h}:{m:02d}:{s % 60:02d}" if with_hours else f"{m + h * 60:02d}:{s % 60:02d}"


def chapters_block(chapters: list[tuple[float, str]]) -> str:
    """One `timestamp title` line per chapter, ready for a YouTube description."""
    with_hours = any(start >= 3600 for start, _ in chapters)
    return "\n".join(f"{format_timestamp(start, with_hours)} {title}" for start, title in chapters)


def compose_description(text: str, chapters: list[tuple[float, str]]) -> str:
    """The LLM's description, then the chapter block when there are chapters, within YouTube's limit."""
    if not chapters:
        return text
    block = _strip_brackets(chapters_block(chapters))
    room = MAX_DESCRIPTION_CHARS - len(block) - 2
    return f"{cut_at_sentence(text, room)}\n\n{block}" if room > 0 else text
