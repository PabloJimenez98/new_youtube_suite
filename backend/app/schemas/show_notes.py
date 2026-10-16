"""Show notes API schemas (M6, docs/decisions/0011 §3).

The limits here are YouTube's and hold for every stored draft,
generated or edited. The generator is stricter on top (at least 3
titles and 5 tags, see `app/shownotes/validate.py`); an edit may go
below those minimums.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TITLE_CHARS = 100
MAX_TITLES = 5
MAX_DESCRIPTION_CHARS = 5000
MAX_TAGS = 15
MAX_TAGS_CHARS = 500
MAX_SUMMARY_CHARS = 600


def tags_length(tags: list[str]) -> int:
    """Characters YouTube counts against its 500 limit.

    Tags are joined with commas, and a tag containing a space is quoted
    (+2).
    """
    return sum(len(t) + (2 if " " in t else 0) for t in tags) + max(len(tags) - 1, 0)


class ShowNotesUpdate(BaseModel):
    """`PATCH` body: any subset of the four fields."""

    model_config = ConfigDict(extra="forbid")

    title_suggestions: list[str] | None = Field(default=None, min_length=1, max_length=MAX_TITLES)
    description: str | None = Field(default=None, min_length=1, max_length=MAX_DESCRIPTION_CHARS)
    tags: list[str] | None = Field(default=None, max_length=MAX_TAGS)
    summary: str | None = Field(default=None, min_length=1, max_length=MAX_SUMMARY_CHARS)

    @field_validator("title_suggestions")
    @classmethod
    def _titles(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        v = [t.strip() for t in v]
        if any(not t or len(t) > MAX_TITLE_CHARS for t in v):
            raise ValueError(f"each title must be 1-{MAX_TITLE_CHARS} characters")
        return v

    @field_validator("tags")
    @classmethod
    def _tags(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        v = [t.strip() for t in v if t.strip()]
        if any("," in t or "<" in t or ">" in t for t in v):
            raise ValueError("tags can't contain commas or angle brackets")
        if tags_length(v) > MAX_TAGS_CHARS:
            raise ValueError(f"tags exceed YouTube's {MAX_TAGS_CHARS}-character limit")
        return v

    @field_validator("description")
    @classmethod
    def _description(cls, v: str | None) -> str | None:
        if v is not None and ("<" in v or ">" in v):
            raise ValueError("YouTube descriptions can't contain angle brackets")
        return v


class ShowNotesOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title_suggestions: list[str]
    description: str
    tags: list[str]
    summary: str
    language: str
    llm_model_used: str
    created_at: datetime
    updated_at: datetime
