"""A clip's opening window and the episode language (docs/decisions/0020 §1). Pure."""

from dataclasses import dataclass

from app.relevance.heuristics import STOPWORDS, tokens
from app.shorts.selection import TimedWord

WINDOW_S = 15.0
# Below this share of language-specific stopwords, the text isn't ES or EN.
MIN_STOPWORD_SHARE = 0.10

# Words only one of the two lists has, so "a", "no", "me" don't vote.
_ONLY = {
    "es": STOPWORDS["es"] - STOPWORDS["en"],
    "en": STOPWORDS["en"] - STOPWORDS["es"],
}


@dataclass(frozen=True)
class Opening:
    text: str
    word_count: int
    # Seconds from the clip start to the first word; None = no word.
    first_word_offset: float | None
    length_s: float  # min(WINDOW_S, clip length)


def opening(words: list[TimedWord], clip_start: float, clip_end: float, window_s: float = WINDOW_S) -> Opening:
    """The words whose start falls in [clip_start, min(clip_start + window_s, clip_end))."""
    end = min(clip_start + window_s, clip_end)
    inside = [w for w in words if w.start is not None and clip_start <= w.start < end]
    text = " ".join(" ".join(w.text.strip() for w in inside).split())
    count = sum(1 for w in inside if any(ch.isalnum() for ch in w.text))
    first = inside[0].start - clip_start if inside else None
    return Opening(text=text, word_count=count, first_word_offset=first, length_s=max(end - clip_start, 0.0))


def detect_language(text: str) -> tuple[str, bool]:
    """(`es`|`en`, fallback): the language with more of its own stopwords.

    `fallback` is True when neither reaches MIN_STOPWORD_SHARE of the
    tokens (another language, or too little text); the result is then `en`.
    """
    toks = tokens(text)
    if not toks:
        return "en", True
    hits = {lang: sum(t in only for t in toks) for lang, only in _ONLY.items()}
    lang = max(hits, key=lambda k: (hits[k], k == "es"))
    if hits[lang] / len(toks) < MIN_STOPWORD_SHARE:
        return "en", True
    return lang, False
