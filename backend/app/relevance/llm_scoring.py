"""Zero-shot LLM relevance, batched (docs/decisions/0012-relevance-v0.md §3).

The LLM rates non-overlapping 15 s blocks (pairs of 7.5 s base blocks),
20 per call, as grammar-constrained JSON `{"scores": [int x N]}`. A
window (base blocks i, i+1) overlaps at most two LLM blocks; its score
is their overlap-weighted mean. A batch that fails every attempt leaves
its blocks unscored (None): the window then falls back to the
heuristic alone.
"""

import json
import logging

from app.llm.chat import ChatLLM
from app.relevance.windows import Block

log = logging.getLogger(__name__)

PROMPT_VERSION = "relevance-p1"
BATCH = 20
MAX_BLOCK_WORDS = 80
ATTEMPTS = ((0.0, 0), (0.7, 1), (0.7, 2))  # (temperature, seed), as 0011 §4

_PROMPTS = {
    "en": (
        "You rate moments of a podcast episode for short-form video clips.",
        "Below are {n} consecutive 15-second excerpts of a podcast episode, numbered.\n\n{blocks}\n\n"
        "Rate each excerpt from 0 to 100 for how well it would work as a standalone YouTube Short: "
        "interesting, self-contained and engaging (funny, insightful, surprising, controversial or "
        "emotional). Filler, logistics, sponsor reads and fragments that make no sense alone score low.\n"
        'Reply with JSON only: {{"scores": [...]}}, exactly {n} integers, in order.',
        "chapter",
    ),
    "es": (
        "Valoras momentos de un episodio de podcast para clips de vídeo cortos.",
        "Estos son {n} fragmentos consecutivos de 15 segundos de un episodio de podcast, numerados.\n\n"
        "{blocks}\n\n"
        "Puntúa cada fragmento de 0 a 100 según lo bien que funcionaría como YouTube Short independiente: "
        "interesante, autocontenido y atractivo (gracioso, revelador, sorprendente, polémico o emotivo). "
        "El relleno, la logística, la publicidad y los fragmentos que no se entienden solos puntúan bajo.\n"
        'Responde solo con JSON: {{"scores": [...]}}, exactamente {n} enteros, en orden.',
        "capítulo",
    ),
}


class ScoresInvalid(ValueError):
    pass


def schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {"scores": {"type": "array", "items": {"type": "integer"}, "minItems": n, "maxItems": n}},
        "required": ["scores"],
        "additionalProperties": False,
    }


def llm_blocks(blocks: list[Block]) -> list[tuple[str, float]]:
    """Base blocks paired into 15 s LLM blocks: (text, start)."""
    out = []
    for j in range(0, len(blocks), 2):
        pair = blocks[j : j + 2]
        words = [w.text.strip() for b in pair for w in b.words if w.text.strip()]
        out.append((" ".join(words[:MAX_BLOCK_WORDS]), pair[0].start))
    return out


def messages(texts: list[str], lang: str, chapters: list[str | None]) -> list[dict]:
    system, user, chapter_word = _PROMPTS[lang if lang in _PROMPTS else "en"]
    lines = [
        f"[{i}]" + (f" ({chapter_word}: {c})" if c else "") + f" {t}"
        for i, (t, c) in enumerate(zip(texts, chapters, strict=True), 1)
    ]
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(n=len(texts), blocks="\n".join(lines))}]


def parse_scores(raw: str, n: int) -> list[float]:
    """Raw output -> n scores in [0, 1]. Out-of-range integers are clamped (local repair)."""
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise ScoresInvalid(f"not valid JSON ({e.msg})") from None
    scores = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(scores, list) or len(scores) != n:
        raise ScoresInvalid(f"expected {n} scores, got {len(scores) if isinstance(scores, list) else 'none'}")
    if not all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in scores):
        raise ScoresInvalid("non-numeric score")
    return [min(max(float(s), 0.0), 100.0) / 100.0 for s in scores]


def score_batch(llm: ChatLLM, texts: list[str], lang: str, chapters: list[str | None]) -> list[float] | None:
    msgs = messages(texts, lang, chapters)
    for attempt, (temperature, seed) in enumerate(ATTEMPTS, 1):
        raw = llm.complete(msgs, max_tokens=8 * len(texts) + 24, temperature=temperature, seed=seed,
                           json_schema=schema(len(texts)))
        try:
            return parse_scores(raw, len(texts))
        except ScoresInvalid as e:
            log.warning("relevance: batch attempt %d invalid (%s): %.200r", attempt, e, raw)
    return None


def block_scores(llm: ChatLLM, blocks: list[Block], lang: str,
                 chapter_at=lambda start: None) -> tuple[list[float | None], int, int]:
    """Score every 15 s LLM block: (scores, batches run, batches failed). Empty blocks score 0 without a call."""
    pairs = llm_blocks(blocks)
    scores: list[float | None] = [0.0 if not t else None for t, _ in pairs]
    todo = [j for j, (t, _) in enumerate(pairs) if t]
    batches = failed = 0
    for k in range(0, len(todo), BATCH):
        idx = todo[k : k + BATCH]
        batches += 1
        result = score_batch(llm, [pairs[j][0] for j in idx], lang, [chapter_at(pairs[j][1]) for j in idx])
        if result is None:
            failed += 1
            continue
        for j, s in zip(idx, result, strict=True):
            scores[j] = s
    return scores, batches, failed


def window_llm_scores(block_llm: list[float | None], n_windows: int) -> list[float | None]:
    """Window i covers base blocks i, i+1: LLM block i/2 when i is even, else the mean of its two neighbours."""
    out: list[float | None] = []
    for i in range(n_windows):
        if i % 2 == 0:
            out.append(block_llm[i // 2] if i // 2 < len(block_llm) else None)
            continue
        parts = [block_llm[j] for j in ((i - 1) // 2, (i + 1) // 2) if j < len(block_llm) and block_llm[j] is not None]
        out.append(sum(parts) / len(parts) if parts else None)
    return out
