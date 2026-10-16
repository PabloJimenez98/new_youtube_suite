"""Show notes from the transcript with the local LLM (docs/decisions/0011).

Map-reduce over a 4k context:

1. a transcript that fits the reduce budget goes straight to the reduce
   step; otherwise it is chunked (`chunking.make_chunks`) and each
   chunk is summarized into bullet notes (map)
2. notes that still don't fit are summarized again in groups (one extra
   level), then cut to fit
3. reduce: one grammar-constrained JSON call, validated and normalized
   (`validate.parse_draft`); a hard failure is retried up to 3 attempts
   in all (greedy, then sampled with seeds 1 and 2)

The model and LID come from the chapters bundle (`CHAPTERS_MODEL_DIR`,
manifest `titles.llm` / `titles.lid`), and the `Llama` is the
process-wide one from `app.llm.runtime`. Everything is synchronous and
CPU-only: callers use `asyncio.to_thread`.
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from app.chapters.llm_titles import sample_text, wrong_language
from app.chapters.text import LanguageId
from app.config import get_settings
from app.llm import runtime
from app.llm.chat import ChatLLM, LlamaChat
from app.shownotes.chunking import ChapterMark, Seg, make_chunks
from app.shownotes.prompts import DRAFT_JSON_SCHEMA, PROMPT_VERSION, map_messages, reduce_messages
from app.shownotes.validate import Draft, DraftInvalid, compose_description, parse_draft

log = logging.getLogger(__name__)

CHUNK_WORDS = 1600
MAP_INPUT_TOKENS = 2600
MAP_MAX_TOKENS = 256
REDUCE_INPUT_TOKENS = 2400
REDUCE_MAX_TOKENS = 900
REDUCE_ATTEMPTS = ((0.0, 0), (0.7, 1), (0.7, 2))  # (temperature, seed)


class ShowNotesUnavailableError(Exception):
    """No usable LLM is configured."""


class ShowNotesGenerationError(Exception):
    """The model gave no valid draft after every attempt."""


@dataclass(frozen=True)
class ShowNotes:
    draft: Draft  # description already includes the chapter block
    language: str


class ShowNotesGenerator:
    def __init__(self, llm: ChatLLM, lid, model_name: str):
        self.llm, self.lid = llm, lid
        self.model_name = model_name  # -> show_notes_drafts.llm_model_used

    def fit(self, text: str, budget: int) -> str:
        """`text`, cut (head + spread windows) until it fits `budget` tokens."""
        words = len(text.split())
        while words > 50 and self.llm.count_tokens(text) > budget:
            words = int(words * 0.85)
            text = sample_text(text, max_words=words)
        return text

    def summarize(self, text: str, lang: str, title: str | None = None) -> str:
        """Bullet notes for one chunk; an empty answer is retried once, then the chunk's head stands in."""
        msgs = map_messages(self.fit(text, MAP_INPUT_TOKENS), lang, title)
        for temperature, seed in ((0.0, 0), (0.7, 1)):
            notes = self.llm.complete(msgs, max_tokens=MAP_MAX_TOKENS, temperature=temperature, seed=seed).strip()
            if notes:
                return notes
        log.warning("show notes: empty map summary twice, using the chunk's head")
        return " ".join(text.split()[:120])

    def material(self, segs: list[Seg], chapters: list[ChapterMark], lang: str) -> tuple[str, bool]:
        """(text for the reduce prompt, whether it is the raw transcript)."""
        transcript = " ".join(s.text.strip() for s in segs if s.text.strip())
        if self.llm.count_tokens(transcript) <= REDUCE_INPUT_TOKENS:
            return transcript, True
        notes = [
            (f"## {c.title}\n" if c.title else "") + self.summarize(c.text, lang, c.title)
            for c in make_chunks(segs, chapters, CHUNK_WORDS)
        ]
        joined = "\n\n".join(notes)
        if self.llm.count_tokens(joined) > REDUCE_INPUT_TOKENS:
            # One more level: summarize the notes in groups that fit a map prompt.
            groups: list[list[str]] = [[]]
            for n in notes:
                if groups[-1] and self.llm.count_tokens("\n\n".join([*groups[-1], n])) > MAP_INPUT_TOKENS:
                    groups.append([])
                groups[-1].append(n)
            joined = "\n\n".join(self.summarize("\n\n".join(g), lang) for g in groups)
        return self.fit(joined, REDUCE_INPUT_TOKENS), False

    def reduce(self, material: str, lang: str, chapter_titles: list[str], from_transcript: bool) -> Draft:
        msgs = reduce_messages(material, lang, chapter_titles, from_transcript)
        reasons = []
        for attempt, (temperature, seed) in enumerate(REDUCE_ATTEMPTS, 1):
            raw = self.llm.complete(msgs, max_tokens=REDUCE_MAX_TOKENS, temperature=temperature, seed=seed,
                                    json_schema=DRAFT_JSON_SCHEMA)
            try:
                draft = parse_draft(raw)
                if wrong_language(draft.description, lang, self.lid):
                    raise DraftInvalid(f"description isn't in {lang!r}")
                return draft
            except DraftInvalid as e:
                reasons.append(str(e))
                log.warning("show notes: attempt %d invalid (%s): %.300r", attempt, e, raw)
        raise ShowNotesGenerationError(
            f"Show notes: model returned no valid draft after {len(REDUCE_ATTEMPTS)} attempts "
            f"({'; '.join(reasons)})"
        )

    def run(self, segs: list[Seg], chapters: list[ChapterMark]) -> ShowNotes:
        text = " ".join(s.text for s in segs)
        detected, _ = self.lid(text[:5000])
        lang = "es" if detected == "es" else "en"
        material, from_transcript = self.material(segs, chapters, lang)
        draft = self.reduce(material, lang, [c.title for c in chapters], from_transcript)
        description = compose_description(draft.description, [(c.start, c.title) for c in chapters])
        return ShowNotes(Draft(draft.title_suggestions, description, draft.tags, draft.summary), lang)


_generator: ShowNotesGenerator | None = None
_lock = threading.Lock()


def get_generator() -> ShowNotesGenerator:
    """Build the generator once per process from the chapters bundle's LLM and LID."""
    global _generator
    with _lock:
        if _generator is None:
            settings = get_settings()
            bundle = settings.chapters_model_dir
            manifest = Path(bundle) / "manifest.json" if bundle else None
            titles = json.loads(manifest.read_text()).get("titles", {}) if manifest and manifest.is_file() else {}
            if "llm" not in titles or "lid" not in titles:
                raise ShowNotesUnavailableError(
                    "No LLM configured for show notes -- set CHAPTERS_MODEL_DIR to a chapter bundle "
                    "with an LLM (see README, 'Chapter models')"
                )
            path = Path(bundle) / titles["llm"]["file"]
            n_ctx = titles["llm"].get("n_ctx", 4096)
            llm = LlamaChat(runtime.get_llama(path, n_ctx, settings.chapters_llm_threads), n_ctx)
            _generator = ShowNotesGenerator(llm, LanguageId(Path(bundle) / titles["lid"]["file"]),
                                            f"{path.stem}+{PROMPT_VERSION}")
        return _generator


def generate_show_notes(segs: list[Seg], chapters: list[ChapterMark]) -> tuple[str, ShowNotes]:
    """(llm_model_used, show notes). Synchronous -- run in a thread."""
    g = get_generator()
    return g.model_name, g.run(segs, chapters)
