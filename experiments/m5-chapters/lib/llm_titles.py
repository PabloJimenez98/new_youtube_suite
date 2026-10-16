"""Chapter titles from a local pretrained instruction LLM (06, plan in README).

llama.cpp through `llama-cpp-python`, CPU-sized GGUF models. The prompt
is the same for every model, so the comparison is between models, not
prompts.

- `sample_text` caps a chapter's transcript: the head, plus evenly
  spaced windows when it's longer.
- `build_messages` builds the chat prompt in the episode's language,
  with the titles already given so they don't repeat.
- `render_prompt` applies the chat template embedded in the GGUF
  (jinja), with `enable_thinking=False` so reasoning models (Qwen 3.5)
  answer directly.
- `postprocess` turns raw model output into one clean title or None.
- `LlamaTitler` runs one call per chapter: greedy, then one sampled
  retry, then None (the caller falls back to mT5).
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from lib.titles import clean_title

LANG_NAME = {"en": "English", "es": "Spanish"}

_PROMPTS = {
    "en": (
        "You write YouTube chapter titles for a podcast.",
        "Here is the transcript of one chapter:\n\n\"\"\"\n{text}\n\"\"\"\n\n"
        "{previous}"
        "Write ONE short title for this chapter, in English, of at most 8 words. "
        "Describe its main topic using only what the transcript says. Don't invent names or facts. "
        "Reply with the title only, no quotes, no numbering, no explanation.",
        "Titles already used for earlier chapters (don't repeat them): {titles}\n\n",
    ),
    "es": (
        "Escribes títulos de capítulos de YouTube para un podcast.",
        "Esta es la transcripción de un capítulo:\n\n\"\"\"\n{text}\n\"\"\"\n\n"
        "{previous}"
        "Escribe UN título corto para este capítulo, en español, de 8 palabras como máximo. "
        "Describe su tema principal usando solo lo que dice la transcripción. No inventes nombres ni datos. "
        "Responde solo con el título, sin comillas, sin numeración y sin explicaciones.",
        "Títulos ya usados en capítulos anteriores (no los repitas): {titles}\n\n",
    ),
}

_THINK = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_PREFIX = re.compile(
    r"^\s*(?:(?:chapter|cap[ií]tulo|t[ií]tulo|title|tema|topic)\s*\d*\s*[:\-–—.]\s*|\d{1,3}\s*[:.\)\-–—]\s+)",
    re.I,
)
_WRAP = "\"'“”‘’«»*_`#"


def sample_text(text: str, max_words: int = 900, n_windows: int = 4) -> str:
    """Whole text if short enough, else the head + evenly spaced windows.

    Half the budget goes to the head (where a chapter usually states its
    topic) and the rest is spread over `n_windows` windows across the
    remainder, joined with " … ".
    """
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    head = max_words // 2
    w = (max_words - head) // n_windows
    rest = words[head:]
    step = (len(rest) - w) / max(n_windows - 1, 1)
    parts = [" ".join(words[:head])]
    for k in range(n_windows):
        s = int(round(k * step))
        parts.append(" ".join(rest[s : s + w]))
    return " … ".join(parts)


def build_messages(text: str, lang: str, previous: list[str] | None = None) -> list[dict]:
    system, user, prev_tpl = _PROMPTS[lang if lang in _PROMPTS else "en"]
    prev = prev_tpl.format(titles="; ".join(previous)) if previous else ""
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(text=text, previous=prev)}]


def render_prompt(template: str, messages: list[dict], bos: str = "", eos: str = "") -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=True, enable_thinking=False,
        bos_token=bos, eos_token=eos,
    )


def collapse_repeats(title: str, max_n: int = 4) -> str:
    """Drop immediately repeated word n-grams: "a b a b a b c" -> "a b c"."""
    w = title.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, max_n + 1):
            i = 0
            while i + 2 * n <= len(w):
                if [x.lower() for x in w[i : i + n]] == [x.lower() for x in w[i + n : i + 2 * n]]:
                    del w[i + n : i + 2 * n]
                    changed = True
                else:
                    i += 1
    return " ".join(w)


def postprocess(raw: str, max_words: int = 10) -> str | None:
    """Raw LLM output -> one clean title, or None if nothing usable."""
    s = _THINK.sub("", raw)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return None
    t = lines[0].strip(_WRAP + " ")
    t = _PREFIX.sub("", t).strip(_WRAP + " ")
    t = collapse_repeats(t)
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    t = t.rstrip(".")
    return clean_title(t)


def wrong_language(title: str, lang: str, lid, min_conf: float = 0.8) -> bool:
    """True only when fastText is confident the title is in the *other* language.

    fastText misfires on short strings (see the 01 log), so an unsure
    or third-language guess is never grounds for rejection.
    """
    pred, conf = lid(title)
    return pred in LANG_NAME and pred != lang and conf >= min_conf


@dataclass
class TitleResult:
    title: str | None
    raw: list[str] = field(default_factory=list)
    seconds: float = 0.0
    prompt_tokens: int = 0
    attempts: int = 0


class LlamaTitler:
    def __init__(self, model_path: Path, n_ctx: int = 4096, n_threads: int = 6, n_gpu_layers: int = 0,
                 max_tokens: int = 32, max_words: int = 900, llm=None):
        # `llm`: an already-loaded Llama to reuse (the backend shares one
        # per process between tools, docs/decisions/0011 §1).
        if llm is None:
            from llama_cpp import Llama

            llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=n_threads,
                        n_gpu_layers=n_gpu_layers, verbose=False)
        self.llm = llm
        md = self.llm.metadata
        self.template = md.get("tokenizer.chat_template")
        if not self.template:
            raise ValueError(f"{model_path.name}: no chat template in GGUF metadata")
        tok = self.llm._model
        # Qwen's GGUF declares no BOS, and llama.cpp then reports a default
        # id (',') it never uses; only use a BOS the file actually declares.
        wants_bos = ("tokenizer.ggml.bos_token_id" in md
                     and md.get("tokenizer.ggml.add_bos_token", "true") != "false")
        self.bos = tok.token_get_text(tok.token_bos()) if wants_bos and tok.token_bos() >= 0 else ""
        self.eos = tok.token_get_text(tok.token_eos()) if tok.token_eos() >= 0 else ""
        self.max_tokens, self.max_words = max_tokens, max_words

    def prompt(self, text: str, lang: str, previous: list[str] | None = None) -> str:
        msgs = build_messages(sample_text(text, self.max_words), lang, previous)
        return render_prompt(self.template, msgs, self.bos, self.eos)

    def title(self, text: str, lang: str, previous: list[str] | None = None, lid=None) -> TitleResult:
        p = self.prompt(text, lang, previous)
        # The template already starts with BOS when the model wants one;
        # tokenize with add_bos=False so it isn't doubled.
        ids = self.llm.tokenize(p.encode(), add_bos=False, special=True)
        res = TitleResult(None, prompt_tokens=len(ids))
        t0 = time.time()
        for attempt, temp in enumerate((0.0, 0.7)):
            out = self.llm.create_completion(ids, max_tokens=self.max_tokens, temperature=temp,
                                             seed=attempt, stop=["\n\n"])
            raw = out["choices"][0]["text"]
            res.raw.append(raw)
            res.attempts = attempt + 1
            t = postprocess(raw)
            if t and not (lid and wrong_language(t, lang, lid)) and t not in (previous or []):
                res.title = t
                break
        res.seconds = time.time() - t0
        return res
