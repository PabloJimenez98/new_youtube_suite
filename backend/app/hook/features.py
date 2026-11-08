"""Engineered hook features from a clip's opening window (spec §2.8, docs/decisions/0020 §2).

Every function takes a parsed spaCy `Doc` (or plain text / numbers) and
is pure, so the tests can run the real ES/EN models on fixed sentences.
"""

import re
from dataclasses import asdict, dataclass, field

from app.hook.window import Opening
from app.relevance.heuristics import STOPWORDS, tokens

CURIOSITY = {
    "en": (
        "but", "until", "the reason why", "turns out", "here's why", "here is why", "what nobody",
        "no one tells you", "the truth is", "the secret", "the problem is",
    ),
    "es": (
        "pero", "hasta que", "la razón por la que", "la razón de que", "resulta que", "lo que nadie",
        "nadie te", "la verdad es que", "el secreto", "el problema es", "sin embargo",
    ),
}
_CURIOSITY_RE = {
    lang: [re.compile(rf"(?<!\w){re.escape(p)}(?!\w)") for p in phrases] for lang, phrases in CURIOSITY.items()
}

SECOND_PERSON = {
    "en": {"you", "your", "yours", "yourself", "yourselves", "ya", "y'all"},
    "es": {
        "tú", "tu", "tus", "te", "ti", "contigo", "usted", "ustedes", "vosotros", "vosotras", "os",
        "vuestro", "vuestra", "vuestros", "vuestras", "tuyo", "tuya", "tuyos", "tuyas",
    },
}
# Sentence-initial ES imperatives the small model often mis-tags (e.g. "escucha" as NOUN).
ES_IMPERATIVES = {
    "mira", "mirad", "miren", "escucha", "escuchad", "escuchen", "oye", "oíd", "oigan", "fíjate",
    "fijaos", "fíjense", "imagina", "imagínate", "imaginad", "imaginen", "piensa", "piénsalo",
    "pensad", "espera", "esperad", "esperen", "dime", "cuéntame", "atento", "atentos",
}
ES_ARTICLES = {"el", "la", "los", "las", "lo"}
ES_COMPARATIVE_SUP = {"mejor", "mejores", "peor", "peores"}

_VOWELS = {"en": "aeiouy", "es": "aeiouáéíóúü"}


@dataclass(frozen=True)
class HookFeatures:
    # Scored (docs/decisions/0020 §3).
    lead_silence_s: float
    words_per_second: float
    first_is_question: bool
    first_curiosity: int
    first_unresolved_demonstratives: int
    second_person: int
    imperatives: int
    # Weight 0: stored for M15.
    words: int
    sentences: int
    avg_sentence_words: float
    readability: float | None
    questions: int
    exclamations: int
    curiosity_connectives: int
    superlatives: int
    numerics: int
    entity_density: float
    emotion: dict[str, float] = field(default_factory=dict)
    emotion_max: float = 0.0
    title_overlap: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _alpha(tok) -> bool:
    return any(ch.isalpha() for ch in tok.text)


def lead_silence(opening: Opening) -> float:
    return round(opening.length_s if opening.first_word_offset is None else max(opening.first_word_offset, 0.0), 3)


def words_per_second(opening: Opening) -> float:
    return round(opening.word_count / opening.length_s, 3) if opening.length_s > 0 else 0.0


def is_question(sentence: str) -> bool:
    s = sentence.strip()
    return s.endswith("?") or "¿" in s


def punct_count(text: str, close: str, open_: str) -> int:
    """`?` (or `!`) marks, counting a Spanish ¿…? pair once."""
    return max(text.count(close), text.count(open_))


def curiosity(text: str, lang: str) -> int:
    low = text.lower()
    return sum(len(rx.findall(low)) for rx in _CURIOSITY_RE[lang])


def unresolved_demonstratives(span) -> int:
    """Demonstrative *pronouns* ("this", "eso"), not determiners ("este libro")."""
    return sum(1 for t in span if t.pos_ == "PRON" and "Dem" in t.morph.get("PronType"))


def _is_imperative(tok, lang: str, sentence_initial: bool) -> bool:
    if lang == "es":
        # The lexicon only counts at a clause start ("Mira, escucha"), so
        # "él piensa" (3rd-person indicative) doesn't match.
        after_punct = tok.i > 0 and tok.doc[tok.i - 1].is_punct
        return "Imp" in tok.morph.get("Mood") or ((sentence_initial or after_punct) and tok.lower_ in ES_IMPERATIVES)
    # EN: a sentence-initial base-form verb with no subject ("Listen to this").
    return (
        sentence_initial
        and tok.tag_ == "VB"
        and not any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in tok.children)
    )


def imperatives_and_second_person(doc, lang: str) -> tuple[int, int]:
    """(imperatives, second person), each token counted once, imperatives first."""
    imp = second = 0
    for sent in doc.sents:
        first = next((t for t in sent if _alpha(t)), None)
        for tok in sent:
            if not _alpha(tok):
                continue
            if tok.pos_ in ("VERB", "AUX", "NOUN", "INTJ", "PROPN", "ADJ") and _is_imperative(tok, lang, tok.i == first.i):
                imp += 1
            elif tok.lower_ in SECOND_PERSON[lang] or (
                tok.pos_ in ("VERB", "AUX", "PRON") and "2" in tok.morph.get("Person")
            ):
                second += 1
    return imp, second


def superlatives(doc, lang: str) -> int:
    if lang == "en":
        return sum(1 for t in doc if "Sup" in t.morph.get("Degree"))
    toks = list(doc)
    n = 0
    for i, t in enumerate(toks):
        prev = toks[i - 1].lower_ if i else ""
        if "Abs" in t.morph.get("Degree") or "Sup" in t.morph.get("Degree"):
            n += 1
        elif t.lower_ in ("más", "menos") and prev in ES_ARTICLES and i + 1 < len(toks) and toks[i + 1].pos_ == "ADJ":
            n += 1
        elif t.lower_ in ES_COMPARATIVE_SUP and prev in ES_ARTICLES:
            n += 1
    return n


def numerics(doc) -> int:
    return sum(1 for t in doc if t.pos_ == "NUM" or t.like_num)


def syllables(word: str, lang: str) -> int:
    w = word.lower()
    groups = len(re.findall(rf"[{_VOWELS[lang]}]+", w))
    if lang == "en" and w.endswith("e") and not w.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(groups, 1)


def readability(words: list[str], sentences: int, lang: str) -> float | None:
    """Flesch reading ease (EN) or Fernández-Huerta (ES); None without words."""
    if not words or not sentences:
        return None
    syl_per_word = sum(syllables(w, lang) for w in words) / len(words)
    words_per_sentence = len(words) / sentences
    if lang == "es":
        return round(206.84 - 60.0 * syl_per_word - 1.02 * words_per_sentence, 2)
    return round(206.835 - 1.015 * words_per_sentence - 84.6 * syl_per_word, 2)


def content_words(text: str, lang: str) -> set[str]:
    return {t for t in tokens(text) if len(t) >= 3 and t not in STOPWORDS[lang]}


def title_overlap(text: str, title: str | None, lang: str) -> float:
    a, b = content_words(text, lang), content_words(title or "", lang)
    return round(len(a & b) / len(a | b), 4) if a and b else 0.0


def extract(opening: Opening, doc, lang: str, emotion: dict[str, float], title: str | None) -> HookFeatures:
    sents = [s for s in doc.sents if any(_alpha(t) for t in s)]
    first = sents[0] if sents else None
    words = [t.text for t in doc if _alpha(t)]
    imp, second = imperatives_and_second_person(doc, lang)
    return HookFeatures(
        lead_silence_s=lead_silence(opening),
        words_per_second=words_per_second(opening),
        first_is_question=bool(first) and is_question(first.text),
        first_curiosity=curiosity(first.text, lang) if first else 0,
        first_unresolved_demonstratives=unresolved_demonstratives(first) if first else 0,
        second_person=second,
        imperatives=imp,
        words=len(words),
        sentences=len(sents),
        avg_sentence_words=round(len(words) / len(sents), 2) if sents else 0.0,
        readability=readability(words, len(sents), lang),
        questions=punct_count(opening.text, "?", "¿"),
        exclamations=punct_count(opening.text, "!", "¡"),
        curiosity_connectives=curiosity(opening.text, lang),
        superlatives=superlatives(doc, lang),
        numerics=numerics(doc),
        entity_density=round(len(doc.ents) / len(words), 4) if words else 0.0,
        emotion=emotion,
        emotion_max=max(emotion.values(), default=0.0),
        title_overlap=title_overlap(opening.text, title, lang),
    )
