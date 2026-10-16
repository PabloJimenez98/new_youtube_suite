"""Heuristic relevance features per window, English and Spanish.

The four spec §2.4 bootstrap features, each in [0, 1]:
topic anchor, question -> answer, speaker-change density and (inverted)
filler ratio. Their weighted mean is the window's heuristic score. See
docs/decisions/0012-relevance-v0.md §2.
"""

import math
import re
from collections import Counter

from app.relevance.windows import Window

WEIGHTS = {"topic": 0.35, "qa": 0.25, "speakers": 0.20, "fillers": 0.20}
TOP_KEYWORDS = 30
MIN_WORDS = 5
ANSWER_WORDS = 8
SPEAKER_CHANGES_PER_MIN_CAP = 6.0
FILLER_RATIO_CAP = 0.15

_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

STOPWORDS = {
    "en": set(
        "a about above after again against all also am an and any are as at be because been before being "
        "below between both but by can could did do does doing down during each few for from further get "
        "got had has have having he her here hers herself him himself his how i if in into is it its itself "
        "just know like me more most my myself no nor not now of off on once only or other our ours "
        "ourselves out over own really right same she should so some such than that the their theirs them "
        "themselves then there these they this those through to too under until up very was we were what "
        "when where which while who whom why will with would yeah you your yours yourself yourselves "
        "gonna going thing things think okay well yes said says say one two lot something actually".split()
    ),
    "es": set(
        "a al algo algunas algunos ante antes aquí así aunque bien bueno cada casi como con contra cual "
        "cuando de del desde donde dos el ella ellas ellos en entonces entre era eran es esa esas ese eso "
        "esos esta estaba estamos están estar este esto estos estoy fue fueron ha había hace hacer han "
        "has hasta hay he la las le les lo los mas me mi mis mucho muy más nada ni no nos nosotros o os "
        "otra otro para pero poco por porque pues que qué se sea ser si sido sin sobre son su sus también "
        "tan tanto te tengo tiene tienen todo todos tu tú un una uno unos usted va vale vamos voy ya yo "
        "dice digo decir cosa cosas tal ver sé sí claro creo ahora tener era eh".split()
    ),
}

FILLERS = {
    "en": ({"um", "uh", "uhm", "erm", "er", "ah", "hmm", "mm", "mhm"}, ["you know", "i mean", "sort of", "kind of"]),
    "es": ({"eh", "em", "ehm", "mm", "mmm", "hmm", "mhm"}, ["o sea", "en plan", "digamos", "vamos a ver"]),
}

def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _lang(lang: str) -> str:
    return lang if lang in STOPWORDS else "en"


def episode_keywords(windows: list[Window], lang: str, k: int = TOP_KEYWORDS) -> set[str]:
    """Top-k recurring content words: total count x IDF across windows, stopwords out."""
    stop = STOPWORDS[_lang(lang)]
    docs = [[t for t in tokens(w.text) if len(t) >= 4 and t not in stop] for w in windows]
    n = len(docs)
    if n == 0:
        return set()
    df = Counter(t for d in docs for t in set(d))
    total = Counter(t for d in docs for t in d)
    # A keyword has to recur (df >= 2) without being everywhere.
    scored = {t: c * math.log(n / df[t]) for t, c in total.items() if df[t] >= 2 and df[t] < n}
    return {t for t, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}


def topic_density(window: Window, keywords: set[str]) -> float:
    toks = tokens(window.text)
    return sum(t in keywords for t in toks) / len(toks) if toks else 0.0


def question_answer(window: Window, following: Window | None) -> float:
    """1 when a question is followed by >= ANSWER_WORDS words (here or in the next window), else partial.

    A question is a sentence ending in "?": Whisper punctuates questions
    in both English and Spanish ("¿...?").
    """
    text = window.text
    after_window = len(tokens(following.text)) if following else 0
    best = 0.0
    for m in re.finditer(r"[^.!?]*[^\W_][^.!?]*\?", text):
        answer = len(tokens(text[m.end():])) + after_window
        best = max(best, min(1.0, answer / ANSWER_WORDS))
    return best


def speaker_changes(window: Window) -> float:
    labels = [w.speaker for w in window.words if w.speaker]
    changes = sum(a != b for a, b in zip(labels, labels[1:]))
    minutes = max(window.end - window.start, 1e-6) / 60
    return min(1.0, changes / minutes / SPEAKER_CHANGES_PER_MIN_CAP)


def filler_score(window: Window, lang: str) -> float:
    """1 - filler ratio (saturating at FILLER_RATIO_CAP): fewer fillers, higher score."""
    singles, phrases = FILLERS[_lang(lang)]
    toks = tokens(window.text)
    if not toks:
        return 0.0
    joined = " " + " ".join(toks) + " "
    count = sum(t in singles for t in toks) + sum(joined.count(f" {p} ") * len(p.split()) for p in phrases)
    return 1.0 - min(1.0, (count / len(toks)) / FILLER_RATIO_CAP)


def heuristic_scores(windows: list[Window], lang: str) -> list[float]:
    """The weighted mean of the features per window; the speaker feature only with diarization."""
    if not windows:
        return []
    keywords = episode_keywords(windows, lang)
    densities = [topic_density(w, keywords) for w in windows]
    top = max(densities) or 1.0
    diarized = any(w.speaker for win in windows for w in win.words)
    weights = {k: v for k, v in WEIGHTS.items() if diarized or k != "speakers"}
    norm = sum(weights.values())

    out = []
    for i, w in enumerate(windows):
        if len(tokens(w.text)) < MIN_WORDS:
            out.append(0.0)
            continue
        features = {
            "topic": densities[i] / top,
            "qa": question_answer(w, windows[i + 1] if i + 1 < len(windows) else None),
            "fillers": filler_score(w, lang),
        }
        if diarized:
            features["speakers"] = speaker_changes(w)
        out.append(sum(weights[k] * features[k] for k in weights) / norm)
    return out
