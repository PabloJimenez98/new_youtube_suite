"""Title cleaning and language ID for the LLM titles (docs/decisions/0010 §1-2).

`clean_title` is copied from experiments/m5-chapters/lib/titles.py
(drift-tested). `LanguageId` is the experiment's lib/data.py class
without the download: the model file ships in the bundle.
"""

import re
from pathlib import Path

_ENUM = re.compile(r"^\s*(?:\(?\d{1,3}[\.\)\-:]|[-•–—*#])\s*")
_PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]


def clean_title(title: str) -> str | None:
    """Light cleaning; None if nothing usable remains.

    Collapse whitespace, drop leading list enumerators ("1.", "02)",
    "- ", "•"), strip unbalanced quotes and brackets, and cap the length
    at 100 chars. A title needs at least 2 letters.
    """
    t = " ".join(str(title).split())
    t = _ENUM.sub("", t).strip()
    if t.count('"') % 2:
        t = t.replace('"', "")
    for o, c in _PAIRS:
        if t.count(o) != t.count(c):
            t = t.replace(o, "").replace(c, "")
    t = t.strip(" -–—:|,;").strip()[:100].strip()
    return t if sum(ch.isalpha() for ch in t) >= 2 else None


class LanguageId:
    """fastText lid.176 (CC BY-SA 3.0): text -> (ISO code, confidence)."""

    def __init__(self, model_path: Path):
        import fasttext  # fasttext-numpy2-wheel

        self.model = fasttext.load_model(str(model_path))

    def __call__(self, text: str) -> tuple[str, float]:
        text = " ".join(text.split())[:2000]
        if not text:
            return "unk", 0.0
        labels, probs = self.model.predict(text)
        return labels[0].removeprefix("__label__"), float(probs[0])
