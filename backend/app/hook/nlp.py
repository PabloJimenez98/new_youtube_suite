"""spaCy and emotion models for the hook scorer, loaded lazily once per process.

CPU only; no GPU lock (docs/decisions/0020 §5). A missing model raises
`HookModelMissingError` with the install step.
"""

import threading

from app.config import get_settings

SPACY_MODELS = {"es": "es_core_news_sm", "en": "en_core_web_sm"}
EMOTION_LABELS = ("anger", "fear", "joy", "sadness")

_lock = threading.Lock()
_spacy: dict = {}
_emotion = None


class HookModelMissingError(Exception):
    """A model the hook scorer needs isn't installed or can't be loaded."""


def spacy_model(lang: str):
    name = SPACY_MODELS[lang]
    with _lock:
        if name not in _spacy:
            import spacy

            try:
                _spacy[name] = spacy.load(name)
            except OSError as exc:
                raise HookModelMissingError(
                    f"spaCy model {name} is not installed -- pip install -r requirements.txt (see README)"
                ) from exc
        return _spacy[name]


def emotion_classifier():
    global _emotion
    with _lock:
        if _emotion is None:
            from transformers import pipeline

            name = get_settings().hook_emotion_model
            try:
                _emotion = pipeline("text-classification", model=name, top_k=None, device=-1)
            except (OSError, ValueError) as exc:  # HFValidationError is a ValueError
                raise HookModelMissingError(
                    f"Emotion model {name} could not be loaded -- check HOOK_EMOTION_MODEL and the network "
                    "for the first download (see README)"
                ) from exc
        return _emotion


def emotion(text: str) -> dict[str, float]:
    """{anger, fear, joy, sadness} probabilities; all 0 for empty text."""
    if not text.strip():
        return dict.fromkeys(EMOTION_LABELS, 0.0)
    out = emotion_classifier()(text[:2000], truncation=True)
    rows = out[0] if out and isinstance(out[0], list) else out
    probs = {r["label"]: round(float(r["score"]), 4) for r in rows}
    return {label: probs.get(label, 0.0) for label in EMOTION_LABELS}


def require(lang: str) -> None:
    """Load everything one job needs, so a missing model fails the job once."""
    spacy_model(lang)
    emotion_classifier()
