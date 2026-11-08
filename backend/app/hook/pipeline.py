"""One clip's opening -> (score, feature_vector). Synchronous -- run in a thread."""

from dataclasses import dataclass

from app.hook import features, nlp, scorer
from app.hook.window import Opening


@dataclass(frozen=True)
class HookResult:
    score: float
    feature_vector: dict


def score_opening(opening: Opening, lang: str, language_fallback: bool,
                  title: str | None, title_source: str) -> HookResult:
    doc = nlp.spacy_model(lang)(opening.text)
    feats = features.extract(opening, doc, lang, nlp.emotion(opening.text), title)
    s = scorer.score(feats)
    return HookResult(
        score=s.score,
        feature_vector={
            "version": scorer.VERSION_LABEL,
            "language": lang,
            "language_fallback": language_fallback,
            "title_source": title_source,
            "window": {"text": opening.text, "length_s": round(opening.length_s, 3)},
            "features": feats.as_dict(),
            "sub_scores": s.sub_scores,
            "weights": scorer.WEIGHTS,
        },
    )
