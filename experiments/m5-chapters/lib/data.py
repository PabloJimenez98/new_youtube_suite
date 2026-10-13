"""VidChapters-7M subset selection (docs/decisions/0009 §1).

Source files, under `$YTS_EXPERIMENT_DATA/m5-chapters/raw/`, from the
Chapter-Llama mirror (HF `lucas-ventura/chapter-llama`, MIT):
- `chapters.json` (2 GB): {vid: {duration, title, description,
  channel_id, view_count, chapters: {"<sec>": "<title>"}}}
- `asrs.json` (22 GB): {vid: {"text": [...], "start": [...], "end": [...]}}
- `split_{train,val,test}.json`: VidChapters' official video-id lists.

Neither big file fits in RAM; both are streamed with `ijson`.

Pipeline: `iter_chapters` -> `parse_chapters` (validation) ->
`is_candidate` (duration/chapter-count) -> language ID on titles ->
`assign_split` -> `sample_english` -> `iter_asr` keeps only the chosen
videos -> language re-checked on the ASR text itself.
"""

import hashlib
import json
import os
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import ijson

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def data_root() -> Path:
    """`$YTS_EXPERIMENT_DATA/m5-chapters` (set in experiments/.env)."""
    root = os.environ.get("YTS_EXPERIMENT_DATA")
    if not root:
        raise RuntimeError("YTS_EXPERIMENT_DATA is not set -- see experiments/.env.example")
    return Path(root) / "m5-chapters"


# --- chapters ---------------------------------------------------------------


@dataclass(frozen=True)
class Criteria:
    min_minutes: float = 15.0
    max_minutes: float = 120.0
    min_chapters: int = 3
    # A "chapter" shorter than this is usually a timestamp list of
    # moments, not topic chapters.
    min_chapter_s: float = 10.0


def iter_chapters(path: Path) -> Iterator[tuple[str, dict]]:
    with open(path, "rb") as f:
        yield from ijson.kvitems(f, "", use_float=True)


def parse_chapters(record: dict, duration: float | None = None) -> list[tuple[float, str]] | None:
    """Sorted (start, title) pairs, or None if the annotation is unusable.

    Unusable: missing/empty chapters, non-numeric or duplicate starts,
    or a start at/after the end of the video.
    """
    duration = duration if duration is not None else record.get("duration")
    raw = record.get("chapters") or {}
    if not duration or not raw:
        return None
    try:
        pairs = sorted((float(k), str(v).strip()) for k, v in raw.items())
    except (TypeError, ValueError):
        return None
    starts = [s for s, _ in pairs]
    if len(set(starts)) != len(starts) or starts[0] < 0 or starts[-1] >= duration:
        return None
    return pairs


def is_candidate(record: dict, chapters: list[tuple[float, str]] | None, c: Criteria) -> bool:
    if chapters is None:
        return False
    minutes = record["duration"] / 60
    if not c.min_minutes <= minutes <= c.max_minutes or len(chapters) < c.min_chapters:
        return False
    bounds = [s for s, _ in chapters] + [record["duration"]]
    return all(b - a >= c.min_chapter_s for a, b in zip(bounds, bounds[1:]))


def duration_bucket(seconds: float) -> str:
    m = seconds / 60
    return "15-30" if m < 30 else "30-60" if m < 60 else "60-120"


# --- language ID ------------------------------------------------------------


class LanguageId:
    """fastText lid.176 (CC BY-SA 3.0), downloaded once next to the data."""

    def __init__(self, model_path: Path):
        import fasttext  # fasttext-numpy2-wheel

        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(LID_URL, model_path)
        self.model = fasttext.load_model(str(model_path))

    def __call__(self, text: str) -> tuple[str, float]:
        text = " ".join(text.split())[:2000]
        if not text:
            return "unk", 0.0
        labels, probs = self.model.predict(text)
        return labels[0].removeprefix("__label__"), float(probs[0])


def titles_text(record: dict) -> str:
    return " . ".join([record.get("title") or "", *(record.get("chapters") or {}).values()])


def asr_text(asr: dict, max_sentences: int = 60) -> str:
    """A sample from the middle of the transcript -- intros are often music/ads."""
    text = asr.get("text") or []
    mid = len(text) // 2
    return " ".join(text[max(0, mid - max_sentences // 2) : mid + max_sentences // 2])


# --- splits -----------------------------------------------------------------


def load_official_splits(raw_dir: Path) -> dict[str, str]:
    """vid -> 'train' | 'val' | 'test' from VidChapters' own lists."""
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for vid in json.loads((raw_dir / f"split_{split}.json").read_text()):
            out[vid] = split
    return out


def _unit_hash(key: str) -> float:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_split(
    vid: str,
    lang: str,
    channel_id: str | None,
    official: dict[str, str],
    es_test: float = 0.40,
    es_val: float = 0.20,
) -> str | None:
    """English keeps the official split. Spanish is re-split *by channel*.

    The official test set holds only a handful of Spanish videos, so
    Spanish gets its own held-out split. Grouping by channel keeps a
    channel's chaptering style from leaking between train and test.
    Only ~200 usable Spanish-speech videos exist (01_data), hence the
    large held-out shares: a 25 % test split would be ~50 videos.
    """
    if lang == "es":
        h = _unit_hash(channel_id or vid)
        return "es_test" if h < es_test else "es_val" if h < es_test + es_val else "train"
    return official.get(vid)


def sample_english(
    rows: Iterable[dict], per_bucket: dict[str, int], seed: str = "m5"
) -> list[dict]:
    """Deterministic stratified sample: at most per_bucket[split] videos
    per (split, duration bucket), taking the lowest hash(seed + vid)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["split"], r["bucket"]), []).append(r)
    out = []
    for (split, _), items in sorted(groups.items()):
        items.sort(key=lambda r: _unit_hash(seed + r["vid"]))
        out.extend(items[: per_bucket.get(split, 0)])
    return out


# --- ASR --------------------------------------------------------------------


def iter_asr(path: Path, wanted: set[str]) -> Iterator[tuple[str, dict]]:
    """Stream asrs.json, yielding only the wanted videos."""
    with open(path, "rb") as f:
        for vid, asr in ijson.kvitems(f, "", use_float=True):
            if vid in wanted:
                yield vid, asr


def asr_is_usable(asr: dict, duration: float, min_coverage: float = 0.5) -> bool:
    """Non-empty, aligned lists whose speech spans at least `min_coverage` of the video."""
    text, start, end = asr.get("text"), asr.get("start"), asr.get("end")
    if not text or not start or not end or not len(text) == len(start) == len(end):
        return False
    return (max(end) - min(start)) >= min_coverage * duration
