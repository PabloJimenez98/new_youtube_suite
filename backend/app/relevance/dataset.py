"""Training data for the relevance BiLSTM (docs/decisions/0024 §2, §4).

Episodes come from `relevance_pseudo_labels` (one teacher), never from
`relevance_scores`. Each episode's windows are rebuilt from its transcript
with the same code as inference, and must line up with the labels one to one.
The split is by episode, stratified by language, and frozen in `split.json`.
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.models.model_version import ModelVersion
from app.models.relevance import RelevancePseudoLabel
from app.models.show_notes import ShowNotesDraft
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.relevance.pipeline import combine
from app.relevance.windows import Window, build_blocks, build_windows

SPLIT_SEED = 15
N_VAL, N_TEST, N_AB = 7, 8, 5


class TeacherNotFoundError(Exception):
    """No usable teacher version (or the named one is a trained model)."""


@dataclass
class Episode:
    upload_id: int
    name: str
    language: str
    windows: list[Window]
    heuristic: np.ndarray  # [T]
    llm: np.ndarray  # [T], NaN where the teacher's LLM gave nothing
    emb: np.ndarray | None = field(default=None, repr=False)  # [T, D], filled by the caller

    @property
    def target(self) -> np.ndarray:
        """The teacher's combined score (p3's `score`)."""
        return np.array([combine(float(h), None if math.isnan(l) else float(l))
                         for h, l in zip(self.heuristic, self.llm, strict=True)], dtype=np.float32)

    @property
    def texts(self) -> list[str]:
        return [w.text for w in self.windows]

    @property
    def starts(self) -> np.ndarray:
        return np.array([w.start for w in self.windows], dtype=np.float32)


async def resolve_teacher(db: AsyncSession, label: str | None) -> ModelVersion:
    """The named teacher, or the teacher version with labels for the most uploads."""
    if label is not None:
        version = await db.scalar(
            select(ModelVersion).where(ModelVersion.model_name == "relevance", ModelVersion.version_label == label)
        )
        if version is None:
            raise TeacherNotFoundError(f"No relevance version {label!r}")
        if version.artifact_path is not None:
            raise TeacherNotFoundError(f"{label!r} is a trained model, not a teacher")
        return version
    row = (
        await db.execute(
            select(ModelVersion, func.count(func.distinct(RelevancePseudoLabel.upload_id)).label("n"))
            .join(RelevancePseudoLabel, RelevancePseudoLabel.teacher_version_id == ModelVersion.id)
            .where(ModelVersion.model_name == "relevance", ModelVersion.artifact_path.is_(None))
            .group_by(ModelVersion.id)
            .order_by(func.count(func.distinct(RelevancePseudoLabel.upload_id)).desc(), ModelVersion.id.desc())
            .limit(1)
        )
    ).first()
    if row is None:
        raise TeacherNotFoundError("No pseudo-labels yet -- run `python -m app.cli relevance pseudo-label`")
    return row[0]


def _words(segments: list[TranscriptSegment]):
    # Imported here: the handler module imports this package's pipeline.
    from app.worker.handlers.relevance import words_from_segments

    return words_from_segments(segments)


async def load_episodes(
    session_factory: async_sessionmaker[AsyncSession], teacher: ModelVersion
) -> tuple[list[Episode], list[tuple[int, str]]]:
    """(episodes, skipped as (upload_id, reason)), ordered by upload id."""
    episodes: list[Episode] = []
    skipped: list[tuple[int, str]] = []
    async with session_factory() as db:
        upload_ids = list(
            (
                await db.execute(
                    select(RelevancePseudoLabel.upload_id)
                    .where(RelevancePseudoLabel.teacher_version_id == teacher.id)
                    .distinct()
                    .order_by(RelevancePseudoLabel.upload_id)
                )
            ).scalars()
        )
        for upload_id in upload_ids:
            upload = await db.get(Upload, upload_id)
            labels = list(
                (
                    await db.execute(
                        select(RelevancePseudoLabel)
                        .where(RelevancePseudoLabel.upload_id == upload_id,
                               RelevancePseudoLabel.teacher_version_id == teacher.id)
                        .order_by(RelevancePseudoLabel.window_index)
                    )
                ).scalars()
            )
            segments = list(
                (
                    await db.execute(
                        select(TranscriptSegment)
                        .where(TranscriptSegment.upload_id == upload_id)
                        .order_by(TranscriptSegment.segment_index)
                        .options(selectinload(TranscriptSegment.words))
                    )
                ).scalars()
            )
            if not segments:
                skipped.append((upload_id, "no transcript"))
                continue
            duration = upload.duration_seconds or max(s.end_seconds for s in segments)
            windows = build_windows(build_blocks(_words(segments), duration))
            if [lab.window_index for lab in labels] != list(range(len(windows))):
                skipped.append((upload_id, f"{len(labels)} labels vs {len(windows)} windows (transcript changed?)"))
                continue
            language = await db.scalar(
                select(ShowNotesDraft.language)
                .where(ShowNotesDraft.upload_id == upload_id)
                .order_by(ShowNotesDraft.id.desc())
                .limit(1)
            )
            episodes.append(
                Episode(
                    upload_id=upload_id,
                    name=upload.filename,
                    language=language or "unknown",
                    windows=windows,
                    heuristic=np.array([lab.heuristic_score for lab in labels], dtype=np.float32),
                    llm=np.array([np.nan if lab.llm_score is None else lab.llm_score for lab in labels],
                                 dtype=np.float32),
                )
            )
    return episodes, skipped


# ---- split -----------------------------------------------------------------


def _allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Per-language share of n, proportional, with ≥ 1 for a language that has ≥ 2 episodes."""
    total = sum(counts.values())
    alloc = {lang: (max(1, round(n * c / total)) if c >= 2 else 0) for lang, c in counts.items()}
    biggest = max(counts, key=lambda k: (counts[k], k))
    alloc[biggest] += n - sum(alloc.values())
    return alloc


def make_split(items: list[tuple[int, str]], seed: int = SPLIT_SEED, n_val: int = N_VAL, n_test: int = N_TEST,
               n_ab: int = N_AB) -> dict:
    """Episodes (upload_id, language) -> {train, val, test, ab}; ab ⊂ test (0024 §4)."""
    if len(items) < n_val + n_test + 1:
        raise ValueError(f"Need at least {n_val + n_test + 1} labelled episodes, have {len(items)}")
    rng = random.Random(seed)
    by_lang: dict[str, list[int]] = {}
    for upload_id, lang in sorted(items):
        by_lang.setdefault(lang, []).append(upload_id)
    for ids in by_lang.values():
        rng.shuffle(ids)
    counts = {k: len(v) for k, v in by_lang.items()}
    test_alloc = _allocate(counts, n_test)
    rest = {k: counts[k] - test_alloc[k] for k in counts}
    val_alloc = _allocate(rest, n_val)
    test, val, train = [], [], []
    for lang in sorted(by_lang):
        ids = by_lang[lang]
        t, v = test_alloc[lang], val_alloc[lang]
        test += ids[:t]
        val += ids[t : t + v]
        train += ids[t + v :]
    ab = sorted(rng.sample(sorted(test), n_ab))
    return {"seed": seed, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "train": sorted(train), "val": sorted(val), "test": sorted(test), "ab": ab}


def split_hash(split: dict) -> str:
    frozen = {k: split[k] for k in ("seed", "val", "test", "ab")}
    return hashlib.sha1(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:12]


def load_or_make_split(path: Path, items: list[tuple[int, str]]) -> tuple[dict, bool]:
    """(split, created). An existing split keeps val/test/ab frozen; new episodes join train."""
    if path.is_file():
        split = json.loads(path.read_text())
        known = set(split["train"]) | set(split["val"]) | set(split["test"])
        new = sorted(i for i, _ in items if i not in known)
        split = {**split, "train": sorted(split["train"] + new)}
        return split, False
    split = make_split(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))
    return split, True


def partition(episodes: list[Episode], split: dict) -> dict[str, list[Episode]]:
    by_id = {e.upload_id: e for e in episodes}
    return {k: [by_id[i] for i in split[k] if i in by_id] for k in ("train", "val", "test", "ab")}
